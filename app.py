"""Интерфейс: streamlit run app.py (на Windows запускается через start.bat)."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tradelab import backtest, config, models, store
from tradelab.bcs import TIMEFRAMES, BcsClient, BcsError

st.set_page_config(page_title="Trade Lab", layout="wide")
st.title("Trade Lab: анализ и прогнозы по данным БКС")
st.caption("Исследовательский инструмент. Не торгует и не даёт инвестиционных рекомендаций: "
           "здесь можно проверять модели и смотреть, насколько они вообще работают.")

con = store.connect()

with st.sidebar:
    st.header("Настройки")
    tf = st.selectbox("Таймфрейм", TIMEFRAMES, index=0)
    tickers = st.multiselect("Бумаги", config.DEFAULT_TICKERS, default=config.DEFAULT_TICKERS)
    extra = st.text_input("Добавить свои тикеры (через запятую)")
    if extra:
        tickers += [t.strip().upper() for t in extra.split(",") if t.strip()]
    years = st.slider("Глубина загрузки, лет", 1, 10, 4)
    cost_bps = st.number_input("Комиссия + проскальзывание, б.п. за смену позиции", 0.0, 100.0, 5.0)
    allow_short = st.checkbox("Разрешить short", value=False)
    if not config.get("BCS_REFRESH_TOKEN"):
        st.warning("Токен БКС не задан: скопируй .env.example в .env и вставь токен.")

tab_data, tab_chart, tab_bt, tab_scan = st.tabs(["Данные", "График", "Бэктест", "Сканер"])

with tab_data:
    c1, c2 = st.columns(2)
    if c1.button("Проверить подключение к БКС"):
        try:
            client = BcsClient()
            client._access_token()
            end = datetime.now(timezone.utc)
            test = client.candles("SBER", end - timedelta(days=10), end)
            st.success(f"Подключение работает, получено баров SBER: {len(test)}")
        except BcsError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Сеть или формат ответа: {exc!r}")
    if c2.button("Загрузить / обновить данные", type="primary"):
        try:
            client = BcsClient()
            progress = st.progress(0.0)
            for i, ticker in enumerate(tickers):
                last = store.last_ts(con, ticker, tf)
                start = (last.to_pydatetime() if last is not None
                         else datetime.now(timezone.utc) - timedelta(days=365 * years))
                df = client.candles(ticker, start, timeframe=tf)
                n = store.upsert(con, ticker, tf, df) if not df.empty else 0
                st.write(f"{ticker}: {n} баров")
                progress.progress((i + 1) / len(tickers))
        except BcsError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Ошибка загрузки: {exc!r}")
    st.subheader("Что лежит в локальной базе")
    st.dataframe(store.summary(con), use_container_width=True)


def load(ticker):
    df = store.load(con, ticker, tf)
    return df if not df.empty else None


def pct(x):
    return f"{x:.1%}"


with tab_chart:
    ticker = st.selectbox("Бумага", tickers, key="chart_ticker")
    df = load(ticker)
    if df is None:
        st.info("Нет данных: загрузи их на вкладке «Данные».")
    else:
        fig = go.Figure(go.Candlestick(x=df["ts"], open=df["open"], high=df["high"],
                                       low=df["low"], close=df["close"]))
        fig.update_layout(xaxis_rangeslider_visible=False, height=520)
        st.plotly_chart(fig, use_container_width=True)

with tab_bt:
    c1, c2 = st.columns(2)
    ticker = c1.selectbox("Бумага", tickers, key="bt_ticker")
    model_name = c2.selectbox("Модель", list(models.MODELS), index=2)
    df = load(ticker)
    if df is None:
        st.info("Нет данных: загрузи их на вкладке «Данные».")
    else:
        try:
            res = backtest.walk_forward(df, model_name, cost_bps=cost_bps, allow_short=allow_short)
            m = backtest.metrics(res)
            cols = st.columns(4)
            for i, (k, v) in enumerate(m.items()):
                cols[i % 4].metric(k, str(v) if k == "Баров вне выборки"
                                   else (f"{v:.3f}" if k in ("IC (Спирмен)", "Шарп") else pct(v)))
            fig = go.Figure()
            fig.add_scatter(x=res["ts"], y=res["equity"], name="Стратегия")
            fig.add_scatter(x=res["ts"], y=res["equity_bh"], name="Buy&hold")
            fig.update_layout(height=450, yaxis_title="Капитал (старт = 1)")
            st.plotly_chart(fig, use_container_width=True)
            value, ts = backtest.latest_forecast(df, model_name)
            st.metric(f"Прогноз доходности следующего бара (после {ts:%Y-%m-%d})", f"{value:+.2%}")
            st.caption("IC около нуля и доля верных знаков около 50% означают, что модель "
                       "не лучше случайной. Это нормальный и частый результат.")
        except ValueError as exc:
            st.warning(str(exc))

with tab_scan:
    model_name = st.selectbox("Модель для сканирования", list(models.MODELS), index=2, key="scan_model")
    if st.button("Прогнать по всем бумагам"):
        rows = []
        progress = st.progress(0.0)
        for i, ticker in enumerate(tickers):
            df = load(ticker)
            if df is not None:
                try:
                    res = backtest.walk_forward(df, model_name, cost_bps=cost_bps, allow_short=allow_short)
                    m = backtest.metrics(res)
                    value, _ = backtest.latest_forecast(df, model_name)
                    rows.append({"Тикер": ticker, "Прогноз": value, "IC": m["IC (Спирмен)"],
                                 "Верных знаков": m["Доля верных знаков"],
                                 "Стратегия": m["Доходность стратегии"],
                                 "Buy&hold": m["Доходность buy&hold"], "Шарп": m["Шарп"],
                                 "Просадка": m["Макс. просадка"]})
                except ValueError:
                    pass
            progress.progress((i + 1) / len(tickers))
        if rows:
            table = pd.DataFrame(rows).sort_values("IC", ascending=False)
            st.dataframe(table.style.format({
                "Прогноз": "{:+.2%}", "IC": "{:.3f}", "Верных знаков": "{:.1%}",
                "Стратегия": "{:.1%}", "Buy&hold": "{:.1%}", "Шарп": "{:.2f}", "Просадка": "{:.1%}"}),
                use_container_width=True)
            st.warning("Осторожно: из 20 бумаг хоть одна почти всегда покажет «хороший» результат "
                       "случайно. Смотри на устойчивость IC по многим бумагам и периодам, "
                       "а не на лучшую строку.")
        else:
            st.info("Нет данных для сканирования.")
