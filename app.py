"""Trade Lab Terminal. Запуск: streamlit run app.py (на Windows через start.bat)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tradelab import (autotrade, backtest, config, demo, indicators, models, optimize, paper,
                      risk, store)
from tradelab.bcs import TIMEFRAMES, BcsClient, BcsError

C = dict(up="#26C281", down="#FF4D4F", accent="#FF9F1A", blue="#4DA3FF", violet="#B38CFF",
         grid="#1E2735", text="#C9D1DC", bg="#0A0D12", panel="#131923")

st.set_page_config(page_title="Trade Lab Terminal", layout="wide", page_icon="📈")
st.markdown(f"""
<style>
.block-container {{padding-top: 3.5rem; max-width: 1700px;}}
[data-testid="stMetricValue"] {{font-size: 1.45rem;}}
[data-testid="stMetric"] {{background: {C['panel']}; border-left: 3px solid {C['accent']};
    padding: 10px 14px; border-radius: 4px;}}
[data-testid="stMetricLabel"] {{color: #8B96A5;}}
.tape {{overflow: hidden; white-space: nowrap; background: {C['panel']}; border-bottom: 2px solid {C['accent']};
    padding: 7px 0; margin-bottom: 10px; font-size: 14px;}}
.tape-inner {{display: inline-block; animation: scroll 80s linear infinite;}}
.tape-inner span.item {{margin: 0 22px;}}
@keyframes scroll {{from {{transform: translateX(0);}} to {{transform: translateX(-50%);}}}}
.badge {{display:inline-block; padding:2px 10px; border-radius:3px; font-size:12px; font-weight:bold;}}
.title {{font-size: 26px; font-weight: bold; letter-spacing: 2px; color: {C['accent']};}}
.stTabs [data-baseweb="tab"] {{font-weight: bold; letter-spacing: 1px;}}
</style>""", unsafe_allow_html=True)


def style(fig, height=450):
    fig.update_layout(template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                      font=dict(family="monospace", color=C["text"]), height=height,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    return fig


def pct(x, d=1):
    return "н/д" if x is None or not np.isfinite(x) else f"{x:.{d}%}"


# ---------------------------------------------------------------- боковая панель
with st.sidebar:
    st.markdown(f"<div class='title'>TRADE LAB</div>", unsafe_allow_html=True)
    has_token = bool(config.get("BCS_REFRESH_TOKEN"))
    use_demo = st.toggle("Демо-режим (синтетические данные)", value=not has_token,
                         help="Случайные котировки для знакомства с интерфейсом. Не реальный рынок.")
    tf = st.selectbox("Таймфрейм", TIMEFRAMES, index=0)
    all_tickers = list(config.DEFAULT_TICKERS)
    extra = st.text_input("Свои тикеры (через запятую)")
    if extra:
        all_tickers += [t.strip().upper() for t in extra.split(",") if t.strip()]
    tickers = st.multiselect("Бумаги", all_tickers, default=config.DEFAULT_TICKERS)
    years = st.slider("Глубина загрузки, лет", 1, 10, 4)
    cost_bps = st.number_input("Комиссия + проскальзывание, б.п.", 0.0, 100.0, 5.0)
    allow_short = st.checkbox("Разрешить short в бэктесте", value=False)
    if not has_token and not use_demo:
        st.warning("Токен БКС не задан: скопируй .env.example в .env и вставь токен.")

if "version" not in st.session_state:
    st.session_state.version = 0
VERSION = st.session_state.version
tickers = tuple(tickers)


@st.cache_data(show_spinner=False)
def get_frames(is_demo, timeframe, tickers_, version):
    if is_demo:
        return demo.universe(list(tickers_))
    con = store.connect()
    out = {}
    for t in tickers_:
        df = store.load(con, t, timeframe)
        if len(df):
            out[t] = df
    return out


@st.cache_data(show_spinner=False)
def run_bt(is_demo, timeframe, ticker, model_name, cost, short, version):
    frames = get_frames(is_demo, timeframe, (ticker,), version)
    return backtest.walk_forward(frames[ticker], model_name, cost_bps=cost, allow_short=short)


@st.cache_data(show_spinner=False)
def run_signals(is_demo, timeframe, tickers_, model_name, cost, version):
    return autotrade.compute_signals(get_frames(is_demo, timeframe, tickers_, version), model_name, cost)


frames = get_frames(use_demo, tf, tickers, VERSION)
if use_demo and tf != "D":
    st.sidebar.info("В демо-режиме есть только дневные свечи.")

broker = paper.PaperBroker(config.DATA_DIR / ("paper_demo.db" if use_demo else "paper.db"))
last_prices = {t: float(f["close"].iloc[-1]) for t, f in frames.items()}


def stats_table(frames):
    rows = []
    for t, f in frames.items():
        c = f["close"]
        if len(c) < 30:
            continue
        r = c.pct_change()
        rows.append({
            "Тикер": t, "Цена": float(c.iloc[-1]),
            "1д %": float(c.iloc[-1] / c.iloc[-2] - 1) * 100,
            "5д %": float(c.iloc[-1] / c.iloc[-6] - 1) * 100,
            "1м %": float(c.iloc[-1] / c.iloc[-22] - 1) * 100,
            "Вол-ть 20д %": float(r.tail(20).std() * np.sqrt(252)) * 100,
            "RSI": float(indicators.rsi(c).iloc[-1]),
            "Оборот (оценка)": float((c * f["volume"]).tail(20).mean()),
            "Тренд 60": c.tail(60).tolist(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- шапка
badge = (f"<span class='badge' style='background:{C['violet']};color:#000'>DEMO · СИНТЕТИКА</span>"
         if use_demo else f"<span class='badge' style='background:{C['up']};color:#000'>БКС · РЕАЛЬНЫЕ ДАННЫЕ</span>")
st.markdown(f"<span class='title'>TRADE LAB TERMINAL</span> &nbsp; {badge} &nbsp; "
            f"<span style='color:#8B96A5'>таймфрейм {tf} · бумаг в базе: {len(frames)} · "
            f"{datetime.now():%d.%m.%Y %H:%M}</span>", unsafe_allow_html=True)

ST = stats_table(frames)
if len(ST):
    items = "".join(
        f"<span class='item'><span style='color:#8B96A5'>{r['Тикер']}</span> <b>{r['Цена']:.2f}</b> "
        f"<span style='color:{C['up'] if r['1д %'] >= 0 else C['down']}'>"
        f"{'▲' if r['1д %'] >= 0 else '▼'}{abs(r['1д %']):.2f}%</span></span>" for _, r in ST.iterrows())
    st.markdown(f"<div class='tape'><div class='tape-inner'>{items}{items}</div></div>", unsafe_allow_html=True)

tabs = st.tabs(["ОБЗОР", "ГРАФИК", "РИСК", "ОПТИМИЗАТОР", "МОДЕЛИ", "СКАНЕР", "АВТОТОРГОВЛЯ", "ДАННЫЕ"])
(tab_ov, tab_chart, tab_risk, tab_opt, tab_models, tab_scan, tab_auto, tab_data) = tabs

# ---------------------------------------------------------------- ОБЗОР
with tab_ov:
    if not len(ST):
        st.info("Нет данных. Включи демо-режим слева или загрузи данные на вкладке «ДАННЫЕ».")
    else:
        up = int((ST["1д %"] > 0).sum())
        m = st.columns(5)
        m[0].metric("Растут / падают", f"{up} / {len(ST) - up}")
        m[1].metric("Средняя доходность 1д", f"{ST['1д %'].mean():+.2f}%")
        m[2].metric("Средняя доходность 1м", f"{ST['1м %'].mean():+.2f}%")
        m[3].metric("Средняя волатильность", f"{ST['Вол-ть 20д %'].mean():.1f}%")
        m[4].metric("Перекупленных RSI>70", int((ST["RSI"] > 70).sum()))
        left, right = st.columns([5, 6])
        with left:
            tm = go.Figure(go.Treemap(
                labels=ST["Тикер"], parents=[""] * len(ST), values=ST["Оборот (оценка)"],
                text=[f"{t}<br>{d:+.2f}%" for t, d in zip(ST["Тикер"], ST["1д %"])],
                textinfo="text", marker=dict(
                    colors=ST["1д %"], cmid=0, cmin=-3, cmax=3, line=dict(width=1, color=C["bg"]),
                    colorscale=[[0, C["down"]], [0.5, "#1B2230"], [1, C["up"]]]),
                hovertemplate="%{label}<br>%{text}<extra></extra>"))
            st.markdown("**Тепловая карта рынка** (размер = оборот, цвет = изменение за день)")
            st.plotly_chart(style(tm, 430), width="stretch")
        with right:
            st.markdown("**Монитор бумаг**")
            st.dataframe(
                ST.drop(columns=["Оборот (оценка)"]), hide_index=True, width="stretch", height=430,
                column_config={
                    "Цена": st.column_config.NumberColumn(format="%.2f"),
                    "1д %": st.column_config.NumberColumn(format="%+.2f"),
                    "5д %": st.column_config.NumberColumn(format="%+.2f"),
                    "1м %": st.column_config.NumberColumn(format="%+.2f"),
                    "Вол-ть 20д %": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=80),
                    "RSI": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
                    "Тренд 60": st.column_config.LineChartColumn("Тренд 60 баров"),
                })

# ---------------------------------------------------------------- ГРАФИК
with tab_chart:
    if not frames:
        st.info("Нет данных.")
    else:
        c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
        ticker = c1.selectbox("Бумага", list(frames), key="chart_ticker")
        n_bars = c2.slider("Баров на графике", 60, 900, 250)
        overlays = c3.multiselect("Наложения", ["SMA 20", "SMA 50", "Bollinger 20"], default=["SMA 20", "SMA 50"])
        lower = c4.selectbox("Нижняя панель", ["RSI", "MACD", "Нет"])
        df = frames[ticker]
        close = df["close"]
        view = df.tail(n_bars)
        rows = 3 if lower != "Нет" else 2
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            row_heights=[0.62, 0.14, 0.24][:rows])
        fig.add_trace(go.Candlestick(
            x=view["ts"], open=view["open"], high=view["high"], low=view["low"], close=view["close"],
            increasing_line_color=C["up"], decreasing_line_color=C["down"], name=ticker), row=1, col=1)
        if "SMA 20" in overlays:
            fig.add_scatter(x=df["ts"].tail(n_bars), y=indicators.sma(close, 20).tail(n_bars),
                            name="SMA 20", line=dict(color=C["accent"], width=1.2), row=1, col=1)
        if "SMA 50" in overlays:
            fig.add_scatter(x=df["ts"].tail(n_bars), y=indicators.sma(close, 50).tail(n_bars),
                            name="SMA 50", line=dict(color=C["blue"], width=1.2), row=1, col=1)
        if "Bollinger 20" in overlays:
            lo, mid, hi = indicators.bollinger(close)
            for band, nm in ((lo, "BB low"), (hi, "BB high")):
                fig.add_scatter(x=df["ts"].tail(n_bars), y=band.tail(n_bars), name=nm,
                                line=dict(color=C["violet"], width=1, dash="dot"), row=1, col=1)
        vol_colors = np.where(view["close"] >= view["open"], C["up"], C["down"])
        fig.add_bar(x=view["ts"], y=view["volume"], marker_color=vol_colors, name="Объём", row=2, col=1)
        if lower == "RSI":
            fig.add_scatter(x=df["ts"].tail(n_bars), y=indicators.rsi(close).tail(n_bars), name="RSI",
                            line=dict(color=C["accent"]), row=3, col=1)
            for lvl in (30, 70):
                fig.add_hline(y=lvl, line=dict(color="#555", dash="dash"), row=3, col=1)
        elif lower == "MACD":
            line, sig, hist = indicators.macd(close)
            fig.add_scatter(x=df["ts"].tail(n_bars), y=line.tail(n_bars), name="MACD",
                            line=dict(color=C["blue"]), row=3, col=1)
            fig.add_scatter(x=df["ts"].tail(n_bars), y=sig.tail(n_bars), name="Signal",
                            line=dict(color=C["accent"]), row=3, col=1)
            fig.add_bar(x=df["ts"].tail(n_bars), y=hist.tail(n_bars), name="Гистограмма",
                        marker_color=np.where(hist.tail(n_bars) >= 0, C["up"], C["down"]), row=3, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified")
        if tf in ("D", "W", "MN"):
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(style(fig, 720), width="stretch")

# ---------------------------------------------------------------- РИСК
with tab_risk:
    if len(frames) < 2:
        st.info("Нужно минимум две бумаги с данными.")
    else:
        R = risk.returns_frame(frames)
        src = st.radio("Портфель", ["Paper-счёт", "Равные веса", "Свои веса"], horizontal=True)
        equity_now = broker.equity(last_prices)
        if src == "Paper-счёт":
            pos = broker.positions()
            weights = {t: q * last_prices.get(t, p) / equity_now for t, (q, p) in pos.items()}
            if not weights:
                st.info("На paper-счёте пока нет позиций: показываю равные веса. "
                        "Позиции появятся после ребаланса на вкладке «АВТОТОРГОВЛЯ».")
                weights = {t: 1 / len(R.columns) for t in R.columns}
        elif src == "Равные веса":
            weights = {t: 1 / len(R.columns) for t in R.columns}
        else:
            ed = st.data_editor(pd.DataFrame({"Тикер": R.columns, "Вес": 1 / len(R.columns)}),
                                hide_index=True, key="w_editor", disabled=["Тикер"])
            total = ed["Вес"].sum() or 1
            weights = dict(zip(ed["Тикер"], ed["Вес"] / total))
        capital = st.number_input("Капитал для расчёта потерь, ₽", 10_000.0, 1e10, float(equity_now), 10_000.0)
        pr = risk.portfolio_returns(R, weights)
        s = risk.summary(pr)
        bench = R.mean(axis=1)
        m = st.columns(6)
        m[0].metric("Доходность годовая", pct(s["ann_return"]))
        m[1].metric("Волатильность", pct(s["ann_vol"]))
        m[2].metric("Шарп (rf=0)", f"{s['sharpe']:.2f}")
        m[3].metric("Макс. просадка", pct(s["max_drawdown"]))
        m[4].metric("VaR 95% (1д)", f"{s['var95'] * capital:,.0f} ₽", pct(-s["var95"], 2), delta_color="off")
        m[5].metric("CVaR 95% (1д)", f"{s['cvar95'] * capital:,.0f} ₽", pct(-s["cvar95"], 2), delta_color="off")
        st.caption(f"VaR 99%: {s['var99'] * capital:,.0f} ₽ · CVaR 99%: {s['cvar99'] * capital:,.0f} ₽ · "
                   f"бета к равновзвешенному индексу: {risk.beta(pr, bench):.2f}. "
                   "Историческая оценка: прошлое не гарантирует будущее.")
        a, b = st.columns(2)
        with a:
            rc = risk.risk_contributions(R, weights)
            w = pd.Series(weights).reindex(R.columns).fillna(0)
            show = pd.DataFrame({"Вес": w, "Вклад в риск": rc})
            show = show[(show["Вес"] > 0) | (show["Вклад в риск"].abs() > 0)].sort_values("Вклад в риск", ascending=False)
            fig = go.Figure()
            fig.add_bar(x=show.index, y=show["Вес"], name="Вес", marker_color=C["blue"])
            fig.add_bar(x=show.index, y=show["Вклад в риск"], name="Вклад в риск", marker_color=C["accent"])
            fig.update_layout(barmode="group", yaxis_tickformat=".0%")
            st.markdown("**Вес против вклада в риск**: если оранжевое сильно выше синего, бумага «тянет» риск")
            st.plotly_chart(style(fig, 360), width="stretch")
        with b:
            corr = R.corr()
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
                                       colorscale=[[0, C["blue"]], [0.5, "#1B2230"], [1, C["down"]]]))
            st.markdown("**Корреляции доходностей**")
            st.plotly_chart(style(fig, 360), width="stretch")
        a, b = st.columns(2)
        with a:
            paths = risk.bootstrap_paths(pr, 21) * capital
            days = np.arange(1, paths.shape[1] + 1)
            q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
            fig = go.Figure()
            fig.add_scatter(x=days, y=q[4], line=dict(width=0), showlegend=False)
            fig.add_scatter(x=days, y=q[0], fill="tonexty", fillcolor="rgba(255,159,26,0.15)",
                            line=dict(width=0), name="5–95%")
            fig.add_scatter(x=days, y=q[3], line=dict(width=0), showlegend=False)
            fig.add_scatter(x=days, y=q[1], fill="tonexty", fillcolor="rgba(255,159,26,0.30)",
                            line=dict(width=0), name="25–75%")
            fig.add_scatter(x=days, y=q[2], line=dict(color=C["accent"], width=2), name="Медиана")
            fig.update_layout(xaxis_title="Торговых дней вперёд", yaxis_title="Капитал, ₽")
            st.markdown("**Веер сценариев на 21 день** (бутстрап исторических дней)")
            st.plotly_chart(style(fig, 360), width="stretch")
        with b:
            table, port_beta = risk.stress_table(R, weights)
            table["Потери, ₽"] = table["Потери портфеля"] * capital
            st.markdown(f"**Стресс-тест** (бета портфеля {port_beta:.2f})")
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "Потери портфеля": st.column_config.NumberColumn(format="percent"),
                "Потери, ₽": st.column_config.NumberColumn(format="localized")})
            worst = risk.worst_days(pr, 5)
            st.markdown("**5 худших дней в истории портфеля**")
            st.dataframe(pd.DataFrame({"Дата": worst.index.strftime("%Y-%m-%d"), "Доходность": worst.values}),
                         hide_index=True, width="stretch",
                         column_config={"Доходность": st.column_config.NumberColumn(format="percent")})

# ---------------------------------------------------------------- ОПТИМИЗАТОР
with tab_opt:
    if len(frames) < 3:
        st.info("Нужно минимум три бумаги с данными.")
    else:
        c1, c2, c3 = st.columns([2, 2, 3])
        objective = c1.selectbox("Цель", ["min_vol", "max_sharpe", "risk_parity", "equal"], format_func=lambda x: {
            "min_vol": "Минимальный риск", "max_sharpe": "Максимальный Шарп",
            "risk_parity": "Равный вклад в риск", "equal": "Равные веса"}[x])
        max_w = c2.slider("Макс. вес одной бумаги", 0.05, 1.0, 0.30, 0.05)
        subset = c3.multiselect("Бумаги", list(frames), default=list(frames))
        if len(subset) >= 2:
            R2 = risk.returns_frame({t: frames[t] for t in subset})
            w_opt = optimize.optimize(R2, objective, max_weight=max_w)
            w_eq = optimize.optimize(R2, "equal")
            ret_o, vol_o = optimize.portfolio_point(R2, w_opt)
            ret_e, vol_e = optimize.portfolio_point(R2, w_eq)
            m = st.columns(4)
            m[0].metric("Доходность (оптим.)", pct(ret_o), f"{(ret_o - ret_e):+.1%} к равным", delta_color="off")
            m[1].metric("Волатильность (оптим.)", pct(vol_o), f"{(vol_o - vol_e):+.1%} к равным", delta_color="off")
            m[2].metric("Шарп (оптим.)", f"{ret_o / vol_o:.2f}")
            m[3].metric("Бумаг с весом > 1%", sum(v > 0.01 for v in w_opt.values()))
            a, b = st.columns(2)
            with a:
                ws = pd.Series(w_opt).sort_values(ascending=False)
                fig = go.Figure(go.Bar(x=ws.index, y=ws.values, marker_color=C["accent"]))
                fig.update_layout(yaxis_tickformat=".0%")
                st.markdown("**Предлагаемые веса**")
                st.plotly_chart(style(fig, 380), width="stretch")
            with b:
                pts = optimize.random_portfolios(R2)
                fig = go.Figure()
                fig.add_scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
                                marker=dict(size=4, color=[p[1] / p[0] for p in pts], colorscale="Viridis",
                                            showscale=False), name="Случайные портфели")
                fig.add_scatter(x=[vol_e], y=[ret_e], mode="markers", name="Равные веса",
                                marker=dict(size=13, color=C["blue"], symbol="diamond"))
                fig.add_scatter(x=[vol_o], y=[ret_o], mode="markers", name="Оптимум",
                                marker=dict(size=16, color=C["accent"], symbol="star"))
                fig.update_layout(xaxis_title="Риск (волатильность)", yaxis_title="Ожидаемая доходность")
                fig.update_xaxes(tickformat=".0%")
                fig.update_yaxes(tickformat=".0%")
                st.markdown("**Риск и доходность** (ожидаемая доходность = историческая, ненадёжна)")
                st.plotly_chart(style(fig, 380), width="stretch")
            if st.button("Применить к paper-счёту (ребаланс)"):
                done = broker.rebalance(w_opt, last_prices, max_weight=max_w, note=f"optimizer:{objective}")
                st.success(f"Сделок: {len(done)}. Результат на вкладке «АВТОТОРГОВЛЯ».")

# ---------------------------------------------------------------- МОДЕЛИ
with tab_models:
    if not frames:
        st.info("Нет данных.")
    else:
        c1, c2 = st.columns(2)
        ticker = c1.selectbox("Бумага", list(frames), key="bt_ticker")
        model_name = c2.selectbox("Модель", list(models.MODELS), index=2)
        try:
            with st.spinner("Считаю walk-forward…"):
                res = run_bt(use_demo, tf, ticker, model_name, cost_bps, allow_short, VERSION)
            met = backtest.metrics(res)
            cols = st.columns(8)
            fmt = {"Баров вне выборки": lambda v: str(v), "IC (Спирмен)": lambda v: f"{v:.3f}",
                   "Шарп": lambda v: f"{v:.2f}"}
            for col, (k, v) in zip(cols, met.items()):
                col.metric(k, fmt.get(k, pct)(v))
            fig = go.Figure()
            fig.add_scatter(x=res["ts"], y=res["equity"], name="Стратегия", line=dict(color=C["accent"], width=2))
            fig.add_scatter(x=res["ts"], y=res["equity_bh"], name="Buy&hold", line=dict(color=C["blue"], width=1.5))
            fig.update_layout(yaxis_title="Капитал (старт = 1)")
            st.plotly_chart(style(fig, 420), width="stretch")
            value, ts = backtest.latest_forecast(frames[ticker], model_name)
            st.metric(f"Прогноз доходности следующего бара (после {ts:%Y-%m-%d})", f"{value:+.2%}")
            with st.expander("Сравнить все модели на этой бумаге"):
                rows, fig = [], go.Figure()
                for name in models.MODELS:
                    r_ = run_bt(use_demo, tf, ticker, name, cost_bps, allow_short, VERSION)
                    mm = backtest.metrics(r_)
                    rows.append({"Модель": name, "IC": mm["IC (Спирмен)"], "Верных знаков": mm["Доля верных знаков"],
                                 "Стратегия": mm["Доходность стратегии"], "Шарп": mm["Шарп"],
                                 "Просадка": mm["Макс. просадка"]})
                    fig.add_scatter(x=r_["ts"], y=r_["equity"], name=name)
                fig.add_scatter(x=res["ts"], y=res["equity_bh"], name="Buy&hold", line=dict(color="#888", dash="dot"))
                st.dataframe(pd.DataFrame(rows).style.format({
                    "IC": "{:.3f}", "Верных знаков": "{:.1%}", "Стратегия": "{:.1%}", "Шарп": "{:.2f}",
                    "Просадка": "{:.1%}"}), hide_index=True, width="stretch")
                st.plotly_chart(style(fig, 380), width="stretch")
            st.caption("IC около нуля и доля верных знаков около 50% означают, что модель не лучше случайной. "
                       "Это нормальный и частый результат.")
        except ValueError as exc:
            st.warning(str(exc))

# ---------------------------------------------------------------- СКАНЕР
with tab_scan:
    model_name = st.selectbox("Модель для сканирования", list(models.MODELS), index=2, key="scan_model")
    if st.button("Прогнать по всем бумагам"):
        with st.spinner("Считаю…"):
            sig = run_signals(use_demo, tf, tickers, model_name, cost_bps, VERSION)
        if sig.empty:
            st.info("Нет данных для сканирования.")
        else:
            table = sig.sort_values("ic", ascending=False)
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "ticker": "Тикер", "forecast": st.column_config.NumberColumn("Прогноз", format="percent"),
                "ic": st.column_config.NumberColumn("IC", format="%.3f"),
                "hit": st.column_config.NumberColumn("Верных знаков", format="percent"),
                "sharpe": st.column_config.NumberColumn("Шарп", format="%.2f"),
                "price": st.column_config.NumberColumn("Цена", format="%.2f")})
            st.warning("Осторожно: из 20 бумаг хоть одна почти всегда покажет «хороший» результат случайно. "
                       "Смотри на устойчивость IC по многим бумагам и периодам, а не на лучшую строку.")

# ---------------------------------------------------------------- АВТОТОРГОВЛЯ
with tab_auto:
    st.markdown(f"<div style='background:{C['panel']};border-left:3px solid {C['violet']};padding:10px 14px;"
                f"border-radius:4px'><b>Бумажный счёт.</b> Сделки симулируются локально по последним ценам из базы. "
                f"Реальные заявки в БКС не отправляются, токен только на чтение.</div>", unsafe_allow_html=True)
    eq = broker.equity(last_prices)
    pnl = eq - broker.start_cash
    m = st.columns(5)
    m[0].metric("Капитал", f"{eq:,.0f} ₽", f"{pnl / broker.start_cash:+.2%}")
    m[1].metric("Свободные деньги", f"{broker.cash:,.0f} ₽")
    m[2].metric("Позиций", len(broker.positions()))
    m[3].metric("Старт", f"{broker.start_cash:,.0f} ₽")
    killed = broker.kill_switch()
    m[4].metric("Аварийный стоп", "ВКЛЮЧЁН" if killed else "выключен")

    left, right = st.columns([3, 2])
    with left:
        c1, c2, c3, c4 = st.columns(4)
        model_name = c1.selectbox("Модель сигналов", list(models.MODELS), index=2, key="auto_model")
        top_n = c2.number_input("Макс. бумаг", 1, 20, 5)
        min_ic = c3.number_input("Мин. IC", -0.1, 0.5, 0.0, 0.01)
        max_w = c4.slider("Макс. вес бумаги", 0.05, 1.0, 0.25, 0.05)
        if st.button("1. Рассчитать сигналы", type="primary"):
            with st.spinner("Считаю сигналы по всем бумагам…"):
                sig = run_signals(use_demo, tf, tickers, model_name, cost_bps, VERSION)
            st.session_state["signals"] = sig
        sig = st.session_state.get("signals")
        if sig is not None and len(sig):
            tw = autotrade.target_weights(sig, int(top_n), float(min_ic), float(max_w))
            sig = sig.assign(target=[tw.get(t, 0.0) for t in sig["ticker"]]).sort_values("target", ascending=False)
            st.dataframe(sig, hide_index=True, width="stretch", column_config={
                "ticker": "Тикер", "forecast": st.column_config.NumberColumn("Прогноз", format="percent"),
                "ic": st.column_config.NumberColumn("IC", format="%.3f"),
                "hit": st.column_config.NumberColumn("Верных знаков", format="percent"),
                "sharpe": st.column_config.NumberColumn("Шарп", format="%.2f"),
                "price": st.column_config.NumberColumn("Цена", format="%.2f"),
                "target": st.column_config.ProgressColumn("Целевой вес", format="percent", min_value=0, max_value=1)})
            if st.button("2. Выполнить ребаланс на paper-счёте"):
                try:
                    done = broker.rebalance(tw, last_prices, max_weight=float(max_w), note=f"signals:{model_name}")
                    st.success(f"Исполнено сделок: {len(done)}")
                except ValueError as exc:
                    st.error(str(exc))
            if not tw:
                st.info("Ни одна бумага не прошла фильтры: ребаланс переведёт всё в деньги. Это нормально, "
                        "если модели не находят устойчивого преимущества.")
    with right:
        st.markdown("**Управление**")
        if st.button("Выключить аварийный стоп" if killed else "АВАРИЙНЫЙ СТОП (блок сделок)"):
            broker.set_kill_switch(not killed)
            st.rerun()
        if st.button("Записать текущую оценку капитала"):
            broker.record_equity(last_prices)
            st.rerun()
        with st.expander("Сбросить paper-счёт"):
            new_cash = st.number_input("Стартовый капитал, ₽", 10_000.0, 1e10, 1_000_000.0, 10_000.0)
            if st.button("Сбросить"):
                broker.reset(new_cash)
                st.rerun()
        ec = broker.equity_curve()
        if len(ec) >= 2:
            fig = go.Figure(go.Scatter(x=ec["ts"], y=ec["equity"], line=dict(color=C["accent"], width=2)))
            fig.update_layout(title="Капитал paper-счёта")
            st.plotly_chart(style(fig, 260), width="stretch")

    pos = broker.positions()
    if pos:
        st.markdown("**Позиции**")
        prow = [{"Тикер": t, "Кол-во": q, "Ср. цена": p, "Цена": last_prices.get(t, p),
                 "Стоимость": q * last_prices.get(t, p), "P&L %": last_prices.get(t, p) / p - 1}
                for t, (q, p) in pos.items()]
        st.dataframe(pd.DataFrame(prow), hide_index=True, width="stretch", column_config={
            "Ср. цена": st.column_config.NumberColumn(format="%.2f"), "Цена": st.column_config.NumberColumn(format="%.2f"),
            "Стоимость": st.column_config.NumberColumn(format="localized"),
            "P&L %": st.column_config.NumberColumn(format="percent")})
    with st.expander("Журнал сделок"):
        st.dataframe(broker.orders(), hide_index=True, width="stretch")
    st.caption("Автоматический режим без браузера: `python scripts/autotrader.py` обновляет данные из БКС "
               "и делает ребаланс paper-счёта по расписанию.")

# ---------------------------------------------------------------- ДАННЫЕ
with tab_data:
    if use_demo:
        st.info("Включён демо-режим: данные синтетические и в базу не пишутся. "
                "Выключи его слева, чтобы работать с настоящими данными БКС.")
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
            con = store.connect()
            progress = st.progress(0.0)
            for i, ticker in enumerate(tickers):
                last = store.last_ts(con, ticker, tf)
                start = (last.to_pydatetime() if last is not None
                         else datetime.now(timezone.utc) - timedelta(days=365 * years))
                df = client.candles(ticker, start, timeframe=tf)
                n = store.upsert(con, ticker, tf, df) if not df.empty else 0
                st.write(f"{ticker}: {n} баров")
                progress.progress((i + 1) / len(tickers))
            st.session_state.version += 1
        except BcsError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Ошибка загрузки: {exc!r}")
    st.subheader("Что лежит в локальной базе")
    st.dataframe(store.summary(store.connect()), hide_index=True, width="stretch")
