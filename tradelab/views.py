"""Вкладки мировой аналитики: МИР (глобус), МАКРО, ОТЧЁТНОСТЬ, СТРУКТУРА."""
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from . import fundamentals, globe, markets, optimize, risk, store, structure, worlddata
from .ui import C, PALETTE, pct, style


def _changes(prices):
    """Изменения за 1д / 5д / 1м / с начала года по таблице цен (колонки - инструменты)."""
    p = prices.ffill()
    last = p.iloc[-1]
    ytd = p[p.index >= pd.Timestamp(datetime.now().year, 1, 1, tz="UTC")]
    return pd.DataFrame({
        "1д": last / p.iloc[-2] - 1, "5д": last / p.iloc[-6] - 1, "1м": last / p.iloc[-22] - 1,
        "С начала года": last / (ytd.iloc[0] if len(ytd) else p.iloc[0]) - 1})


def _empty(msg="Нет данных. Включи демо-режим слева или загрузи данные на вкладке «ДАННЫЕ»."):
    st.info(msg)


# ================================================================ МИР
def view_world(world, is_demo):
    idx = world["index"]
    if idx.empty or len(idx.columns) < 3:
        return _empty()
    R = worlddata.returns(idx)
    chg = _changes(idx)
    now = datetime.now(timezone.utc)
    status = {n: markets.market_status(n, now) for n in idx.columns if n in markets.MARKETS}

    c1, c2, c3 = st.columns([3, 3, 2])
    paint = c1.selectbox("Окраска стран", ["Изменение за день", "Изменение за месяц", "Ключевая ставка ЦБ",
                                           "Инфляция", "Рост ВВП", "Безработица"])
    flow = c2.selectbox("Потоки", ["Кто кого ведёт (lead-lag)", "Сильнейшие корреляции", "Нет"])
    n_arcs = c3.slider("Число потоков", 4, 30, 14)

    names = [n for n in idx.columns if n in markets.MARKETS]
    mk = []
    for n in names:
        iso, _, _, city, lat, lon, tz, *_ = markets.MARKETS[n]
        d1 = float(chg.loc[n, "1д"]) if n in chg.index else float("nan")
        mk.append({"name": n, "iso": iso, "city": city, "lat": lat, "lon": lon,
                   "color": globe.diverging(d1, 0.03) or "#7a8697", "open": bool(status[n][0]),
                   "label": f"{idx[n].iloc[-1]:,.2f} · 1д {d1:+.2%} · 1м {chg.loc[n, '1м']:+.2%}"})

    colors, legend = {}, []
    if paint.startswith("Изменение"):
        col = "1д" if paint.endswith("день") else "1м"
        scale = 0.03 if col == "1д" else 0.10
        for n in names:
            colors[markets.MARKETS[n][0]] = globe.diverging(float(chg.loc[n, col]), scale)
        legend = [("рынок растёт", C["up"]), ("рынок падает", C["down"])]
    elif paint == "Ключевая ставка ЦБ":
        rt = world["rates"]
        vals = {iso: float(rt[name].dropna().iloc[-1]) for name, (_, iso) in markets.POLICY_RATES.items()
                if name in rt.columns and rt[name].notna().any()}
        if vals:
            lo, hi = min(vals.values()), max(vals.values())
            colors = {iso: globe.sequential(v, lo, hi) for iso, v in vals.items()}
            legend = [(f"ставка {lo:.1f}% (тёмный)", "#1e2634"), (f"ставка {hi:.1f}% (яркий)", C["accent"])]
    else:
        mc = world["macro"]
        sub = mc[mc["indicator"] == {"Инфляция": "Инфляция, %", "Рост ВВП": "Рост ВВП, %",
                                    "Безработица": "Безработица, %"}[paint]]
        if len(sub):
            latest = sub.sort_values("year").groupby("country")["value"].last()
            lo, hi = float(latest.min()), float(latest.max())
            colors = {iso: globe.sequential(v, lo, hi) for iso, v in latest.items()}
            legend = [(f"{lo:.1f}%", "#1e2634"), (f"{hi:.1f}%", C["accent"])]

    arcs = []
    if flow != "Нет" and len(names) > 3:
        Rn = R[names]
        pos = {n: i for i, n in enumerate(names)}
        if flow.startswith("Кто"):
            L = structure.lead_lag(Rn)
            net = (L - L.T).stack()
            net = net[net > 0].nlargest(n_arcs)
            top = float(net.max()) if len(net) else 1.0
            for (a, b), v in net.items():
                arcs.append({"a": pos[a], "b": pos[b], "w": float(v / top), "color": C["accent"],
                             "label": f"{a} → {b}"})
            legend.append(("поток: вчерашний рынок → сегодняшний", C["accent"]))
        else:
            corr = Rn.corr().where(np.triu(np.ones((len(names), len(names))), 1) == 1).stack().nlargest(n_arcs)
            top = float(corr.max())
            for (a, b), v in corr.items():
                arcs.append({"a": pos[a], "b": pos[b], "w": float(max(v, 0) / top), "color": C["blue"],
                             "label": f"{a} ↔ {b}"})
            legend.append(("связь: корреляция доходностей", C["blue"]))
    title = ("ДЕМО · СИНТЕТИЧЕСКИЕ ДАННЫЕ · " if is_demo else "") + paint.upper()
    html = globe.build(mk, arcs, colors, title=title, legend=legend, height=700)
    components.html(html, height=705)
    if flow.startswith("Кто"):
        st.caption("Поток A → B: вчерашняя доходность A предсказывает сегодняшнюю доходность B лучше, чем наоборот "
                   "(разница корреляций с лагом 1 день). Часть эффекта объясняется часовыми поясами: Азия "
                   "закрывается раньше Европы и США. Это связь, а не доказательство причинности.")

    st.subheader("Мировые рынки")
    rows = []
    for n in names:
        is_open, local = status[n]
        rows.append({"Рынок": n, "Город": markets.MARKETS[n][3], "Уровень": float(idx[n].iloc[-1]),
                     "1д": chg.loc[n, "1д"], "5д": chg.loc[n, "5д"], "1м": chg.loc[n, "1м"],
                     "С начала года": chg.loc[n, "С начала года"],
                     "Статус": "● открыт" if is_open else "○ закрыт", "Местное время": f"{local:%H:%M}",
                     "Тренд": (idx[n].tail(60) / idx[n].tail(60).iloc[0]).tolist()})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=420, column_config={
        "Уровень": st.column_config.NumberColumn(format="%.2f"),
        **{k: st.column_config.NumberColumn(format="percent") for k in ("1д", "5д", "1м", "С начала года")},
        "Тренд": st.column_config.LineChartColumn("Тренд 60д")})
    st.caption("Статус торгов считается по расписанию и дням недели, без учёта праздников и перерывов.")


# ================================================================ МАКРО
def _rebased(df, cols):
    sub = df[cols].dropna(how="all").ffill()
    return sub / sub.iloc[0] * 100


def view_macro(world):
    tabs = st.tabs(["Ставки ЦБ", "Валюты", "Кривая доходности", "Сырьё и риск", "Страны: макро", "Связи активов"])
    rates, fx, yl, cross, macro, idx = (world[k] for k in ("rates", "fx", "yields", "cross", "macro", "index"))
    with tabs[0]:
        if rates.empty:
            _empty()
        else:
            last = rates.ffill().iloc[-1]
            year_ago = rates.ffill().iloc[max(len(rates) - 253, 0)]
            tbl = pd.DataFrame({"Ставка, %": last, "Год назад, %": year_ago, "Изменение, п.п.": last - year_ago})
            a, b = st.columns([2, 3])
            with a:
                st.dataframe(tbl.sort_values("Ставка, %", ascending=False), width="stretch", height=480,
                             column_config={c: st.column_config.NumberColumn(format="%.2f") for c in tbl.columns})
            with b:
                pick = st.multiselect("Центробанки", list(rates.columns), default=list(rates.columns)[:5])
                fig = go.Figure()
                for i, c in enumerate(pick):
                    fig.add_scatter(x=rates.index, y=rates[c], name=c, line=dict(color=PALETTE[i % len(PALETTE)],
                                    shape="hv"))
                fig.update_layout(yaxis_title="Ставка, %")
                st.plotly_chart(style(fig, 440), width="stretch")
            st.caption("Ряды берутся из FRED (ЕЦБ, ФРС, OECD/IMF). Для некоторых стран это не ключевая ставка, а ставка "
                       "дисконта или денежного рынка: смотри пояснение в источнике.")
    with tabs[1]:
        if fx.empty:
            _empty()
        else:
            pairs = list(fx.columns)
            pick = st.multiselect("Пары (индексированы к 100 на старте)", pairs,
                                  default=[p for p in pairs if p.endswith("/RUB")][:4] or pairs[:4])
            if pick:
                reb = _rebased(fx, pick)
                fig = go.Figure()
                for i, c in enumerate(pick):
                    fig.add_scatter(x=reb.index, y=reb[c], name=c, line=dict(color=PALETTE[i % len(PALETTE)]))
                st.plotly_chart(style(fig, 400), width="stretch")
            ch = _changes(fx)
            st.dataframe(pd.concat([fx.ffill().iloc[-1].rename("Курс"), ch], axis=1), width="stretch", height=380,
                         column_config={"Курс": st.column_config.NumberColumn(format="%.4f"),
                                        **{k: st.column_config.NumberColumn(format="percent") for k in ch.columns}})
    with tabs[2]:
        if yl.empty or "US 10Y" not in yl or "US 2Y" not in yl:
            _empty("Нужны ряды US 2Y и US 10Y (FRED).")
        else:
            spread = (yl["US 10Y"] - yl["US 2Y"]).dropna()
            m = st.columns(3)
            m[0].metric("US 10Y", f"{yl['US 10Y'].dropna().iloc[-1]:.2f}%")
            m[1].metric("US 2Y", f"{yl['US 2Y'].dropna().iloc[-1]:.2f}%")
            m[2].metric("Наклон 10Y−2Y", f"{spread.iloc[-1]:+.2f} п.п.", "инверсия" if spread.iloc[-1] < 0 else "норма",
                        delta_color="inverse" if spread.iloc[-1] < 0 else "normal")
            fig = go.Figure()
            fig.add_scatter(x=spread.index, y=spread, name="10Y − 2Y", line=dict(color=C["accent"]))
            fig.add_hline(y=0, line=dict(color="#888", dash="dash"))
            fig.update_layout(yaxis_title="п.п.")
            st.plotly_chart(style(fig, 340), width="stretch")
            fig = go.Figure()
            for i, c in enumerate(yl.columns):
                fig.add_scatter(x=yl.index, y=yl[c], name=c, line=dict(color=PALETTE[i]))
            st.plotly_chart(style(fig, 340), width="stretch")
            st.caption("Инверсия (10Y ниже 2Y) исторически часто предшествовала рецессиям, но с большим и переменным лагом.")
    with tabs[3]:
        if cross.empty:
            _empty()
        else:
            pick = st.multiselect("Активы (индексированы к 100)", list(cross.columns), default=list(cross.columns)[:5])
            if pick:
                reb = _rebased(cross, pick)
                fig = go.Figure()
                for i, c in enumerate(pick):
                    fig.add_scatter(x=reb.index, y=reb[c], name=c, line=dict(color=PALETTE[i % len(PALETTE)]))
                st.plotly_chart(style(fig, 400), width="stretch")
            ch = _changes(cross)
            st.dataframe(ch, width="stretch", column_config={k: st.column_config.NumberColumn(format="percent")
                                                              for k in ch.columns})
    with tabs[4]:
        if macro.empty:
            _empty()
        else:
            latest = (macro.sort_values("year").groupby(["country", "indicator"])["value"].last().unstack())
            st.dataframe(latest, width="stretch", height=360,
                         column_config={c: st.column_config.NumberColumn(format="%.2f") for c in latest.columns})
            ind = st.selectbox("Показатель", sorted(macro["indicator"].unique()))
            cs = st.multiselect("Страны", sorted(macro["country"].unique()), default=["USA", "CHN", "DEU", "RUS"])
            fig = go.Figure()
            for i, c in enumerate(cs):
                s = macro[(macro["indicator"] == ind) & (macro["country"] == c)].sort_values("year")
                fig.add_scatter(x=s["year"], y=s["value"], name=c, line=dict(color=PALETTE[i % len(PALETTE)]))
            st.plotly_chart(style(fig, 360), width="stretch")
            st.caption("World Bank публикует годовые данные с задержкой: последний год может быть прошлым или позапрошлым.")
    with tabs[5]:
        parts = [df for df in (idx, cross, fx) if not df.empty]
        if not parts:
            _empty()
        else:
            allp = pd.concat(parts, axis=1)
            Rall = allp.sort_index().ffill(limit=3).pct_change().dropna(how="all")
            Rall = Rall.loc[:, Rall.notna().mean() > 0.8].dropna()
            window = st.slider("Окно, дней (0 = вся история)", 0, 750, 0, 30)
            if window:
                Rall = Rall.tail(window)
            _, order = structure.cluster_order(Rall.corr().to_numpy())
            names = list(Rall.columns[order])
            corr = Rall[names].corr()
            fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1,
                                       colorscale=[[0, C["blue"]], [0.5, "#1B2230"], [1, C["down"]]]))
            st.markdown("**Корреляции индексов, сырья и валют** (упорядочены кластеризацией: близкие рядом)")
            st.plotly_chart(style(fig, 640), width="stretch")


# ================================================================ ОТЧЁТНОСТЬ
def view_fundamentals(world, con, is_demo, bump):
    facts = world["facts"]
    if "sec_report" in st.session_state:
        st.dataframe(st.session_state["sec_report"], hide_index=True, width="stretch")
    if not is_demo:
        with st.expander("Загрузить отчётность из SEC EDGAR (США и иностранные эмитенты в SEC)", expanded=not facts):
            tickers = st.text_input("Тикеры через запятую", "AAPL, MSFT, NVDA, JPM")
            st.caption("SEC требует идентификацию клиента: укажи свою почту в .env (TRADELAB_CONTACT=you@mail.com). "
                       "Для российских компаний открытого API нет: используй импорт CSV ниже.")
            if st.button("Загрузить из SEC"):
                res = worlddata.refresh_fundamentals(con, [t.strip() for t in tickers.split(",") if t.strip()])
                st.session_state["sec_report"] = pd.DataFrame(res, columns=["Тикер", "OK", "Сообщение"])
                bump()
        with st.expander("Импорт отчётности из CSV (любая компания, в т.ч. российская)"):
            st.caption("Колонки: ticker, concept, period_end (YYYY-MM-DD), value. Названия показателей как в таблице "
                       "(Выручка, Чистая прибыль, Активы, Капитал, Обязательства …).")
            up = st.file_uploader("CSV", type="csv")
            if up is not None:
                df = pd.read_csv(up)
                for t, grp in df.groupby("ticker"):
                    grp = grp.assign(form="CSV", filed=datetime.now().date().isoformat())
                    store.upsert_facts(con, str(t), grp[["concept", "period_end", "value", "form", "filed"]])
                st.success("Импортировано")
                bump()
    if not facts:
        return _empty("Нет данных отчётности. Загрузи её выше (SEC) или импортируй CSV.")
    ticker = st.selectbox("Компания", sorted(facts))
    t = fundamentals.table(facts[ticker])
    if t.empty:
        return _empty("По компании нет годовых данных.")
    r = fundamentals.ratios(t)
    last = r.iloc[-1]
    m = st.columns(5)
    m[0].metric("Рост выручки", pct(last["Рост выручки"]))
    m[1].metric("Чистая маржа", pct(last["Чистая маржа"]))
    m[2].metric("ROE", pct(last["ROE"]))
    m[3].metric("Долг / капитал", f"{last['Долг / капитал']:.2f}" if np.isfinite(last["Долг / капитал"]) else "н/д")
    m[4].metric("FCF маржа", pct(last["FCF маржа"]))
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for col, color in (("Выручка", C["blue"]), ("Чистая прибыль", C["accent"])):
            if col in t:
                fig.add_bar(x=t.index, y=t[col], name=col, marker_color=color)
        fig.update_layout(barmode="group")
        st.markdown("**Выручка и чистая прибыль**")
        st.plotly_chart(style(fig, 340), width="stretch")
    with b:
        fig = go.Figure()
        for col, color in (("Операционная маржа", C["up"]), ("Чистая маржа", C["accent"]), ("ROE", C["violet"])):
            fig.add_scatter(x=r.index, y=r[col], name=col, line=dict(color=color))
        fig.update_layout(yaxis_tickformat=".0%")
        st.markdown("**Рентабельность**")
        st.plotly_chart(style(fig, 340), width="stretch")
    st.markdown("**Годовые показатели**")
    st.dataframe(t.T, width="stretch")
    st.markdown("**Коэффициенты**")
    st.dataframe(r.T, width="stretch", column_config={})
    st.caption("Данные из годовых форм (10-K, 20-F, 40-F) без пересчёта: при пересмотрах берётся последнее раскрытие.")


# ================================================================ СТРУКТУРА
@st.cache_data(show_spinner=False)
def _topology(R, window, step):
    return structure.rolling_topology(R, window, step)


@st.cache_data(show_spinner=False)
def _shift(R, window, step):
    return structure.rolling_cov_shift(R, window, step)


def _zscore(s):
    return (s - s.mean()) / s.std()


def view_structure(R, label):
    if R is None or R.shape[1] < 5 or len(R) < 250:
        return _empty("Нужно минимум 5 бумаг и около 250 баров истории.")
    names = list(R.columns)
    corr = R.corr()
    st.caption(f"Универсум: {label}. Бумаг: {len(names)}, наблюдений: {len(R)}.")
    sec = st.tabs(["Карта рынка", "Шум и сигнал (RMT)", "Топология и хрупкость", "Режимы", "Связи и хвосты", "HRP"])

    with sec[0]:
        k = st.slider("Число кластеров", 2, min(10, len(names) - 1), 4)
        cl = structure.clusters(corr.to_numpy(), k)
        edges = structure.mst_edges(corr.to_numpy(), names)
        G = nx.Graph()
        G.add_nodes_from(names)
        for r in edges.itertuples():
            G.add_edge(r.u, r.v, weight=r.dist)
        pos = nx.spring_layout(G, seed=3, weight=None, k=1.2 / np.sqrt(len(names)), iterations=200)
        fig = go.Figure()
        for r in edges.itertuples():
            fig.add_scatter(x=[pos[r.u][0], pos[r.v][0]], y=[pos[r.u][1], pos[r.v][1]], mode="lines",
                            line=dict(color="rgba(150,170,200,.45)", width=max(0.6, 3.2 - 2.2 * r.dist)),
                            hoverinfo="skip", showlegend=False)
        deg = dict(G.degree())
        vol = R.std() * np.sqrt(252)
        fig.add_scatter(x=[pos[n][0] for n in names], y=[pos[n][1] for n in names], mode="markers+text",
                        text=names, textposition="top center", textfont=dict(size=11, color=C["text"]),
                        marker=dict(size=[10 + 4 * deg[n] for n in names], color=[PALETTE[(c - 1) % len(PALETTE)] for c in cl],
                                    line=dict(color="#fff", width=1)),
                        hovertext=[f"{n}: связей {deg[n]}, вол-ть {vol[n]:.0%}" for n in names], showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Минимальное остовное дерево** корреляций: цвет = кластер, размер = число связей. "
                        "Хабы дерева связывают весь рынок: их падение тянет остальных")
            st.plotly_chart(style(fig, 560), width="stretch")
        with c2:
            use_dn = st.checkbox("Очистить шум (RMT)", value=False)
            C_used = structure.rmt(R)["denoised"] if use_dn else corr
            _, order = structure.cluster_order(C_used.to_numpy())
            ordered = [names[i] for i in order]
            fig2 = go.Figure(go.Heatmap(z=C_used.loc[ordered, ordered].values, x=ordered, y=ordered, zmin=-1, zmax=1,
                                        colorscale=[[0, C["blue"]], [0.5, "#1B2230"], [1, C["down"]]]))
            st.markdown("**Корреляции, упорядоченные по кластерам**")
            st.plotly_chart(style(fig2, 560), width="stretch")
        members = pd.DataFrame({"Бумага": names, "Кластер": cl}).sort_values(["Кластер", "Бумага"])
        st.dataframe(members.groupby("Кластер")["Бумага"].apply(", ".join).reset_index(), hide_index=True, width="stretch")

    with sec[1]:
        res = structure.rmt(R)
        vals = res["eigenvalues"]
        m = st.columns(3)
        m[0].metric("Значимых факторов", res["n_signal"], f"из {len(names)}", delta_color="off")
        m[1].metric("Граница шума (Марченко–Пастур)", f"{res['lambda_plus']:.2f}")
        m[2].metric("Доля дисперсии у первого фактора", pct(float(vals[0] / vals.sum())))
        fig = go.Figure()
        fig.add_bar(x=list(range(1, len(vals) + 1)), y=vals, marker_color=[C["accent"] if v > res["lambda_plus"] else "#556070" for v in vals])
        fig.add_hline(y=res["lambda_plus"], line=dict(color=C["down"], dash="dash"), annotation_text="граница шума")
        fig.update_layout(xaxis_title="Номер собственного числа", yaxis_title="Собственное число корреляций")
        st.plotly_chart(style(fig, 360), width="stretch")
        st.caption("Собственные числа выше красной линии несут структуру (рынок, секторы), ниже неотличимы от шума "
                   "конечной выборки. Очистка заменяет шумовые числа средним и даёт более устойчивую корреляцию.")

    with sec[2]:
        c1, c2 = st.columns(2)
        window = c1.slider("Окно, дней", 30, 180, 60, 10)
        step = c2.slider("Шаг, дней", 2, 20, 5)
        with st.spinner("Считаю топологию…"):
            topo = _topology(R, window, step)
            shift = _shift(R, window, max(step, 10))
        fig = go.Figure()
        fig.add_scatter(x=topo.index, y=_zscore(topo["mst_length"]), name="Длина MST (z)", line=dict(color=C["blue"]))
        fig.add_scatter(x=topo.index, y=_zscore(topo["absorption"]), name="Коэфф. поглощения (z)", line=dict(color=C["accent"]))
        fig.add_scatter(x=topo.index, y=_zscore(topo["mean_corr"]), name="Средняя корреляция (z)", line=dict(color=C["up"]))
        if topo["h1"].notna().any():
            fig.add_scatter(x=topo.index, y=_zscore(topo["h1"]), name="Персистентность H1 (z)", line=dict(color=C["violet"]))
        if len(shift):
            fig.add_scatter(x=shift.index, y=_zscore(shift["shift"]), name="Сдвиг ковариации (z)",
                            line=dict(color=C["down"], dash="dot"))
        fig.update_layout(yaxis_title="Стандартизированное значение")
        st.plotly_chart(style(fig, 440), width="stretch")
        if not topo["h1"].notna().any():
            st.info("Топологические петли H1 недоступны: установи пакет ripser (`pip install ripser`). "
                    "Остальные индикаторы работают без него.")
        hot = (_zscore(topo["absorption"]) + _zscore(topo["mean_corr"]) - _zscore(topo["mst_length"])) / 3
        m = st.columns(3)
        m[0].metric("Индекс хрупкости сейчас", f"{hot.iloc[-1]:+.2f}", "выше нормы" if hot.iloc[-1] > 1 else "в норме",
                    delta_color="inverse" if hot.iloc[-1] > 1 else "off")
        m[1].metric("Перцентиль за историю", pct(float((hot <= hot.iloc[-1]).mean()), 0))
        m[2].metric("Последний сдвиг ковариации", f"{shift['shift'].iloc[-1]:.2f}" if len(shift) else "н/д")
        st.caption("Как читать: в кризис рынок «слипается»: средняя корреляция и коэффициент поглощения растут, остовное "
                   "дерево сжимается (Onnela и др.), в облаке дневных векторов меняются петли (Gidea–Katz, топологический "
                   "анализ данных), а риманова метрика между ковариациями соседних окон показывает скачок «состояния рынка». "
                   "Индекс хрупкости = среднее (поглощение + корреляция − длина MST) в z-оценках. Это диагностика, а не "
                   "прогноз: рост бывает и без обвала, и обвал без роста.")

    with sec[3]:
        k = st.slider("Число режимов", 2, 4, 3)
        state, feats = structure.regimes(R, k=k)
        idx_ret = R.mean(axis=1).loc[state.index]
        equity = (1 + idx_ret).cumprod()
        fig = go.Figure()
        fig.add_scatter(x=equity.index, y=equity, line=dict(color="#E8ECF1", width=1.5), name="Индекс (равные веса)")
        shades = ["rgba(38,194,129,.14)", "rgba(255,159,26,.16)", "rgba(255,77,79,.20)", "rgba(179,140,255,.2)"]
        blocks = (state != state.shift()).cumsum()
        for _, seg in state.groupby(blocks):
            fig.add_vrect(x0=seg.index[0], x1=seg.index[-1], fillcolor=shades[int(seg.iloc[0])], line_width=0, layer="below")
        st.plotly_chart(style(fig, 400), width="stretch")
        stats = pd.DataFrame({
            "Доля времени": state.value_counts(normalize=True).sort_index(),
            "Доходность (год)": [float(np.exp(np.log1p(idx_ret[state == s]).mean() * 252) - 1) for s in range(k)],
            "Волатильность": [float(idx_ret[state == s].std() * np.sqrt(252)) for s in range(k)],
            "Средняя корреляция": [float((lambda c: (c.sum() - len(c)) / (len(c) * (len(c) - 1)))(
                np.corrcoef(R.loc[state.index][state == s].to_numpy().T)) if (state == s).sum() > 20 else np.nan)
                for s in range(k)]})
        stats.index = ["Спокойный", "Умеренный", "Напряжённый", "Экстремальный"][:k]
        st.dataframe(stats, width="stretch", column_config={c: st.column_config.NumberColumn(format="percent")
                                                            for c in stats.columns[:3]} | {
            "Средняя корреляция": st.column_config.NumberColumn(format="%.2f")})
        st.caption(f"Сейчас режим: **{stats.index[int(state.iloc[-1])]}**. Режимы найдены кластеризацией (смесь гауссиан) "
                   "по скользящим доходности и волатильности индекса, поэтому границы немного размыты и пересматриваются "
                   "при добавлении данных.")

    with sec[4]:
        L = structure.lead_lag(R)
        net = (L - L.T).stack()
        top = net[net > 0].nlargest(10).reset_index()
        top.columns = ["Ведущий", "Ведомый", "Разность лаговых корреляций"]
        a, b = st.columns(2)
        with a:
            order = list(L.mean(axis=1).sort_values(ascending=False).index)
            fig = go.Figure(go.Heatmap(z=L.loc[order, order].values, x=order, y=order, zmid=0,
                                       colorscale=[[0, C["blue"]], [0.5, "#1B2230"], [1, C["accent"]]]))
            fig.update_xaxes(title="сегодня (ведомый)")
            fig.update_yaxes(title="вчера (ведущий)", autorange="reversed")
            st.markdown("**Lead-lag**: корреляция вчерашней доходности строки с сегодняшней доходностью столбца")
            st.plotly_chart(style(fig, 480), width="stretch")
        with b:
            T = structure.tail_dependence(R, 0.05, lower=True)
            order2 = [names[i] for i in structure.cluster_order(corr.to_numpy())[1]]
            fig = go.Figure(go.Heatmap(z=T.loc[order2, order2].values, x=order2, y=order2, zmin=0, zmax=1,
                                       colorscale=[[0, "#1B2230"], [1, C["down"]]]))
            st.markdown("**Хвостовая зависимость**: если строка в худших 5% дней, как часто и столбец")
            st.plotly_chart(style(fig, 480), width="stretch")
        st.dataframe(top, hide_index=True, width="stretch", column_config={
            "Разность лаговых корреляций": st.column_config.NumberColumn(format="%.3f")})
        st.caption("При независимости вероятность совместного падения равна 5%. Значения заметно выше показывают, что "
                   "бумаги падают вместе именно в плохие дни: на хвостах диверсификация работает хуже, чем говорит "
                   "обычная корреляция.")

    with sec[5]:
        w_hrp = structure.hrp_weights(R)
        w_eq = pd.Series(1 / len(names), index=names)
        w_mv = pd.Series(optimize.optimize(R, "min_vol", max_weight=0.3, shrink=True))
        out = pd.DataFrame({"HRP": w_hrp, "Равные": w_eq, "Мин. риск": w_mv}).sort_values("HRP", ascending=False)
        fig = go.Figure()
        for i, c in enumerate(out.columns):
            fig.add_bar(x=out.index, y=out[c], name=c, marker_color=[C["accent"], C["blue"], C["violet"]][i])
        fig.update_layout(barmode="group", yaxis_tickformat=".0%")
        st.plotly_chart(style(fig, 380), width="stretch")
        rows = []
        for name, w in out.items():
            pr = risk.portfolio_returns(R, w.to_dict())
            s = risk.summary(pr)
            rows.append({"Метод": name, "Волатильность": s["ann_vol"], "Макс. просадка": s["max_drawdown"],
                         "VaR 95% (1д)": s["var95"], "Эфф. число бумаг": risk.concentration(w.to_dict())[1]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Волатильность": st.column_config.NumberColumn(format="percent"),
            "Макс. просадка": st.column_config.NumberColumn(format="percent"),
            "VaR 95% (1д)": st.column_config.NumberColumn(format="percent"),
            "Эфф. число бумаг": st.column_config.NumberColumn(format="%.1f")})
        st.caption("HRP (иерархический паритет риска, López de Prado) строит веса по дереву кластеров без обращения "
                   "матрицы ковариации: обычно устойчивее на шумных данных. Метрики посчитаны на той же истории, на "
                   "которой строились веса, поэтому это описание, а не прогноз.")
