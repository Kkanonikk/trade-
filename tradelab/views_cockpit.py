"""Вкладка КОКПИТ: единый взгляд на портфель (риск, страны, климат, факторные сценарии, лимиты)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import cockpit, compliance, config, countries, crisis, gdata, hunger, panel as panel_mod, risk, views_global, views_worldmode
from .ui import C, PALETTE, pct, style


def _overlay_values(w, key, year=None):
    s = gdata.cross_section(w, key, year or views_worldmode.default_year(w, key))
    return s


def _hub_key(w, needle):
    meta = gdata.meta()
    for k in w.columns:
        m = meta.get(k, {})
        if m.get("source") == "owid_hub" and needle.lower() in (m.get("dataset", "") + " " + m.get("label", "")).lower():
            return k
    return None


def view_cockpit(frames, weights, weights_fallback, equity, world, version, limits, use_demo):
    if len(frames) < 2:
        return st.info("Нужно минимум две бумаги с данными (вкладка ДАННЫЕ).")
    if weights_fallback:
        st.info("На paper-счёте пока нет позиций: показываю равные веса. Позиции появятся после ребаланса на вкладке АВТОТОРГОВЛЯ.")
    R = risk.returns_frame(frames)
    pr = risk.portfolio_returns(R, weights)
    s = risk.summary(pr)
    bench = R.mean(axis=1)
    hhi, eff_n = risk.concentration(weights)
    breaches = compliance.check(weights, limits, var95=s["var95"])
    st.markdown("### Портфель одним взглядом")
    m = st.columns(6)
    m[0].metric("Капитал", f"{equity:,.0f} $")
    m[1].metric("VaR 95% (1 день)", f"{s['var95'] * equity:,.0f} $", pct(-s["var95"], 2), delta_color="off")
    m[2].metric("CVaR 95%", f"{s['cvar95'] * equity:,.0f} $", pct(-s["cvar95"], 2), delta_color="off")
    m[3].metric("Волатильность", pct(s["ann_vol"], 1))
    m[4].metric("Макс. просадка", pct(s["max_drawdown"], 1))
    m[5].metric("Лимиты", "в норме" if not breaches else f"нарушений: {len(breaches)}", delta_color="off")
    if breaches:
        st.dataframe(pd.DataFrame(breaches), hide_index=True, width="stretch")

    a, b = st.columns(2)
    exposure = cockpit.look_through(weights)
    with a:
        st.markdown("**По странам риска (с разложением ETF по корзинам)**")
        e = exposure.head(12)
        fig = go.Figure(go.Bar(x=e.values, y=[countries.short_ru(i) if i in countries.table().index else i for i in e.index], orientation="h", marker_color=C["accent"]))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(style(fig, 340), width="stretch")
    with b:
        st.markdown("**По классам активов и секторам**")
        cl = cockpit.class_exposure(weights)
        sec = pd.Series(risk.sector_exposure(weights, config.sector)).sort_values(ascending=False)
        fig = go.Figure(go.Pie(labels=list(cl.index), values=list(cl.values), hole=0.55, marker=dict(colors=PALETTE), textinfo="label+percent"))
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 230), width="stretch")
        st.dataframe(pd.DataFrame({"Сектор": sec.index, "Доля": sec.values}), hide_index=True, height=130,
                     column_config={"Доля": st.column_config.NumberColumn(format="percent")})
    st.caption(f"Концентрация: эффективное число позиций {eff_n:.1f}, индекс Херфиндаля {hhi:.2f}. Страны ETF на корзины EM "
               "заданы приблизительными весами; тикеры вне справочника попадают в «н/д».")

    st.markdown("### Страновые и климатические риски портфеля")
    w = views_worldmode.get_wide(version)
    iso_exp = exposure[exposure.index.isin(countries.table().index)]
    cover = float(iso_exp.sum() / exposure.sum()) if exposure.sum() else 0.0
    st.caption(f"Данные по странам есть для {cover:.0%} портфеля (остальное: сырьё, крипта, неизвестные тикеры).")
    rows = []
    hidx, _ = hunger.index(w, 2022)
    specs = [("Уязвимость к голоду, 0–100 (выше = хуже)", hidx),
             ("Выбросы CO₂ на душу, т", _overlay_values(w, "co2:co2_per_capita")),
             ("Доля возобновляемых в электроэнергии, %", _overlay_values(w, "en:renewables_share_elec"))]
    ck = _hub_key(w, "Corruption Perception")
    if ck:
        specs.append(("Индекс восприятия коррупции (выше = чище)", _overlay_values(w, ck)))
    for label, vals in specs:
        avg, cov, tbl = cockpit.weighted_overlay(iso_exp, vals)
        rows.append({"Показатель": label, "Среднее по портфелю": avg, "Покрытие": cov * (cover or 1),
                     "Главный вклад": countries.short_ru(tbl.index[0]) if len(tbl) else "н/д"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Среднее по портфелю": st.column_config.NumberColumn(format="%.1f"), "Покрытие": st.column_config.NumberColumn(format="percent")})
    views_worldmode.src_box(["co2:co2_per_capita", "en:renewables_share_elec"] + ([ck] if ck else []) + list(hunger.COMPONENTS)[:1],
                            "взвешивание по долям портфеля рассчитано приложением")

    with st.expander("Вероятность кризисов в странах портфеля (модели раннего предупреждения)"):
        st.caption("Модели из вкладки КРИЗИСЫ на встроенном снимке (макро IMF до 2013 года): пока не загружены свежие данные, это вероятность на год после последнего "
                   "года снимка, то есть показатель уязвимости, а не прогноз на сегодня.")
        if st.button("Рассчитать (около 30–60 с при первом запуске)", key="ck_crisis"):
            with st.spinner("Обучаю модели…"):
                models, _ = views_global.get_models("snapshot", version, 1)
                out = {}
                for kind, mdl in models.items():
                    pr_, fy = mdl.predict()
                    out[crisis.KINDS[kind]] = (pr_["prob"], fy + 1)
            st.session_state["ck_crisis_res"] = out
        res = st.session_state.get("ck_crisis_res")
        if res:
            rows = []
            for name, (probs, year) in res.items():
                avg, cov, tbl = cockpit.weighted_overlay(iso_exp, probs)
                rows.append({"Кризис": name, "Год прогноза": year, "Средняя вероятность по портфелю": avg,
                             "Главный вклад": countries.short_ru(tbl.index[0]) if len(tbl) else "н/д"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"Средняя вероятность по портфелю": st.column_config.NumberColumn(format="percent")})

    st.markdown("### Факторные сценарии")
    F = cockpit.factor_returns(world)
    if F.empty or F.shape[1] < 2:
        return st.info("Для сценариев нужны мировые данные (индексы, сырьё): вкладка ДАННЫЕ → «Обновить мировые данные», либо включи демо-режим.")
    try:
        exp = cockpit.exposures(pr, F)
    except ValueError as exc:
        return st.warning(str(exc))
    st.caption(f"Чувствительность портфеля к факторам оценена по {exp.attrs['n']} неделям (МНК, устойчивые SE). R² = {exp.attrs['r2']:.2f}: "
               "чем он ниже, тем меньше портфель объясняется этими факторами. Интервалы ±1.96 SE.")
    fig = go.Figure(go.Bar(x=exp["beta"], y=exp.index, orientation="h", error_x=dict(type="data", array=1.96 * exp["se"]), marker_color=C["accent"]))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Доходность портфеля на +1% фактора, п.п.")
    st.plotly_chart(style(fig, 300), width="stretch")
    rows = []
    for name, shocks in cockpit.SCENARIOS.items():
        est, lo, hi = cockpit.scenario_pnl(exp, shocks)
        rows.append({"Сценарий": name, "Оценка": est, "Нижняя (95%)": lo, "Верхняя (95%)": hi, "$ (оценка)": est * equity})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Оценка": st.column_config.NumberColumn(format="percent"), "Нижняя (95%)": st.column_config.NumberColumn(format="percent"),
        "Верхняя (95%)": st.column_config.NumberColumn(format="percent"), "$ (оценка)": st.column_config.NumberColumn(format="localized")})
    with st.expander("Свой сценарий"):
        cols = st.columns(len(F.columns))
        shocks = {f: cols[i].slider(f, -50, 80, 0, 5, key=f"ck_sh_{i}") / 100 for i, f in enumerate(F.columns)}
        est, lo, hi = cockpit.scenario_pnl(exp, shocks)
        st.metric("Оценка изменения портфеля", pct(est, 2), f"интервал {pct(lo, 1)} … {pct(hi, 1)}", delta_color="off")
    st.caption("Шоки считаются мгновенными и независимыми, чувствительности взяты из истории и могут не повториться в стрессе (корреляции в кризис растут). "
               "Это арифметика, а не прогноз." + (" Режим демо: данные синтетические, оценки лишь иллюстрация." if use_demo else ""))
