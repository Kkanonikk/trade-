"""Страницы ПРОБЛЕМЫ (тепловые карты: где больше и где меньше проблем) и ПАСПОРТ (досье страны по измерениям)."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import countries, gdata, globe, hunger, problems, views_global, views_worldmode as vw
from .ui import C, PALETTE, pct, style

HEAT = [[0.0, "#12202f"], [0.35, "#2b4a3a"], [0.5, "#8a6a1c"], [0.75, "#FF9F1A"], [1.0, "#FF3C3C"]]
BAD = 65.0      # выше этого процентиля страну считаем «красной зоной»


def _goto(page, state):
    """Переход на страницу из кнопки: значения виджетов можно менять только в обработчике нажатия."""
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state["wm_page"] = page


def _sovereign(iso):
    t = countries.table()
    return iso in t.index and str(t.loc[iso, "independent"]).strip().lower() == "yes"


@st.cache_data(show_spinner=False)
def _scored(_w, version, year, wkey, sov_only):
    res = problems.score(_w, year, dict(wkey))
    comp = res["composite"]
    if sov_only:
        keep = [i for i in comp.index if _sovereign(i)]
        comp = comp.loc[keep]
        for k in ("dims", "pct", "values"):
            res[k] = res[k].loc[[i for i in res[k].index if i in set(keep)]]
    res["composite"] = comp
    return res


@st.cache_data(show_spinner=False)
def _history(_w, version, years, wkey, sov_only):
    h = pd.DataFrame({y: _scored(_w, version, y, wkey, sov_only)["composite"] for y in years})
    return h


def _weights_ui(prefix):
    """Веса измерений в сайдбар-подобной панели настроек индекса."""
    st.caption("Вес измерения в общем индексе. 0 исключает измерение. Выбор весов спорен, поэтому ниже есть проверка устойчивости.")
    cols = st.columns(4)
    wk = {}
    for i, (k, (name, _)) in enumerate(problems.DIMENSIONS.items()):
        wk[k] = cols[i % 4].slider(name, 0.0, 3.0, float(st.session_state.get(f"set_w_{k}", 1.0)), 0.25, key=f"{prefix}_w_{k}")
    return tuple(sorted(wk.items()))


def _heat_color(v, lo=0.0, hi=100.0):
    """Цвет проблемности между lo и hi (по умолчанию 0–100)."""
    if v is None or not np.isfinite(v) or hi <= lo:
        return None
    return globe.heat(float(np.clip((v - lo) / (hi - lo), 0, 1)), 1.0)


def page_problems(version):
    w = vw.get_wide(version)
    st.markdown("### Тепловая карта проблем мира")
    st.caption("Сравнительный индекс по семи измерениям на открытых данных: 0 = меньше всего проблем среди стран, 100 = больше всего. Это не прогноз и не оценка людей, "
               "а положение страны относительно других по выбранным показателям.")
    c1, c2, c3 = st.columns([2, 2, 3])
    year = c1.slider("Состояние на год", 1995, 2023, int(st.session_state.get("set_year", 2021)), key="pr_year")
    sov = c2.checkbox("Только суверенные страны", value=True, key="pr_sov")
    layer_opts = ["Общий индекс"] + [v[0] for v in problems.DIMENSIONS.values()] + ["Ухудшение за 5 лет"]
    layer = c3.selectbox("Слой карты", layer_opts, key="pr_layer")
    with st.expander("Настройки индекса: веса измерений"):
        wkey = _weights_ui("pr")
    res = _scored(w, version, year, wkey, sov)
    comp, dims = res["composite"], res["dims"].loc[res["composite"].index]
    if comp.empty:
        return st.info("Недостаточно данных.")
    names = lambda i: countries.short_ru(i)
    path = vw.get_pop_path(version)
    popcol = min(year, 2024)
    pop = path[popcol] if popcol in path.columns else pd.Series(dtype=float)
    red = comp[comp > BAD].index
    m = st.columns(5)
    m[0].metric("Стран в оценке", len(comp))
    m[1].metric("Больше всего проблем", names(comp.idxmax()), f"индекс {comp.max():.0f}", delta_color="off")
    m[2].metric("Меньше всего проблем", names(comp.idxmin()), f"индекс {comp.min():.0f}", delta_color="off")
    m[3].metric(f"«Красная зона» (> {BAD:.0f})", f"{len(red)} стран")
    inred = float(pop.reindex(red).sum() / pop.reindex(comp.index).sum()) if len(pop) and pop.reindex(comp.index).sum() > 0 else float("nan")
    m[4].metric("Население в красной зоне", pct(inred, 0) if np.isfinite(inred) else "н/д")

    if layer == "Общий индекс":
        series, title = comp, "Общий индекс проблем"
    elif layer == "Ухудшение за 5 лет":
        past = _scored(w, version, year - 5, wkey, sov)["composite"]
        series, title = (comp - past.reindex(comp.index)).dropna(), "Изменение общего индекса за 5 лет (плюс = стало хуже)"
    else:
        key = next(k for k, v in problems.DIMENSIONS.items() if v[0] == layer)
        series, title = dims[key].dropna(), f"Проблемы: {layer.lower()}"
    if layer == "Ухудшение за 5 лет":
        lim = float(series.abs().quantile(0.95)) or 1.0
        colors = {i: globe.scale_color(float(v), -lim, lim, "down") if v > 0 else globe.scale_color(float(v), -lim, lim, "up") for i, v in series.items()}
        colors = {i: globe.scale_color(float(v), -lim, lim, "up" if False else "down") for i, v in series.items()}
        grad = views_global.gradient_legend(lambda x: globe.scale_color(x, -lim, lim, "down"), -lim, lim, f"лучше · {-lim:+.0f}", f"хуже · {lim:+.0f}")
        fmt = lambda x: f"{x:+.1f}"
    else:
        lo_, hi_ = float(series.quantile(0.03)), float(series.quantile(0.97))
        colors = {i: _heat_color(v, lo_, hi_) for i, v in series.items()}
        grad = views_global.gradient_legend(lambda x: _heat_color(x, lo_, hi_), lo_, hi_, f"меньше проблем · {lo_:.0f}", f"больше проблем · {hi_:.0f}")
        fmt = lambda x: f"{x:.0f}"
    infos = {}
    for iso in series.index:
        row = dims.loc[iso] if iso in dims.index else pd.Series(dtype=float)
        bars = "".join(f"<div style='display:flex;align-items:center;gap:6px;font-size:11px'><span style='width:112px;color:#8B96A5'>{v[0]}</span>"
                       f"<span style='display:inline-block;height:7px;width:{max(2, 0 if np.isnan(row.get(k, np.nan)) else row[k]) * 0.9:.0f}px;background:{_heat_color(row.get(k, np.nan)) or '#333'}'></span>"
                       f"<span>{'' if np.isnan(row.get(k, np.nan)) else f'{row[k]:.0f}'}</span></div>" for k, v in problems.DIMENSIONS.items())
        infos[iso] = (f"<b style='color:#FF9F1A'>{names(iso)}</b><br>{title}: <b>{fmt(float(series[iso]))}</b>"
                      f"<br>место {int(series.rank(ascending=False, method='min')[iso])} из {len(series)}<br>{bars}")
    rk = views_global.ranking_panel(series, lambda i: colors.get(i), names, fmt, "Больше проблем" if layer != "Ухудшение за 5 лет" else "Сильнее ухудшилось",
                                    "Меньше проблем" if layer != "Ухудшение за 5 лет" else "Сильнее улучшилось", n=10)
    mode = st.radio("Вид карты", ["Карта", "Глобус"], horizontal=True, key="pr_mode", label_visibility="collapsed")
    views_global.world_map(colors, infos, title.upper(), [], mode="flat" if mode == "Карта" else "globe", height=620, grad=grad, ranking=rk,
                           subtitle=f"{year} · {len(series)} стран")
    vw.src_box(list(res["values"].columns), "индекс рассчитан приложением: процентили показателей, усреднённые по измерениям")

    t1, t2, t3, t4 = st.tabs(["Матрица стран × измерений", "Регионы", "Связи между проблемами", "Устойчивость и горячие точки"])
    with t1:
        a, b, c = st.columns(3)
        n = a.slider("Сколько стран", 10, 60, 25, key="pr_n")
        which = b.radio("Показать", ["Больше всего проблем", "Меньше всего проблем"], horizontal=True, key="pr_which", label_visibility="collapsed")
        order = comp.sort_values(ascending=(which == "Меньше всего проблем")).head(n).index
        Z = dims.loc[order]
        fig = go.Figure(go.Heatmap(z=Z.values, x=[problems.DIMENSIONS[k][0] for k in Z.columns], y=[names(i) for i in Z.index], colorscale=HEAT, zmin=0, zmax=100,
                                   text=np.round(Z.values, 0), texttemplate="%{text:.0f}", hoverongaps=False, colorbar=dict(title="баллов", thickness=10)))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(side="top")
        st.plotly_chart(style(fig, max(380, 22 * n + 80)), width="stretch")
        st.caption("Строка это страна, столбец это измерение, цвет и число это процентиль среди стран (100 = хуже всех). Пустая клетка: данных недостаточно.")
    with t2:
        reg = pd.Series({i: vw.region_of(i) for i in dims.index})
        pw = pop.reindex(dims.index)
        byreg = {}
        for r, idx in reg.groupby(reg).groups.items():
            if r == "Прочее" or len(idx) < 3:
                continue
            sub = dims.loc[idx]
            wts = pw.reindex(idx).fillna(pw.median())
            byreg[vw.REGION_RU.get(r, r)] = (sub.mul(wts, axis=0).sum() / sub.notna().mul(wts, axis=0).sum())
        R = pd.DataFrame(byreg).T
        R["Общий"] = pd.Series({vw.REGION_RU.get(r, r): float((comp.loc[idx].mul(pw.reindex(idx).fillna(pw.median())).sum() / pw.reindex(idx).fillna(pw.median()).sum()))
                                for r, idx in reg.groupby(reg).groups.items() if r != "Прочее" and len(idx) >= 3})
        Z = R.rename(columns={k: problems.DIMENSIONS[k][0] for k in problems.DIMENSIONS})
        fig = go.Figure(go.Heatmap(z=Z.values, x=list(Z.columns), y=list(Z.index), colorscale=HEAT, zmin=0, zmax=100, text=np.round(Z.values, 0),
                                   texttemplate="%{text:.0f}", colorbar=dict(title="баллов", thickness=10)))
        fig.update_xaxes(side="top")
        st.plotly_chart(style(fig, 300), width="stretch")
        st.caption("Средние по регионам, взвешенные населением стран (население по ООН).")
    with t3:
        corr = dims.corr(method="spearman")
        lab = [problems.DIMENSIONS[k][0] for k in corr.columns]
        fig = go.Figure(go.Heatmap(z=corr.values, x=lab, y=lab, colorscale="RdBu_r", zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
                                   colorbar=dict(title="ρ Спирмена", thickness=10)))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 420), width="stretch")
        st.caption("Корреляция рангов между измерениями по странам: показывает, какие проблемы идут вместе. Это связь, не причинность.")
    with t4:
        rng = np.random.default_rng(0)
        base = comp.rank(ascending=False)
        sp = []
        D0 = dims.copy()
        for _ in range(100):
            wv = pd.Series(rng.uniform(0.25, 2.0, D0.shape[1]), index=D0.columns)
            alt = (D0.fillna(0) * wv).sum(axis=1) / (D0.notna() * wv).sum(axis=1)
            sp.append(base.corr(alt.rank(ascending=False), method="spearman"))
        st.metric("Устойчивость рейтинга к весам (Спирмен, 100 случайных наборов)", f"{np.mean(sp):.3f}", f"минимум {np.min(sp):.3f}", delta_color="off")
        past = _scored(w, version, year - 5, wkey, sov)["composite"]
        chg = (comp - past.reindex(comp.index)).dropna().sort_values()
        a, b = st.columns(2)
        with a:
            st.markdown("**Сильнее всего ухудшились за 5 лет**")
            top = chg.tail(10).iloc[::-1]
            st.dataframe(pd.DataFrame({"Страна": [names(i) for i in top.index], "Изменение": top.values, "Сейчас": comp.reindex(top.index).values}), hide_index=True,
                         column_config={"Изменение": st.column_config.NumberColumn(format="%+.1f"), "Сейчас": st.column_config.NumberColumn(format="%.0f")})
        with b:
            st.markdown("**Сильнее всего улучшились за 5 лет**")
            top = chg.head(10)
            st.dataframe(pd.DataFrame({"Страна": [names(i) for i in top.index], "Изменение": top.values, "Сейчас": comp.reindex(top.index).values}), hide_index=True,
                         column_config={"Изменение": st.column_config.NumberColumn(format="%+.1f"), "Сейчас": st.column_config.NumberColumn(format="%.0f")})
        st.caption("Изменение индекса это сдвиг относительного положения: страна может улучшиться в абсолютных цифрах и всё равно «ухудшиться», если другие росли быстрее.")
    st.markdown("**Открыть паспорт страны**")
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("Страна", comp.sort_values(ascending=False).index.tolist(), format_func=names, key="pr_pick", label_visibility="collapsed")
    c2.button("Паспорт →", key="pr_open", on_click=_goto, args=("ПАСПОРТ", {"cty_iso": pick}))


# ================================================================ ПАСПОРТ
def _gauge(score_, rank, n, delta=None, cover=None):
    col = _heat_color(score_) or "#333"
    arrow = ""
    if delta is not None and np.isfinite(delta):
        arrow = f"<span style='color:{C['down'] if delta > 1 else C['up'] if delta < -1 else '#8B96A5'}'>{'▲' if delta > 1 else '▼' if delta < -1 else '•'} {delta:+.0f} за 5 л.</span>"
    return (f"<div class='gauge'><div class='gbar'><div style='width:{max(2, score_):.0f}%;background:{col}'></div></div>"
            f"<div class='gval'>{score_:.0f}</div><div class='grk'>место {rank} из {n}</div><div class='gdl'>{arrow}</div></div>")


def page_passport(version):
    w = vw.get_wide(version)
    opts = vw.country_options(w)
    iso = st.selectbox("Страна", opts, index=opts.index(st.session_state.get("cty_iso", "RUS")) if st.session_state.get("cty_iso", "RUS") in opts else 0,
                       format_func=countries.short_ru, key="cty_iso", label_visibility="collapsed")
    t = countries.table()
    year = int(st.session_state.get("set_year", 2021))
    wkey = tuple(sorted({k: float(st.session_state.get(f"set_w_{k}", 1.0)) for k in problems.DIMENSIONS}.items()))
    res = _scored(w, version, year, wkey, False)
    past = _scored(w, version, year - 5, wkey, False)
    comp, dims = res["composite"], res["dims"]
    reg = vw.region_of(iso)
    st.markdown(
        f"<div class='passport'><div class='pp-flag'>{t.loc[iso, 'iso2']}</div><div class='pp-main'><div class='pp-name'>{html.escape(countries.short_ru(iso))}</div>"
        f"<div class='pp-sub'>{html.escape(str(t.loc[iso, 'name_en']))} · {html.escape(vw.REGION_RU.get(reg, reg))} · {html.escape(str(t.loc[iso, 'subregion']))} · валюта {html.escape(str(t.loc[iso, 'currency']))}</div></div>"
        f"<div class='pp-score'><div class='pp-big'>{comp.get(iso, float('nan')):.0f}</div><div class='pp-cap'>индекс проблем {year}<br>"
        + (f"место {int(comp.rank(ascending=False, method='min')[iso])} из {len(comp)}" if iso in comp.index else "нет оценки") + "</div></div></div>", unsafe_allow_html=True)

    # ключевые цифры
    meta = gdata.meta()
    kp = [("gm:total_population_with_projections", "Население"), ("gm:gdppercapita_us_inflation_adjusted", "ВВП на душу"),
          ("gm:life_expectancy_at_birth_data_from_ihme", "Продолжительность жизни"), ("gm:median_age_years", "Медианный возраст"),
          ("co2:co2_per_capita", "CO₂ на душу"), ("gm:internet_users", "Интернет")]
    cols = st.columns(len(kp))
    for c, (k, lab) in zip(cols, kp):
        s = gdata.series(w, k, iso)
        cap = gdata.last_actual_year(k)
        s = s[s.index <= cap] if cap else s
        if k == "gm:total_population_with_projections":          # у источника факты по населению кончаются в 2013: берём откалиброванную траекторию
            pp = vw.get_pop_path(version)
            s = pp.loc[iso].dropna().loc[:2025] if iso in pp.index else s
        if s.empty:
            c.metric(lab, "н/д")
            continue
        y, v = int(s.index[-1]), float(s.iloc[-1])
        xs = gdata.cross_section(w, k, y)
        rk = int(xs.rank(ascending=False, method="min").get(iso, np.nan)) if iso in xs.index else None
        val = f"{v / 1e6:,.1f} млн" if k.endswith("with_projections") and v > 1e6 else vw.fmt_val(v, "")
        c.metric(lab + (f", {meta[k]['unit']}" if meta[k]["unit"] and not k.endswith("with_projections") else ""), val, f"{y} · #{rk}/{len(xs)}" if rk else f"{y}", delta_color="off")

    a, b = st.columns([3, 2])
    with a:
        st.markdown("### Оценка по измерениям")
        pdims = past["dims"]
        for k, (name, _) in problems.DIMENSIONS.items():
            sc = dims[k].get(iso, np.nan) if k in dims else np.nan
            if not np.isfinite(sc):
                st.markdown(f"<div class='gline'><div class='gname'>{name}</div><div class='gauge'><div class='gval'>н/д</div></div></div>", unsafe_allow_html=True)
                continue
            col = dims[k].dropna()
            rank = int(col.rank(ascending=False, method="min")[iso])
            d5 = sc - pdims[k].get(iso, np.nan) if k in pdims else None
            st.markdown(f"<div class='gline'><div class='gname'>{name}</div>{_gauge(sc, rank, len(col), d5)}</div>", unsafe_allow_html=True)
        st.caption("0 = меньше всего проблем среди стран, 100 = больше всего. Место 1 = хуже всех. Стрелка показывает сдвиг за 5 лет (▲ стало хуже).")
    with b:
        st.markdown("### Профиль на фоне мира")
        short = {"hunger": "Голод", "health": "Здоровье", "economy": "Экономика", "institutions": "Институты", "environment": "Среда", "capital": "Чел. капитал", "demography": "Демография"}
        labels = [short[k] for k in dims.columns]
        world_med = dims.median().reindex(dims.columns)
        reg_isos = [i for i in dims.index if vw.region_of(i) == reg]
        reg_med = dims.loc[reg_isos].median().reindex(dims.columns) if reg_isos else world_med
        fig = go.Figure()
        for name, vals, col, fill in (("Мир (медиана)", world_med, "#7D8898", None), (f"Регион: {vw.REGION_RU.get(reg, reg)}", reg_med, C["blue"], None),
                                      (countries.short_ru(iso), dims.loc[iso] if iso in dims.index else world_med * np.nan, C["accent"], "rgba(255,159,26,.22)")):
            v = list(vals.fillna(0)) if hasattr(vals, "fillna") else list(vals)
            fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=labels + labels[:1], name=name, line=dict(color=col, width=2), fill="toself" if fill else None, fillcolor=fill))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(size=9)), bgcolor=C["bg"]), legend=dict(orientation="h", y=-0.08),
                          margin=dict(l=40, r=40, t=20, b=40))
        st.plotly_chart(style(fig, 380), width="stretch")

    worst, best = problems.diagnosis(res, iso)
    pc = res["pct"]
    st.markdown("### Диагноз")
    d1, d2 = st.columns(2)

    def line(k, p):
        v = res["values"].loc[iso, k] if k in res["values"].columns and iso in res["values"].index else np.nan
        lab = problems.label_of(k)
        return f"- **{lab}**: {vw.fmt_val(float(v), meta.get(k, {}).get('unit', ''))} (хуже, чем у {p:.0f}% стран)" if p >= 50 else f"- **{lab}**: {vw.fmt_val(float(v), meta.get(k, {}).get('unit', ''))} (лучше, чем у {100 - p:.0f}% стран)"
    with d1:
        st.markdown("<div class='dx bad'>СЛАБЫЕ СТОРОНЫ</div>", unsafe_allow_html=True)
        st.markdown("\n".join(line(k, p) for k, p in worst) or "Ярко выраженных слабых сторон по этим показателям нет.")
    with d2:
        st.markdown("<div class='dx good'>СИЛЬНЫЕ СТОРОНЫ</div>", unsafe_allow_html=True)
        st.markdown("\n".join(line(k, p) for k, p in best) or "Ярко выраженных сильных сторон по этим показателям нет.")

    st.markdown("### Похожие страны")
    pr = problems.peers(res, iso)
    if not pr.empty:
        tbl = dims.loc[pr.index].copy()
        tbl.columns = [problems.DIMENSIONS[k][0] for k in tbl.columns]
        tbl.insert(0, "Страна", [countries.short_ru(i) for i in tbl.index])
        tbl.insert(1, "Индекс", comp.reindex(tbl.index).values)
        tbl.insert(2, "Расстояние", pr["distance"].values)
        st.dataframe(tbl, hide_index=True, width="stretch", column_config={c: st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100)
                                                                           for c in tbl.columns[3:]} | {"Индекс": st.column_config.NumberColumn(format="%.0f"),
                                                                                                         "Расстояние": st.column_config.NumberColumn(format="%.1f")})
        st.button("Сравнить с похожими →", key="pp_cmp", on_click=_goto, args=("СРАВНЕНИЕ", {"cmp_c": [iso] + list(pr.index[:5])}))

    st.markdown("### Динамика индекса проблем")
    years = list(range(1995, 2024, 4)) + [2021]
    hist = _history(w, version, tuple(sorted(set(years))), wkey, False)
    if iso in hist.index:
        s = hist.loc[iso].dropna()
        regm = hist.loc[[i for i in hist.index if vw.region_of(i) == reg]].median()
        fig = go.Figure()
        fig.add_scatter(x=hist.columns, y=hist.median(), name="Мир (медиана)", line=dict(color="#7D8898", dash="dot"))
        fig.add_scatter(x=regm.index, y=regm.values, name="Регион (медиана)", line=dict(color=C["blue"]))
        fig.add_scatter(x=s.index, y=s.values, name=countries.short_ru(iso), line=dict(color=C["accent"], width=3))
        fig.update_layout(yaxis_title="Индекс проблем (0–100)", yaxis_range=[0, 100])
        st.plotly_chart(style(fig, 320), width="stretch")
    with st.expander("Все показатели паспорта: значение, год, место"):
        rows = []
        for k in res["values"].columns:
            if iso not in res["values"].index or pd.isna(res["values"].loc[iso, k]):
                continue
            col = res["values"][k].dropna()
            rows.append({"Показатель": problems.label_of(k), "Значение": float(res["values"].loc[iso, k]), "Место (1 = наибольшее значение)": int(col.rank(ascending=False, method="min")[iso]),
                         "Стран": len(col), "Проблемность, процентиль": float(pc.loc[iso, k]) if k in pc.columns and pd.notna(pc.loc[iso, k]) else np.nan})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={"Значение": st.column_config.NumberColumn(format="%.2f"),
                                                                                           "Проблемность, процентиль": st.column_config.NumberColumn(format="%.0f")})
    vw.src_box(list(res["values"].columns), "индекс и диагноз рассчитаны приложением")
    st.divider()
    vw.page_country(version, iso=iso, show_header=False)
