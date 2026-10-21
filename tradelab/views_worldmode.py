"""Глобальный режим: единая рабочая среда по странам мира на открытых данных (прошлое, прогнозы, источники).

Страницы: ОБЗОР, ИССЛЕДОВАТЕЛЬ, СРАВНЕНИЕ, СТРАНА, ПУЗЫРЬКИ, ДЕМОГРАФИЯ, ГОЛОД, КЛИМАТ, КРИЗИСЫ, ТОРГОВЛЯ, МОРЕ, КОНФЛИКТЫ, ГОРИЗОНТ,
ВАЛЮТЫ, МЕТОДЫ, ИСТОЧНИКИ. Под каждым графиком показан источник из реестра.
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import json
from pathlib import Path

from . import aging, ask, brief, catalog, countries, crisis, forecast, gdata, globe, hunger, panel as panel_mod, quality, registry, store
from . import views_global
from .ui import C, PALETTE, pct, style

REGION_COLORS = {"Africa": "#FF9F1A", "Americas": "#26C281", "Asia": "#FF4D4F", "Europe": "#4DA3FF", "Oceania": "#B38CFF"}
REGION_RU = {"Africa": "Африка", "Americas": "Америка", "Asia": "Азия", "Europe": "Европа", "Oceania": "Океания"}
KEY_PROFILE = [
    "gm:total_population_with_projections", "gm:median_age_years", "gm:children_and_elderly_per_100_adults",
    "gm:gdppercapita_us_inflation_adjusted", "gm:life_expectancy_at_birth_data_from_ihme", "gm:hdi_human_development_index",
    "wp:inflation", "co2:co2_per_capita", "en:renewables_share_elec", "gm:internet_users",
    "gm:food_supply_kilocalories_per_person_and_day", "gm:malnutrition_weight_for_age_percent_of_children_under_5"]


# ================================================================ данные и вспомогательные
@st.cache_data(show_spinner=False)
def get_wide(version):
    live = store.load_panel_long(store.connect_macro())
    extra = panel_mod.from_long(live) if len(live) else None
    return gdata.load(extra)


@st.cache_data(show_spinner=False)
def get_avail(version):
    return gdata.available(get_wide(version))


@st.cache_data(show_spinner=False)
def get_pop_path(version):
    return aging.population_path(get_wide(version))


def src_box(keys, note=""):
    """Источник под графиком: названия, организации, ссылки и лицензии из реестра."""
    ids, lines = [], []
    meta = gdata.meta()
    if any(str(k).startswith("user:") for k in keys):
        note = (note + " · " if note else "") + "«Мои данные»: загружены пользователем, источник и качество на его ответственности"
    for k in keys:
        s = meta.get(k, {}).get("source")
        if s and s not in ids:
            ids.append(s)
    for i in ids:
        d = registry.SOURCES.get(i)
        if d:
            lines.append(f"**{d['name']}**, {d['org']}: {d['url']} · лицензия: {d['license']}" + (f" · {d['note']}" if d["note"] else ""))
    updated = panel_mod.snapshot_info().get("global_updated") or panel_mod.snapshot_info().get("updated", "н/д")
    st.caption("📎 Источник: " + " | ".join(lines) + f" · снимок от {updated}" + (f" · {note}" if note else ""))


def fmt_val(v, unit=""):
    if v is None or not np.isfinite(v):
        return "н/д"
    a = abs(v)
    txt = f"{v:,.0f}" if a >= 1000 else (f"{v:.1f}" if a >= 10 else f"{v:.2f}")
    return f"{txt} {unit}".strip()


def indicator_picker(avail, key, default=None):
    """Выбор показателя из всех доступных; поле поддерживает поиск набором текста."""
    options = avail["key"].tolist()
    info = avail.set_index("key")
    labels = {k: f"{info.loc[k, 'group']} · {info.loc[k, 'label']}" + (f" ({info.loc[k, 'unit']})" if info.loc[k, "unit"] else "") for k in options}
    idx = options.index(default) if default in options else 0
    return st.selectbox("Показатель (можно искать по названию)", options, index=idx, format_func=lambda k: labels[k], key=key)


def default_year(w, k):
    """Год по умолчанию: последний фактический, где данных больше всего."""
    s = w[k].dropna()
    yrs = s.index.get_level_values("year")
    counts = s.groupby(yrs).size()
    cap = gdata.last_actual_year(k)
    ok = counts[counts >= 0.6 * counts.max()]
    y = int(ok.index.max()) if len(ok) else int(yrs.max())
    return min(y, cap) if cap else y


def country_options(w):
    isos = sorted(w.index.get_level_values("iso3").unique(), key=countries.short_ru)
    return [i for i in isos if i in countries.table().index]


def region_of(iso):
    t = countries.table()
    return t.loc[iso, "region"] if iso in t.index and isinstance(t.loc[iso, "region"], str) else "Прочее"


def value_map(values, k, year, title_extra="", height=600, key_suffix=""):
    """Карта со значениями показателя: подсказка по странам с рангом, единицей и пометкой прогноза."""
    m = gdata.meta()[k]
    if values.empty:
        st.info("Для выбранного года данных нет.")
        return
    lo, hi = float(values.quantile(0.03)), float(values.quantile(0.97))
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max()) + 1e-9
    rank = values.rank(ascending=False, method="min")
    proj = m.get("projection_from")
    colors, infos = {}, {}
    for iso, v in values.items():
        colors[iso] = globe.scale_color(float(v), lo, hi, m["better"])
        tag = "<br><span style='color:#B38CFF'>прогноз ООН, не факт</span>" if proj and year >= proj else ""
        infos[iso] = (f"<b style='color:#FF9F1A'>{countries.short_ru(iso)}</b><br>{m['label']} ({year}): <b>{fmt_val(float(v), m['unit'])}</b>"
                      f"<br>место {int(rank[iso])} из {len(values)}{tag}")
    leg = {"down": [("ниже (лучше)", "#1e2634"), ("выше (хуже)", "#FF3C3C")], "up": [("ниже (хуже)", "#FF4D4F"), ("выше (лучше)", "#26C281")],
           "none": [("ниже", "#1e2634"), ("выше", "#FF9F1A")]}[m["better"]]
    leg = [(f"{leg[0][0]} · {fmt_val(lo)}", leg[0][1]), (f"{leg[1][0]} · {fmt_val(hi)}", leg[1][1])]
    mode = st.radio("Вид карты", ["Карта", "Глобус"], horizontal=True, key=f"vm_mode_{k}_{key_suffix}")
    views_global.world_map(colors, infos, f"{m['label'].upper()} · {year}{title_extra}", leg, mode="flat" if mode == "Карта" else "globe",
                           height=height)


def series_chart(w, k, isos, y0=None, height=380):
    """Линии стран по показателю: факт сплошной, прогноз ООН пунктиром."""
    m = gdata.meta()[k]
    proj = m.get("projection_from")
    fig = go.Figure()
    for i, iso in enumerate(isos):
        s = gdata.series(w, k, iso)
        if y0:
            s = s[s.index >= y0]
        if s.empty:
            continue
        col = PALETTE[i % len(PALETTE)]
        if proj:
            a, b = s[s.index < proj], s[s.index >= proj - 1]
            fig.add_scatter(x=a.index, y=a.values, name=countries.short_ru(iso), line=dict(color=col), mode="lines")
            fig.add_scatter(x=b.index, y=b.values, name=f"{countries.short_ru(iso)} (прогноз ООН)", line=dict(color=col, dash="dot"),
                            mode="lines", showlegend=False)
        else:
            fig.add_scatter(x=s.index, y=s.values, name=countries.short_ru(iso), line=dict(color=col), mode="lines")
    fig.update_layout(yaxis_title=f"{m['label']}" + (f", {m['unit']}" if m["unit"] else ""))
    st.plotly_chart(style(fig, height), width="stretch")


# ================================================================ ОБЗОР МИРА
def page_overview(version):
    w, avail = get_wide(version), get_avail(version)
    st.markdown("### Мир одним взглядом")
    path = get_pop_path(version)
    cols = st.columns(5)
    world_2025 = float(path[2025].sum()) if 2025 in path.columns else float("nan")
    world_2100 = float(path[2100].sum()) if 2100 in path.columns else float("nan")
    cols[0].metric("Население 2025 (сумма стран)", f"{world_2025 / 1e9:.2f} млрд", f"{(world_2100 / world_2025 - 1):+.0%} к 2100 (ООН)", delta_color="off")
    peaks = aging.population_peaks(w)
    cols[1].metric("Стран с уже достигнутым пиком населения", f"{int((peaks['peak_year'] <= 2025).sum())} из {len(peaks)}")
    age25 = gdata.cross_section(w, aging.MEDIAN_AGE, 2025)
    cols[2].metric("Старейшая страна 2025", countries.short_ru(age25.idxmax()), f"медианный возраст {age25.max():.0f} лет", delta_color="off")
    try:
        t = pd.read_csv(panel_mod.SNAP / "global_temp_annual.csv.gz")
        t = t[t["Source"] == "GCAG"].sort_values("Year")
        cols[3].metric("Температура: аномалия", f"{t['Mean'].iloc[-1]:+.2f} °C", f"{int(t['Year'].iloc[-1])} год (к 1951–1980)", delta_color="off")
    except Exception:
        pass
    try:
        c = pd.read_csv(panel_mod.SNAP / "co2_ppm_annual.csv.gz").sort_values("Year")
        cols[4].metric("CO₂ в атмосфере", f"{c['Mean'].iloc[-1]:.0f} ppm", f"{int(c['Year'].iloc[-1])} год, Мауна-Лоа", delta_color="off")
    except Exception:
        pass
    layer = st.radio("Слой карты", ["Медианный возраст", "Прирост населения", "Продовольственная уязвимость", "CO₂ на душу", "ВВП на душу"],
                     horizontal=True)
    ykey = {"Медианный возраст": ("gm:median_age_years", 2025), "Прирост населения": ("gm:population_growth_annual_percent_with_projections", 2025),
            "CO₂ на душу": ("co2:co2_per_capita", None), "ВВП на душу": ("gm:gdppercapita_us_inflation_adjusted", None)}
    if layer == "Продовольственная уязвимость":
        yr = 2022
        idx, _ = hunger.index(w, yr)
        colors = {i: globe.scale_color(float(v), float(idx.quantile(.03)), float(idx.quantile(.97)), "down") for i, v in idx.items()}
        infos = {i: f"<b style='color:#FF9F1A'>{countries.short_ru(i)}</b><br>индекс уязвимости {yr}: <b>{v:.0f}</b> из 100" for i, v in idx.items()}
        views_global.world_map(colors, infos, f"ПРОДОВОЛЬСТВЕННАЯ УЯЗВИМОСТЬ · {yr}", [("ниже", "#1e2634"), ("выше", "#FF3C3C")], height=560)
        src_box(list(hunger.COMPONENTS), "индекс 0–100 рассчитан приложением, см. страницу ГОЛОД")
    else:
        k, y = ykey[layer]
        y = y or default_year(w, k)
        value_map(gdata.cross_section(w, k, y), k, y, height=560, key_suffix="ov")
        src_box([k])
    a, b = st.columns(2)
    with a:
        st.markdown("**Где население растёт быстрее всего к 2050 (ООН)**")
        pk = peaks.assign(g=(peaks["y2050"] / peaks["now"] - 1)).sort_values("g", ascending=False).head(8)
        st.dataframe(pd.DataFrame({"Страна": [countries.short_ru(i) for i in pk.index], "К 2050": pk["g"]}), hide_index=True,
                     column_config={"К 2050": st.column_config.NumberColumn(format="percent")})
    with b:
        st.markdown("**Где население сократится сильнее всего к 2050 (ООН)**")
        pk = peaks.assign(g=(peaks["y2050"] / peaks["now"] - 1)).sort_values("g").head(8)
        st.dataframe(pd.DataFrame({"Страна": [countries.short_ru(i) for i in pk.index], "К 2050": pk["g"]}), hide_index=True,
                     column_config={"К 2050": st.column_config.NumberColumn(format="percent")})
    src_box([aging.POPULATION, aging.GROWTH_POP])
    st.info(f"В базе {len(avail)} показателей по {w.index.get_level_values('iso3').nunique()} странам и территориям; "
            "все карты и графики ниже показывают источник. Прогнозы ООН помечены как прогноз, а статистические прогнозы приложения "
            "всегда сопровождаются проверкой на истории.")


# ================================================================ ИССЛЕДОВАТЕЛЬ
def page_explorer(version):
    w, avail = get_wide(version), get_avail(version)
    k = indicator_picker(avail, "ex_k", "gm:median_age_years")
    m = gdata.meta()[k]
    s_all = w[k].dropna()
    ys = sorted(s_all.index.get_level_values("year").unique())
    y = st.select_slider("Год", options=ys, value=default_year(w, k), key=f"ex_y_{k}")
    proj = m.get("projection_from")
    if proj and y >= proj:
        st.warning(f"{y} год это прогноз ООН (средний вариант), а не наблюдение.")
    allowed = global_filter(version, "gf_ex")
    vals = gdata.cross_section(w, k, y)
    vals = vals[vals.index.isin(allowed)]
    c = st.columns(4)
    c[0].metric("Стран с данными", len(vals))
    c[1].metric("Медиана", fmt_val(float(vals.median()), m["unit"]) if len(vals) else "н/д")
    if len(vals):
        c[2].metric("Максимум", f"{countries.short_ru(vals.idxmax())}: {fmt_val(float(vals.max()), m['unit'])}")
        c[3].metric("Минимум", f"{countries.short_ru(vals.idxmin())}: {fmt_val(float(vals.min()), m['unit'])}")
    value_map(vals, k, y, key_suffix="ex")
    src_box([k])
    a, b = st.columns(2)
    with a:
        st.markdown("**Лидеры и аутсайдеры**")
        top = vals.sort_values(ascending=False)
        tbl = pd.concat([top.head(10), top.tail(10)])
        st.dataframe(pd.DataFrame({"Страна": [countries.short_ru(i) for i in tbl.index], "Значение": tbl.values}), hide_index=True, height=420)
    with b:
        st.markdown("**Распределение по странам**")
        fig = go.Figure(go.Histogram(x=vals.values, nbinsx=30, marker_color=C["accent"]))
        fig.update_layout(xaxis_title=f"{m['label']}, {m['unit']}", yaxis_title="Стран")
        st.plotly_chart(style(fig, 420), width="stretch")
    st.markdown("**Динамика выбранных стран**")
    sel = st.multiselect("Страны", country_options(w), default=[i for i in ("RUS", "USA", "CHN", "IND", "DEU", "BRA") if i in country_options(w)],
                         format_func=countries.short_ru, key=f"ex_c_{k}")
    series_chart(w, k, sel, height=400)
    src_box([k])
    out = pd.DataFrame({"iso3": vals.index, "country": [countries.short_ru(i) for i in vals.index], "value": vals.values, "year": y, "indicator": m["label"],
                        "unit": m["unit"], "source": registry.SOURCES[m["source"]]["name"]})
    st.download_button("Скачать срез (CSV)", out.to_csv(index=False).encode("utf-8-sig"), f"{k.replace(':', '_')}_{y}.csv", "text/csv")


# ================================================================ СТРАНА (досье и прогноз)
@st.cache_data(show_spinner=False)
def run_backtest_cached(version, k, horizon):
    w = get_wide(version)
    ser = {iso: w.loc[iso, k].dropna() for iso in w.index.get_level_values("iso3").unique() if w.loc[iso, k].notna().any()}
    cap = gdata.last_actual_year(k)
    return forecast.run_backtest(ser, horizon=horizon, last_year_cap=cap)


def page_country(version):
    w, avail = get_wide(version), get_avail(version)
    iso = st.selectbox("Страна", country_options(w), index=country_options(w).index("RUS") if "RUS" in country_options(w) else 0,
                       format_func=countries.short_ru, key="cty_iso")
    t = countries.table()
    st.markdown(f"### {countries.short_ru(iso)}")
    st.caption(f"{t.loc[iso, 'name_en']} · регион: {REGION_RU.get(region_of(iso), region_of(iso))} · валюта: {t.loc[iso, 'currency']}")
    meta = gdata.meta()
    with st.expander("📄 Досье для печати / отправки (HTML, из браузера сохраняется в PDF)"):
        if st.button("Подготовить досье", key=f"brief_btn_{iso}"):
            with st.spinner("Собираю досье…"):
                b = brief.build(w, iso, get_pop_path(version))
            st.session_state["brief_html"] = (iso, brief.to_html(b), b["bullets"])
        got = st.session_state.get("brief_html")
        if got and got[0] == iso:
            for t, _ in got[2]:
                st.markdown(f"- {t}")
            st.download_button("Скачать досье (HTML)", got[1].encode("utf-8"), f"dossier_{iso}.html", "text/html")
    cols = st.columns(4)
    shown = 0
    for k in KEY_PROFILE:
        s = gdata.series(w, k, iso)
        if s.empty:
            continue
        last_actual = gdata.last_actual_year(k)
        s_act = s[s.index <= last_actual] if last_actual else s
        if s_act.empty:
            continue
        y = int(s_act.index[-1])
        v = float(s_act.iloc[-1])
        xs = gdata.cross_section(w, k, y)
        rk = int(xs.rank(ascending=False, method="min").get(iso, np.nan)) if iso in xs.index else None
        with cols[shown % 4]:
            st.metric(meta[k]["label"], fmt_val(v, meta[k]["unit"]), f"{y} · место {rk} из {len(xs)}" if rk else f"{y}", delta_color="off")
            spark = s_act.tail(30)
            fig = go.Figure(go.Scatter(x=spark.index, y=spark.values, line=dict(color=C["accent"], width=1.6), mode="lines"))
            fig.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=C["bg"], plot_bgcolor=C["bg"])
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"spark_{iso}_{k}")
        shown += 1
    src_box([k for k in KEY_PROFILE if not gdata.series(w, k, iso).empty])

    st.markdown("#### Прогноз показателя")
    pool = avail[avail["key"].map(lambda k: len(gdata.series(w, k, iso)) >= 12)]
    kk = indicator_picker(pool, "cty_k", "gm:median_age_years" if "gm:median_age_years" in pool["key"].values else pool["key"].iloc[0])
    m = meta[kk]
    hz = st.radio("Горизонт, лет", [5, 10], horizontal=True, key="cty_h")
    with st.spinner("Проверяю методы на истории всех стран…"):
        bt = run_backtest_cached(version, kk, hz)
    if not bt.mase:
        st.warning("Для этого показателя мало данных, чтобы проверить прогноз на истории.")
        return
    cap = gdata.last_actual_year(kk)
    s = gdata.series(w, kk, iso)
    s_act = s[s.index <= cap] if cap else s
    fc = forecast.forecast_series(s_act, bt, hz)
    fig = go.Figure()
    hist = s_act.tail(45)
    fig.add_scatter(x=hist.index, y=hist.values, name="Факт", line=dict(color="#E8ECF1", width=2), mode="lines")
    if fc is not None:
        fig.add_scatter(x=fc["year"], y=fc["p90"], line=dict(width=0), showlegend=False, mode="lines")
        fig.add_scatter(x=fc["year"], y=fc["p10"], fill="tonexty", fillcolor="rgba(255,159,26,0.20)", line=dict(width=0), name="P10–P90 (по ошибкам на истории)", mode="lines")
        fig.add_scatter(x=fc["year"], y=fc["mean"], name=f"Прогноз приложения: {forecast.LABELS[fc['method'].iloc[0]]}", line=dict(color=C["accent"], width=2.2), mode="lines")
    if cap:
        un = s[s.index >= cap]
        fig.add_scatter(x=un.index, y=un.values, name="Прогноз ООН (средний вариант)", line=dict(color=C["violet"], dash="dot"), mode="lines")
    fig.update_layout(yaxis_title=f"{m['label']}, {m['unit']}")
    st.plotly_chart(style(fig, 420), width="stretch")
    best = bt.best
    sk = forecast.skill_table(bt)
    base = sk[sk["Метод"] == forecast.LABELS["naive"]]
    kskill = float(1 - bt.mase[best][min(hz, len(bt.mase[best])) - 1] / bt.mase["naive"][min(hz, len(bt.mase["naive"])) - 1])
    c = st.columns(4)
    c[0].metric("Лучший метод по истории", forecast.LABELS[best])
    c[1].metric(f"Навык на {hz} л. против «ничего не меняется»", f"{kskill:+.0%}")
    c[2].metric("Рядов в проверке", bt.n_series)
    c[3].metric("Покрытие интервала P10–P90 на поздних годах", pct(bt.coverage, 0) if np.isfinite(bt.coverage) else "н/д")
    if kskill < 0.05:
        st.warning("Метод почти не лучше наивного прогноза «останется как сейчас»: прогнозу этого показателя доверять осторожно.")
    cb, ratio = forecast.country_backtest(s_act, best, hz)
    if cb is not None:
        verdict = "лучше" if ratio < 0.95 else "примерно так же" if ratio < 1.05 else "хуже"
        st.caption(f"**Именно для этой страны** ({len(cb)} прошлых прогнозов на {hz} л.): метод ошибался в {ratio:.2f} раза от ошибки «останется как сейчас», "
                   f"то есть {verdict}, чем наивный прогноз. Общий навык по всем странам выше может не соответствовать этой стране.")
        with st.expander("Прошлые прогнозы для этой страны"):
            fig2 = go.Figure()
            fig2.add_scatter(x=cb["target"], y=cb["actual"], name="Факт", line=dict(color="#E8ECF1"))
            fig2.add_scatter(x=cb["target"], y=cb["forecast"], name=f"Прогноз за {hz} л. до даты", line=dict(color=C["accent"], dash="dash"))
            st.plotly_chart(style(fig2, 300), width="stretch")
    with st.expander("Все методы и их точность на истории"):
        st.dataframe(sk, hide_index=True, width="stretch")
        st.caption("MASE: средняя ошибка в единицах типичного годового изменения ряда страны; навык = 1 − MASE метода / MASE наивного. "
                   "Проверка на прошлых прогнозах всех стран с шагом 1–3 года.")
    src_box([kk], "статистический прогноз рассчитан приложением")

    st.markdown("#### Профиль относительно мира")
    keys = [k for k in KEY_PROFILE if not gdata.series(w, k, iso).empty][:10]
    if len(keys) >= 4:
        pcts = []
        labels = []
        for k in keys:
            ca = gdata.last_actual_year(k)
            s = gdata.series(w, k, iso)
            s = s[s.index <= ca] if ca else s
            y = int(s.index[-1])
            xs = gdata.cross_section(w, k, y)
            r = float((xs < s.iloc[-1]).mean())
            pcts.append(1 - r if meta[k]["better"] == "down" else r)
            labels.append(meta[k]["label"])
        fig = go.Figure(go.Scatterpolar(r=pcts + [pcts[0]], theta=labels + [labels[0]], fill="toself", line=dict(color=C["accent"]),
                                        fillcolor="rgba(255,159,26,0.25)"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%"), bgcolor=C["bg"]))
        st.plotly_chart(style(fig, 440), width="stretch")
        st.caption("Чем дальше от центра, тем лучше относительно других стран (для «чем ниже, тем лучше» шкала перевёрнута).")
    idx, contrib = hunger.index(w, 2022)
    if iso in idx.index:
        st.markdown("#### Риски")
        c = st.columns(3)
        c[0].metric("Продовольственная уязвимость", f"{idx[iso]:.0f} из 100", f"место {int(idx.rank(ascending=False)[iso])} из {len(idx)}", delta_color="off")
        c[1].metric("Население к 2050 (ООН)", pct(float(get_pop_path(version).loc[iso, 2050] / get_pop_path(version).loc[iso, 2025] - 1), 0) if iso in get_pop_path(version).index else "н/д")
        age = gdata.series(w, aging.MEDIAN_AGE, iso)
        if not age.empty:
            c[2].metric("Медианный возраст 2025 → 2050", f"{age.get(2025, np.nan):.0f} → {age.get(2050, np.nan):.0f}")


# ================================================================ ФИЛЬТРЫ и СРАВНЕНИЕ
GROUPS = {
    "G7": ["USA", "JPN", "DEU", "GBR", "FRA", "ITA", "CAN"],
    "БРИКС": ["BRA", "RUS", "IND", "CHN", "ZAF", "EGY", "ETH", "IRN", "ARE"],
    "G20 (основные)": ["USA", "CHN", "JPN", "DEU", "IND", "GBR", "FRA", "ITA", "BRA", "CAN", "RUS", "KOR", "AUS", "MEX", "IDN", "TUR", "SAU", "ARG", "ZAF"],
    "ЕС (крупнейшие)": ["DEU", "FRA", "ITA", "ESP", "POL", "NLD", "BEL", "SWE", "AUT", "GRC", "PRT", "CZE"],
    "СНГ": ["RUS", "BLR", "KAZ", "UZB", "ARM", "AZE", "KGZ", "TJK", "MDA", "UKR"],
    "Самые населённые": ["CHN", "IND", "USA", "IDN", "PAK", "NGA", "BRA", "BGD", "RUS", "MEX"],
}


def global_filter(version, key="gf"):
    """Фильтр стран: регион, население, до трёх условий по значениям показателей. Возвращает множество iso3."""
    w, avail = get_wide(version), get_avail(version)
    allc = country_options(w)
    with st.expander("🔎 Фильтры стран", expanded=False):
        c1, c2 = st.columns(2)
        regs = c1.multiselect("Регион", list(REGION_RU), format_func=REGION_RU.get, key=f"{key}_reg")
        pop = get_pop_path(version)
        lo_hi = c2.slider("Население 2024, млн (лог. шкала)", 0.0, 1500.0, (0.0, 1500.0), key=f"{key}_pop")
        n_rules = st.number_input("Условий по показателям", 0, 3, 0, key=f"{key}_n")
        rules = []
        for i in range(int(n_rules)):
            a, b, c = st.columns([3, 1, 2])
            k = a.selectbox("Показатель", avail["key"].tolist(), format_func=lambda x: avail.set_index("key").loc[x, "label"], key=f"{key}_k{i}")
            op = b.selectbox("Условие", [">", "<"], key=f"{key}_o{i}")
            y = default_year(w, k)
            xs = gdata.cross_section(w, k, y)
            if len(xs):
                thr = c.number_input(f"Порог ({y})", value=float(xs.median()), key=f"{key}_t{i}")
                rules.append((k, op, thr, y))
        picks = st.multiselect("Только эти страны (пусто = все)", allc, format_func=countries.short_ru, key=f"{key}_pick")
    out = set(allc)
    if regs:
        out &= {i for i in allc if region_of(i) in regs}
    if lo_hi != (0.0, 1500.0) and 2024 in pop.columns:
        m = pop[2024] / 1e6
        out &= set(m[(m >= lo_hi[0]) & (m <= lo_hi[1])].index)
    for k, op, thr, y in rules:
        xs = gdata.cross_section(w, k, y)
        out &= set(xs[xs > thr].index if op == ">" else xs[xs < thr].index)
    if picks:
        out &= set(picks)
    if len(out) != len(allc):
        st.caption(f"Фильтр активен: выбрано {len(out)} из {len(allc)} стран.")
    return out


def page_compare(version):
    w, avail = get_wide(version), get_avail(version)
    meta = gdata.meta()
    st.markdown("### Сравнение стран")
    opts = country_options(w)
    c1, c2 = st.columns([1, 3])
    preset = c1.selectbox("Быстрый выбор группы", ["—"] + list(GROUPS), key="cmp_preset")
    if preset != "—" and st.session_state.get("cmp_last_preset") != preset:
        st.session_state["cmp_c"] = [i for i in GROUPS[preset] if i in opts]
        st.session_state["cmp_last_preset"] = preset
    if "cmp_c" not in st.session_state:
        st.session_state["cmp_c"] = [i for i in ("RUS", "CHN", "USA", "IND", "DEU") if i in opts]
    sel = c2.multiselect("Страны (до 12)", opts, format_func=countries.short_ru, key="cmp_c", max_selections=12)
    if len(sel) < 2:
        return st.info("Выбери минимум две страны.")
    defaults = [k for k in KEY_PROFILE if k in avail["key"].values][:8]
    up = st.file_uploader("Добавить свои данные (CSV: колонки country или iso3, year, value)", type="csv", key="cmp_up")
    user_keys = []
    if up is not None:
        try:
            ser, bad = gdata.from_user_csv(pd.read_csv(up), "user:" + Path(up.name).stem)
            w = w.join(ser, how="outer")
            meta = dict(meta)
            meta[ser.name] = {"label": f"Мои данные: {Path(up.name).stem}", "unit": "", "group": "Мои данные", "better": "none", "source": None, "projection_from": None}
            avail = pd.concat([avail, pd.DataFrame([{"key": ser.name, "label": meta[ser.name]["label"], "group": "Мои данные", "unit": "", "source": "user",
                                                    "better": "none", "countries": ser.index.get_level_values("iso3").nunique(),
                                                    "first": int(ser.index.get_level_values("year").min()), "last": int(ser.index.get_level_values("year").max())}])])
            user_keys = [ser.name]
            st.success(f"Загружено рядов по странам: {ser.index.get_level_values('iso3').nunique()}" + (f"; пропущено строк: {bad}" if bad else ""))
        except Exception as exc:
            st.error(f"Не удалось прочитать файл: {exc}")
    keys = st.multiselect("Показатели", avail["key"].tolist(), default=defaults + user_keys, key="cmp_k" + ("_u" if user_keys else ""),
                          format_func=lambda k: f"{meta[k]['group']} · {meta[k]['label']}")
    if not keys:
        return st.info("Выбери хотя бы один показатель.")
    rows = {}
    for k in keys:
        cap = gdata.last_actual_year(k)
        r = {}
        for iso in sel:
            s = gdata.series(w, k, iso)
            s = s[s.index <= cap] if cap else s
            r[iso] = (float(s.iloc[-1]), int(s.index[-1])) if len(s) else (np.nan, None)
        rows[k] = r
    tab = pd.DataFrame({countries.short_ru(i): [rows[k][i][0] for k in keys] for i in sel}, index=[meta[k]["label"] + (f", {meta[k]['unit']}" if meta[k]["unit"] else "") for k in keys])
    years = pd.DataFrame({countries.short_ru(i): [rows[k][i][1] for k in keys] for i in sel}, index=tab.index)
    st.markdown("**Последние фактические значения** (зелёное — лучшее среди выбранных, красное — худшее, где у показателя есть «лучше/хуже»)")

    def shade(row):
        k = keys[list(tab.index).index(row.name)]
        b = meta[k]["better"]
        if b == "none" or row.notna().sum() < 2:
            return [""] * len(row)
        best, worst = (row.max(), row.min()) if b == "up" else (row.min(), row.max())
        return ["background-color: rgba(38,194,129,.28)" if v == best else "background-color: rgba(255,77,79,.28)" if v == worst else "" for v in row]
    st.dataframe(tab.style.apply(shade, axis=1).format("{:,.2f}", na_rep="н/д"), width="stretch")
    with st.expander("За какой год значения"):
        st.dataframe(years, width="stretch")
    st.download_button("Скачать таблицу (CSV)", tab.to_csv().encode("utf-8-sig"), "compare.csv", "text/csv")
    src_box(keys)

    t1, t2, t3 = st.tabs(["Динамика", "Профиль (радар)", "Две оси"])
    with t1:
        k = st.selectbox("Показатель", keys, format_func=lambda k: meta[k]["label"], key="cmp_dyn")
        a, b = st.columns(2)
        y0 = a.slider("С какого года", 1950, 2020, 1990, key="cmp_y0")
        rebase = b.checkbox("Индекс = 100 в начале периода", key="cmp_rb")
        fig = go.Figure()
        proj = meta[k].get("projection_from")
        for i, iso in enumerate(sel):
            s = gdata.series(w, k, iso)
            s = s[s.index >= y0]
            if s.empty:
                continue
            if rebase and s.iloc[0] != 0:
                s = s / s.iloc[0] * 100
            col = PALETTE[i % len(PALETTE)]
            if proj:
                fig.add_scatter(x=s[s.index < proj].index, y=s[s.index < proj].values, name=countries.short_ru(iso), line=dict(color=col))
                fig.add_scatter(x=s[s.index >= proj - 1].index, y=s[s.index >= proj - 1].values, line=dict(color=col, dash="dot"), showlegend=False)
            else:
                fig.add_scatter(x=s.index, y=s.values, name=countries.short_ru(iso), line=dict(color=col))
        fig.update_layout(yaxis_title="Индекс" if rebase else f"{meta[k]['label']}, {meta[k]['unit']}")
        st.plotly_chart(style(fig, 420), width="stretch")
        src_box([k], "пунктир: прогноз ООН")
    with t2:
        if len(keys) < 3:
            st.info("Для радара нужно минимум три показателя.")
        else:
            fig = go.Figure()
            labels = [meta[k]["label"] for k in keys]
            for i, iso in enumerate(sel):
                vals = []
                for k in keys:
                    v, y = rows[k][iso]
                    xs = gdata.cross_section(w, k, y) if y else pd.Series(dtype=float)
                    p = float((xs < v).mean()) if len(xs) and np.isfinite(v) else 0.0
                    vals.append(1 - p if meta[k]["better"] == "down" else p)
                fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], name=countries.short_ru(iso),
                                              line=dict(color=PALETTE[i % len(PALETTE)])))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%"), bgcolor=C["bg"]))
            st.plotly_chart(style(fig, 480), width="stretch")
            st.caption("Положение страны среди всех стран мира по показателю (процентиль); для «чем ниже, тем лучше» шкала перевёрнута.")
    with t3:
        a, b = st.columns(2)
        kx = a.selectbox("Ось X", keys, index=0, format_func=lambda k: meta[k]["label"], key="cmp_x")
        ky = b.selectbox("Ось Y", keys, index=min(1, len(keys) - 1), format_func=lambda k: meta[k]["label"], key="cmp_y")
        yx, yy = default_year(w, kx), default_year(w, ky)
        allx, ally = gdata.cross_section(w, kx, yx), gdata.cross_section(w, ky, yy)
        d = pd.concat([allx.rename("x"), ally.rename("y")], axis=1).dropna()
        fig = go.Figure(go.Scatter(x=d["x"], y=d["y"], mode="markers", marker=dict(color="#3a4558", size=5), name="Все страны", hoverinfo="skip"))
        pts = d.loc[[i for i in sel if i in d.index]]
        fig.add_scatter(x=pts["x"], y=pts["y"], mode="markers+text", text=[countries.short_ru(i) for i in pts.index], textposition="top center",
                        marker=dict(color=C["accent"], size=12), name="Выбранные")
        fig.update_layout(xaxis_title=f"{meta[kx]['label']} ({yx})", yaxis_title=f"{meta[ky]['label']} ({yy})")
        st.plotly_chart(style(fig, 460), width="stretch")
        if len(d) > 20:
            st.caption(f"Корреляция по всем {len(d)} странам: Спирмен ρ = {d['x'].corr(d['y'], method='spearman'):.2f}. Корреляция не доказывает причинность.")
        src_box([kx, ky])


# ================================================================ ПУЗЫРЬКИ
def page_bubbles(version):
    w, avail = get_wide(version), get_avail(version)
    path = get_pop_path(version)
    st.markdown("### Пузырьки: как страны движутся во времени")
    st.caption("Две оси по выбору, размер = население (по годовой траектории, согласованной с прогнозом ООН), цвет = регион. "
               "Нажми ▶ под графиком или двигай ползунок.")
    c1, c2, c3 = st.columns(3)
    with c1:
        kx = indicator_picker(avail, "bb_x", "gm:gdppercapita_us_inflation_adjusted")
    with c2:
        ky = indicator_picker(avail, "bb_y", "gm:life_expectancy_at_birth_data_from_ihme")
    logx = c3.checkbox("Логарифмическая ось X", value=True)
    allowed = global_filter(version, "gf_bb")
    mx, my = gdata.meta()[kx], gdata.meta()[ky]
    capy = min(filter(None, [gdata.last_actual_year(kx), gdata.last_actual_year(ky)]), default=None)
    ys = sorted(set(w[kx].dropna().index.get_level_values("year")) & set(w[ky].dropna().index.get_level_values("year")))
    ys = [y for y in ys if y >= 1960 and (capy is None or y <= capy)]
    if len(ys) < 5 or not allowed:
        return st.info("У этой пары показателей мало общих лет.")
    step = max(1, len(ys) // 70)
    ys = ys[::step] + ([ys[-1]] if ys[-1] not in ys[::step] else [])
    frames, allx, ally = [], [], []
    for y in ys:
        d = pd.concat([gdata.cross_section(w, kx, y).rename("x"), gdata.cross_section(w, ky, y).rename("y")], axis=1).dropna()
        d = d[d.index.isin(countries.table().index) & d.index.isin(allowed)]
        if logx:
            d = d[d["x"] > 0]
        pop = path[y].reindex(d.index) if y in path.columns else pd.Series(np.nan, index=d.index)
        d = d.assign(pop=pop.fillna(pop.median() if pop.notna().any() else 1e6))
        d["region"] = [region_of(i) for i in d.index]
        frames.append((y, d))
        allx += list(d["x"]); ally += list(d["y"])
    xr = [float(np.quantile(allx, 0.002)), float(np.quantile(allx, 0.998))]
    yr = [float(np.quantile(ally, 0.002)), float(np.quantile(ally, 0.998))]
    if logx:
        xr = [np.log10(xr[0]), np.log10(xr[1])]
    pad = (yr[1] - yr[0]) * 0.05
    yr = [yr[0] - pad, yr[1] + pad]
    sizeref = 2.0 * max(float(f["pop"].max()) for _, f in frames) / (55.0 ** 2)

    def trace(d):
        return go.Scatter(x=d["x"], y=d["y"], mode="markers", text=[countries.short_ru(i) for i in d.index], customdata=d["pop"] / 1e6,
                          marker=dict(size=d["pop"], sizemode="area", sizeref=sizeref, sizemin=3, opacity=0.78,
                                      color=[REGION_COLORS.get(r, "#8B96A5") for r in d["region"]], line=dict(width=0.5, color="#0b0f14")),
                          hovertemplate="<b>%{text}</b><br>X: %{x:,.2f}<br>Y: %{y:,.2f}<br>население: %{customdata:,.0f} млн<extra></extra>")
    fig = go.Figure(data=[trace(frames[0][1])],
                    frames=[go.Frame(data=[trace(d)], name=str(y)) for y, d in frames])
    fig.update_layout(
        xaxis=dict(title=f"{mx['label']}" + (f", {mx['unit']}" if mx["unit"] else ""), type="log" if logx else "linear",
                   range=xr),
        yaxis=dict(title=f"{my['label']}" + (f", {my['unit']}" if my["unit"] else ""), range=yr),
        updatemenus=[dict(type="buttons", x=0.0, y=-0.18, direction="left", buttons=[
            dict(label="▶ Пуск", method="animate", args=[None, dict(frame=dict(duration=350, redraw=True), fromcurrent=True, transition=dict(duration=250))]),
            dict(label="❚❚ Пауза", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")])])],
        sliders=[dict(x=0.12, len=0.88, y=-0.1, currentvalue=dict(prefix="Год: "), steps=[
            dict(label=str(y), method="animate", args=[[str(y)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
            for y, _ in frames])])
    st.plotly_chart(style(fig, 640), width="stretch")
    st.caption("Цвета: " + ", ".join(f"<span style='color:{c}'>■</span> {REGION_RU[r]}" for r, c in REGION_COLORS.items()), unsafe_allow_html=True)
    st.caption("Размер пузырька до 2013 года факт, дальше годовая траектория по приросту ООН, откалиброванная к точкам 2050 и 2100 источника.")
    src_box([kx, ky, aging.POPULATION])


# ================================================================ ДЕМОГРАФИЯ
def page_demography(version):
    w, avail = get_wide(version), get_avail(version)
    path = get_pop_path(version)
    st.markdown("### Демография: где постареет мир и что это значит для экономики")
    tabs = st.tabs(["Возраст на карте", "Рейтинги", "Население и рождаемость", "Влияние на экономику"])
    with tabs[0]:
        ys = sorted(w[aging.MEDIAN_AGE].dropna().index.get_level_values("year").unique())
        y = st.select_slider("Год (после 2024 это прогноз ООН)", options=ys, value=2025, key="dm_year")
        if y >= 2024:
            st.warning(f"{y} год это прогноз ООН (средний вариант), а не наблюдение.")
        vals = gdata.cross_section(w, aging.MEDIAN_AGE, y)
        value_map(vals, aging.MEDIAN_AGE, y, key_suffix="dm")
        src_box([aging.MEDIAN_AGE])
        dep = gdata.cross_section(w, aging.DEPENDENCY, y)
        if len(dep):
            st.markdown(f"**Нагрузка на работающих в {y} году: дети и пожилые на 100 взрослых.** "
                        f"Выше всех: {countries.short_ru(dep.idxmax())} ({dep.max():.0f}), ниже всех: {countries.short_ru(dep.idxmin())} ({dep.min():.0f}).")
            src_box([aging.DEPENDENCY])
    with tabs[1]:
        a, b = st.columns(2)
        y0 = a.selectbox("Год для рейтинга «самые старые»", [2000, 2025, 2050, 2075, 2100], index=1)
        old, young = aging.ranking(w, y0, n=15)
        with a:
            st.markdown(f"**Самый высокий медианный возраст, {y0}**")
            st.dataframe(pd.DataFrame({"Страна": [countries.short_ru(i) for i in old.index], "Лет": old.values}), hide_index=True)
        with b:
            sp = aging.aging_speed(w, 2025, 2050).sort_values(ascending=False)
            st.markdown("**Быстрее всего стареют 2025→2050 (прибавка медианного возраста)**")
            st.dataframe(pd.DataFrame({"Страна": [countries.short_ru(i) for i in sp.head(15).index], "+лет": sp.head(15).values}), hide_index=True)
        fig = go.Figure(go.Histogram(x=sp.values, nbinsx=30, marker_color=C["accent"]))
        fig.update_layout(xaxis_title="Прибавка медианного возраста 2025→2050, лет", yaxis_title="Стран")
        st.plotly_chart(style(fig, 280), width="stretch")
        src_box([aging.MEDIAN_AGE])
    with tabs[2]:
        peaks = aging.population_peaks(w)
        chk = aging.path_consistency(w, path)
        st.caption("Траектория населения восстановлена из годового прироста ООН и откалибрована к точкам 2050 и 2100; "
                   f"медианное расхождение с исходными точками: {chk.get(2050, float('nan')):.1%} (2050), {chk.get(2100, float('nan')):.1%} (2100).")
        c = st.columns(3)
        c[0].metric("Пик уже позади (по ООН)", f"{int((peaks['peak_year'] <= 2025).sum())} стран")
        c[1].metric("Население сократится к 2100", f"{int((peaks['change_2100'] < 0).sum())} из {len(peaks)} стран")
        c[2].metric("Вырастет более чем вдвое к 2100", f"{int((peaks['change_2100'] > 1).sum())} стран")
        t = peaks.assign(Страна=[countries.short_ru(i) for i in peaks.index]).sort_values("change_2100")
        t = t.rename(columns={"now": "2024, чел.", "peak_year": "Год пика", "y2050": "2050, чел.", "y2100": "2100, чел.", "change_2100": "Изменение к 2100"})
        st.dataframe(t[["Страна", "2024, чел.", "Год пика", "2050, чел.", "2100, чел.", "Изменение к 2100"]], hide_index=True, height=360,
                     column_config={"Изменение к 2100": st.column_config.NumberColumn(format="percent"),
                                    "2024, чел.": st.column_config.NumberColumn(format="%.0f"),
                                    "2050, чел.": st.column_config.NumberColumn(format="%.0f"),
                                    "2100, чел.": st.column_config.NumberColumn(format="%.0f"), "Год пика": st.column_config.NumberColumn(format="%d")})
        sel = st.multiselect("Траектории стран", country_options(w), default=[i for i in ("RUS", "CHN", "IND", "NGA", "JPN", "USA") if i in path.index],
                             format_func=countries.short_ru, key="dm_pop")
        fig = go.Figure()
        for i, iso in enumerate(sel):
            if iso in path.index:
                s = path.loc[iso].dropna()
                fig.add_scatter(x=s.index, y=s.values / 1e6, name=countries.short_ru(iso), line=dict(color=PALETTE[i % len(PALETTE)]))
        fig.add_vline(x=2024.5, line_dash="dot", line_color="#8B96A5", annotation_text="дальше прогноз ООН")
        fig.update_layout(yaxis_title="Население, млн", yaxis_type="log")
        st.plotly_chart(style(fig, 380), width="stretch")
        tfr = aging.fertility(w)
        if len(tfr) and sel:
            fig = go.Figure()
            for i, iso in enumerate(sel):
                if iso in tfr.index.get_level_values("iso3"):
                    s = tfr.loc[iso].dropna()
                    fig.add_scatter(x=s.index, y=s.values, name=countries.short_ru(iso), line=dict(color=PALETTE[i % len(PALETTE)]))
            fig.add_hline(y=2.1, line_dash="dot", line_color="#8B96A5", annotation_text="уровень замещения ≈ 2.1")
            fig.update_layout(yaxis_title="Детей на женщину")
            st.markdown("**Рождаемость (приближение Gapminder: выжившие + умершие дети на женщину)**")
            st.plotly_chart(style(fig, 320), width="stretch")
        src_box([aging.POPULATION, aging.GROWTH_POP, aging.KIDS_SURV])
    with tabs[3]:
        st.markdown("**Связана ли скорость старения с ростом, сбережениями и бюджетом?**")
        st.caption("Панель 5-летних окон 1980–2013 по странам (IMF WEO + ООН). Оцениваем, как изменение медианного возраста за окно связано "
                   "со средним значением показателя в окне. Две спецификации: «просто связь» между странами и «внутри страны» "
                   "(фиксированные эффекты страны и периода, стандартные ошибки по кластерам стран).")
        with st.spinner("Считаю панельные регрессии…"):
            pooled = aging.economic_effect(w, country_fe=False)
            fe = aging.economic_effect(w, country_fe=True)
        if fe.empty:
            return st.info("Недостаточно данных для оценки.")
        fe["ci"] = 1.96 * fe["se"]
        pooled["ci"] = 1.96 * pooled["se"]
        fig = go.Figure()
        for name, d, col in (("Между странами (без фикс. эффектов)", pooled, C["violet"]), ("Внутри страны (фикс. эффекты)", fe, C["accent"])):
            fig.add_scatter(x=d["coef"], y=d["label"], mode="markers", name=name, marker=dict(color=col, size=10),
                            error_x=dict(type="data", array=d["ci"], color=col))
        fig.add_vline(x=0, line_color="#8B96A5")
        fig.update_layout(xaxis_title="Эффект +1 года медианного возраста за 5 лет, 95% интервал")
        st.plotly_chart(style(fig, 360), width="stretch")
        tbl = fe[["label", "coef", "se", "t", "n", "countries"]].rename(columns={"label": "Показатель", "coef": "Коэф.", "se": "SE", "t": "t", "n": "Наблюдений", "countries": "Стран"})
        st.dataframe(tbl, hide_index=True, column_config={"Коэф.": st.column_config.NumberColumn(format="%.3f"), "SE": st.column_config.NumberColumn(format="%.3f"),
                                                          "t": st.column_config.NumberColumn(format="%.2f")})
        sig = fe[fe["t"].abs() >= 2]
        if sig.empty:
            st.info("Вывод: внутри страны ни одна из связей не отличима от нуля (|t| < 2). Старение объясняет экономику хуже, чем принято думать: "
                    "эффект может быть, но на этих данных его не выделить. Прогнозы ниже показывают лишь «если связь реальна».")
        else:
            st.success("Значимые связи внутри страны: " + ", ".join(f"{r.label} (t={r.t:.1f})" for r in sig.itertuples()) +
                       ". Это статистическая связь, а не доказанная причинность.")
        outcome = st.selectbox("Сценарий: подразумеваемый эффект старения 2025→2050 на", list(aging.OUTCOMES), format_func=lambda k: aging.OUTCOMES[k][1], key="dm_out")
        sc = aging.scenario_effect(w, fe, outcome=outcome)
        if sc.empty:
            st.info("Нет оценки для этого показателя.")
        else:
            lim = float(sc["effect"].abs().quantile(0.97)) or 1.0
            colors = {i: globe.scale_color(float(v), -lim, lim, "up") for i, v in sc["effect"].items()}
            infos = {i: f"<b style='color:#FF9F1A'>{countries.short_ru(i)}</b><br>старение: +{r.aging_per_year * 25:.1f} лет медианного возраста за 25 лет"
                        f"<br>подразумеваемый эффект: <b>{r.effect:+.2f}</b> (от {r.low:+.2f} до {r.high:+.2f}) в год" for i, r in sc.iterrows()}
            views_global.world_map(colors, infos, f"СЦЕНАРИЙ: ЭФФЕКТ СТАРЕНИЯ 2025→2050 · {aging.OUTCOMES[outcome][1]}",
                                   [("хуже", "#FF4D4F"), ("лучше", "#26C281")], height=520)
            st.caption("Это не прогноз, а арифметика «коэффициент × скорость старения». Интервалы включают ноль, если связь не значима.")
        with st.expander("Проверка устойчивости вывода (другие окна, богатые/бедные страны, перестановочный тест)"):
            with st.spinner("Пересчитываю в разных спецификациях (около минуты при первом запуске)…"):
                rb = robustness_cached(version, outcome)
            if rb.empty:
                st.info("Недостаточно данных.")
            else:
                st.dataframe(rb, hide_index=True, width="stretch", column_config={
                    "Коэф.": st.column_config.NumberColumn(format="%.3f"), "SE": st.column_config.NumberColumn(format="%.3f"),
                    "t": st.column_config.NumberColumn(format="%.2f"), "p": st.column_config.NumberColumn(format="%.2f")})
                coefs = rb["Коэф."].dropna()
                same = (np.sign(coefs) == np.sign(coefs.iloc[0])).all() if len(coefs) else False
                nsig = int((rb["t"].abs() >= 2).sum())
                st.caption(f"Значимых (|t| ≥ 2) спецификаций: {nsig} из {len(rb)}; знак коэффициента " + ("одинаков во всех." if same else "меняется между спецификациями.")
                           + " Если значимых нет и знак меняется, вывод «эффект не выделяется» устойчив, а не артефакт одного выбора.")
        src_box([aging.MEDIAN_AGE, "wp:gdp_growth"], "регрессия рассчитана приложением")


@st.cache_data(show_spinner=False)
def robustness_cached(version, outcome):
    return aging.robustness(get_wide(version), outcome)


@st.cache_data(show_spinner=False)
def hunger_robust_cached(version, year):
    return hunger.weight_robustness(get_wide(version), year, n=60)


# ================================================================ ГОЛОД
@st.cache_data(show_spinner=False)
def hunger_history(version, years):
    w = get_wide(version)
    return pd.DataFrame({y: hunger.index(w, y)[0] for y in years})


def page_hunger(version):
    w = get_wide(version)
    st.markdown("### Продовольственная уязвимость: где риск голода выше")
    st.warning("Это индекс уязвимости (0–100), а не прогноз голода: открытых готовых рядов объявленных голодоморов (IPC/FEWS NET) в базе нет. "
               "Индекс объединяет 10 факторов: недоедание детей, калории, бедность, доход, зависимость от сельского хозяйства, воду, засухи, "
               "боевые потери, инфляцию, зависимость от помощи.")
    ys = list(range(1995, 2023))
    y = st.select_slider("Год", options=ys, value=2022, key="hg_year")
    idx, contrib = hunger.index(w, y)
    c = st.columns(4)
    c[0].metric("Стран в оценке", len(idx))
    c[1].metric("Самая уязвимая", f"{countries.short_ru(idx.idxmax())}: {idx.max():.0f}")
    c[2].metric("Выше 60 баллов", int((idx > 60).sum()))
    pop = get_pop_path(version)
    hi = idx[idx > 60].index.intersection(pop.index)
    c[3].metric("Население в странах выше 60", f"{pop.loc[hi, min(y, 2024)].sum() / 1e9:.2f} млрд" if len(hi) and min(y, 2024) in pop.columns else "н/д")
    colors, infos = {}, {}
    lo, hi_q = float(idx.quantile(.03)), float(idx.quantile(.97))
    for i, v in idx.items():
        colors[i] = globe.scale_color(float(v), lo, hi_q, "down")
        top = contrib.loc[i].sort_values(ascending=False).head(3)
        why = ", ".join(hunger.COMPONENTS[k][2] for k in top.index)
        infos[i] = (f"<b style='color:#FF9F1A'>{countries.short_ru(i)}</b><br>уязвимость {y}: <b>{v:.0f}</b> из 100"
                    f"<br><span style='color:#8B96A5'>главные факторы: {why}</span>")
    views_global.world_map(colors, infos, f"ПРОДОВОЛЬСТВЕННАЯ УЯЗВИМОСТЬ · {y}", [("ниже", "#1e2634"), ("выше", "#FF3C3C")], height=560)
    src_box(list(hunger.COMPONENTS), "индекс рассчитан приложением")
    a, b = st.columns([2, 3])
    with a:
        st.markdown("**Рейтинг**")
        top = idx.sort_values(ascending=False).head(25)
        st.dataframe(pd.DataFrame({"Страна": [countries.short_ru(i) for i in top.index], "Индекс": top.values}), hide_index=True, height=520,
                     column_config={"Индекс": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100)})
    with b:
        iso = st.selectbox("Страна подробно", idx.sort_values(ascending=False).index.tolist(), format_func=countries.short_ru, key="hg_iso")
        row = contrib.loc[iso].dropna().sort_values()
        fig = go.Figure(go.Bar(x=row.values, y=[hunger.COMPONENTS[k][2] for k in row.index], orientation="h", marker_color=C["down"]))
        fig.update_layout(xaxis_title="Вклад в индекс, баллов")
        st.markdown(f"**Из чего складывается уязвимость: {countries.short_ru(iso)} ({idx[iso]:.0f})**")
        st.plotly_chart(style(fig, 360), width="stretch")
    hist = hunger_history(version, tuple(range(1995, 2023, 3)))
    sel = st.multiselect("Динамика индекса", hist.index.tolist(), default=[i for i in ("NGA", "YEM", "ETH", "IND", "RUS", "BRA") if i in hist.index],
                         format_func=countries.short_ru, key="hg_hist")
    fig = go.Figure()
    for i, iso2 in enumerate(sel):
        s = hist.loc[iso2].dropna()
        fig.add_scatter(x=s.index, y=s.values, name=countries.short_ru(iso2), line=dict(color=PALETTE[i % len(PALETTE)]))
    fig.update_layout(yaxis_title="Индекс уязвимости")
    st.plotly_chart(style(fig, 340), width="stretch")
    with st.expander("Можно ли верить индексу: проверка на истории", expanded=True):
        v = hunger.validate(w)
        if v.empty:
            st.info("Недостаточно данных для проверки.")
        else:
            st.caption("Для каждого года-происхождения индекс строится БЕЗ показателя недоедания и сравнивается с недоеданием детей через 5 лет "
                       "(корреляция рангов Спирмена). «Инерция» это тот же прогноз, но по недоеданию в год происхождения.")
            st.dataframe(v.rename(columns={"origin": "Год", "n": "Стран", "rho_index": "ρ индекса", "rho_persistence": "ρ инерции",
                                           "top_quartile_precision": "Точность верхнего квартиля"}), hide_index=True,
                         column_config={"Точность верхнего квартиля": st.column_config.NumberColumn(format="percent"),
                                        "ρ индекса": st.column_config.NumberColumn(format="%.2f"), "ρ инерции": st.column_config.NumberColumn(format="%.2f")})
            st.caption(f"В среднем ρ индекса = {v['rho_index'].mean():.2f}, ρ инерции = {v['rho_persistence'].mean():.2f}. Индекс без самого "
                       "недоедания уступает прямой инерции, поэтому служит для сравнения уязвимости, а не замены наблюдаемых данных.")
        src_box([hunger.TARGET])
    with st.expander("Устойчивость рейтинга к весам (60 случайных наборов весов ±50%)"):
        with st.spinner("Считаю…"):
            rob = hunger_robust_cached(version, y)
        st.caption(f"Средний Спирмен между базовым и пересчитанными рейтингами: {rob.attrs['spearman_mean']:.3f} "
                   "(1 = рейтинг не меняется). Если он близок к 1, выводы не зависят от экспертного выбора весов.")
        top = rob.sort_values("base_rank").head(15)
        st.dataframe(pd.DataFrame({"Страна": [countries.short_ru(i) for i in top.index], "Базовое место": top["base_rank"].values,
                                   "Медианное место": top["median_rank"].values, "Диапазон P5–P95": [f"{a:.0f}–{b:.0f}" for a, b in zip(top["p5"], top["p95"])]}),
                     hide_index=True)
    with st.expander("Сценарий для страны: шок доходов, цен и безопасности"):
        iso_s = st.selectbox("Страна", idx.index.tolist(), index=idx.index.tolist().index("NGA") if "NGA" in idx.index else 0,
                             format_func=countries.short_ru, key="hg_sc_iso")
        a1, a2, a3 = st.columns(3)
        inc_m = a1.slider("Доход на душу, ×", 0.5, 1.5, 1.0, 0.05, key="hg_sc_inc")
        inf_m = a2.slider("Инфляция, ×", 0.5, 3.0, 1.0, 0.1, key="hg_sc_inf")
        war_m = a3.slider("Боевые потери, ×", 0.5, 5.0, 1.0, 0.1, key="hg_sc_war")
        base_v, new_v = hunger.what_if(w, y, iso_s, {"gm:gdppercapita_us_inflation_adjusted": inc_m, "gm:inflation_annual_percent": inf_m,
                                                    "gm:battle_deaths_per_100_000": war_m})
        st.metric(f"Индекс уязвимости: {countries.short_ru(iso_s)}", f"{new_v:.1f}", f"{new_v - base_v:+.1f} к базе ({base_v:.1f})", delta_color="inverse")
        st.caption("Пересчитываются процентильные ранги относительно остальных стран (остальные неизменны). Это арифметика индекса, а не прогноз последствий шока.")
    with st.expander("Что если: изменить веса факторов"):
        wts = {}
        cols = st.columns(2)
        for n, (k, (wt, sign, lab)) in enumerate(hunger.COMPONENTS.items()):
            wts[k] = cols[n % 2].slider(lab, 0.0, 0.4, float(wt), 0.01, key=f"hg_w_{k}")
        orig = dict(hunger.COMPONENTS)
        try:
            for k in wts:
                hunger.COMPONENTS[k] = (wts[k], orig[k][1], orig[k][2])
            alt, _ = hunger.index(w, y)
        finally:
            hunger.COMPONENTS.clear(); hunger.COMPONENTS.update(orig)
        d = pd.concat([idx.rename("a"), alt.rename("b")], axis=1).dropna()
        d["shift"] = d["b"] - d["a"]
        mv = d.reindex(d["shift"].abs().sort_values(ascending=False).index).head(10)
        st.dataframe(pd.DataFrame({"Страна": [countries.short_ru(i) for i in mv.index], "Базовый индекс": mv["a"].values, "С новыми весами": mv["b"].values}),
                     hide_index=True, column_config={"Базовый индекс": st.column_config.NumberColumn(format="%.1f"), "С новыми весами": st.column_config.NumberColumn(format="%.1f")})


# ================================================================ КЛИМАТ
def _trend_ci(s, n=30):
    """Наклон МНК за последние n лет с 95% интервалом и поправкой на автокорреляцию остатков (эффективное число наблюдений)."""
    s = s.dropna().tail(n)
    x = s.index.to_numpy(float)
    slope, icpt = np.polyfit(x, s.values, 1)
    res = s.values - (icpt + slope * x)
    r1 = float(np.corrcoef(res[:-1], res[1:])[0, 1]) if len(res) > 3 else 0.0
    neff = max(len(s) * (1 - r1) / (1 + r1), 4) if r1 > 0 else len(s)
    se = np.sqrt(res.var(ddof=2) / ((x - x.mean()) ** 2).sum() * len(s) / neff)
    return slope, 1.96 * se


def _read(name):
    return pd.read_csv(panel_mod.SNAP / name)


def page_climate(version):
    w, avail = get_wide(version), get_avail(version)
    st.markdown("### Климат и энергия")
    tabs = st.tabs(["Планета", "Страны: выбросы и энергия"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        try:
            t = _read("global_temp_annual.csv.gz")
            t = t[t["Source"] == "GCAG"].sort_values("Year").set_index("Year")["Mean"]
            fig = go.Figure(go.Bar(x=t.index, y=t.values, marker_color=[C["down"] if v > 0 else C["blue"] for v in t.values]))
            fig.update_layout(yaxis_title="Аномалия температуры, °C (к 1951–1980)")
            with c1:
                st.markdown("**Глобальная температура**")
                st.plotly_chart(style(fig, 340), width="stretch")
                sl, ci = _trend_ci(t, 30)
                st.caption(f"Тренд за последние 30 лет: {sl * 10:+.2f} ± {ci * 10:.2f} °C за десятилетие (95%, с поправкой на автокорреляцию).")
                d = registry.SOURCES["temp"]
                st.caption(f"📎 Источник: **{d['name']}**, {d['org']}: {d['url']} · лицензия: {d['license']}")
        except Exception as exc:
            c1.info(f"Нет данных по температуре: {exc}")
        try:
            a = _read("co2_ppm_annual.csv.gz").sort_values("Year").set_index("Year")["Mean"]
            fig = go.Figure(go.Scatter(x=a.index, y=a.values, line=dict(color=C["accent"])))
            fig.update_layout(yaxis_title="CO₂, ppm")
            with c2:
                st.markdown("**CO₂ в атмосфере (Мауна-Лоа)**")
                st.plotly_chart(style(fig, 340), width="stretch")
                sl, ci = _trend_ci(a, 20)
                st.caption(f"Рост за последние 20 лет: {sl:+.2f} ± {ci:.2f} ppm в год (95%).")
                d = registry.SOURCES["noaa_gml"]
                st.caption(f"📎 Источник: **{d['name']}**, {d['org']}: {d['url']} · лицензия: {d['license']}")
        except Exception as exc:
            c2.info(f"Нет данных по CO₂: {exc}")
        c3, c4 = st.columns(2)
        try:
            sl_df = _read("sea_level_csiro.csv.gz")
            sl_df = sl_df.assign(Year=sl_df["Time"].astype(float).round().astype(int)).groupby("Year")["GMSL"].mean()
            fig = go.Figure(go.Scatter(x=sl_df.index, y=sl_df.values, line=dict(color="#4DA3FF")))
            fig.update_layout(yaxis_title="Уровень моря, мм (CSIRO)")
            with c3:
                st.markdown("**Уровень океана**")
                st.plotly_chart(style(fig, 320), width="stretch")
                sl, ci = _trend_ci(sl_df, 30)
                st.caption(f"Тренд за последние 30 лет: {sl:+.1f} ± {ci:.1f} мм в год.")
                d = registry.SOURCES["sea_level"]
                st.caption(f"📎 Источник: **{d['name']}**, {d['org']}: {d['url']} · лицензия: {d['license']}")
        except Exception as exc:
            c3.info(f"Нет данных по уровню моря: {exc}")
        try:
            g = _read("glaciers.csv.gz").set_index("Year")["Mean cumulative mass balance"]
            fig = go.Figure(go.Scatter(x=g.index, y=g.values, line=dict(color="#B38CFF")))
            fig.update_layout(yaxis_title="Накопленный баланс массы, м в.э.")
            with c4:
                st.markdown("**Ледники (накопленная потеря массы)**")
                st.plotly_chart(style(fig, 320), width="stretch")
                d = registry.SOURCES["wgms"]
                st.caption(f"📎 Источник: **{d['name']}**, {d['org']}: {d['url']} · лицензия: {d['license']}")
        except Exception as exc:
            c4.info(f"Нет данных по ледникам: {exc}")
    with tabs[1]:
        pool = avail[avail["key"].str.startswith(("co2:", "en:"))]
        k = indicator_picker(pool, "cl_k", "co2:co2_per_capita")
        m = gdata.meta()[k]
        y = default_year(w, k)
        value_map(gdata.cross_section(w, k, y), k, y, key_suffix="cl")
        src_box([k])
        iso = st.selectbox("Страна", country_options(w), index=country_options(w).index("CHN") if "CHN" in country_options(w) else 0,
                           format_func=countries.short_ru, key="cl_iso")
        s = gdata.series(w, k, iso)
        if len(s) < 12:
            return st.info("По этой стране слишком короткий ряд для прогноза.")
        bt = run_backtest_cached(version, k, 10)
        fc = forecast.forecast_series(s, bt, 10) if bt.mase else None
        fig = go.Figure()
        fig.add_scatter(x=s.index, y=s.values, name="Факт", line=dict(color="#E8ECF1"))
        if fc is not None:
            fig.add_scatter(x=fc["year"], y=fc["p90"], line=dict(width=0), showlegend=False)
            fig.add_scatter(x=fc["year"], y=fc["p10"], fill="tonexty", fillcolor="rgba(255,159,26,0.2)", line=dict(width=0), name="P10–P90")
            fig.add_scatter(x=fc["year"], y=fc["mean"], name="Прогноз приложения", line=dict(color=C["accent"]))
        fig.update_layout(yaxis_title=f"{m['label']}, {m['unit']}")
        st.plotly_chart(style(fig, 380), width="stretch")
        if bt.mase:
            sk = 1 - bt.mase[bt.best][-1] / bt.mase["naive"][-1]
            st.caption(f"Метод: {forecast.LABELS[bt.best]}. Навык на 10 лет против «останется как сейчас»: {sk:+.0%} "
                       f"(по {bt.n_series} рядам). Это экстраполяция тренда, она не учитывает будущую политику.")
        src_box([k], "прогноз рассчитан приложением")


# ================================================================ ИСТОЧНИКИ
@st.cache_data(show_spinner=False)
def quality_cached(version):
    return quality.report(get_wide(version))


def provenance():
    import hashlib
    man = json.loads((panel_mod.SNAP / "manifest.json").read_text(encoding="utf-8")) if (panel_mod.SNAP / "manifest.json").exists() else {}
    files = man.get("files", {})
    rows = []
    for f in sorted(panel_mod.SNAP.glob("*")):
        if f.name == "manifest.json" or not f.is_file():
            continue
        info = files.get(f.name) or files.get(f.stem) or files.get(f.name.replace(".csv.gz", "")) or {}
        rows.append({"Файл": f.name, "Размер, КБ": round(f.stat().st_size / 1024), "Строк": info.get("rows"),
                     "Источник": info.get("source"), "SHA-256": hashlib.sha256(f.read_bytes()).hexdigest()[:16] + "…"})
    return rows


def page_sources(version):
    w, avail = get_wide(version), get_avail(version)
    st.markdown("### Источники данных и покрытие")
    rows = [{"Источник": d["name"], "Организация": d["org"], "Ссылка": d["url"], "Лицензия": d["license"], "Примечание": d["note"],
             "Показателей": int((avail["source"] == sid).sum())} for sid, d in registry.SOURCES.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={"Ссылка": st.column_config.LinkColumn()})
    st.markdown("**Покрытие по группам показателей**")
    g = avail.groupby("group").agg(Показателей=("key", "size"), Стран=("countries", "max"), Лет=("last", "max")).reset_index()
    st.dataframe(g.rename(columns={"group": "Группа", "Лет": "Последний год"}), hide_index=True, width="stretch")
    st.markdown("**Все показатели**")
    q = st.text_input("Поиск по названию")
    t = avail if not q else avail[avail["label"].str.contains(q, case=False) | avail["key"].str.contains(q, case=False)]
    st.dataframe(t, hide_index=True, width="stretch", height=420)
    st.caption("Правила честности: прогнозы ООН помечены как прогнозы; статистические прогнозы приложения сопровождаются проверкой на истории; "
               "модели, у которых навыка нет, показываются с этой пометкой. Карты с индексами (уязвимость, риск кризиса) рассчитаны приложением.")
    st.markdown("### Свежесть и качество данных")
    with st.spinner("Проверяю ряды на пропуски, скачки и возраст…"):
        qrep = quality_cached(version)
    c = st.columns(4)
    c[0].metric("Показателей", len(qrep))
    c[1].metric("Свежих (≤ 1 года)", int((qrep["Статус"] == "свежий").sum()))
    c[2].metric("Устаревших (> 3 лет)", int((qrep["Статус"] == "устарел").sum()))
    c[3].metric("Со скачками у >10% стран", int((qrep["Со скачками"] > 0.10).sum()))
    st.dataframe(qrep.drop(columns=["key"]).sort_values("Возраст, лет", ascending=False), hide_index=True, width="stretch", height=380, column_config={
        "С пропусками": st.column_config.NumberColumn(format="percent"), "Со скачками": st.column_config.NumberColumn(format="percent"),
        "Залипание": st.column_config.NumberColumn(format="percent")})
    st.caption("Скачки и «залипание» это подозрения: скачок может быть реальным событием (война, дефолт), а повторяющиеся значения признаком заполнения пропусков в источнике. "
               "Строки с пометкой «устарел» нельзя использовать для суждений о сегодняшнем дне.")
    with st.expander("Происхождение данных: файлы снимка и контрольные суммы"):
        st.dataframe(pd.DataFrame(provenance()), hide_index=True, width="stretch")
        st.caption("SHA-256 позволяет проверить, что файл не менялся; даты и строки берутся из manifest.json, который пишут скрипты обновления.")
    with st.expander("Обновить из первоисточников (World Bank, IMF; нужен интернет)"):
        st.caption("Загружает ~25 макропоказателей напрямую из API World Bank и IMF DataMapper и подмешивает к снимку. Если сеть закрыта, ошибки видны в отчёте.")
        if st.button("Загрузить свежие данные", key="wm_refresh"):
            bar = st.progress(0.0)
            wide, report = panel_mod.refresh_live(lambda f, label: bar.progress(f, text=label))
            if len(wide):
                store.upsert_panel(store.connect_macro(), panel_mod.to_long(wide))
            st.session_state["wm_refresh_report"] = pd.DataFrame(report, columns=["Что", "OK", "Сообщение"])
            st.session_state["wm_bump"] = True
        if "wm_refresh_report" in st.session_state:
            rep = st.session_state["wm_refresh_report"]
            st.write(f"Успешно {int(rep['OK'].sum())} из {len(rep)}")
            st.dataframe(rep.sort_values("OK"), hide_index=True, height=220)
            if st.session_state.pop("wm_bump", False):
                st.session_state.version += 1
    st.divider()
    views_global.view_sources()


# ================================================================ точка входа
EVAL_PATH = Path(__file__).resolve().parent.parent / "assets" / "crisis_horizon_eval.json"


def crisis_horizon_table():
    """Измеренная точность моделей на горизонтах 1/3/5 лет (scripts/eval_crisis_horizons.py), если файл есть."""
    if not EVAL_PATH.exists():
        return
    d = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    with st.expander(f"Измеренная точность по горизонтам (walk-forward, проверено {d['measured']})"):
        t = pd.DataFrame(d["rows"])
        t = t.assign(Период=[f"{a}–{b}" for a, b in t["years"]])
        t = t.rename(columns={"label": "Кризис", "horizon": "Горизонт, лет", "obs": "Наблюдений", "events": "Событий", "base_rate": "Базовая частота",
                              "auc": "AUC модели", "auc_naive": "AUC наивного", "top_decile_capture": "Верхний дециль ловит"})
        st.dataframe(t[["Кризис", "Горизонт, лет", "Период", "Наблюдений", "Событий", "Базовая частота", "AUC модели", "AUC наивного", "Верхний дециль ловит"]],
                     hide_index=True, width="stretch", column_config={
                         "Базовая частота": st.column_config.NumberColumn(format="percent"), "AUC модели": st.column_config.NumberColumn(format="%.3f"),
                         "AUC наивного": st.column_config.NumberColumn(format="%.3f"), "Верхний дециль ловит": st.column_config.NumberColumn(format="percent")})
        st.caption("AUC 0.5 = случайно, 1.0 = идеально. «Наивный» это один простой признак. Если AUC модели не выше наивного, модель ничего не добавляет.")


METHODS_MD = """
### Как устроены расчёты

**Прогноз рядов по странам.** Методы: «останется как сейчас», линейный тренд (последние 10 лет), демпфированный тренд Хольта и версии в логарифмах.
Каждый метод прогоняется на прошлых прогнозах всех стран (скользящие origin), ошибка измеряется в единицах типичного годового изменения ряда (MASE).
Лучший метод выбирается по средней ошибке, интервал P10–P90 берётся из эмпирических квантилей ошибок на том же горизонте и проверяется на поздних годах
(покрытие). Допущения: ряд продолжает вести себя как в прошлом; не учитываются политика, войны, технологии. Навык показан против наивного прогноза.

**Кризисные модели (раннее предупреждение).** Метки: валютный (Frankel–Rose: ослабление ≥ 25% и ≥ 10 п.п. быстрее прошлого года), инфляционный (первый год ≥ 20%),
рецессия (первый год отрицательного роста), конфликт (≥ 25 погибших, UCDP). Признаки известны на конец предыдущего года. Модель: ансамбль логистической
регрессии и градиентного бустинга с поправкой на базовую частоту. Проверка: walk-forward (учится только на прошлом), AUC и доля событий в верхнем дециле риска.
Горизонт 3 и 5 лет: метка «начнётся в течение h лет». Ограничение: банковские и долговые кризисы не прогнозируются (нет открытых меток).

**Население и старение.** Население до 2013 года факт ООН; дальше годовая траектория из прироста ООН, откалиброванная к точкам 2050 и 2100 источника.
Влияние старения на экономику: панель 5-летних окон 1980–2013, изменение медианного возраста за окно против среднего значения исхода, фиксированные эффекты
страны и периода, стандартные ошибки кластеризованы по странам. Это статистическая связь, не причинность; устойчивость проверяется на других окнах, подвыборках и
перестановочным тестом.

**Индекс продовольственной уязвимости.** Взвешенная средняя процентильных рангов 10 компонентов (недоедание, калории, бедность, доход, сельское хозяйство, вода, засухи,
боевые потери, инфляция, помощь). Веса экспертные; устойчивость к весам проверяется случайными наборами. Проверка на истории: индекс без компонента недоедания против недоедания
через 5 лет. Это индекс сравнительной уязвимости, а не прогноз голода.

**Долгосрочные оценки (ГОРИЗОНТ).** Доходность облигаций по стартовой ставке, акции по CAPE со сжатием к среднему, Монте-Карло портфеля, рост стран со сходимостью, динамика
госдолга. Каждая оценка сопровождается проверкой вне выборки; у акций и роста на 10 лет навык слабый или отсутствует, и это указано.

**Качество данных.** Для каждого ряда ищутся пропуски, скачки (робастный z по MAD) и «залипание»; для каждого показателя считается возраст последних данных.

**Чего здесь нет.** Данных ниже уровня страны (регионы, города), причинных моделей с идентификацией, банковских и долговых кризисов, оценок голода по IPC/FEWS NET,
автоматического обновления без запуска скриптов (для этого есть GitHub Actions, см. README).
"""


def page_methods(version):
    st.markdown(METHODS_MD)
    st.info("Каждая модель в приложении сопровождается проверкой на реальной истории прямо рядом с результатом. Если навыка нет, об этом написано.")


PAGES = ["ОБЗОР", "ИССЛЕДОВАТЕЛЬ", "СРАВНЕНИЕ", "СТРАНА", "ПУЗЫРЬКИ", "ДЕМОГРАФИЯ", "ГОЛОД", "КЛИМАТ", "КРИЗИСЫ", "ТОРГОВЛЯ", "МОРЕ", "КОНФЛИКТЫ",
         "ГОРИЗОНТ", "ВАЛЮТЫ", "МЕТОДЫ", "ИСТОЧНИКИ"]


def _ask_cb(version):
    q = st.session_state.get("wm_ask", "")
    if not q.strip():
        return
    avail = get_avail(version)
    r = ask.interpret(q, avail)
    st.session_state["wm_ask_msg"] = r["explanation"]
    if not r["page"]:
        return
    st.session_state["wm_page"] = r["page"]
    w = get_wide(version)
    opts = country_options(w)
    if r["iso"] in opts:
        st.session_state["cty_iso"] = r["iso"]
        st.session_state["cl_iso"] = r["iso"]
    keys = set(avail["key"])
    if r["indicators"] and r["indicators"][0] in keys:
        st.session_state["ex_k"] = r["indicators"][0]
        if r["indicators"][0].startswith(("co2:", "en:")):
            st.session_state["cl_k"] = r["indicators"][0]


def render(version, bump, use_demo=False):
    st.markdown("<div class='title'>МИР · ГЛОБАЛЬНЫЙ РЕЖИМ</div>", unsafe_allow_html=True)
    st.caption("История и прогнозы по странам на открытых данных. Под каждой картой и графиком указан источник.")
    st.text_input("Спросить: «где больше всего стариков», «риск голода в Нигерии», «кризис в Турции»", key="wm_ask", on_change=_ask_cb, args=(version,),
                  placeholder="Введи вопрос и нажми Enter")
    if st.session_state.get("wm_ask_msg"):
        st.caption("➜ " + st.session_state["wm_ask_msg"])
    page = st.radio("Раздел", PAGES, horizontal=True, label_visibility="collapsed", key="wm_page")
    if page == "ОБЗОР":
        page_overview(version)
    elif page == "ИССЛЕДОВАТЕЛЬ":
        page_explorer(version)
    elif page == "СРАВНЕНИЕ":
        page_compare(version)
    elif page == "СТРАНА":
        page_country(version)
    elif page == "ПУЗЫРЬКИ":
        page_bubbles(version)
    elif page == "ДЕМОГРАФИЯ":
        page_demography(version)
    elif page == "ГОЛОД":
        page_hunger(version)
    elif page == "КЛИМАТ":
        page_climate(version)
    elif page == "КРИЗИСЫ":
        crisis_horizon_table()
        views_global.view_crises(version, bump, horizon_choice=True)
    elif page == "ТОРГОВЛЯ":
        views_global.view_trade(version, bump, use_demo)
    elif page == "МОРЕ":
        views_global.view_sea(version, bump)
    elif page == "КОНФЛИКТЫ":
        views_global.view_conflicts(version, bump, use_demo)
    elif page == "ГОРИЗОНТ":
        views_global.view_horizon(version)
    elif page == "ВАЛЮТЫ":
        views_global.view_fx()
    elif page == "МЕТОДЫ":
        page_methods(version)
    else:
        page_sources(version)
