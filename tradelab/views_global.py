"""Вкладки глобальной аналитики: КРИЗИСЫ, ВАЛЮТЫ (FX-сценарии), ТОРГОВЛЯ, ИСТОЧНИКИ."""
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from . import catalog, countries, crisis, fxscen, globe, panel as panel_mod, store, trade
from .ui import C, PALETTE, pct, style

HEAT_LEGEND = [("низкий риск", "#1e2634"), ("средний", "#FF9F1A"), ("высокий", "#FF3C3C")]


# ---------------------------------------------------------------- общее
def world_map(colors, info, title, legend, arcs=None, markers=None, mode="flat", height=600, lon0=20.0, grad=None, ranking=None, subtitle=""):
    html = globe.build(markers or [], arcs or [], colors, title=title, legend=legend, height=height, lon0=lon0,
                       info=info, mode=mode, grad=grad, ranking=ranking, subtitle=subtitle,
                       auto_rotate=st.session_state.get("set_rotate", True))
    components.html(html, height=height + 6)


def gradient_legend(color_fn, lo, hi, lo_label, hi_label, n=7):
    """Данные для градиентной шкалы под картой: цвета от lo к hi по функции раскраски."""
    cols = [color_fn(lo + (hi - lo) * i / (n - 1)) or "#141c29" for i in range(n)]
    return {"colors": cols, "lo": lo_label, "hi": hi_label}


def ranking_panel(values, color_of, name_of, fmt, top_title="Больше всего", bottom_title="Меньше всего", n=8):
    """Рейтинг для боковой панели карты: верх и низ по значению. color_of(iso) -> цвет, fmt(v) -> текст."""
    v = values.dropna().sort_values(ascending=False)
    row = lambda iso, x: [iso, name_of(iso), fmt(float(x)), color_of(iso) or "#556"]
    return {"top_title": top_title, "bottom_title": bottom_title, "top": [row(i, x) for i, x in v.head(n).items()],
            "bottom": [row(i, x) for i, x in v.tail(n).iloc[::-1].items()]}


def _fmt(v, suffix=""):
    return "н/д" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.1f}{suffix}"


# ================================================================ КРИЗИСЫ
@st.cache_data(show_spinner=False)
def load_panel(source, version):
    if source == "live":
        long = store.load_panel_long(store.connect_macro())
        return panel_mod.from_long(long) if len(long) else pd.DataFrame()
    return panel_mod.snapshot_panel()


@st.cache_resource(show_spinner=False)
def get_models(source, version, horizon=1):
    p = load_panel(source, version)
    models = {k: crisis.CrisisModel(p, k, horizon) for k in crisis.available_kinds(p)}
    return models, crisis.FxModel(p)


@st.cache_data(show_spinner=False)
def cached_replay(source, version, kind, year, horizon=1):
    return crisis.replay(load_panel(source, version), kind, year, horizon)


@st.cache_data(show_spinner=False)
def cached_eval(source, version, kind, first_year, last_year, horizon=1):
    return crisis.evaluate(load_panel(source, version), kind, first_year, last_year, horizon=horizon)


def view_crises(version, bump, horizon_choice=False):
    info = panel_mod.snapshot_info()
    live_exists = len(store.load_panel_long(store.connect_macro())) > 0
    c1, c2 = st.columns([3, 2])
    source = c1.radio("Данные", ["Встроенный снимок (реальные, IMF/World Bank)", "Живые данные (WB + IMF API)"],
                      horizontal=True, index=1 if live_exists else 0)
    source_key = "live" if source.startswith("Живые") else "snapshot"
    with c2.expander("Загрузить живые данные (World Bank, IMF)"):
        st.caption("25 макропоказателей: резервы, внешний долг, кредит, курсы, институты. Нужен интернет, ключи не нужны.")
        if st.button("Загрузить макро-панель"):
            bar = st.progress(0.0)
            wide, report = panel_mod.refresh_live(lambda f, label: bar.progress(f, text=label))
            if len(wide):
                store.upsert_panel(store.connect_macro(), panel_mod.to_long(wide))
            st.session_state["panel_report"] = pd.DataFrame(report, columns=["Что", "OK", "Сообщение"])
            bump()
        if "panel_report" in st.session_state:
            rep = st.session_state["panel_report"]
            st.write(f"Успешно {int(rep['OK'].sum())} из {len(rep)}")
            st.dataframe(rep.sort_values("OK"), hide_index=True, height=200)
    if source_key == "live" and not live_exists:
        return st.info("Живых данных пока нет. Загрузи их кнопкой справа или переключись на встроенный снимок.")
    panel = load_panel(source_key, version)
    if panel.empty:
        return st.info("Нет данных.")
    if source_key == "snapshot":
        st.caption(f"Снимок от {info['updated']}: макро IMF до {panel_mod.WEO_LAST_ACTUAL} года, инфляция/ВВП/население "
                   "World Bank до 2023. Он нужен, чтобы проверить модели на реальной истории и работать офлайн; "
                   "для актуального прогноза загрузи живые данные.")

    mode = st.radio("Режим", ["Машина времени (проверка на истории)", "Прогноз на ближайший год"], horizontal=True,
                    index=0 if source_key == "snapshot" else 1)
    kinds = crisis.available_kinds(panel)
    horizon = 1
    if horizon_choice:
        horizon = st.radio("Горизонт прогноза: кризис начнётся в течение …", [1, 3, 5], horizontal=True,
                           format_func=lambda h: "1 года" if h == 1 else f"{h} лет", key="crisis_horizon")
        if horizon > 1:
            st.caption("Многолетний горизонт: метка = «кризис начнётся в течение h лет». Точность падает с горизонтом, "
                       "реальную точность см. в блоке проверки ниже.")
    kind = st.selectbox("Вид кризиса", kinds, format_func=lambda k: crisis.KINDS[k])
    if kind == "recession":
        st.warning("Рецессии прогнозируются плохо: на реальной истории модель не лучше простого признака «рост ВВП "
                   "уже замедлился». Карту стоит читать как индикатор уязвимости, а не прогноз.")

    if mode.startswith("Машина"):
        max_year = panel_mod.WEO_LAST_ACTUAL if source_key == "snapshot" else int(
            panel.index.get_level_values("year").max())
        max_year -= horizon - 1
        year = st.slider("Год, на который модель строит прогноз (учится только на более ранних годах)", 1990, max_year,
                         min(2008, max_year))
        try:
            out, summ = cached_replay(source_key, version, kind, year, horizon)
        except ValueError as exc:
            return st.warning(str(exc))
        probs = out["prob"]
        m = st.columns(4)
        m[0].metric("Стран в оценке", summ["countries"])
        m[1].metric("Реально началось кризисов", summ["events"])
        m[2].metric("Из них в верхнем дециле риска", f"{summ['events_in_top_decile']} из {summ['events']}"
                    if summ["events"] else "н/д")
        m[3].metric("AUC этого года", f"{summ['auc']:.2f}" if np.isfinite(summ["auc"]) else "н/д")
        title = f"{crisis.KINDS[kind].upper()} · РИСК НА {year}" + (f"–{year + horizon - 1}" if horizon > 1 else " ГОД") + " (модель знает только прошлое)"
        _crisis_map(probs, title, actual=out.index[out["actual"] == 1].tolist(), panel=panel, year=year - 1)
        st.markdown("**Что реально произошло в этом году**")
        hit = out[out["actual"] == 1].copy()
        if hit.empty:
            st.info("В риск-наборе года кризисов этого вида не началось.")
        else:
            rank = out["prob"].rank(ascending=False)
            tbl = pd.DataFrame({"Страна": [countries.short_ru(i) for i in hit.index],
                                "Риск по модели": hit["prob"], "Место в рейтинге": [f"{int(rank[i])} из {len(out)}" for i in hit.index]})
            st.dataframe(tbl.sort_values("Риск по модели", ascending=False), hide_index=True, width="stretch",
                         column_config={"Риск по модели": st.column_config.NumberColumn(format="percent")})
        _ranking(out[["prob"]].rename(columns={"prob": "Риск"}), out["actual"])
        st.caption("Метка кризиса считается по данным (см. методику ниже). Страны, уже находившиеся в кризисе в "
                   "предыдущий год, в оценку не входят. Модель не видит ни ответа, ни данных будущих лет.")
    else:
        with st.spinner("Обучаю модели на всей истории (около 30 с при первом запуске)…"):
            models, fxm = get_models(source_key, version, horizon)
        model = models[kind]
        probs_df, fyear = model.predict()
        title = f"{crisis.KINDS[kind].upper()} · ПРОГНОЗ НА {fyear + 1}" + (f"–{fyear + horizon}" if horizon > 1 else " ГОД")
        if source_key == "snapshot" and fyear < 2020:
            st.warning(f"Данные снимка по макро доходят до {fyear}: это прогноз на {fyear + 1}, а не на текущий год.")
        probs = probs_df["prob"]
        m = st.columns(4)
        m[0].metric("Стран в оценке", len(probs))
        m[1].metric("Средняя вероятность", pct(float(probs.mean())))
        m[2].metric("Выше 30%", int((probs > 0.3).sum()))
        m[3].metric("Максимум", f"{countries.short_ru(probs.idxmax())} {pct(float(probs.max()), 0)}")
        _crisis_map(probs, title, panel=panel, year=fyear, model=model)
        _ranking(probs_df[["prob", "logit", "gbm"]].rename(columns={"prob": "Риск", "logit": "Логит", "gbm": "Бустинг"}))
        _country_detail(models, fxm, panel, kind)

    with st.expander("Насколько моделям можно верить: проверка вне выборки"):
        st.caption("Для каждого блока лет модель обучается только на предыдущих годах и прогнозирует следующие. "
                   "Считаются AUC (0.5 = случайно, 1.0 = идеально) и доля реальных кризисов в верхних 10% по риску "
                   "(при случайном выборе было бы 10%). Сравнение с наивным ориентиром: прогноз по одному признаку.")
        first = st.slider("Начало периода проверки", 1990, 2005, 1996)
        if st.button("Запустить проверку (1–3 минуты)"):
            rows = []
            last = panel_mod.WEO_LAST_ACTUAL if source_key == "snapshot" else None
            for k in kinds:
                with st.spinner(f"Проверяю: {crisis.KINDS[k]}…"):
                    try:
                        met, _ = cached_eval(source_key, version, k, first, last, horizon)
                    except ValueError as exc:
                        rows.append({"Кризис": crisis.KINDS[k], "Комментарий": str(exc)})
                        continue
                rows.append({"Кризис": crisis.KINDS[k], "Периодов": f"{met['years'][0]}–{met['years'][1]}",
                             "Наблюдений": met["obs"], "Событий": met["events"], "Базовая частота": met["base_rate"],
                             "AUC ансамбля": met["ensemble"]["auc"], "AUC наивного": met.get("naive", {}).get("auc", np.nan),
                             "Верхний дециль ловит": met["ensemble"]["top_decile_capture"]})
            st.session_state["eval_rows"] = pd.DataFrame(rows)
        if "eval_rows" in st.session_state:
            st.dataframe(st.session_state["eval_rows"], hide_index=True, width="stretch", column_config={
                "Базовая частота": st.column_config.NumberColumn(format="percent"),
                "AUC ансамбля": st.column_config.NumberColumn(format="%.3f"),
                "AUC наивного": st.column_config.NumberColumn(format="%.3f"),
                "Верхний дециль ловит": st.column_config.NumberColumn(format="percent")})
    with st.expander("Методика, определения и ограничения"):
        st.markdown(
            "- **Валютный кризис** (Frankel–Rose): курс к USD ослаб на ≥25% и на ≥10 п.п. сильнее, чем годом ранее; "
            "считается начало эпизода. Курс берётся официальный (World Bank) или из отношения ВВП в нац. валюте к ВВП в USD (IMF).\n"
            "- **Инфляционный кризис**: инфляция впервые поднялась до ≥20%.\n"
            "- **Рецессия**: рост реального ВВП впервые стал отрицательным.\n"
            "- **Признаки**: уровни и изменения макропоказателей на конец предыдущего года (инфляция, рост, счёт текущих "
            "операций, госдолг, бюджет, безработица, сбережения, инвестиции, при живых данных: резервы, внешний долг, кредит, "
            "институты), ослабление валюты, размер и доход страны.\n"
            "- **Модель**: ансамбль логистической регрессии и градиентного бустинга; редкие события учитываются балансом классов.\n"
            "- **Не прогнозируются**: банковские и суверенные долговые кризисы (открытых готовых меток нет), политические и "
            "военные шоки. Вероятность относится к началу кризиса в следующем году, а не к его масштабу.\n"
            "- Прошлое не гарантирует будущего; модель описывает уязвимость, а не предсказывает конкретное событие.")


def _crisis_map(probs, title, panel, year, actual=None, model=None):
    vmax = float(max(probs.quantile(0.98), 0.05))
    colors, infos = {}, {}
    contrib = model.contributions() if model is not None else None
    for iso, p in probs.items():
        colors[iso] = globe.heat(float(p), vmax)
        row = panel.loc[(iso, year)] if (iso, year) in panel.index else None
        extra = ""
        if row is not None:
            extra = (f"<br>инфляция {_fmt(row.get('inflation'), '%')} · рост {_fmt(row.get('gdp_growth'), '%')}"
                     f"<br>счёт тек. оп. {_fmt(row.get('current_account'), '% ВВП')} · госдолг {_fmt(row.get('gov_debt'), '% ВВП')}")
        drivers = ""
        if contrib is not None and iso in contrib.index:
            row_c = contrib.loc[iso]
            top3 = row_c.reindex(row_c.abs().sort_values(ascending=False).index).head(3)
            drivers = "<br><span style='color:#8B96A5'>главные причины: " + ", ".join(
                ("↑ " if c > 0 else "↓ ") + panel_mod.LABELS.get(f, crisis.FEATURE_LABELS.get(f, f))
                for f, c in top3.items()) + "</span>"
        infos[iso] = f"<b style='color:#FF9F1A'>{countries.short_ru(iso)}</b><br>риск: <b>{probs[iso]:.1%}</b>{extra}{drivers}"
    markers = []
    if actual:
        for iso in actual:
            c = countries.centroid(iso)
            if c:
                markers.append({"name": countries.short_ru(iso), "iso": iso, "city": "", "lat": c[0], "lon": c[1],
                                "color": "#FFFFFF", "open": True, "label": "кризис реально начался в этом году"})
    legend = HEAT_LEGEND + ([("белый маркер: кризис случился", "#FFFFFF")] if actual else [])
    mode = st.radio("Вид карты", ["Карта", "Глобус"], horizontal=True, key=f"mapmode_{title[:8]}", label_visibility="collapsed")
    grad = gradient_legend(lambda x: globe.heat(x, vmax), 0.0, vmax, "ниже · 0%", f"выше · {vmax:.0%}")
    rk = ranking_panel(probs, lambda i: colors.get(i), countries.short_ru, lambda x: f"{x:.1%}", "Выше риск", "Ниже риск")
    world_map(colors, infos, title, [l for l in legend if "белый" in l[0]], markers=markers, mode="flat" if mode == "Карта" else "globe",
              grad=grad, ranking=rk, height=600, subtitle=f"{len(probs)} стран")


def _ranking(df, actual=None, n=25):
    t = df.sort_values(df.columns[0], ascending=False).head(n).copy()
    t.insert(0, "Страна", [countries.short_ru(i) for i in t.index])
    if actual is not None:
        t["Кризис начался"] = ["да" if actual.get(i, 0) == 1 else "" for i in t.index]
    st.markdown(f"**Топ-{n} по риску**")
    st.dataframe(t, hide_index=True, width="stretch", height=360,
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in t.columns[1:]
                                if t[c].dtype != object})


def _country_detail(models, fxm, panel, kind):
    st.subheader("Страна подробно")
    pool = models[kind].predict()[0].sort_values("prob", ascending=False).index.tolist()
    iso = st.selectbox("Страна", pool, format_func=countries.short_ru)
    model = models[kind]
    c1, c2 = st.columns(2)
    with c1:
        vals = {crisis.KINDS[k]: float(models[k].predict()[0]["prob"].get(iso, np.nan)) for k in models}
        fig = go.Figure(go.Bar(x=list(vals), y=list(vals.values()), marker_color=C["accent"]))
        fig.update_layout(yaxis_tickformat=".0%")
        st.markdown("**Вероятность начала кризиса в следующем году**")
        st.plotly_chart(style(fig, 280), width="stretch")
    with c2:
        d = model.drivers(iso, top=8)
        if len(d):
            names = [panel_mod.LABELS.get(f, crisis.FEATURE_LABELS.get(f, f)) for f in d["feature"]]
            fig = go.Figure(go.Bar(x=d["contribution"], y=names, orientation="h",
                                   marker_color=[C["down"] if v > 0 else C["up"] for v in d["contribution"]]))
            fig.update_yaxes(autorange="reversed")
            st.markdown("**Что повышает (красное) и снижает (зелёное) риск** по сравнению с типичной страной")
            st.plotly_chart(style(fig, 280), width="stretch")
    fxp, fy = fxm.predict()
    if iso in fxp.index:
        r = fxp.loc[iso]
        m = st.columns(3)
        m[0].metric("Курс нац. валюты к USD, медиана", f"{r['p50']:+.1%}", "плюс = валюта слабеет", delta_color="off")
        m[1].metric("Оптимистичный сценарий (P10)", f"{r['p10']:+.1%}")
        m[2].metric("Пессимистичный сценарий (P90)", f"{r['p90']:+.1%}")
        st.caption("Квантильный бустинг по тем же признакам: интервал P10–P90 охватывает примерно 80% исходов.")
    with st.expander("Что если: изменить показатели страны (по логит-модели)"):
        base = float(model.predict()[0]["prob"].get(iso, np.nan))
        feats = [f for f in ("inflation", "gdp_growth", "current_account", "gov_debt", "fiscal_balance", "unemployment")
                 if f in model.columns]
        Xf, _ = model.forecast_frame()
        over = {}
        cols = st.columns(3)
        for i, f in enumerate(feats):
            cur = Xf.loc[iso, f] if iso in Xf.index else np.nan
            if np.isfinite(cur):
                lo, hi = float(Xf[f].quantile(0.02)), float(Xf[f].quantile(0.98))
                over[f] = cols[i % 3].slider(panel_mod.LABELS.get(f, f), lo, hi, float(np.clip(cur, lo, hi)))
        if over:
            new = model.what_if(iso, over)
            st.metric("Вероятность при выбранных значениях (логит-модель)", pct(new), f"{new - base:+.1%} к базе",
                      delta_color="inverse")
    series = [c for c in ("inflation", "gdp_growth", "current_account", "gov_debt", "dep") if c in model.df.columns]
    hist = model.df.loc[iso, series].dropna(how="all") if iso in model.df.index.get_level_values("iso3") else None
    if hist is not None and len(hist):
        fig = go.Figure()
        for i, s in enumerate(series):
            y = hist[s] * (100 if s == "dep" else 1)
            fig.add_scatter(x=hist.index, y=y, name=panel_mod.LABELS.get(s, "Ослабление валюты, %"),
                            line=dict(color=PALETTE[i % len(PALETTE)]))
        st.markdown("**История ключевых показателей**")
        st.plotly_chart(style(fig, 330), width="stretch")


# ================================================================ ВАЛЮТЫ
@st.cache_data(show_spinner=False)
def fx_sensitivities(years):
    fx, f = fxscen.snapshot_fx(), fxscen.snapshot_factors()
    r, ff = fxscen.build_dataset(fx, f)
    return fxscen.sensitivities(r, ff, years=years), fx.index.max()


def view_fx():
    st.caption("Чувствительность валют к силе доллара, нефти и страху рынка (VIX) по реальным недельным данным ФРС США, "
               "Brent и VIX (встроенный снимок). Сценарий переводит заданные шоки в ожидаемое движение курса.")
    years = st.slider("Окно оценки, лет", 3, 20, 8)
    sens, last = fx_sensitivities(years)
    st.caption(f"Последняя дата в данных: {last:%d.%m.%Y}")
    st.markdown("**Сценарий**")
    c1, c2, c3, c4 = st.columns(4)
    usd = c1.slider("Доллар к корзине валют, %", -15, 15, 0) / 100
    oil = c2.slider("Нефть Brent, %", -60, 60, -30) / 100
    vix = c3.slider("VIX (страх рынка), %", -50, 150, 50) / 100
    horizon = c4.slider("Горизонт, недель", 1, 52, 13)
    res = fxscen.scenario(sens, {"usd": np.log1p(usd), "oil": np.log1p(oil), "vix": np.log1p(vix)}, horizon)
    res = res.sort_values("expected", ascending=False)
    names = [fxscen.CURRENCY_RU.get(c, c) for c in res.index]
    fig = go.Figure()
    fig.add_bar(x=names, y=res["expected"], marker_color=[C["down"] if v > 0 else C["up"] for v in res["expected"]],
                error_y=dict(type="data", symmetric=False, array=(res["p90"] - res["expected"]).to_numpy(),
                             arrayminus=(res["expected"] - res["p10"]).to_numpy(), color="#8B96A5"))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Изменение курса нац. валюты за USD (плюс = слабеет)")
    st.plotly_chart(style(fig, 420), width="stretch")
    colors, infos = {}, {}
    scale = max(0.05, float(res["expected"].abs().quantile(0.9)))
    for cur, r in res.iterrows():
        iso = fxscen.FX_COUNTRY.get(cur)
        if iso:
            colors[iso] = globe.diverging(-float(r["expected"]), scale)  # красный = валюта слабеет
            infos[iso] = (f"<b style='color:#FF9F1A'>{fxscen.CURRENCY_RU.get(cur, cur)}</b><br>ожидаемо {r['expected']:+.1%}"
                          f"<br>интервал {r['p10']:+.1%} … {r['p90']:+.1%}")
    world_map(colors, infos, "ВАЛЮТЫ: СЦЕНАРИЙ", [("валюта слабеет", C["down"]), ("валюта крепнет", C["up"]),
                                                  ("нет данных", "#1b2536")], height=480)
    st.markdown("**Коэффициенты чувствительности**")
    tbl = sens.copy()
    tbl.insert(0, "Валюта", [fxscen.CURRENCY_RU.get(c, c) for c in tbl.index])
    st.dataframe(tbl[["Валюта", "b_usd", "b_oil", "b_vix", "t_usd", "t_oil", "t_vix", "r2"]], hide_index=True, width="stretch",
                 height=420, column_config={
                     "b_usd": st.column_config.NumberColumn("β доллар", format="%.2f"),
                     "b_oil": st.column_config.NumberColumn("β нефть", format="%.3f"),
                     "b_vix": st.column_config.NumberColumn("β VIX", format="%.3f"),
                     "t_usd": st.column_config.NumberColumn("t доллар", format="%.1f"),
                     "t_oil": st.column_config.NumberColumn("t нефть", format="%.1f"),
                     "t_vix": st.column_config.NumberColumn("t VIX", format="%.1f"),
                     "r2": st.column_config.NumberColumn("R²", format="%.2f")})
    st.caption("β: на сколько растёт курс нац. валюты за USD (валюта слабеет) при росте фактора на 1 (в логарифмах). "
               "Отрицательная β по нефти у CAD, NOK, MXN означает: при подорожании нефти валюта крепнет. "
               "|t| выше 2 означает статистически заметную связь. Связи оценены по прошлому и в кризисы меняются; "
               "валюты без устойчивой связи (в т.ч. административные курсы) не показаны.")


# ================================================================ ТОРГОВЛЯ
@st.cache_data(show_spinner=False)
def load_trade(source, version):
    if source == "live":
        return store.load_trade(store.connect_macro())
    return trade.demo()


def view_trade(version, bump, use_demo):
    live = store.load_trade(store.connect_macro())
    has_live = len(live) > 0
    src = st.radio("Данные о торговле", ["Демо (синтетика по гравитационной модели)", "Реальные (Comtrade / CSV)"],
                   horizontal=True, index=1 if has_live and not use_demo else 0)
    key = "live" if src.startswith("Реальные") else "demo"
    with st.expander("Загрузить реальные торговые потоки (UN Comtrade или CSV)", expanded=key == "live" and not has_live):
        st.caption("UN Comtrade preview API без ключа: до 500 строк за запрос, поэтому грузим по странам. "
                   "Можно также загрузить свой CSV (BACI, WITS, выгрузка Comtrade).")
        c1, c2 = st.columns(2)
        year = c1.number_input("Год", 2000, datetime.now().year - 1, 2022)
        reps = c2.multiselect("Страны-отчитывающиеся", countries.sovereigns(),
                              default=["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "IND", "RUS", "BRA", "TUR", "KOR", "SAU"],
                              format_func=countries.short_ru)
        if st.button("Загрузить из Comtrade"):
            bar = st.progress(0.0)
            df, report = trade.fetch_comtrade(reps, int(year), lambda f, label: bar.progress(f, text=label))
            if len(df):
                store.upsert_trade(store.connect_macro(), df)
            st.session_state["trade_report"] = pd.DataFrame(report, columns=["Что", "OK", "Сообщение"])
            bump()
        up = st.file_uploader("…или CSV (колонки: reporter, partner, year, value, при наличии flow)", type="csv")
        if up is not None:
            raw = pd.read_csv(up)
            df = trade.from_csv(raw, flow="flow" if "flow" in raw.columns else None)
            store.upsert_trade(store.connect_macro(), df)
            st.success(f"Импортировано записей: {len(df)}")
            bump()
        if "trade_report" in st.session_state:
            rep = st.session_state["trade_report"]
            st.write(f"Успешно {int(rep['OK'].sum())} из {len(rep)}")
            st.dataframe(rep.sort_values("OK"), hide_index=True, height=180)
    if key == "live" and not has_live:
        return st.info("Реальных данных о торговле пока нет: загрузи их выше.")
    df = load_trade(key, version)
    if key == "demo":
        st.warning("ДЕМО: торговые потоки сгенерированы гравитационной моделью (ВВП и расстояние), это не реальные "
                   "данные. Они нужны, чтобы показать интерфейс. Реальные загружаются выше.")
    years = sorted(df["year"].unique())
    yr = st.selectbox("Год", years, index=len(years) - 1)
    E = trade.export_matrix(df, yr)
    gdp = trade.gdp_usd(int(yr))
    bal = trade.balances(E)
    cen = trade.centrality(E)

    c1, c2, c3 = st.columns([3, 3, 2])
    metric = c1.selectbox("Окраска стран", ["Экспорт, USD", "Сальдо торговли, % ВВП", "Открытость (экспорт, % ВВП)",
                                            "Значимость в сети (PageRank)", "Торговое сообщество",
                                            "Уязвимость к заражению"])
    n_arcs = c2.slider("Число потоков на карте", 5, 80, 30)
    mapmode = c3.radio("Вид", ["Карта", "Глобус"], horizontal=True, key="trade_mode")

    colors, infos, legend = {}, {}, []
    gdp_e = gdp.reindex(E.index)
    cont = None
    if metric == "Уязвимость к заражению":
        st.caption("Выбери страны, у которых предполагается кризис, и силу шока: торговые партнёры потеряют спрос на "
                   "экспорт и часть поставок.")
        s1, s2 = st.columns([3, 2])
        crisis_iso = s1.multiselect("Страны в кризисе", list(E.index), default=[c for c in ("TUR",) if c in E.index],
                                    format_func=countries.short_ru)
        drop = s2.slider("Падение импорта кризисной страной, %", 5, 50, 20) / 100
        cont = trade.contagion(E, gdp, {c: 1.0 for c in crisis_iso}, import_drop=drop)
    for iso in E.index:
        if metric == "Экспорт, USD":
            v = float(bal.loc[iso, "export"])
            col = globe.sequential(np.log10(max(v, 1)), 9, 13)
            txt = f"экспорт {v / 1e9:,.0f} млрд USD"
        elif metric == "Сальдо торговли, % ВВП":
            v = float(bal.loc[iso, "balance"] / gdp_e[iso]) if iso in gdp_e.index and gdp_e[iso] > 0 else np.nan
            col = globe.diverging(v, 0.1)
            txt = f"сальдо {v:+.1%} ВВП"
        elif metric == "Открытость (экспорт, % ВВП)":
            v = float(bal.loc[iso, "export"] / gdp_e[iso]) if iso in gdp_e.index and gdp_e[iso] > 0 else np.nan
            col = globe.sequential(v, 0, 0.6)
            txt = f"экспорт {v:.0%} ВВП"
        elif metric == "Значимость в сети (PageRank)":
            v = float(cen.loc[iso, "pagerank"])
            col = globe.sequential(v, 0, float(cen["pagerank"].quantile(0.95)))
            txt = f"PageRank {v:.3f}"
        elif metric == "Торговое сообщество":
            k = int(cen.loc[iso, "community"])
            col = ["#B8451A", "#2d7fb8", "#2a8f6a", "#b8962a", "#8a5bb8", "#b84a7a"][k % 6]
            txt = f"сообщество {k + 1}"
        else:
            v = float(cont.loc[iso, "total"]) if iso in cont.index else np.nan
            col = globe.heat(v, max(float(cont["total"].quantile(0.97)), 1e-4)) if cont is not None else None
            txt = f"ожидаемый удар {v:.2%} ВВП" if np.isfinite(v) else "н/д"
        colors[iso] = col
        infos[iso] = f"<b style='color:#FF9F1A'>{countries.short_ru(iso)}</b><br>{txt}"
    legend = [(metric, C["accent"])]
    flows = trade.top_flows(E, n_arcs)
    mx = float(flows["value"].max()) if len(flows) else 1.0
    markers, idx = [], {}
    for iso in sorted(set(flows["from"]) | set(flows["to"])):
        c = countries.centroid(iso)
        if c:
            idx[iso] = len(markers)
            markers.append({"name": countries.short_ru(iso), "iso": iso, "city": "", "lat": c[0], "lon": c[1],
                            "color": "#4DA3FF", "open": False, "label": f"экспорт {bal.loc[iso, 'export'] / 1e9:,.0f} млрд USD"})
    arcs = [{"a": idx[r.from_], "b": idx[r.to], "w": float(r.value / mx), "color": "#FF9F1A",
             "label": f"{r.from_} → {r.to}"} for r in flows.rename(columns={"from": "from_"}).itertuples()
            if r.from_ in idx and r.to in idx]
    world_map(colors, infos, f"ТОРГОВЛЯ {yr}: {metric.upper()}", legend, arcs=arcs, markers=markers,
              mode="flat" if mapmode == "Карта" else "globe", height=620)

    a, b = st.columns(2)
    with a:
        iso = st.selectbox("Страна", list(E.index), format_func=countries.short_ru, index=list(E.index).index("RUS") if "RUS" in E.index else 0)
        pt = trade.partner_table(E, iso).head(12)
        if len(pt):
            fig = go.Figure()
            fig.add_bar(x=[countries.short_ru(i) for i in pt.index], y=pt["export"] / 1e9, name="Экспорт", marker_color=C["accent"])
            fig.add_bar(x=[countries.short_ru(i) for i in pt.index], y=pt["import"] / 1e9, name="Импорт", marker_color=C["blue"])
            fig.update_layout(barmode="group", yaxis_title="млрд USD")
            st.markdown(f"**Главные партнёры: {countries.short_ru(iso)}**")
            st.plotly_chart(style(fig, 340), width="stretch")
    with b:
        top = trade.top_flows(E, 18)
        labels = sorted(set(top["from"]) | set(top["to"]))
        pos = {n: i for i, n in enumerate(labels)}
        sk = go.Figure(go.Sankey(
            node=dict(label=[countries.short_ru(n) for n in labels], pad=12, color=C["accent"]),
            link=dict(source=[pos[x] for x in top["from"]], target=[pos[x] + 0 for x in top["to"]],
                      value=(top["value"] / 1e9).tolist(), color="rgba(77,163,255,.35)")))
        st.markdown("**Крупнейшие торговые потоки** (млрд USD)")
        st.plotly_chart(style(sk, 340), width="stretch")
    summary = bal.copy()
    summary.insert(0, "Страна", [countries.short_ru(i) for i in summary.index])
    summary["Сальдо, % ВВП"] = [summary.loc[i, "balance"] / gdp_e[i] if i in gdp_e.index and gdp_e[i] > 0 else np.nan
                                for i in summary.index]
    summary["PageRank"] = cen["pagerank"]
    summary["Сообщество"] = cen["community"].astype(int) + 1
    for c in ("export", "import", "balance"):
        summary[c] = summary[c] / 1e9
    st.dataframe(summary.sort_values("export", ascending=False), hide_index=True, width="stretch", height=380,
                 column_config={"export": st.column_config.NumberColumn("Экспорт, млрд USD", format="%.0f"),
                                "import": st.column_config.NumberColumn("Импорт, млрд USD", format="%.0f"),
                                "balance": st.column_config.NumberColumn("Сальдо, млрд USD", format="%.0f"),
                                "export_hhi": st.column_config.NumberColumn("Концентрация экспорта (HHI)", format="%.2f"),
                                "Сальдо, % ВВП": st.column_config.NumberColumn(format="percent"),
                                "PageRank": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Сеть строится по взвешенным потокам; PageRank тем выше, чем больше важных стран зависят от рынка сбыта. "
               "Сообщества найдены по модулярности (жадный алгоритм). Заражение: канал спроса (потери экспорта) плюс "
               "канал поставок (с весом 0.5), при допущении, что кризисная страна сокращает импорт на выбранную долю.")


# ================================================================ ИСТОЧНИКИ
def view_sources():
    t = catalog.table()
    info = panel_mod.snapshot_info()
    m = st.columns(3)
    m[0].metric("Источников в каталоге", len(t))
    m[1].metric("Проверено на реальных данных", int(t["Статус"].str.startswith("✅").sum()))
    m[2].metric("Подключено по документации", int(t["Статус"].str.startswith("🟡").sum()))
    c1, c2 = st.columns([2, 3])
    cat = c1.multiselect("Категория", sorted(t["Категория"].unique()))
    q = c2.text_input("Поиск")
    if cat:
        t = t[t["Категория"].isin(cat)]
    if q:
        t = t[t.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    st.dataframe(t, hide_index=True, width="stretch", height=560)
    st.caption("✅ данные есть во встроенном снимке и проверены расчётами; 🟡 код написан по документации и проверен на "
               "образцах ответов, живой запрос нужно проверить у себя (кнопка «Проверить источники»); ⚪ источник описан, "
               "но не подключён: можно подключить по запросу.")
    st.subheader("Встроенный снимок")
    st.write(f"Обновлён: **{info['updated']}**. Обновить: `python scripts/update_snapshot.py`.")
    rows = [{"Файл": k, "Описание": v["desc"], "Строк": v["rows"], "Источник": v["source"]}
            for k, v in info.get("files", {}).items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ================================================================ МОРЕ: танкеры и проливы
def _choke_markers(meta_rows, anomalies, col_label):
    out = []
    for name, (lat, lon, ru) in meta_rows.items():
        a = anomalies.get(name)
        if a is None:
            out.append({"name": ru, "lat": lat, "lon": lon, "color": "#7a8697", "size": 6, "label": "данных PortWatch нет",
                        "spark": []})
            continue
        ch = a["change"]
        color = C["down"] if ch < -0.15 else (C["up"] if ch > 0.15 else C["accent"])
        out.append({"name": ru, "lat": lat, "lon": lon, "color": color, "size": float(np.clip(5 + np.log1p(a["recent"]) * 1.2, 6, 14)),
                    "label": f"{col_label}: {a['recent']:.0f}/сут, {ch:+.0%} к среднему за 90 дн.", "spark": a["spark"],
                    "detail": f"{col_label} в среднем за 7 дней: <b>{a['recent']:.1f}</b> в сутки<br>среднее за 90 дней: {a['baseline']:.1f}<br>"
                              f"отклонение: <b>{ch:+.0%}</b>"})
    return out


def view_sea(version, bump):
    from . import maritime
    st.caption("Движение судов: реальные агрегаты IMF PortWatch (проходы по проливам и каналам, AIS, без ключа), живые "
               "позиции (Digitraffic, Балтика, без ключа; AISStream, бесплатный ключ) и условная модель маршрутов "
               "танкеров для демонстрации. Открытых мировых AIS-данных в реальном времени без регистрации не существует.")
    con = store.connect_macro()
    sc = st.columns(3)
    if sc[0].button("Загрузить PortWatch (проливы)"):
        try:
            with st.spinner("Запрашиваю IMF PortWatch…"):
                n_ports, n_rows = maritime.refresh_portwatch(con)
            st.success(f"Загружено проливов: {n_ports}, строк трафика: {n_rows}")
            bump()
        except Exception as exc:
            st.error(f"PortWatch: {exc}")
    live = sc[1].button("Живые суда: Балтика (Digitraffic)")
    with sc[2].expander("AISStream (бесплатный ключ, любая область)"):
        key = st.text_input("Ключ AISStream", type="password")
        area = st.selectbox("Область", ["Ормузский пролив", "Суэц и Красное море", "Сингапур и Малакка", "Ла-Манш", "Мексиканский залив"])
        boxes = {"Ормузский пролив": (24.0, 54.0, 28.0, 58.0), "Суэц и Красное море": (12.0, 32.0, 31.0, 44.0),
                 "Сингапур и Малакка": (0.5, 99.0, 6.0, 105.0), "Ла-Манш": (49.0, -2.0, 52.0, 3.0),
                 "Мексиканский залив": (24.0, -98.0, 30.0, -88.0)}
        if st.button("Собрать позиции (25 с)") and key:
            try:
                with st.spinner("Слушаю поток AISStream 25 секунд…"):
                    df = maritime.collect_aisstream(key, boxes[area])
                st.session_state["ais"] = df
                st.success(f"Судов в области: {len(df)}")
            except Exception as exc:
                st.error(str(exc))
    if live:
        try:
            with st.spinner("Запрашиваю Digitraffic…"):
                st.session_state["ais"] = maritime.fetch_digitraffic()
        except Exception as exc:
            st.error(f"Digitraffic: {exc}")

    series = store.load_series(con, "choke")
    pos = store.load_series(con, "choke_pos")
    cols = sorted({c.split("|")[1] for c in series.columns}) if len(series) else []
    anomalies, meta_rows = {}, dict(maritime.CHOKEPOINTS)
    tanker_col = next((c for c in cols if c.lower() == "n_tanker"), None)
    label_cols = {c: c for c in cols}
    if len(series) and tanker_col:
        daily = series[[k for k in series.columns if k.endswith("|" + tanker_col)]]
        daily.columns = [k.split("|")[0] for k in daily.columns]
        rows = {}
        for name in daily.columns:
            st_ = maritime.choke_stats(daily[name])
            if st_:
                rows[name] = st_
        meta_rows = {}
        for pw_name, a in rows.items():
            ref = maritime.match_chokepoint(pw_name)
            if ref:
                lat, lon, ru = maritime.CHOKEPOINTS[ref]
            elif f"{pw_name}|lat" in pos.columns:
                lat, lon, ru = float(pos[f"{pw_name}|lat"].dropna().iloc[-1]), float(pos[f"{pw_name}|lon"].dropna().iloc[-1]), pw_name
            else:
                continue
            meta_rows[pw_name] = (lat, lon, ru)
        anomalies = rows
        last = max(a["last"] for a in rows.values()) if rows else None
        st.caption(f"Реальные данные PortWatch: проливов {len(rows)}, тип судов «{tanker_col}», последняя дата в данных "
                   f"{last:%d.%m.%Y}. Окна считаются по календарю (7 и 90 суток до этой даты)." if last is not None else "")
    else:
        st.info("Реальные данные по проливам не загружены: нажми «Загрузить PortWatch». Ниже на карте условная модель "
                "(демо) и справочное положение проливов.")
    markers = _choke_markers(meta_rows, anomalies, "танкеров")
    ais = st.session_state.get("ais")
    ship_pts = maritime.ships_from_ais(ais, "Танкер" if st.checkbox("Только танкеры", value=True) else None) if ais is not None and len(ais) else []
    if ship_pts:
        st.success(f"Показаны живые позиции судов: {len(ship_pts)} (Digitraffic/AISStream).")
    demo_routes = maritime.demo_ships() if st.checkbox("Показать условные маршруты танкеров (демо)", value=not anomalies and not ship_pts) else []
    m = st.columns(4)
    m[0].metric("Проливов на карте", len(markers))
    m[1].metric("Реальных данных", len(anomalies))
    m[2].metric("Живых судов", len(ship_pts))
    if anomalies:
        worst = min(anomalies.items(), key=lambda kv: kv[1]["change"])
        m[3].metric("Сильнее всего упал трафик", worst[0], f"{worst[1]['change']:+.0%}", delta_color="inverse")
    _sea_map(markers, demo_routes, ship_pts)
    if anomalies:
        t = pd.DataFrame([{"Пролив": meta_rows[k][2], "Танкеров/сут (7д)": a["recent"], "Среднее 90д": a["baseline"],
                           "Отклонение": a["change"], "Тренд": a["spark"]} for k, a in anomalies.items() if k in meta_rows]).sort_values("Отклонение")
        st.dataframe(t, hide_index=True, width="stretch", height=380, column_config={
            "Танкеров/сут (7д)": st.column_config.NumberColumn(format="%.1f"), "Среднее 90д": st.column_config.NumberColumn(format="%.1f"),
            "Отклонение": st.column_config.NumberColumn(format="percent"), "Тренд": st.column_config.LineChartColumn("60 дней")})
        st.caption("Данные PortWatch: оценки по AIS, обновляются еженедельно с задержкой в несколько дней, возможны ревизии. "
                   "Падение трафика через пролив может означать конфликт, штормы, смену маршрутов или лаг публикации.")
    if ais is not None and len(ais):
        st.markdown("**Суда в выбранной области**")
        show = ais.copy()
        if "class" in show:
            st.dataframe(show.groupby("class").size().rename("Судов").reset_index(), hide_index=True)
        st.dataframe(show.head(300), hide_index=True, width="stretch", height=300)


def _sea_map(markers, demo_routes, ship_pts):
    html = globe.build([], [], {}, title="МОРЕ: ТАНКЕРЫ И ПРОЛИВЫ" + (" · ДЕМО-МАРШРУТЫ" if demo_routes else ""),
                       legend=[("трафик выше нормы", C["up"]), ("в норме", C["accent"]), ("упал более чем на 15%", C["down"]),
                               ("танкер (реальная позиция)", "#FF9F1A")],
                       height=660, lon0=55.0, ships=demo_routes, ship_points=ship_pts, chokes=markers, mode="flat")
    components.html(html, height=666)


# ================================================================ КОНФЛИКТЫ
def view_conflicts(version, bump, use_demo):
    from . import conflicts
    con = store.connect_events()
    stored = store.load_ged(con)
    has_real = len(stored) > 0
    with st.expander("Загрузить события UCDP GED (реальные, без ключа)", expanded=not has_real):
        st.caption("Файл с сайта Uppsala Conflict Data Program (события с гибелью людей, 1989+, с координатами). Если сайт недоступен "
                   "из сети, скачай ZIP вручную и загрузи ниже; для API нужен бесплатный токен.")
        c1, c2 = st.columns(2)
        if c1.button("Скачать GED с ucdp.uu.se"):
            try:
                with st.spinner("Скачиваю и разбираю (десятки МБ)…"):
                    ged = conflicts.fetch_ged()
                st.success(f"События: {len(ged)}, сопоставлено со странами: {ged['iso3'].notna().mean():.0%}")
                store.upsert_ged(con, ged)
                bump()
            except Exception as exc:
                st.error(str(exc))
        up = c2.file_uploader("…или CSV/ZIP GED", type=["csv", "zip"])
        if up is not None:
            try:
                if up.name.endswith(".zip"):
                    import io, zipfile
                    with zipfile.ZipFile(io.BytesIO(up.read())) as zf:
                        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                        raw = pd.read_csv(zf.open(name), low_memory=False)
                else:
                    raw = pd.read_csv(up, low_memory=False)
                ged = conflicts.parse_ged(raw)
                store.upsert_ged(con, ged)
                st.success(f"Импортировано {len(ged)} событий")
                bump()
            except Exception as exc:
                st.error(str(exc))
        token = st.text_input("Токен UCDP API (необязательно)", type="password")
        if token and st.button("Загрузить через API (последние годы)"):
            try:
                ged = conflicts.fetch_ged_api(token, StartDate=f"{datetime.now().year - 2}-01-01")
                store.upsert_ged(con, ged)
                bump()
            except Exception as exc:
                st.error(str(exc))
    if has_real:
        ged = stored.assign(date=pd.to_datetime(stored["date"], errors="coerce"))
        src = "UCDP GED (реальные события)"
    else:
        ged = conflicts.demo_events()
        src = "ДЕМО (синтетика вокруг известных очагов, НЕ реальные события)"
        st.warning("Показаны синтетические события вокруг известных очагов: чтобы увидеть реальные, загрузи UCDP GED выше.")
    years = sorted(int(y) for y in ged["year"].dropna().unique())
    c1, c2, c3 = st.columns([3, 2, 3])
    yr = c1.select_slider("Период", options=years, value=(years[max(0, len(years) - 3)], years[-1]))
    vtypes = c2.multiselect("Тип насилия", list(conflicts.VIOLENCE), default=list(conflicts.VIOLENCE), format_func=lambda k: conflicts.VIOLENCE[k])
    cell = c3.select_slider("Детализация карты, градусов", options=[1.0, 2.0, 3.0, 5.0], value=2.0)
    d = ged[(ged["year"] >= yr[0]) & (ged["year"] <= yr[1]) & (ged["type_of_violence"].isin(vtypes) | ged["type_of_violence"].isna())]
    g = conflicts.grid(d, cell).head(160)
    m = st.columns(4)
    s = conflicts.summary(d, list(range(yr[0], yr[1] + 1)))
    m[0].metric("Событий", f"{s['events']:,}")
    m[1].metric("Погибших (лучшая оценка)", f"{s['deaths']:,.0f}")
    m[2].metric("Из них гражданских", f"{s['civilians']:,.0f}")
    m[3].metric("Стран", s["countries"])
    ref = conflicts.reference_table()
    layer = st.radio("Слой карты", ["События (ячейки)", "Справочник очагов (вручную)", "Оба"], horizontal=True)
    pts = []
    if layer != "Справочник очагов (вручную)":
        for r in g.itertuples():
            pts.append({"name": f"ячейка {r.lat:.0f}°, {r.lon:.0f}°", "lat": r.lat, "lon": r.lon, "deaths": float(r.deaths), "scale": 0.55,
                        "label": f"погибших: <b>{r.deaths:,.0f}</b><br>событий: {r.events:,}"})
    if layer != "События (ячейки)":
        for r in ref.itertuples():
            pts.append({"name": r.name, "lat": r.lat, "lon": r.lon, "deaths": 150.0 if r.type != "очаг напряжённости" else 20.0, "scale": 0.7,
                        "label": f"<b>{r.type}</b>, с {r.since}<br>{r.note}",
                        "detail": f"<b>{r.type}</b>, с {r.since} года<br>{r.note}<br><span style='color:#8B96A5'>справочник составлен вручную "
                                  "по состоянию на 2025 год и может устареть; события берутся из UCDP</span>"})
    html = globe.build([], [], {}, title=f"КОНФЛИКТЫ · {src}", legend=[("очаги насилия (размер = число погибших)", "#FF3C3C")],
                       height=640, lon0=30.0, conflicts=pts, mode="flat")
    components.html(html, height=646)
    a, b = st.columns(2)
    with a:
        by_year = ged.groupby("year")["best"].sum()
        fig = go.Figure(go.Bar(x=by_year.index, y=by_year.values, marker_color=C["down"]))
        fig.update_layout(yaxis_title="Погибших в год")
        st.markdown("**Погибшие по годам**")
        st.plotly_chart(style(fig, 320), width="stretch")
    with b:
        top = d.dropna(subset=["conflict_name"]).groupby("conflict_name")["best"].sum().nlargest(12)
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=C["accent"]))
        fig.update_yaxes(autorange="reversed")
        st.markdown("**Самые смертоносные конфликты периода**")
        st.plotly_chart(style(fig, 320), width="stretch")
    cy = d.dropna(subset=["iso3"]).groupby("iso3")["best"].sum().nlargest(15)
    st.dataframe(pd.DataFrame({"Страна": [countries.short_ru(i) for i in cy.index], "Погибших": cy.values}), hide_index=True, width="stretch")
    # модель риска начала конфликта
    with st.expander("Модель риска начала нового вооружённого конфликта (на данных UCDP и макро-панели)"):
        if not has_real:
            st.info("Нужны реальные события UCDP: загрузи их выше. На синтетике модель не запускается.")
        else:
            st.caption("Конфликт «начинается», когда в стране за год погибло ≥25 человек, а годом ранее меньше. Признаки: макро-панель, "
                       "прошлые конфликты, доход, размер. Проверка вне выборки как у остальных кризисов.")
            if st.button("Обучить и проверить"):
                p0 = panel_mod.snapshot_panel() if not len(store.load_panel_long(store.connect_macro())) else panel_mod.from_long(
                    store.load_panel_long(store.connect_macro()))
                pc = panel_mod.with_conflicts(p0, conflicts.country_year(ged.assign(iso3=ged["iso3"])))
                if "conflict" not in crisis.available_kinds(pc):
                    st.warning("Недостаточно данных для меток.")
                else:
                    met, _ = crisis.evaluate(pc, "conflict", first_test_year=2000, last_year=None)
                    st.json({"AUC ансамбля": round(met["ensemble"]["auc"], 3), "событий": met["events"], "наблюдений": met["obs"],
                             "верхний дециль ловит": f"{met['ensemble']['top_decile_capture']:.0%}",
                             "наивный (прошлые конфликты)": round(met.get("naive", {}).get("auc", float('nan')), 3)})
    st.caption("Справочник очагов составлен вручную по общеизвестным данным на 2025 год: это ориентир, а не данные событий. "
               "Источники событий: UCDP GED (без ключа), ACLED (бесплатная регистрация), GDELT (без ключа).")


# ================================================================ ГОРИЗОНТ: 5-10 лет
@st.cache_resource(show_spinner=False)
def eq_bond_model(years):
    from . import longrun
    return longrun.EquityBondModel(years)


@st.cache_data(show_spinner=False)
def cached_growth_backtest(source, version, horizon):
    from . import longrun
    return longrun.growth_backtest(load_panel(source, version), horizon, first_origin=1995)


@st.cache_data(show_spinner=False)
def cached_growth_forecast(source, version, horizon):
    from . import longrun
    return longrun.growth_forecast(load_panel(source, version), horizon)


def view_horizon(version):
    from . import longrun
    st.caption("Оценки на 5–10 лет с явной неопределённостью и честной проверкой вне выборки. Это не предсказание будущего: "
               "здесь видно, что поддаётся оценке (доходность облигаций по ставке), что слабо (акции по CAPE) и что почти не поддаётся "
               "(рост стран на 10 лет сверх мирового среднего).")
    with st.expander("Что можно и нельзя предсказать на 5–10 лет (проверено на реальной истории)", expanded=True):
        st.markdown(
            "- **Облигации США, 10 лет**: номинальная доходность определяется стартовой ставкой, R² вне выборки около 0.89.\n"
            "- **Акции США, 10 лет** по CAPE: R² вне выборки около 0.08 (слабо); оценка сжимается к историческому среднему, "
            "интервалы расширены по реальным ошибкам.\n"
            "- **Рост ВВП стран**: на 5 лет модель лучше наивного «как последние 5 лет» (на 20%), но от мирового среднего почти не "
            "отличается; на 10 лет сверх мирового среднего навыка нет.\n"
            "- **Не предсказываются**: кризисы, войны, технологические сдвиги, рецессии заранее. Для них есть только сценарии.")
    tabs = st.tabs(["Рынки", "Портфель", "Рост стран", "Госдолг"])
    with tabs[0]:
        yrs = st.radio("Горизонт", [10, 5], horizontal=True)
        with st.spinner("Обучаю и проверяю модель на истории с 1871 года…"):
            m = eq_bond_model(yrs)
        c1, c2, c3 = st.columns(3)
        infl = c1.slider("Ожидаемая инфляция, % в год", 0.0, 8.0, 2.7, 0.1) / 100
        cape_now, last, now = m.current_cape(infl)
        cape = c2.slider("CAPE (по умолчанию оценка на сегодня)", 8.0, 60.0, float(round(cape_now, 1)), 0.5)
        y_def = float(pd.read_csv(panel_mod.SNAP / "us10y.csv.gz")["Rate"].iloc[-1])
        yld = c3.slider("Доходность US 10Y, %", 0.5, 12.0, y_def, 0.05) / 100
        est = m.estimate(cape, yld, infl)
        k = st.columns(4)
        k[0].metric(f"Акции США, реально, {yrs} л", pct(est["equity"]["mean"], 1), f"интервал {est['equity']['p10']:.1%} … {est['equity']['p90']:.1%}", delta_color="off")
        k[1].metric("Сырая CAPE-модель (без сжатия)", pct(est["equity"]["raw"], 1))
        k[2].metric(f"Облигации US 10Y, номинально", pct(est["bond"]["nominal"], 1), f"реально {est['bond']['mean']:.1%}", delta_color="off")
        k[3].metric("Историческое среднее акций (реально)", pct(est["hist_equity"], 1))
        st.warning(f"CAPE на сегодня оценён грубо: данные Шиллера в снимке заканчиваются {last:%m.%Y}, цена S&P 500 известна до {now:%m.%Y}; "
                   "оценка пересчитывает последний CAPE на движение цены с поправкой на рост прибыли и инфляцию. Передвинь ползунок, "
                   "если знаешь точное значение.")
        ev = m.evaluate()
        a, b = st.columns(2)
        for col, key, title, color in ((a, "equity", "Акции: прогноз по CAPE против факта", C["accent"]), (b, "bond", "Облигации: прогноз по ставке против факта", C["blue"])):
            df, met = ev[key]
            with col:
                fig = go.Figure()
                fig.add_scatter(x=df["pred"], y=df["actual"], mode="markers", marker=dict(color=color, size=6, opacity=.7), name="даты проверки")
                lim = [float(min(df["pred"].min(), df["actual"].min())), float(max(df["pred"].max(), df["actual"].max()))]
                fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color="#888", dash="dash"), name="идеал")
                fig.update_layout(xaxis_title="Прогноз (CAGR за горизонт)", yaxis_title="Фактическая доходность", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
                st.markdown(f"**{title}**")
                st.plotly_chart(style(fig, 330), width="stretch")
                st.caption(f"Проверка вне выборки с 1920 г.: точек {met['n']}, R² {met['oos_r2']:.2f}, корреляция {met['corr']:.2f}, "
                           f"вес модели при сжатии {met['shrink_weight']:.0%}, σ ошибки {met['oos_sigma']:.1%}")
    with tabs[1]:
        hz = st.radio("Горизонт, лет", [5, 10, 20], horizontal=True, index=1, key="pf_h")
        m = eq_bond_model(10 if hz != 5 else 5)
        infl = st.slider("Ожидаемая инфляция, % в год", 0.0, 8.0, 2.7, 0.1, key="pf_infl") / 100
        est = m.estimate(infl=infl)
        c1, c2, c3 = st.columns(3)
        start = c1.number_input("Стартовый капитал, $ (в сегодняшних ценах)", 10_000.0, 1e9, 1_000_000.0, 50_000.0)
        contrib = c2.number_input("Ежегодный взнос, $ (в сегодняшних ценах)", 0.0, 1e8, 100_000.0, 10_000.0)
        target = c3.number_input("Цель к концу срока, $", 0.0, 1e10, 3_000_000.0, 100_000.0)
        w1, w2, w3, w4 = st.columns(4)
        weights = {"equity": w1.slider("Акции США, %", 0, 100, 60), "bond": w2.slider("Облигации США, %", 0, 100, 30),
                   "cash": w3.slider("Деньги, %", 0, 100, 5), "gold": w4.slider("Золото, %", 0, 100, 5)}
        if sum(weights.values()) == 0:
            return st.warning("Задай ненулевые веса")
        a1, a2 = st.columns(2)
        cash_real = a1.slider("Реальная доходность денег (допущение), %", -2.0, 3.0, 0.0, 0.1) / 100
        gold_real = a2.slider("Реальная доходность золота (допущение), %", -3.0, 5.0, 0.0, 0.1) / 100
        scen = st.selectbox("Сценарий", list(longrun.SCENARIOS))
        hist = m.annual_history()
        mu = {"equity": est["equity"]["mean"], "bond": est["bond"]["mean"]}
        wealth, _ = longrun.simulate_portfolio(hist, mu, weights, years=hz, n=4000, start=start, contribution=contrib,
                                               cash_real=cash_real, gold_real=gold_real, shift=longrun.SCENARIOS[scen])
        sm = longrun.summarize_wealth(wealth, start, contrib, target)
        k = st.columns(5)
        k[0].metric("Медиана (реально)", f"{sm['p50']:,.0f} $")
        k[1].metric("Плохой исход (P5)", f"{sm['p5']:,.0f} $")
        k[2].metric("Хороший исход (P95)", f"{sm['p95']:,.0f} $")
        k[3].metric("Вероятность реального убытка", pct(sm["prob_real_loss"], 0), f"вложено {sm['invested']:,.0f} $", delta_color="off")
        k[4].metric("Вероятность достичь цели", pct(sm.get("prob_target", float("nan")), 0))
        t = np.arange(wealth.shape[1])
        q = np.percentile(wealth, [5, 25, 50, 75, 95], axis=0)
        fig = go.Figure()
        fig.add_scatter(mode="lines", x=t, y=q[4], line=dict(width=0), showlegend=False)
        fig.add_scatter(mode="lines", x=t, y=q[0], fill="tonexty", fillcolor="rgba(255,159,26,0.15)", line=dict(width=0), name="5–95%")
        fig.add_scatter(mode="lines", x=t, y=q[3], line=dict(width=0), showlegend=False)
        fig.add_scatter(mode="lines", x=t, y=q[1], fill="tonexty", fillcolor="rgba(255,159,26,0.30)", line=dict(width=0), name="25–75%")
        fig.add_scatter(mode="lines", x=t, y=q[2], line=dict(color=C["accent"], width=2), name="Медиана")
        if target:
            fig.add_hline(y=target, line=dict(color=C["up"], dash="dash"), annotation_text="цель")
        fig.update_layout(xaxis_title="Годы", yaxis_title="Капитал в сегодняшних ценах, $")
        st.plotly_chart(style(fig, 420), width="stretch")
        rows = []
        for name, shift in longrun.SCENARIOS.items():
            w_, _ = longrun.simulate_portfolio(hist, mu, weights, years=hz, n=1500, start=start, contribution=contrib,
                                               cash_real=cash_real, gold_real=gold_real, shift=shift, seed=1)
            s_ = longrun.summarize_wealth(w_, start, contrib, target)
            rows.append({"Сценарий": name, "Медиана": s_["p50"], "P5": s_["p5"], "Убыток": s_["prob_real_loss"], "Цель": s_.get("prob_target")})
        st.markdown("**Сравнение сценариев** (поправки к реальным доходностям: стагфляция −3 п.п. акции, −2.5 облигации и т.д.)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Медиана": st.column_config.NumberColumn(format="localized"), "P5": st.column_config.NumberColumn(format="localized"),
            "Убыток": st.column_config.NumberColumn(format="percent"), "Цель": st.column_config.NumberColumn(format="percent")})
        st.caption("Модель: средние доходности акций и облигаций США берутся из CAPE- и ставка-моделей, форма распределения и связь "
                   "акций с облигациями из реальной истории с 1871 года (блочный бутстреп по 3 года). Деньги и золото это допущения. "
                   "Реальная доходность считается в долларовых активах и не учитывает курсы валют, налоги и комиссии.")
    with tabs[2]:
        source = "snapshot" if not len(store.load_panel_long(store.connect_macro())) else "live"
        hz = st.radio("Горизонт, лет", [5, 10], horizontal=True, key="g_h")
        with st.spinner("Проверяю модель роста на истории…"):
            met, r = cached_growth_backtest(source, version, hz)
        c = st.columns(4)
        c[0].metric("Ошибка модели (MAE, п.п.)", f"{met['mae_model']:.2f}")
        c[1].metric("«Как последние 5 лет»", f"{met['mae_naive_past']:.2f}", f"модель лучше на {met['skill_vs_naive']:.0%}", delta_color="off")
        c[2].metric("«Мировое среднее»", f"{met['mae_world_mean']:.2f}", f"навык {met['skill_vs_world']:+.0%}", delta_color="off")
        c[3].metric("Корреляция прогноза с фактом", f"{met['corr']:.2f}")
        if met["skill_vs_world"] < 0.05:
            st.warning("Сверх мирового среднего модель почти ничего не добавляет: долгосрочный рост отдельных стран плохо предсказуем.")
        fc, year = cached_growth_forecast(source, version, hz)
        st.caption(f"Прогноз от {year} года: средний годовой рост реального ВВП на следующие {hz} лет. Макро в снимке до 2013 г., "
                   "для актуального прогноза загрузи живые данные на вкладке КРИЗИСЫ.")
        scale = max(float(fc["growth"].quantile(0.95)), 3.0)
        colors = {i: globe.heat(float(v), scale) for i, v in fc["growth"].items()}
        infos = {i: f"<b style='color:#FF9F1A'>{countries.short_ru(i)}</b><br>рост {r_['growth']:.1f}% в год ({r_['p10']:.1f} … {r_['p90']:.1f})"
                 for i, r_ in fc.iterrows()}
        world_map(colors, infos, f"РОСТ ВВП НА {hz} ЛЕТ (ОТ {year} Г.)", [("медленнее", "#1e2634"), ("быстрее", "#FF3C3C")], height=520)
        tab = fc.assign(Страна=[countries.short_ru(i) for i in fc.index]).sort_values("growth", ascending=False)
        st.dataframe(tab[["Страна", "growth", "p10", "p90"]], hide_index=True, width="stretch", height=340, column_config={
            "growth": st.column_config.NumberColumn("Рост, % в год", format="%.1f"), "p10": st.column_config.NumberColumn("P10", format="%.1f"),
            "p90": st.column_config.NumberColumn("P90", format="%.1f")})
    with tabs[3]:
        source = "snapshot" if not len(store.load_panel_long(store.connect_macro())) else "live"
        p = load_panel(source, version)
        pool = [i for i in p.index.get_level_values("iso3").unique() if "gov_debt" in p and p.loc[i, "gov_debt"].notna().any()]
        iso = st.selectbox("Страна", sorted(pool, key=countries.short_ru), format_func=countries.short_ru,
                           index=sorted(pool, key=countries.short_ru).index("USA") if "USA" in pool else 0)
        g = p.loc[iso]
        d0 = float(g["gov_debt"].dropna().iloc[-1])
        ylast = int(g["gov_debt"].dropna().index[-1])
        g10 = float(g["gdp_growth"].dropna().tail(10).mean()) if "gdp_growth" in g and g["gdp_growth"].notna().any() else 2.0
        sd = float(g["gdp_growth"].dropna().tail(20).std()) if "gdp_growth" in g and g["gdp_growth"].notna().sum() > 5 else 2.5
        i5 = float(g["inflation"].dropna().tail(5).mean()) if "inflation" in g and g["inflation"].notna().any() else 3.0
        st.caption(f"Исходные данные: долг {d0:.0f}% ВВП на {ylast} год; средний рост за 10 лет {g10:.1f}%, инфляция за 5 лет {i5:.1f}%. "
                   + ("Снимок до 2013 года: загрузи живые данные для актуальных значений." if source == "snapshot" else ""))
        c1, c2, c3, c4 = st.columns(4)
        gm = c1.slider("Реальный рост, % в год", -2.0, 8.0, float(np.clip(g10, -2, 8)), 0.1)
        pi = c2.slider("Инфляция, % в год", 0.0, 30.0, float(np.clip(i5, 0, 30)), 0.1)
        rn = c3.slider("Номинальная ставка по долгу, %", 0.0, 30.0, float(np.clip(pi + 1.5, 0, 30)), 0.1)
        pb = c4.slider("Первичный баланс, % ВВП (плюс = профицит)", -10.0, 8.0, 0.0, 0.1)
        years = st.radio("Горизонт, лет", [5, 10], horizontal=True, key="debt_h", index=1)
        paths = longrun.debt_paths(d0, gm, pi, rn, pb, years=years, n=4000, g_sd=max(sd, 1.0), r_sd=1.0, pb_sd=1.0)
        star = longrun.stabilizing_balance(d0, gm, pi, rn)
        t = np.arange(years + 1)
        q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        fig = go.Figure()
        fig.add_scatter(mode="lines", x=t, y=q[4], line=dict(width=0), showlegend=False)
        fig.add_scatter(mode="lines", x=t, y=q[0], fill="tonexty", fillcolor="rgba(255,77,79,0.15)", line=dict(width=0), name="5–95%")
        fig.add_scatter(mode="lines", x=t, y=q[3], line=dict(width=0), showlegend=False)
        fig.add_scatter(mode="lines", x=t, y=q[1], fill="tonexty", fillcolor="rgba(255,77,79,0.30)", line=dict(width=0), name="25–75%")
        fig.add_scatter(mode="lines", x=t, y=q[2], line=dict(color=C["down"], width=2), name="Медиана")
        fig.update_layout(xaxis_title="Годы", yaxis_title="Долг, % ВВП")
        st.plotly_chart(style(fig, 380), width="stretch")
        k = st.columns(4)
        k[0].metric("Долг через горизонт, медиана", f"{q[2][-1]:.0f}% ВВП", f"{q[2][-1] - d0:+.0f} п.п.", delta_color="inverse")
        k[1].metric("Стабилизирующий первичный баланс", f"{star:+.1f}% ВВП", f"сейчас {pb:+.1f}", delta_color="off")
        k[2].metric("Вероятность долга выше 100% ВВП", pct(float((paths[:, -1] > 100).mean()), 0))
        k[3].metric("Вероятность роста долга", pct(float((paths[:, -1] > d0).mean()), 0))
        st.caption("Динамика долга: d' = d(1+r)/((1+g)(1+π)) − pb со случайными шоками роста (по истории страны), ставки и баланса. "
                   "Это учебная модель устойчивости долга, а не прогноз: она не учитывает дефолты, реструктуризации и смену политики.")
