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
def world_map(colors, info, title, legend, arcs=None, markers=None, mode="flat", height=620, lon0=20.0):
    html = globe.build(markers or [], arcs or [], colors, title=title, legend=legend, height=height, lon0=lon0,
                       info=info, mode=mode)
    components.html(html, height=height + 6)


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
def get_models(source, version):
    p = load_panel(source, version)
    models = {k: crisis.CrisisModel(p, k) for k in crisis.KINDS}
    return models, crisis.FxModel(p)


@st.cache_data(show_spinner=False)
def cached_replay(source, version, kind, year):
    return crisis.replay(load_panel(source, version), kind, year)


@st.cache_data(show_spinner=False)
def cached_eval(source, version, kind, first_year, last_year):
    return crisis.evaluate(load_panel(source, version), kind, first_year, last_year)


def view_crises(version, bump):
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
    kind = st.selectbox("Вид кризиса", list(crisis.KINDS), format_func=lambda k: crisis.KINDS[k])
    if kind == "recession":
        st.warning("Рецессии прогнозируются плохо: на реальной истории модель не лучше простого признака «рост ВВП "
                   "уже замедлился». Карту стоит читать как индикатор уязвимости, а не прогноз.")

    if mode.startswith("Машина"):
        max_year = panel_mod.WEO_LAST_ACTUAL if source_key == "snapshot" else int(
            panel.index.get_level_values("year").max())
        year = st.slider("Год, на который модель строит прогноз (учится только на более ранних годах)", 1990, max_year,
                         min(2008, max_year))
        try:
            out, summ = cached_replay(source_key, version, kind, year)
        except ValueError as exc:
            return st.warning(str(exc))
        probs = out["prob"]
        m = st.columns(4)
        m[0].metric("Стран в оценке", summ["countries"])
        m[1].metric("Реально началось кризисов", summ["events"])
        m[2].metric("Из них в верхнем дециле риска", f"{summ['events_in_top_decile']} из {summ['events']}"
                    if summ["events"] else "н/д")
        m[3].metric("AUC этого года", f"{summ['auc']:.2f}" if np.isfinite(summ["auc"]) else "н/д")
        title = f"{crisis.KINDS[kind].upper()} · РИСК НА {year} ГОД (модель знает только прошлое)"
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
            models, fxm = get_models(source_key, version)
        model = models[kind]
        probs_df, fyear = model.predict()
        title = f"{crisis.KINDS[kind].upper()} · ПРОГНОЗ НА {fyear + 1} ГОД"
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
            for k in crisis.KINDS:
                with st.spinner(f"Проверяю: {crisis.KINDS[k]}…"):
                    try:
                        met, _ = cached_eval(source_key, version, k, first, last)
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
    mode = st.radio("Вид карты", ["Карта", "Глобус"], horizontal=True, key=f"mapmode_{title[:8]}")
    world_map(colors, infos, title, legend, markers=markers, mode="flat" if mode == "Карта" else "globe")


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
        vals = {crisis.KINDS[k]: float(models[k].predict()[0]["prob"].get(iso, np.nan)) for k in crisis.KINDS}
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
