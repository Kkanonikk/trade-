"""Страница НАСТРОЙКИ: подключения к онлайн-источникам, параметры аналитики и интерфейса, ключи API, кэш."""
from datetime import datetime

import pandas as pd
import streamlit as st

from . import config, connections, problems, settings, sources, store
from . import views_worldmode as vw

STATUS_COLOR = {"онлайн": "#1c4a35", "ошибка": "#5a1f22", "формат?": "#5a4a14", "не проверен": "#222C3B"}


def init_state():
    """Загружает сохранённые настройки в session_state один раз за сессию."""
    if st.session_state.get("_settings_loaded"):
        return
    cur = settings.load()
    st.session_state["set_page"] = cur["start_page"]
    st.session_state["set_map"] = cur["map_mode"]
    st.session_state["set_year"] = int(cur["as_of_year"])
    st.session_state["set_red"] = int(cur["red_zone"])
    st.session_state["set_hz"] = int(cur["forecast_horizon"])
    st.session_state["set_rank"] = int(cur["rank_size"])
    st.session_state["set_tape"] = bool(cur["show_tape"])
    st.session_state["set_rotate"] = bool(cur["auto_rotate"])
    st.session_state["set_autorefresh"] = bool(cur["auto_refresh_on_start"])
    st.session_state["set_hist"] = int(cur["history_years"])
    st.session_state["set_demo_warn"] = bool(cur["show_demo_warnings"])
    st.session_state["set_dec"] = int(cur["decimals"])
    for k in problems.DIMENSIONS:
        st.session_state[f"set_w_{k}"] = float(cur["dim_weights"].get(k, 1.0))
    st.session_state["_settings_loaded"] = True


def _save():
    s = st.session_state
    settings.save({"start_page": s["set_page"], "map_mode": s["set_map"], "as_of_year": s["set_year"], "red_zone": s["set_red"],
                   "forecast_horizon": s["set_hz"], "rank_size": s["set_rank"], "show_tape": s["set_tape"], "auto_rotate": s["set_rotate"],
                   "auto_refresh_on_start": s["set_autorefresh"], "history_years": s["set_hist"], "show_demo_warnings": s["set_demo_warn"],
                   "decimals": s["set_dec"], "dim_weights": {k: s[f"set_w_{k}"] for k in problems.DIMENSIONS}})


def _style(df):
    def color(v):
        return f"background-color:{STATUS_COLOR.get(v, '#222C3B')};color:#E8ECF1;font-weight:600"
    return df.style.map(color, subset=["Статус"])


def page_settings(version, bump):
    init_state()
    st.markdown("### Настройки и подключения")
    t1, t2, t3, t4 = st.tabs(["Подключения", "Параметры", "Ключи API", "Данные и кэш"])
    with t1:
        if st.session_state.get("probe_df") is None:          # при первом открытии проверяем сразу
            with st.spinner("Проверяю подключения к источникам (до 10 секунд)…"):
                st.session_state["probe_df"] = connections.probe_all()
                st.session_state["probe_time"] = datetime.now().strftime("%d.%m %H:%M:%S")
        probe = st.session_state.get("probe_df")
        c = st.columns([1, 1, 3])
        if c[0].button("Проверить подключения", type="primary", key="st_probe"):
            with st.spinner("Проверяю источники параллельно (до 10 секунд)…"):
                st.session_state["probe_df"] = connections.probe_all()
                st.session_state["probe_time"] = datetime.now().strftime("%d.%m %H:%M:%S")
            st.rerun()
        if c[1].button("Глубокая проверка форматов", key="st_deep"):
            with st.spinner("Запрашиваю образцы данных и разбираю ответы (может занять минуту)…"):
                st.session_state["deep_df"] = pd.DataFrame(sources.probe_all(), columns=["Источник", "OK", "Сообщение"])
        c[2].caption("Быстрая проверка делает один запрос к каждому источнику и показывает доступность и задержку. Глубокая проверка разбирает образцы данных так же, как загрузка.")
        if probe is None:
            st.info("Подключения ещё не проверялись. Нажми «Проверить подключения».")
        else:
            ok = int((probe["Статус"] == "онлайн").sum())
            m = st.columns(4)
            m[0].metric("Онлайн", f"{ok} из {len(probe)}")
            m[1].metric("С ошибкой", int((probe["Статус"] == "ошибка").sum()))
            m[2].metric("Формат под вопросом", int((probe["Статус"] == "формат?").sum()))
            m[3].metric("Последняя проверка", st.session_state.get("probe_time", ""))
            if ok == len(probe):
                st.success("Все источники доступны.")
            elif ok == 0:
                st.error("Ни один источник не отвечает: проверь интернет, VPN или прокси. Встроенные снимки данных продолжают работать офлайн.")
            else:
                st.warning("Часть источников недоступна: данные из них будут браться из встроенного снимка или будут пропущены. Остальное загрузится.")
            show = probe.drop(columns=["id"]).copy()
            st.dataframe(_style(show), hide_index=True, width="stretch", height=36 * len(show) + 40,
                         column_config={"мс": st.column_config.NumberColumn("Задержка, мс", format="%d")})
        if st.session_state.get("deep_df") is not None:
            st.markdown("**Глубокая проверка форматов**")
            st.dataframe(st.session_state["deep_df"], hide_index=True, width="stretch")
        st.markdown("### Что лежит в локальной базе")
        st.dataframe(connections.local_freshness(), hide_index=True, width="stretch")
        rep = st.session_state.get("load_all_report")
        if rep is not None:
            st.markdown(f"### Последняя загрузка всех данных ({st.session_state.get('load_all_time', '')})")
            st.dataframe(rep, hide_index=True, width="stretch", column_config={"ОК": st.column_config.CheckboxColumn()})
        st.caption("Загрузить все онлайн-данные одной кнопкой можно в боковой панели слева. Ключи для этих источников не нужны, "
                   "необязательные ключи (FRED, AISStream, UCDP, SEC) задаются на вкладке «Ключи API».")
    with t2:
        st.caption("Настройки сохраняются в data/settings.json при каждом изменении и применяются ко всем страницам.")
        a, b = st.columns(2)
        with a:
            st.markdown("#### Интерфейс")
            st.selectbox("Стартовая страница глобального режима", vw.PAGES, key="set_page", on_change=_save)
            st.radio("Вид карт по умолчанию", ["Карта", "Глобус"], horizontal=True, key="set_map", on_change=_save)
            st.checkbox("Показывать бегущую строку рынков", key="set_tape", on_change=_save)
            st.checkbox("Автовращение глобуса", key="set_rotate", on_change=_save)
            st.checkbox("Показывать предупреждения о демо-данных", key="set_demo_warn", on_change=_save)
            st.slider("Размер рейтингов на картах", 5, 15, key="set_rank", on_change=_save)
        with b:
            st.markdown("#### Аналитика")
            st.slider("Год для индекса проблем и паспортов", 1995, 2023, key="set_year", on_change=_save)
            st.slider("Граница «красной зоны» (процентиль)", 50, 90, key="set_red", on_change=_save)
            st.radio("Горизонт прогноза по умолчанию, лет", [5, 10], horizontal=True, key="set_hz", on_change=_save)
            st.slider("Глубина загрузки котировок, лет", 1, 10, key="set_hist", on_change=_save)
            st.checkbox("Загружать все онлайн-данные при запуске приложения", key="set_autorefresh", on_change=_save,
                        help="Один раз за сессию, как нажатие кнопки «Загрузить все онлайн-данные». Нужен интернет.")
        st.markdown("#### Веса измерений индекса проблем")
        cols = st.columns(4)
        for i, (k, (name, _)) in enumerate(problems.DIMENSIONS.items()):
            cols[i % 4].slider(name, 0.0, 3.0, step=0.25, key=f"set_w_{k}", on_change=_save)
        st.caption("Веса применяются на страницах ПРОБЛЕМЫ и ПАСПОРТ. 0 исключает измерение. Устойчивость рейтинга к весам показана на странице ПРОБЛЕМЫ.")
    with t3:
        st.caption("Ключи сохраняются только в локальный файл .env на этом компьютере (он в .gitignore) и не отправляются никуда, кроме самого источника. "
                   "Все основные источники работают без ключей.")
        for env, (name, hint) in settings.KEYS.items():
            cur = config.get(env)
            c1, c2, c3 = st.columns([2, 4, 1])
            c1.markdown(f"**{name}**<br><span style='color:#7D8898;font-size:.74rem'>{hint}</span>", unsafe_allow_html=True)
            val = c2.text_input(env, value="", type="password", placeholder=("задан: " + settings.mask(cur)) if cur else "не задан", key=f"key_{env}", label_visibility="collapsed")
            if c3.button("Сохранить", key=f"save_{env}") and val:
                settings.save_env(env, val)
                st.success(f"{name}: сохранён")
                st.rerun()
    with t4:
        info = vw.panel_mod.snapshot_info()
        c = st.columns(3)
        c[0].metric("Снимок данных от", info.get("global_updated") or info.get("updated", "н/д"))
        db = config.DB_PATH
        c[1].metric("Локальная база", f"{db.stat().st_size / 1e6:.1f} МБ" if db.exists() else "нет")
        c[2].metric("Папка данных", str(config.DATA_DIR))
        a, b = st.columns(2)
        if a.button("Сбросить кэш вычислений", key="st_cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Кэш сброшен: следующая страница пересчитает данные.")
        if b.button("Сбросить настройки к умолчанию", key="st_reset"):
            settings.save(settings.DEFAULTS)
            st.session_state.pop("_settings_loaded", None)
            st.rerun()
        st.caption("Встроенные снимки лежат в assets/snapshot и обновляются скриптами update_snapshot.py, update_global_snapshot.py, update_hub.py или автоматически через GitHub Actions.")
