"""Старение населения и его связь с экономикой (реальные данные ООН/Gapminder и IMF/World Bank).

Показатели старения без живых данных: медианный возраст и коэффициент иждивенчества (дети и пожилые на 100 взрослых).
Доля 65+ доступна через World Bank API (живая загрузка).
Оценка влияния на экономику: панельная регрессия с фиксированными эффектами страны и периода на 5-летних окнах,
стандартные ошибки кластеризованы по странам. Это статистическая связь, а не доказанная причинность.
"""
import numpy as np
import pandas as pd

from . import gdata

MEDIAN_AGE = "gm:median_age_years"
DEPENDENCY = "gm:children_and_elderly_per_100_adults"
POPULATION = "gm:total_population_with_projections"
GROWTH_POP = "gm:population_growth_annual_percent_with_projections"
KIDS_SURV, KIDS_DEAD = "gm:surviving_kids_per_woman", "gm:dead_kids_per_woman"
FIRST_PROJECTION = 2024


def ranking(w, year, key=MEDIAN_AGE, n=15):
    s = gdata.cross_section(w, key, year)
    return s.sort_values(ascending=False).head(n), s.sort_values().head(n)


def fertility(w):
    """Суммарный коэффициент рождаемости = выживающие + умершие дети на женщину (приближение Gapminder). Прогнозные годы
    в источнике заданы редкими точками (2020, 2030, 2050, 2099): между ними значения линейно интерполируются."""
    if KIDS_SURV not in w.columns or KIDS_DEAD not in w.columns:
        return pd.Series(dtype=float)
    parts = []
    for k in (KIDS_SURV, KIDS_DEAD):
        wide = w[k].unstack("year")
        parts.append(wide.interpolate(axis=1, limit_area="inside"))
    tfr = (parts[0] + parts[1]).stack().rename("tfr")
    tfr.index.names = ["iso3", "year"]
    return tfr


def _calibrated_segment(start_pop, rates, target):
    """Накопление годовых приростов (%) от start_pop со сдвигом на постоянную δ так, чтобы последняя точка попала в target."""
    logs = np.log1p(np.asarray(rates) / 100.0)
    delta = (np.log(target / start_pop) - logs.sum()) / len(logs)
    return start_pop * np.exp(np.cumsum(logs + delta))


def population_path(w, calibrate=True):
    """Годовая траектория населения: факты до последнего года с данными, дальше накопление годового прироста ООН.
    В источнике население после 2013 года дано только точками 2050 и 2100, а прирост есть на каждый год: форма траектории
    берётся из прироста, а уровень калибруется так, чтобы точно попасть в точки 2050 и 2100 (calibrate=False: без калибровки,
    для проверки согласованности). Возвращает DataFrame страна x год."""
    pop = w[POPULATION].dropna().unstack("year")
    gr = w[GROWTH_POP].dropna().unstack("year")
    years = list(range(1950, 2101))
    out = {}
    for iso in pop.index.intersection(gr.index):
        p0 = pop.loc[iso].dropna()
        anchors = p0[p0.index <= 2013]
        if anchors.empty:
            continue
        y0 = int(anchors.index.max())
        path = {y: float(pop.loc[iso, y]) for y in anchors.index if y >= 1950}
        cur_y, cur = y0, float(anchors.iloc[-1])
        for target_year in (2050, 2100):
            ys = list(range(cur_y + 1, target_year + 1))
            rates = [gr.loc[iso].get(y) for y in ys]
            if not ys or any(r is None or pd.isna(r) for r in rates):
                break
            tgt = pop.loc[iso].get(target_year)
            if calibrate and tgt is not None and pd.notna(tgt) and tgt > 0:
                seg = _calibrated_segment(cur, rates, float(tgt))
            else:
                seg = cur * np.exp(np.cumsum(np.log1p(np.asarray(rates) / 100.0)))
            path.update(dict(zip(ys, seg)))
            cur_y, cur = target_year, float(seg[-1])
        out[iso] = pd.Series(path).reindex(years)
    return pd.DataFrame(out).T


def path_consistency(w, path):
    """Относительная ошибка восстановленной траектории против точек 2050 и 2100 из источника (медиана по странам)."""
    pop = w[POPULATION].dropna().unstack("year")
    res = {}
    for y in (2050, 2100):
        if y in pop.columns and y in path.columns:
            both = pd.concat([path[y], pop[y]], axis=1, keys=["rec", "src"]).dropna()
            res[y] = float(((both["rec"] / both["src"]) - 1).abs().median()) if len(both) else float("nan")
    return res


def population_peaks(w, since=2024):
    """Для каждой страны: год пика населения по прогнозу ООН, население сейчас, в 2050 и 2100 и изменение since→2100."""
    path = population_path(w)
    rows = []
    for iso, r in path.iterrows():
        proj = r.loc[since:2100].dropna()
        if len(proj) < 60 or pd.isna(r.get(since)):
            continue
        rows.append({"iso3": iso, "now": float(r[since]), "peak_year": int(proj.idxmax()), "peak": float(proj.max()),
                     "y2050": float(r.get(2050, np.nan)), "y2100": float(r.get(2100, np.nan)),
                     "change_2100": float(r.get(2100, np.nan) / r[since] - 1)})
    return pd.DataFrame(rows).set_index("iso3") if rows else pd.DataFrame(columns=["now", "peak_year", "peak", "y2050", "y2100", "change_2100"])


def aging_speed(w, y0=2025, y1=2050):
    """Рост медианного возраста за период (лет) по прогнозу ООН."""
    a, b = gdata.cross_section(w, MEDIAN_AGE, y0), gdata.cross_section(w, MEDIAN_AGE, y1)
    return (b - a).dropna().rename("delta_median_age")


# ---------------------------------------------------------------- влияние на экономику
OUTCOMES = {
    "growth": ("wp:gdp_growth", "Рост реального ВВП, п.п. в год", "IMF WEO"),
    "savings": ("wp:savings", "Норма сбережений, % ВВП", "IMF WEO"),
    "fiscal": ("wp:fiscal_balance", "Баланс бюджета, % ВВП", "IMF WEO"),
    "investment": ("wp:investment", "Норма инвестиций, % ВВП", "IMF WEO"),
}


def _windows(w, outcome_key, driver_key, first=1980, last=2013, width=5):
    """Панель 5-летних окон: среднее значение исхода и изменение драйвера внутри окна, начальный доход."""
    rows = []
    income = w.get("gm:income_per_person_with_projections")
    for start in range(first, last - width + 2, width):
        end = start + width - 1
        rng = range(start, end + 1)
        for iso, g in w.groupby(level="iso3"):
            g = g.droplevel("iso3")
            y = g[outcome_key].reindex(rng) if outcome_key in g else None
            d0, d1 = g[driver_key].get(start - 1, g[driver_key].get(start)), g[driver_key].get(end)
            if y is None or y.notna().sum() < 3 or pd.isna(d0) or pd.isna(d1):
                continue
            inc = income.loc[iso].get(start - 1) if income is not None and iso in income.index.get_level_values("iso3") else np.nan
            rows.append({"iso3": iso, "period": start, "y": float(y.mean()), "d_driver": float(d1 - d0), "driver0": float(d0),
                         "log_income0": float(np.log(inc)) if pd.notna(inc) and inc > 0 else np.nan})
    return pd.DataFrame(rows)


def _fe_ols(df, y, xs, entity="iso3", time="period", country_fe=True):
    """Оценка с фиксированными эффектами страны и периода (внутреннее преобразование + дамми периода), кластер-устойчивые SE."""
    d = df.dropna(subset=[y] + xs).copy()
    counts = d.groupby(entity)[y].transform("size")
    d = d[counts >= 2]
    if len(d) < 50:
        raise ValueError("Недостаточно наблюдений")
    X = d[xs].copy()
    for p in sorted(d[time].unique())[1:]:
        X[f"t{p}"] = (d[time] == p).astype(float)
    if country_fe:
        Xw = X - X.groupby(d[entity]).transform("mean")
        yw = d[y] - d.groupby(entity)[y].transform("mean")
    else:                                   # без эффектов страны: смешивает уровень развития и возраст, но показывает ассоциацию
        X.insert(0, "const", 1.0)
        Xw, yw = X, d[y]
    A = Xw.to_numpy()
    beta, *_ = np.linalg.lstsq(A, yw.to_numpy(), rcond=None)
    resid = yw.to_numpy() - A @ beta
    bread = np.linalg.pinv(A.T @ A)
    meat = np.zeros((A.shape[1], A.shape[1]))
    for _, idx in d.groupby(entity).indices.items():
        s = A[idx].T @ resid[idx]
        meat += np.outer(s, s)
    G, n, k = d[entity].nunique(), len(d), A.shape[1]
    cov = bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / max(n - k - G, 1))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = pd.DataFrame({"coef": beta, "se": se}, index=X.columns)
    out["t"] = out["coef"] / out["se"].replace(0, np.nan)
    return out, {"n": n, "countries": G}


def economic_effect(w, driver=MEDIAN_AGE, outcomes=("growth", "savings", "fiscal", "investment"), country_fe=True):
    """Связь изменения драйвера (медианного возраста) за 5 лет со средним исходом окна при фикс. эффектах страны и периода.
    Возвращает таблицу по исходам: коэффициент на +1 год медианного возраста, SE, t, N, число стран."""
    rows = []
    for name in outcomes:
        key, label, src = OUTCOMES[name]
        if key not in w.columns or driver not in w.columns:
            continue
        df = _windows(w, key, driver)
        if df.empty:
            continue
        try:
            res, info = _fe_ols(df, "y", ["d_driver", "log_income0"], country_fe=country_fe)
        except ValueError:
            continue
        r = res.loc["d_driver"]
        rows.append({"outcome": name, "label": label, "coef": float(r["coef"]), "se": float(r["se"]), "t": float(r["t"]),
                     "n": info["n"], "countries": info["countries"], "source": src, "country_fe": country_fe})
    return pd.DataFrame(rows)


def scenario_effect(w, effects, y0=2025, y1=2050, outcome="growth"):
    """Подразумеваемое изменение показателя в год из-за старения по прогнозу ООН: коэф. × (рост медианного возраста за 5 лет).
    Возвращает DataFrame по странам: годовое изменение медианного возраста, эффект и интервал ±1.96 SE (п.п. в год)."""
    row = effects[effects["outcome"] == outcome]
    if row.empty:
        return pd.DataFrame()
    c, se = float(row["coef"].iloc[0]), float(row["se"].iloc[0])
    speed = aging_speed(w, y0, y1) / (y1 - y0)          # лет медианного возраста в год
    per_window = speed * 5.0                            # изменение за 5-летнее окно, как в регрессии
    return pd.DataFrame({"aging_per_year": speed, "effect": c * per_window,
                         "low": (c - 1.96 * se) * per_window, "high": (c + 1.96 * se) * per_window})
