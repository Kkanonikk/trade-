"""Индекс проблем стран: где больше всего и где меньше всего проблем по семи измерениям (голод, здоровье, экономика, институты и безопасность,
среда и климат, человеческий капитал, демографическое давление).

Каждый показатель переводится в процентиль среди стран (100 = хуже всех), показатели измерения усредняются с весами, измерения
сводятся в общий индекс с весами пользователя. Это сравнительный индекс на открытых данных, а не прогноз и не оценка «качества жизни»:
выбор показателей и весов спорен, поэтому веса меняются в интерфейсе и проверяется устойчивость.
"""
import numpy as np
import pandas as pd

from . import gdata, hunger

HUNGER = "computed:hunger_index"
# измерение -> (название, [(ключ или ("hub", начало названия), направление +1 = больше значит хуже / -1 = больше значит лучше, вес)])
DIMENSIONS = {
    "hunger": ("Голод и питание", [(HUNGER, +1, 2.0), ("gm:malnutrition_weight_for_age_percent_of_children_under_5", +1, 1.0),
                                   (("hub", "Prevalence of undernourishment"), +1, 1.0), ("gm:food_supply_kilocalories_per_person_and_day", -1, 1.0)]),
    "health": ("Здоровье", [(("hub", "Infant mortality rate"), +1, 1.0), (("hub", "Healthy Life Expectancy"), -1, 1.0),
                            ("gm:life_expectancy_at_birth_data_from_ihme", -1, 1.0), ("gm:adults_with_hiv_percent_age_15_49", +1, 0.5),
                            ("gm:at_least_basic_water_source_overall_access_percent", -1, 1.0), ("gm:at_least_basic_sanitation_overall_access_percent", -1, 1.0),
                            (("hub", "Share of deaths from air pollution (outdoor and indoor)"), +1, 0.7)]),
    "economy": ("Экономика и бедность", [("gm:gdppercapita_us_inflation_adjusted", -1, 1.5), ("gm:extreme_poverty_percent_people_below_300_a_day", +1, 1.5),
                                          ("gm:inequality_index_gini", +1, 1.0), ("gm:inflation_annual_percent", +1, 0.7),
                                          ("gm:debt_to_foreigners_by_public_and_private_percent_of_gni", +1, 0.7), ("gm:aid_received_percent_of_gni", +1, 0.5),
                                          ("gm:aged_15plus_unemployment_rate_percent", +1, 0.7)]),
    "institutions": ("Институты и безопасность", [(("hub", "Corruption Perception Index"), -1, 1.5), (("hub", "Death rate from all forms of violence"), +1, 1.0),
                                                   (("hub", "Violent deaths in conflicts"), +1, 1.0), ("gm:battle_deaths_per_100_000", +1, 0.5),
                                                   (("hub", "Gender Inequality Index"), +1, 0.7)]),
    "environment": ("Среда и климат", [("co2:co2_per_capita", +1, 0.7), ("co2:co2_per_gdp", +1, 0.7), ("en:renewables_share_elec", -1, 1.0),
                                       ("gm:forest_coverage_percent", -1, 0.5)]),
    "capital": ("Человеческий капитал", [("gm:hdi_human_development_index", -1, 1.5), (("hub", "Expected Years of Schooling"), -1, 1.0),
                                         ("gm:literacy_rate_adult_total_percent_of_people_ages_15_and_above", -1, 1.0), ("gm:internet_users", -1, 0.7),
                                         ("gm:broadband_subscribers_per_100_people", -1, 0.5)]),
    "demography": ("Демографическое давление", [("gm:children_and_elderly_per_100_adults", +1, 1.0), ("gm:teen_fertility_rate_births_per_1000_women_ages_15_19", +1, 1.0)]),
}
DEFAULT_WEIGHTS = {k: 1.0 for k in DIMENSIONS}


def resolve(w):
    """Ключи показателей: строки остаются как есть, ("hub", «начало названия») заменяется ключом с самыми свежими данными. Возвращает {спецификация: ключ}."""
    meta = gdata.meta()
    out = {}
    for _, (_, items) in DIMENSIONS.items():
        for spec, _, _ in items:
            if isinstance(spec, str):
                out[spec] = spec if (spec == HUNGER or spec in w.columns) else None
                continue
            best = None
            for k in w.columns:
                m = meta.get(k, {})
                if m.get("source") == "owid_hub" and m.get("label", "").lower().startswith(spec[1].lower()):
                    s = w[k].dropna()
                    last = int(s.index.get_level_values("year").max()) if len(s) else 0
                    cand = (last, s.index.get_level_values("iso3").nunique(), k)
                    if best is None or cand > best:
                        best = cand
            out[spec] = best[2] if best else None
    return out


def label_of(key):
    if key == HUNGER:
        return "Индекс продовольственной уязвимости"
    return gdata.meta().get(key, {}).get("label", key)


def values(w, year, window=6, refs=None):
    """Таблица страна x показатель: последнее значение не позже year в окне window лет (без заглядывания в будущее)."""
    refs = refs or resolve(w)
    cols = {}
    for spec, key in refs.items():
        if key is None:
            continue
        if key == HUNGER:
            cols[key] = hunger.index(w, year)[0]
            continue
        d = w[key].dropna().reset_index()
        d = d[(d["year"] <= year) & (d["year"] >= year - window)].sort_values("year")
        if not d.empty:
            cols[key] = d.groupby("iso3")[key].last()
    return pd.DataFrame(cols)


def score(w, year, weights=None, window=6, min_share=0.5):
    """Возвращает dict: values (показатели), pct (процентили проблемности 0–100), dims (измерения 0–100), composite (общий индекс 0–100),
    coverage (доля веса показателей, по которым есть данные)."""
    refs = resolve(w)
    V = values(w, year, window, refs)
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    P = pd.DataFrame(index=V.index)
    for dim, (_, items) in DIMENSIONS.items():
        for spec, sign, wt in items:
            key = refs.get(spec)
            if key is None or key not in V:
                continue
            r = V[key].rank(pct=True) * 100
            P[key] = r if sign > 0 else 100 - r
    dims, cover = {}, {}
    for dim, (_, items) in DIMENSIONS.items():
        num = pd.Series(0.0, index=V.index)
        den = pd.Series(0.0, index=V.index)
        tot = 0.0
        for spec, _, wt in items:
            key = refs.get(spec)
            tot += wt
            if key is None or key not in P:
                continue
            have = P[key].notna()
            num += P[key].fillna(0) * wt
            den += have * wt
        dims[dim] = (num / den).where(den >= min_share * tot)
        cover[dim] = den / tot
    D = pd.DataFrame(dims)
    wv = pd.Series({k: weights[k] for k in D.columns})
    have = D.notna()
    comp = (D.fillna(0) * wv).sum(axis=1) / (have * wv).sum(axis=1)
    comp = comp.where(have.sum(axis=1) >= 4)
    n_dims = have.sum(axis=1)
    return {"values": V, "pct": P, "dims": D, "composite": comp.dropna().rename("problems"), "n_dims": n_dims, "refs": refs}


def history(w, years, weights=None):
    """Общий индекс по годам: DataFrame страна x год."""
    return pd.DataFrame({y: score(w, y, weights)["composite"] for y in years})


def diagnosis(res, iso, n=3):
    """Самые сильные и самые слабые стороны страны: показатели с наибольшим и наименьшим процентилем проблемности."""
    if iso not in res["pct"].index:
        return [], []
    row = res["pct"].loc[iso].dropna().sort_values(ascending=False)
    worst = [(k, float(v)) for k, v in row.head(n).items() if v >= 60]
    best = [(k, float(v)) for k, v in row.tail(n).iloc[::-1].items() if v <= 40]
    return worst, best


def peers(res, iso, k=6):
    """Страны с самым похожим профилем по измерениям (евклидово расстояние в 0–100)."""
    D = res["dims"].dropna(thresh=5)
    if iso not in D.index:
        return pd.DataFrame()
    X = D.fillna(D.mean())
    dist = np.sqrt(((X - X.loc[iso]) ** 2).mean(axis=1)).drop(iso).sort_values()
    return pd.DataFrame({"distance": dist.head(k)})
