"""Индекс продовольственной уязвимости стран и его проверка на реальной истории.

Индекс 0–100 (выше = уязвимее) это взвешенная средняя процентильных рангов компонентов. Он описывает уязвимость, а не
прогнозирует голод: объявленные голодоморы (IPC/FEWS NET) в открытых готовых рядах недоступны. Проверка: предсказывает ли
индекс, построенный БЕЗ показателя недоедания, будущее недоедание детей.
"""
import numpy as np
import pandas as pd

# ключ показателя -> (вес, направление: +1 выше = хуже, -1 выше = лучше, подпись для расшифровки)
COMPONENTS = {
    "gm:malnutrition_weight_for_age_percent_of_children_under_5": (0.20, +1, "Недоедание детей"),
    "gm:food_supply_kilocalories_per_person_and_day": (0.15, -1, "Калории на человека"),
    "gm:extreme_poverty_percent_people_below_300_a_day": (0.15, +1, "Крайняя бедность"),
    "gm:gdppercapita_us_inflation_adjusted": (0.10, -1, "Доход на душу"),
    "gm:agriculture_workers_percent_of_employment": (0.08, +1, "Занятость в сельском хозяйстве"),
    "gm:internal_renewable_water_cu_meters_per_person": (0.07, -1, "Водные ресурсы на человека"),
    "gm:drought_affected_annual_number": (0.07, +1, "Засухи (пострадавшие на душу)"),
    "gm:battle_deaths_per_100_000": (0.08, +1, "Боевые потери"),
    "gm:inflation_annual_percent": (0.05, +1, "Инфляция"),
    "gm:aid_received_percent_of_gni": (0.05, +1, "Зависимость от помощи"),
}
TARGET = "gm:malnutrition_weight_for_age_percent_of_children_under_5"
POP = "gm:total_population_with_projections"


def _latest(w, key, year, window=8):
    """Последнее значение показателя по странам в окне [year-window, year] (без заглядывания в будущее)."""
    if key not in w.columns:
        return pd.Series(dtype=float)
    d = w[key].dropna().reset_index()
    d = d[(d["year"] <= year) & (d["year"] >= year - window)].sort_values("year")
    return d.groupby("iso3")[key].last()


def features(w, year):
    """Таблица компонентов по странам на год year."""
    cols = {}
    for k in COMPONENTS:
        s = _latest(w, k, year)
        if k == "gm:drought_affected_annual_number" and POP in w.columns:
            pop = _latest(w, POP, year, window=40)
            s = np.log1p(s.reindex(pop.index).fillna(0) / pop * 1000)        # пострадавшие на 1000 жителей, логарифм
        if k in ("gm:gdppercapita_us_inflation_adjusted", "gm:internal_renewable_water_cu_meters_per_person"):
            s = np.log(s.where(s > 0))
        if k == "gm:inflation_annual_percent":
            s = s.clip(-5, 100)
        cols[k] = s
    return pd.DataFrame(cols)


def index(w, year, exclude=(), min_components=4):
    """Индекс уязвимости 0–100 и вклад компонентов. exclude: ключи, исключаемые из расчёта (для проверки)."""
    F = features(w, year)
    ranks = pd.DataFrame(index=F.index)
    weights = {}
    for k, (wt, sign, _) in COMPONENTS.items():
        if k in exclude or k not in F:
            continue
        r = F[k].rank(pct=True)
        ranks[k] = r if sign > 0 else 1 - r
        weights[k] = wt
    W = pd.Series(weights)
    have = ranks.notna()
    num = (ranks.fillna(0) * W).sum(axis=1)
    den = (have * W).sum(axis=1)
    score = (num / den * 100).where(have.sum(axis=1) >= min_components)
    contrib = ranks.mul(W, axis=1).div(den, axis=0) * 100
    return score.dropna().rename("index"), contrib.loc[score.dropna().index]


def validate(w, origins=range(1995, 2016, 5), horizon=5):
    """Для каждого года-происхождения t: индекс без недоедания по данным ≤ t против недоедания детей в t+horizon.
    Сравнение с «инерцией» (недоедание в t предсказывает t+horizon). Возвращает DataFrame по происхождениям."""
    rows = []
    for t in origins:
        idx, _ = index(w, t, exclude=(TARGET,))
        fut = _latest(w, TARGET, t + horizon, window=2)
        past = _latest(w, TARGET, t, window=5)
        d = pd.concat([idx, fut.rename("future"), past.rename("past")], axis=1).dropna(subset=["index", "future"])
        if len(d) < 25:
            continue
        rho_idx = d["index"].corr(d["future"], method="spearman")
        both = d.dropna(subset=["past"])
        rho_past = both["past"].corr(both["future"], method="spearman") if len(both) >= 25 else np.nan
        top = d["index"] >= d["index"].quantile(0.75)
        hi = d["future"] >= d["future"].quantile(0.75)
        rows.append({"origin": t, "n": len(d), "rho_index": rho_idx, "rho_persistence": rho_past,
                     "top_quartile_precision": float((top & hi).sum() / max(top.sum(), 1))})
    return pd.DataFrame(rows)
