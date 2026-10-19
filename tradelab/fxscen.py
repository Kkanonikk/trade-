"""Чувствительность валют к глобальным факторам и сценарии шоков.

Для каждой валюты по недельным доходностям оцениваются коэффициенты к трём факторам:
сила доллара (корзина остальных валют), нефть Brent и страх рынка (VIX). Затем сценарий (например, «нефть -30%,
VIX +50%») переводится в ожидаемое движение курса с интервалом. Это статистическая связь за выбранное окно, а не
структурная модель: в кризисах связи меняются.
"""
import numpy as np
import pandas as pd

from . import panel as panel_mod

# Страны/зоны в данных ФРС (H.10) -> ISO3 (евро: Германия как представитель зоны)
FX_COUNTRY = {"Australia": "AUS", "Brazil": "BRA", "Canada": "CAN", "China": "CHN", "Denmark": "DNK", "Euro": "DEU",
              "Hong Kong": "HKG", "India": "IND", "Japan": "JPN", "Malaysia": "MYS", "Mexico": "MEX",
              "New Zealand": "NZL", "Norway": "NOR", "Singapore": "SGP", "South Africa": "ZAF", "South Korea": "KOR",
              "Sweden": "SWE", "Switzerland": "CHE", "Taiwan": "TWN", "Thailand": "THA", "United Kingdom": "GBR",
              "Venezuela": "VEN"}
CURRENCY_RU = {"Australia": "Австралийский доллар", "Brazil": "Бразильский реал", "Canada": "Канадский доллар",
               "China": "Китайский юань", "Denmark": "Датская крона", "Euro": "Евро", "Hong Kong": "Гонконгский доллар",
               "India": "Индийская рупия", "Japan": "Японская иена", "Malaysia": "Малайзийский ринггит",
               "Mexico": "Мексиканское песо", "New Zealand": "Новозеландский доллар", "Norway": "Норвежская крона",
               "Singapore": "Сингапурский доллар", "South Africa": "Южноафриканский рэнд", "South Korea": "Вона",
               "Sweden": "Шведская крона", "Switzerland": "Швейцарский франк", "Taiwan": "Тайваньский доллар",
               "Thailand": "Тайский бат", "United Kingdom": "Фунт стерлингов", "Venezuela": "Венесуэльский боливар"}
MAJORS = ["Euro", "Japan", "United Kingdom", "Switzerland", "Canada", "Australia", "Sweden", "Norway", "New Zealand"]
FACTORS = {"usd": "Сила доллара (корзина валют)", "oil": "Нефть Brent", "vix": "Страх рынка (VIX)"}


def snapshot_fx():
    """Дневные курсы 'нац. валюта за 1 USD' (рост = валюта слабеет)."""
    df = pd.read_csv(panel_mod.SNAP / "fx_daily.csv.gz", parse_dates=["Date"])
    df["Exchange rate"] = pd.to_numeric(df["Exchange rate"], errors="coerce")
    return df.pivot_table(index="Date", columns="Country", values="Exchange rate").sort_index()


def snapshot_factors():
    brent = pd.read_csv(panel_mod.SNAP / "brent.csv.gz", parse_dates=["Date"]).set_index("Date")["Price"]
    vix = pd.read_csv(panel_mod.SNAP / "vix.csv.gz", parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    return pd.DataFrame({"brent": brent, "vix": vix}).sort_index()


def weekly_logret(prices):
    return np.log(prices.resample("W-FRI").last()).diff()


def build_dataset(fx, factors):
    """Недельные лог-доходности: валюты (LCU/USD) и факторы; фактор доллара считается как среднее по корзине."""
    r = weekly_logret(fx.where(fx > 0))
    f = pd.DataFrame({"oil": weekly_logret(factors["brent"].where(factors["brent"] > 0)),
                      "vix": weekly_logret(factors["vix"].where(factors["vix"] > 0))})
    return r, f


def _usd_factor(r, exclude):
    basket = [c for c in MAJORS if c in r.columns and c != exclude]
    return r[basket].mean(axis=1)


def _ols_hc1(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    n, k = X1.shape
    XtX_inv = np.linalg.pinv(X1.T @ X1)
    meat = (X1 * resid[:, None] ** 2).T @ X1
    cov = n / max(n - k, 1) * XtX_inv @ meat @ XtX_inv
    ss_res, ss_tot = float((resid ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return beta, cov, 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, float(resid.std(ddof=k))


def sensitivities(r, f, years=8, min_weeks=156):
    """Таблица коэффициентов по валютам за последние years лет: beta (usd, oil, vix), t-статистики, R², остаточная σ."""
    end = min(r.dropna(how="all").index.max(), f.dropna(how="all").index.max())
    start = end - pd.DateOffset(years=years)
    rows = []
    for cur in r.columns:
        usd = _usd_factor(r, cur)
        d = pd.DataFrame({"y": r[cur], "usd": usd, "oil": f["oil"], "vix": f["vix"]}).loc[start:end].dropna()
        if len(d) < min_weeks:
            continue
        beta, cov, r2, sigma = _ols_hc1(d[["usd", "oil", "vix"]].to_numpy(), d["y"].to_numpy())
        se = np.sqrt(np.diag(cov))
        rows.append({"currency": cur, "n": len(d), "alpha": beta[0], "b_usd": beta[1], "b_oil": beta[2], "b_vix": beta[3],
                     "t_usd": beta[1] / se[1], "t_oil": beta[2] / se[2], "t_vix": beta[3] / se[3],
                     "se_usd": se[1], "se_oil": se[2], "se_vix": se[3], "r2": r2, "sigma_week": sigma})
    out = pd.DataFrame(rows).set_index("currency")
    return out[out["r2"] >= 0.03]  # без связи с факторами (в т.ч. административные курсы) не оцениваем


def scenario(sens, shocks, horizon_weeks=13):
    """Ожидаемое изменение курса 'нац. валюта за USD' (в %, плюс = слабеет) при шоках факторов (в долях, лог-масштаб).

    shocks: {'usd': 0.05, 'oil': -0.30, 'vix': 0.5}. Интервал P10-P90 учитывает неопределённость коэффициентов и
    остаточную волатильность за горизонт."""
    s = {"usd": shocks.get("usd", 0.0), "oil": shocks.get("oil", 0.0), "vix": shocks.get("vix", 0.0)}
    mean = sum(sens[f"b_{k}"] * v for k, v in s.items())
    var_beta = sum((sens[f"se_{k}"] * v) ** 2 for k, v in s.items())
    sd = np.sqrt(var_beta + sens["sigma_week"] ** 2 * horizon_weeks)
    z = 1.2816
    out = pd.DataFrame({"expected": np.expm1(mean), "p10": np.expm1(mean - z * sd), "p90": np.expm1(mean + z * sd)})
    return out.clip(-0.9, 2.0)
