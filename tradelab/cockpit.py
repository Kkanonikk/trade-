"""Риск-кокпит в духе корпоративных платформ: один портфель, разложение по странам и классам, страновой и климатический риск,
факторные сценарии с интервалами. Чистые функции без интерфейса; факторные чувствительности оцениваются по истории, а не задаются вручную.
"""
import numpy as np
import pandas as pd

from . import config

FACTORS = {"Акции мира (S&P 500)": ("index", "S&P 500"), "Нефть Brent": ("cross", "Нефть Brent"),
           "Золото": ("cross", "Золото"), "Индекс доллара": ("cross", "Индекс доллара"), "VIX (страх)": ("cross", "VIX")}

SCENARIOS = {
    "Падение рынка −20%": {"Акции мира (S&P 500)": -0.20},
    "Нефтяной шок +40%": {"Нефть Brent": 0.40},
    "Сильный доллар +10%": {"Индекс доллара": 0.10},
    "Стагфляция": {"Акции мира (S&P 500)": -0.15, "Нефть Brent": 0.30, "Золото": 0.15},
    "Бегство в безопасность": {"Акции мира (S&P 500)": -0.30, "Индекс доллара": 0.08, "Золото": 0.10, "VIX (страх)": 0.80},
}


def look_through(weights):
    """Доли портфеля по странам риска: {ISO3 или 'Без страны'/'н/д': доля}. ETF на корзины раскладываются по весам корзины."""
    out = {}
    for t, w in weights.items():
        if w <= 0:
            continue
        if t in config.BASKETS:
            for iso, share in config.BASKETS[t].items():
                out[iso] = out.get(iso, 0.0) + w * share
        elif t.endswith(".MOEX"):
            out["RUS"] = out.get("RUS", 0.0) + w
        elif t in config.COUNTRY_OF_RISK:
            iso = config.COUNTRY_OF_RISK[t]
            out[iso] = out.get(iso, 0.0) + w
        elif config.asset_class(t) in ("Сырьё", "Крипта"):
            out["Без страны"] = out.get("Без страны", 0.0) + w
        else:
            out["н/д"] = out.get("н/д", 0.0) + w
    return pd.Series(out).sort_values(ascending=False)


def class_exposure(weights):
    out = {}
    for t, w in weights.items():
        out[config.asset_class(t)] = out.get(config.asset_class(t), 0.0) + w
    return pd.Series(out).sort_values(ascending=False)


def weighted_overlay(exposure, values):
    """Взвешенное среднее показателя страны по портфелю. exposure: доли по ISO3, values: Series страна -> значение.
    Возвращает (среднее по странам с данными, доля портфеля с данными, таблица вкладов)."""
    e = exposure[exposure.index.isin(values.index)]
    cover = float(e.sum() / exposure.sum()) if exposure.sum() > 0 else 0.0
    if e.empty or e.sum() <= 0:
        return float("nan"), 0.0, pd.DataFrame()
    avg = float((e * values.reindex(e.index)).sum() / e.sum())
    tbl = pd.DataFrame({"доля": e, "значение": values.reindex(e.index)})
    tbl["вклад"] = tbl["доля"] * tbl["значение"] / e.sum()
    return avg, cover, tbl.sort_values("вклад", ascending=False)


def _weekly(series):
    s = series.dropna().copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else pd.DatetimeIndex(s.index)
    return s.resample("W").last().pct_change().dropna()


def factor_returns(world):
    cols = {}
    for name, (kind, col) in FACTORS.items():
        df = world.get(kind)
        if df is not None and len(df) and col in df.columns:
            cols[name] = _weekly(df[col])
    return pd.DataFrame(cols).dropna(how="any") if cols else pd.DataFrame()


def exposures(port_returns, F):
    """Многофакторная регрессия недельной доходности портфеля на факторы (МНК, устойчивые SE HC1).
    Возвращает DataFrame: beta, se, t и число недель; alpha и R² в attrs."""
    r = _weekly((1 + port_returns).cumprod())
    d = pd.concat([r.rename("y"), F], axis=1, join="inner").dropna()
    if len(d) < 60 or F.shape[1] == 0:
        raise ValueError("Мало общих недельных наблюдений (нужно минимум 60)")
    X = np.column_stack([np.ones(len(d)), d[F.columns].to_numpy()])
    y = d["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    meat = (X * res[:, None] ** 2).T @ X
    cov = bread @ meat @ bread * len(d) / max(len(d) - X.shape[1], 1)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = pd.DataFrame({"beta": beta[1:], "se": se[1:]}, index=F.columns)
    out["t"] = out["beta"] / out["se"].replace(0, np.nan)
    out.attrs.update(alpha=float(beta[0]), r2=float(1 - res.var() / y.var()) if y.var() > 0 else 0.0, n=len(d),
                     resid_sd=float(res.std(ddof=X.shape[1])))
    return out


def scenario_pnl(exp, shocks):
    """Оценка изменения портфеля при шоках факторов (доли) по оценённым чувствительностям: (оценка, нижняя, верхняя) на ±1.96 SE.
    Шоки считаются мгновенными и независимыми: это арифметика чувствительностей, а не прогноз."""
    est = lo = hi = 0.0
    for f, sh in shocks.items():
        if f in exp.index:
            b, se = exp.loc[f, "beta"], exp.loc[f, "se"]
            est += b * sh
            lo += min((b - 1.96 * se) * sh, (b + 1.96 * se) * sh)
            hi += max((b - 1.96 * se) * sh, (b + 1.96 * se) * sh)
    return est, lo, hi
