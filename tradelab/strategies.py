"""Портфельный бэктест: регулярная ребалансировка по правилу, с дрейфом весов и комиссиями."""
import numpy as np
import pandas as pd

from . import optimize, risk

METHODS = {
    "equal": "Равные веса",
    "min_vol": "Минимальный риск",
    "risk_parity": "Равный вклад в риск",
    "max_sharpe": "Максимальный Шарп",
    "momentum": "Моментум (топ-N)",
}


def target(window, method, top_n=5, max_weight=0.3):
    """Целевые веса по окну прошлых доходностей (без будущих данных)."""
    if method == "momentum":
        score = (1 + window).prod() - 1
        top = score.nlargest(top_n).index
        return {t: 1.0 / len(top) for t in top}
    return optimize.optimize(window, method, max_weight=max_weight, shrink=True)


def run(R, method, rebalance_days=21, lookback=120, top_n=5, max_weight=0.3, cost_bps=5.0):
    """Возвращает DataFrame: ret (чистая дневная доходность), equity, turnover."""
    if len(R) < lookback + rebalance_days:
        raise ValueError("Слишком мало данных для бэктеста стратегии")
    cols = list(R.columns)
    w = pd.Series(0.0, index=cols)
    rows = []
    for i in range(lookback, len(R)):
        cost = turnover = 0.0
        if (i - lookback) % rebalance_days == 0:
            new = pd.Series(target(R.iloc[i - lookback:i], method, top_n, max_weight)).reindex(cols).fillna(0.0)
            turnover = float((new - w).abs().sum())
            cost = turnover * cost_bps / 1e4
            w = new
        day = R.iloc[i]
        gross = float((w * day).sum())
        rows.append((R.index[i], gross - cost, turnover))
        w = w * (1 + day) / (1 + gross) if gross > -1 else w  # дрейф весов
    out = pd.DataFrame(rows, columns=["ts", "ret", "turnover"]).set_index("ts")
    out["equity"] = (1 + out["ret"]).cumprod()
    return out


def summarize(out, periods=252):
    s = risk.summary(out["ret"])
    s["turnover_year"] = float(out["turnover"].sum() / len(out) * periods)
    return s
