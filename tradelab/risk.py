"""Риск-аналитика портфеля (в духе risk-систем: VaR, CVaR, вклад в риск, стресс-тесты)."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def returns_frame(frames):
    """frames: {ticker: DataFrame[ts, close]} -> простые дневные доходности, общий индекс дат."""
    closes = pd.DataFrame({t: f.set_index("ts")["close"] for t, f in frames.items()})
    return closes.pct_change().dropna(how="any")


def portfolio_returns(R, weights):
    w = pd.Series(weights).reindex(R.columns).fillna(0.0)
    return R @ w


def var_cvar(r, alpha=0.95):
    """Историческая VaR/CVaR как положительные числа (доля капитала за 1 период)."""
    var = -np.quantile(r, 1 - alpha)
    tail = r[r <= -var]
    cvar = -tail.mean() if len(tail) else var
    return float(var), float(cvar)


def summary(r):
    ann_ret = float(np.exp(np.log1p(r).mean() * TRADING_DAYS) - 1)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    equity = (1 + r).cumprod()
    return {
        "ann_return": ann_ret,
        "ann_vol": vol,
        "sharpe": ann_ret / vol if vol > 0 else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "var95": var_cvar(r, 0.95)[0], "cvar95": var_cvar(r, 0.95)[1],
        "var99": var_cvar(r, 0.99)[0], "cvar99": var_cvar(r, 0.99)[1],
    }


def risk_contributions(R, weights):
    """Доля каждой бумаги в общей дисперсии портфеля (сумма = 1)."""
    w = pd.Series(weights).reindex(R.columns).fillna(0.0).to_numpy()
    cov = R.cov().to_numpy()
    var = w @ cov @ w
    if var <= 0:
        return pd.Series(0.0, index=R.columns)
    return pd.Series(w * (cov @ w) / var, index=R.columns)


def beta(r, bench):
    cov = np.cov(r, bench)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan


def stress_table(R, weights, market_shocks=(-0.05, -0.10, -0.20, -0.30)):
    """Сценарии: рынок (равновзвешенный индекс) падает на X, каждая бумага реагирует через свою бету."""
    bench = R.mean(axis=1)
    w = pd.Series(weights).reindex(R.columns).fillna(0.0)
    betas = pd.Series({c: beta(R[c], bench) for c in R.columns})
    port_beta = float((w * betas).sum())
    rows = [{"Сценарий": f"Рынок {s:+.0%}", "Потери портфеля": port_beta * s} for s in market_shocks]
    return pd.DataFrame(rows), port_beta


def worst_days(r, n=5):
    return r.nsmallest(n)


def bootstrap_paths(r, horizon=21, n_paths=2000, seed=0):
    """Веер возможных траекторий капитала: случайные дни из истории (бутстрап)."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(r.to_numpy(), size=(n_paths, horizon), replace=True)
    return np.cumprod(1 + draws, axis=1)
