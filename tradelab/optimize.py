"""Оптимизация портфеля (long-only, сумма весов = 1, ограничение на вес одной бумаги)."""
import numpy as np
from scipy.optimize import minimize

from .risk import TRADING_DAYS


def _stats(R):
    mu = R.mean().to_numpy() * TRADING_DAYS
    cov = R.cov().to_numpy() * TRADING_DAYS
    return mu, cov


def _solve(objective, n, max_weight):
    max_weight = max(max_weight, 1.0 / n + 1e-9)
    res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def optimize(R, objective="min_vol", max_weight=0.3, risk_free=0.0):
    mu, cov = _stats(R)
    n = len(mu)
    if objective == "min_vol":
        w = _solve(lambda w: w @ cov @ w, n, max_weight)
    elif objective == "max_sharpe":
        w = _solve(lambda w: -(w @ mu - risk_free) / np.sqrt(max(w @ cov @ w, 1e-12)), n, max_weight)
    elif objective == "risk_parity":
        def f(w):
            vol = np.sqrt(max(w @ cov @ w, 1e-12))
            rc = w * (cov @ w) / vol
            return float(((rc - vol / n) ** 2).sum())
        w = _solve(f, n, max_weight)
    elif objective == "equal":
        w = np.full(n, 1.0 / n)
    else:
        raise ValueError(objective)
    return dict(zip(R.columns, w))


def portfolio_point(R, weights):
    mu, cov = _stats(R)
    w = np.array([weights.get(c, 0.0) for c in R.columns])
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def random_portfolios(R, n=1500, seed=0):
    mu, cov = _stats(R)
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(len(mu)))
        pts.append((float(np.sqrt(w @ cov @ w)), float(w @ mu)))
    return pts
