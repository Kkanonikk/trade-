"""Оптимизация портфеля (long-only, сумма весов = 1, ограничение на вес одной бумаги)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import risk
from .risk import TRADING_DAYS


def _stats(R, shrink=False):
    mu = R.mean().to_numpy() * TRADING_DAYS
    cov = (risk.shrunk_cov(R) if shrink else R.cov()).to_numpy() * TRADING_DAYS
    return mu, cov


def _solve(objective, n, max_weight):
    max_weight = max(max_weight, 1.0 / n + 1e-9)
    res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def optimize(R, objective="min_vol", max_weight=0.3, risk_free=0.0, shrink=False, mu=None, risk_aversion=3.0):
    mu_h, cov = _stats(R, shrink)
    mu = mu_h if mu is None else np.asarray(mu, dtype=float)
    n = len(mu)
    if objective == "min_vol":
        w = _solve(lambda w: w @ cov @ w, n, max_weight)
    elif objective == "max_sharpe":
        w = _solve(lambda w: -(w @ mu - risk_free) / np.sqrt(max(w @ cov @ w, 1e-12)), n, max_weight)
    elif objective == "mean_variance":
        w = _solve(lambda w: -(w @ mu - 0.5 * risk_aversion * w @ cov @ w), n, max_weight)
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


def portfolio_point(R, weights, shrink=False):
    mu, cov = _stats(R, shrink)
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


def black_litterman(R, views, confidence, tau=0.05, risk_aversion=3.0, shrink=True):
    """Апостериорная доходность Black-Litterman.

    views: {тикер: ожидаемая годовая доходность}, confidence: {тикер: 0..1}.
    Априорная (равновесная) доходность берётся из равных весов, так как рыночных капитализаций у нас нет.
    """
    _, cov = _stats(R, shrink)
    n = len(R.columns)
    pi = risk_aversion * cov @ np.full(n, 1.0 / n)
    names = [t for t in views if t in R.columns]
    if not names:
        return pd.Series(pi, index=R.columns)
    P = np.zeros((len(names), n))
    for i, t in enumerate(names):
        P[i, list(R.columns).index(t)] = 1.0
    Q = np.array([views[t] for t in names])
    conf = np.clip(np.array([confidence[t] for t in names]), 1e-3, 0.999)
    omega = np.diag((1.0 / conf - 1.0) * np.diag(P @ (tau * cov) @ P.T))
    middle = np.linalg.inv(P @ (tau * cov) @ P.T + omega)
    mu = pi + tau * cov @ P.T @ middle @ (Q - P @ pi)
    return pd.Series(mu, index=R.columns)
