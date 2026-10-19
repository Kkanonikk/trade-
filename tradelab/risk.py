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


# ---------------------------------------------------------------- расширенная аналитика
def shrunk_cov(R):
    """Ковариация со сжатием Ledoit-Wolf: устойчивее выборочной при малой истории."""
    from sklearn.covariance import LedoitWolf
    return pd.DataFrame(LedoitWolf().fit(R.to_numpy()).covariance_, index=R.columns, columns=R.columns)


def ewma_cov(R, lam=0.94):
    """Ковариация RiskMetrics: недавние дни весят больше (lam=0.94 для дневных данных)."""
    x = R.to_numpy()
    w = lam ** np.arange(len(x) - 1, -1, -1)
    w /= w.sum()
    cov = (x * w[:, None]).T @ x
    return pd.DataFrame(cov, index=R.columns, columns=R.columns)


def parametric_var(R, weights, alpha=0.95, cov=None):
    """Нормальная VaR и её разложение по бумагам (компонентная VaR, сумма = VaR)."""
    from scipy.stats import norm
    cov = (shrunk_cov(R) if cov is None else cov)
    w = pd.Series(weights).reindex(R.columns).fillna(0.0)
    sigma = float(np.sqrt(w @ cov @ w))
    z = norm.ppf(alpha)
    comp = z * w * (cov @ w) / sigma if sigma > 0 else w * 0
    return z * sigma, comp


def pca_factors(R, k=3):
    """Статистические факторы: главные компоненты стандартизированных доходностей."""
    Z = (R - R.mean()) / R.std()
    vals, vecs = np.linalg.eigh(np.cov(Z.to_numpy().T))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    explained = vals / vals.sum()
    cols = [f"Фактор {i + 1}" for i in range(k)]
    loadings = pd.DataFrame(vecs[:, :k], index=R.columns, columns=cols)
    factors = pd.DataFrame(Z.to_numpy() @ vecs[:, :k], index=R.index, columns=cols)
    return explained, loadings, factors


def factor_decomposition(R, weights, k=3):
    """Какая доля дисперсии портфеля объясняется общими факторами, а какая специфична для бумаг."""
    explained, loadings, factors = pca_factors(R, k)
    r = portfolio_returns(R, weights)
    X = np.column_stack([np.ones(len(factors)), factors.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, r.to_numpy(), rcond=None)
    betas = coef[1:]
    var_total = float(r.var())
    contrib = betas ** 2 * factors.var().to_numpy()  # факторы ортогональны
    r2 = float(contrib.sum() / var_total) if var_total > 0 else 0.0
    return {"explained": explained, "loadings": loadings,
            "factor_share": pd.Series(contrib / var_total if var_total > 0 else contrib * 0, index=factors.columns),
            "systematic": min(r2, 1.0), "idiosyncratic": max(1.0 - r2, 0.0)}


def active_stats(r, bench):
    """Показатели относительно бенчмарка: tracking error, IR, альфа/бета, захват роста/падения."""
    active = r - bench
    te = float(active.std() * np.sqrt(TRADING_DAYS))
    b = beta(r, bench)
    up, down = bench > 0, bench < 0
    return {
        "Активная доходность (год)": float(active.mean() * TRADING_DAYS),
        "Tracking error": te,
        "Information ratio": float(active.mean() * TRADING_DAYS / te) if te > 0 else np.nan,
        "Бета": b,
        "Альфа (год)": float((r.mean() - b * bench.mean()) * TRADING_DAYS),
        "Захват роста": float(r[up].mean() / bench[up].mean()) if up.any() else np.nan,
        "Захват падения": float(r[down].mean() / bench[down].mean()) if down.any() else np.nan,
        "Корреляция": float(np.corrcoef(r, bench)[0, 1]),
    }


def drawdown_series(r):
    equity = (1 + r).cumprod()
    return equity / equity.cummax() - 1


def rolling_stats(r, bench, window=60):
    return pd.DataFrame({
        "vol": r.rolling(window).std() * np.sqrt(TRADING_DAYS),
        "beta": r.rolling(window).cov(bench) / bench.rolling(window).var(),
    })


def concentration(weights):
    """HHI и эффективное число бумаг по инвестированной части портфеля."""
    w = pd.Series(weights)
    w = w[w > 0]
    if w.empty:
        return 0.0, 0.0
    w = w / w.sum()
    hhi = float((w ** 2).sum())
    return hhi, 1.0 / hhi


def sector_exposure(weights, sector_of):
    s = pd.Series(weights)
    s = s[s > 0]
    return s.groupby([sector_of(t) for t in s.index]).sum().sort_values(ascending=False)


def what_if(weights, shocks):
    """Мгновенное изменение портфеля (доля капитала) при заданных шоках цен бумаг."""
    return float(sum(w * shocks.get(t, 0.0) for t, w in weights.items()))
