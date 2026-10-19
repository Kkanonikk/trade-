"""Структура рынка: графы, кластеры, случайные матрицы, топология, геометрия ковариаций, режимы, lead-lag.

Все методы диагностические: показывают, как устроен и меняется рынок. Прибыльных сигналов сами по себе не дают.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.linalg import eigh
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------- корреляции, графы, кластеры
def corr_distance(C):
    """Метрика Мантеньи: d = sqrt(2 (1 - rho))."""
    D = np.sqrt(np.clip(2.0 * (1.0 - np.asarray(C)), 0.0, None))
    np.fill_diagonal(D, 0.0)
    return D


def mst_edges(C, names):
    """Минимальное остовное дерево по корреляционной метрике: DataFrame u, v, dist (n-1 рёбер)."""
    D = corr_distance(C)
    T = minimum_spanning_tree(D + np.where(np.eye(len(D)) == 1, 0, 1e-12)).tocoo()
    return pd.DataFrame({"u": [names[i] for i in T.row], "v": [names[j] for j in T.col], "dist": T.data})


def mst_length(C):
    """Суммарная длина MST = суммарная персистентность H0 для точек-активов. Падает в кризисы (рынок «сжимается»)."""
    return float(mst_edges(C, list(range(len(C)))).dist.sum())


def cluster_order(C, method="average"):
    """Порядок листьев дендрограммы и linkage (для упорядоченной тепловой карты и кластеров)."""
    Z = linkage(squareform(corr_distance(C), checks=False), method=method)
    return Z, leaves_list(Z)


def clusters(C, k, method="average"):
    Z, _ = cluster_order(C, method)
    return fcluster(Z, k, criterion="maxclust")


def hrp_weights(R):
    """Hierarchical Risk Parity (López de Prado): веса без обращения матрицы ковариации."""
    cov = R.cov().to_numpy()
    corr = R.corr().to_numpy()
    Z, order = cluster_order(corr, "single")
    order = list(order)

    def cluster_var(idx):
        sub = cov[np.ix_(idx, idx)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    w = pd.Series(1.0, index=order)
    groups = [order]
    while groups:
        groups = [g[i:j] for g in groups for i, j in ((0, len(g) // 2), (len(g) // 2, len(g))) if len(g) > 1]
        for k in range(0, len(groups), 2):
            left, right = groups[k], groups[k + 1]
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1.0 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    w = w / w.sum()
    return pd.Series(w.sort_index().to_numpy(), index=R.columns)


# ---------------------------------------------------------------- теория случайных матриц
def rmt(R):
    """Сравнение собственных чисел корреляций с границей Марченко-Пастура и очистка шума.

    Возвращает dict: eigenvalues, lambda_plus, n_signal, denoised (очищенная корреляционная матрица).
    """
    T, N = R.shape
    q = N / T
    lam_plus = (1.0 + np.sqrt(q)) ** 2
    C = np.corrcoef(R.to_numpy().T)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    signal = vals > lam_plus
    cleaned = vals.copy()
    if (~signal).any():
        cleaned[~signal] = vals[~signal].mean()  # шумовые значения заменяются средним: след сохраняется
    Cd = (vecs * cleaned) @ vecs.T
    d = np.sqrt(np.diag(Cd))
    Cd = Cd / np.outer(d, d)
    return {"eigenvalues": vals, "lambda_plus": float(lam_plus), "n_signal": int(signal.sum()),
            "denoised": pd.DataFrame(Cd, index=R.columns, columns=R.columns), "q": float(q)}


def absorption_ratio(R, k=3):
    """Доля дисперсии рынка в первых k факторах (Kritzman): рост = рынок «слипается», хрупкость выше."""
    vals = np.linalg.eigvalsh(np.corrcoef(R.to_numpy().T))[::-1]
    return float(vals[:k].sum() / vals.sum())


# ---------------------------------------------------------------- топология (TDA)
def h1_total_persistence(points):
    """Суммарная персистентность H1 (петель) облака точек. None, если ripser не установлен."""
    try:
        from ripser import ripser
    except ImportError:
        return None
    dgm = ripser(np.asarray(points), maxdim=1)["dgms"][1]
    return float((dgm[:, 1] - dgm[:, 0]).sum()) if len(dgm) else 0.0


def _window_features(win):
    """Признаки одного окна: длина MST активов, H1 облака дней (Gidea-Katz), поглощение, средняя корреляция."""
    C = np.corrcoef(win.to_numpy().T)
    Z = ((win - win.mean()) / win.std()).to_numpy()
    n = C.shape[0]
    return {"mst_length": mst_length(C) / (n - 1), "h1": h1_total_persistence(Z),
            "absorption": absorption_ratio(win), "mean_corr": float((C.sum() - n) / (n * (n - 1)))}


def rolling_topology(R, window=60, step=5):
    rows = []
    for end in range(window, len(R) + 1, step):
        win = R.iloc[end - window:end]
        rows.append({"ts": R.index[end - 1], **_window_features(win)})
    return pd.DataFrame(rows).set_index("ts")


# ---------------------------------------------------------------- геометрия ковариаций (SPD-многообразие)
def spd_distance(A, B):
    """Аффинно-инвариантная риманова метрика: sqrt(sum(log^2 собственных чисел A^-1 B))."""
    w = eigh(B, A, eigvals_only=True)
    return float(np.sqrt(np.sum(np.log(w) ** 2)))


def rolling_cov_shift(R, window=60, step=20, ridge=1e-8):
    """Расстояние между ковариациями соседних окон: большие скачки = смена «состояния» рынка."""
    rows = []
    for end in range(2 * window, len(R) + 1, step):
        a = R.iloc[end - 2 * window:end - window].cov().to_numpy() + ridge * np.eye(R.shape[1])
        b = R.iloc[end - window:end].cov().to_numpy() + ridge * np.eye(R.shape[1])
        rows.append({"ts": R.index[end - 1], "shift": spd_distance(a, b)})
    return pd.DataFrame(rows).set_index("ts")


# ---------------------------------------------------------------- режимы, lead-lag, хвосты
def regimes(R, k=3, window=20, seed=0):
    """Режимы рынка: смесь гауссиан по (доходность, волатильность) равновзвешенного индекса.
    Состояния упорядочены по волатильности (0 = спокойный). Это кластеризация, а не HMM."""
    from sklearn.mixture import GaussianMixture
    idx = R.mean(axis=1)
    feats = pd.DataFrame({"ret": idx.rolling(window).mean(), "vol": idx.rolling(window).std()}).dropna()
    X = (feats - feats.mean()) / feats.std()
    gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed).fit(X)
    raw = gm.predict(X)
    order = np.argsort([feats["vol"][raw == s].mean() for s in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    state = pd.Series([remap[s] for s in raw], index=feats.index)
    return state, feats


def lead_lag(R):
    """L[i, j] = corr(r_i[t-1], r_j[t]): влияние «вчера i» на «сегодня j». Диагональ = 0."""
    X = R.to_numpy()
    n = X.shape[1]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i, j] = np.corrcoef(X[:-1, i], X[1:, j])[0, 1]
    return pd.DataFrame(L, index=R.columns, columns=R.columns)


def tail_dependence(R, q=0.05, lower=True):
    """Эмпирическая хвостовая зависимость: P(j в худших q | i в худших q). Верхняя при lower=False."""
    U = R.rank(pct=True).to_numpy()
    n = U.shape[1]
    out = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                a, b = (U[:, i] <= q, U[:, j] <= q) if lower else (U[:, i] >= 1 - q, U[:, j] >= 1 - q)
                out[i, j] = (a & b).sum() / max(a.sum(), 1)
    return pd.DataFrame(out, index=R.columns, columns=R.columns)
