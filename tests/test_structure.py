"""Проверка аналитики на данных с заранее известной структурой."""
import numpy as np
import pandas as pd
import pytest

from tradelab import structure as st


def noise(n=600, k=12, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (n, k)), columns=[f"A{i}" for i in range(k)],
                        index=pd.bdate_range("2020-01-01", periods=n))


def factor_data(n=600, k=12, seed=1, load=0.8):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, (n, 1))
    X = load * f + (1 - load) * rng.normal(0, 0.01, (n, k))
    return pd.DataFrame(X, columns=[f"A{i}" for i in range(k)], index=pd.bdate_range("2020-01-01", periods=n))


def test_mst_has_n_minus_1_edges_and_is_connected():
    R = factor_data()
    e = st.mst_edges(R.corr().to_numpy(), list(R.columns))
    assert len(e) == R.shape[1] - 1
    assert set(e.u) | set(e.v) == set(R.columns)


def test_mst_length_shrinks_when_market_tightens():
    loose = st.mst_length(noise().corr().to_numpy())
    tight = st.mst_length(factor_data(load=0.9).corr().to_numpy())
    assert tight < loose


def test_rmt_noise_has_few_signals_and_factor_has_one():
    assert st.rmt(noise(n=2000))["n_signal"] <= 1
    assert st.rmt(factor_data())["n_signal"] >= 1


def test_rmt_denoised_is_valid_correlation():
    d = st.rmt(factor_data())["denoised"].to_numpy()
    assert np.allclose(np.diag(d), 1.0) and np.allclose(d, d.T)
    assert np.linalg.eigvalsh(d).min() > -1e-9


def test_absorption_ratio_higher_with_common_factor():
    assert st.absorption_ratio(factor_data(load=0.9)) > st.absorption_ratio(noise())


def test_h1_circle_beats_blob():
    pytest.importorskip("ripser")
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 80)
    circle = np.c_[np.cos(th), np.sin(th)] + rng.normal(0, 0.03, (80, 2))
    blob = rng.normal(0, 0.5, (80, 2))
    assert st.h1_total_persistence(circle) > 3 * st.h1_total_persistence(blob)


def test_spd_distance_properties():
    rng = np.random.default_rng(0)
    A = np.cov(rng.normal(size=(200, 4)).T)
    B = np.cov((rng.normal(size=(200, 4)) * np.array([1, 2, 1, 3])).T)
    assert st.spd_distance(A, A) == pytest.approx(0, abs=1e-9)
    assert st.spd_distance(A, B) == pytest.approx(st.spd_distance(B, A))
    M = rng.normal(size=(4, 4)) + 3 * np.eye(4)  # аффинная инвариантность
    assert st.spd_distance(M @ A @ M.T, M @ B @ M.T) == pytest.approx(st.spd_distance(A, B), rel=1e-6)


def test_cov_shift_detects_regime_change():
    calm = noise(n=300, seed=3)
    wild = noise(n=300, seed=4) * 4
    R = pd.concat([calm, wild]).reset_index(drop=True)
    R.index = pd.bdate_range("2020-01-01", periods=len(R))
    s = st.rolling_cov_shift(R, window=60, step=20)["shift"]
    assert s.idxmax() > R.index[250] and s.max() > 3 * s.iloc[:3].mean()


def test_regimes_find_planted_vol_states():
    rng = np.random.default_rng(0)
    vol = np.r_[np.full(400, 0.005), np.full(400, 0.03), np.full(400, 0.005)]
    R = pd.DataFrame(rng.normal(0, 1, (1200, 5)) * vol[:, None], index=pd.bdate_range("2018-01-01", periods=1200))
    state, _ = st.regimes(R, k=2)
    mid = state.loc[R.index[500]:R.index[750]]
    edge = state.loc[R.index[50]:R.index[300]]
    assert mid.mean() > 0.9 and edge.mean() < 0.1


def test_lead_lag_detects_planted_lag():
    rng = np.random.default_rng(0)
    a = rng.normal(size=1000)
    b = np.r_[0, a[:-1]] * 0.8 + rng.normal(size=1000) * 0.3  # b следует за a на 1 шаг
    R = pd.DataFrame({"A": a, "B": b})
    L = st.lead_lag(R)
    assert L.loc["A", "B"] > 0.6 and abs(L.loc["B", "A"]) < 0.15


def test_tail_dependence_higher_for_joint_crashes():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    y_ind = rng.normal(size=n)
    crash = rng.random(n) < 0.03
    y_dep = np.where(crash, x - 3, rng.normal(size=n))
    x2 = np.where(crash, -4 + rng.normal(size=n) * 0.2, x)
    dep = st.tail_dependence(pd.DataFrame({"x": x2, "y": y_dep}))
    ind = st.tail_dependence(pd.DataFrame({"x": x, "y": y_ind}))
    assert dep.loc["x", "y"] > ind.loc["x", "y"] + 0.15
    assert ind.loc["x", "y"] == pytest.approx(0.05, abs=0.04)


def test_hrp_weights_valid_and_downweights_clone_cluster():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, (800, 1))
    X = np.hstack([base + rng.normal(0, 0.001, (800, 1)) for _ in range(4)] + [rng.normal(0, 0.01, (800, 1))])
    R = pd.DataFrame(X, columns=list("ABCDE"))
    w = st.hrp_weights(R)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    assert w["E"] > w["A"]  # независимый актив получает больше, чем один из четырёх клонов


def test_rolling_topology_columns():
    out = st.rolling_topology(factor_data(n=200), window=60, step=20)
    assert {"mst_length", "absorption", "mean_corr", "h1"} <= set(out.columns) and len(out) >= 5
