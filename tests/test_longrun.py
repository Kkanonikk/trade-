import warnings

import numpy as np
import pandas as pd
import pytest

from tradelab import longrun, panel

warnings.filterwarnings("ignore")


def test_par_bond_pricing():
    assert longrun._par_bond_price(0.05, 0.05) == pytest.approx(1.0)
    assert longrun._par_bond_price(0.06, 0.05) < 1.0 < longrun._par_bond_price(0.04, 0.05)


def test_forward_returns_cagr():
    idx = pd.date_range("2000-01-31", periods=48, freq="ME")
    index = pd.Series(1.07 ** (np.arange(48) / 12), index=idx)
    f = longrun.forward_returns(index, 2)
    assert f.iloc[0] == pytest.approx(0.07, abs=1e-9) and f.iloc[-1] != f.iloc[-1]   # хвост: NaN


def synthetic_xy(n=900, slope=-0.8, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1900-01-31", periods=n, freq="ME")
    x = pd.Series(np.cumsum(rng.normal(0, 0.08, n)) * 0.2 + rng.normal(0, 0.3, n), index=idx)
    fwd = 0.06 + slope * x.rolling(1).mean() * 0.1 + rng.normal(0, noise, n)
    return x, pd.Series(fwd, index=idx)


def test_walk_forward_finds_planted_signal_and_not_noise():
    x, fwd = synthetic_xy()
    _, m = longrun.walk_forward(x, fwd, 5, start="1930-01-01")
    assert m["oos_r2"] > 0.2 and m["corr"] > 0.5
    rng = np.random.default_rng(1)
    _, m0 = longrun.walk_forward(x, pd.Series(0.06 + rng.normal(0, 0.03, len(x)), index=x.index), 5, start="1930-01-01")
    assert m0["oos_r2"] < 0.05


def test_walk_forward_never_trains_on_unknown_outcomes():
    x, fwd = synthetic_xy()
    years = 5
    a, _ = longrun.walk_forward(x, fwd, years, start="1930-01-01")
    cutoff = x.index[600]
    poisoned = fwd.copy()
    poisoned[poisoned.index > cutoff] = 100.0                   # исходы после cutoff портим
    b, _ = longrun.walk_forward(x, poisoned, years, start="1930-01-01")
    safe = a.index[a.index <= cutoff + pd.DateOffset(months=years * 12)]   # до cutoff+h модель ничего о порче знать не может
    np.testing.assert_allclose(a.loc[safe, "pred"], b.loc[safe, "pred"])


@pytest.fixture(scope="module")
def model10():
    return longrun.EquityBondModel(10)


def test_real_data_models_have_expected_skill(model10):
    m = model10.metrics
    assert m["bond"]["oos_r2"] > 0.6 and m["bond"]["corr"] > 0.8            # ставка почти определяет номинальную доходность
    assert m["equity"]["corr"] > 0.2 and 0 <= m["equity"]["shrink_weight"] <= 1
    assert m["equity"]["oos_sigma"] > 0.02


def test_estimate_is_ordered_and_sensible(model10):
    e = model10.estimate()
    for k in ("equity", "bond"):
        assert e[k]["p10"] < e[k]["mean"] < e[k]["p90"]
    assert 15 < e["cape"] < 70
    cheap = model10.estimate(cape=12.0)["equity"]["mean"]
    dear = model10.estimate(cape=40.0)["equity"]["mean"]
    assert cheap > dear                                                   # дешёвый рынок -> выше ожидаемая доходность


def test_annual_history_shape(model10):
    h = model10.annual_history()
    assert len(h) > 100 and h["equity"].std() > h["bond"].std() and h["equity"].mean() > 0.04


def test_portfolio_simulation_deterministic_case():
    hist = pd.DataFrame({"equity": np.zeros(20), "bond": np.zeros(20)})
    w, paths = longrun.simulate_portfolio(hist, {"equity": 0.05, "bond": 0.0}, {"equity": 1.0}, years=10, n=50,
                                          start=100.0, contribution=0.0, cash_vol=0.0, gold_vol=0.0)
    np.testing.assert_allclose(w[:, -1], 100 * 1.05 ** 10)
    s = longrun.summarize_wealth(w, 100.0, 0.0, target=150.0)
    assert s["prob_real_loss"] == 0.0 and s["prob_target"] == 1.0


def test_portfolio_contributions_and_scenarios(model10):
    hist = model10.annual_history()
    mu = {"equity": 0.04, "bond": 0.02}
    base, _ = longrun.simulate_portfolio(hist, mu, {"equity": 0.6, "bond": 0.4}, years=10, n=2000, contribution=0.0)
    more, _ = longrun.simulate_portfolio(hist, mu, {"equity": 0.6, "bond": 0.4}, years=10, n=2000, contribution=10_000.0)
    bad, _ = longrun.simulate_portfolio(hist, mu, {"equity": 0.6, "bond": 0.4}, years=10, n=2000,
                                        shift=longrun.SCENARIOS["Стагфляция (реальные ставки вниз)"])
    assert np.median(more[:, -1]) > np.median(base[:, -1]) > np.median(bad[:, -1])
    assert (base > 0).all()
    risky, _ = longrun.simulate_portfolio(hist, mu, {"equity": 1.0}, years=10, n=2000)
    assert np.std(risky[:, -1]) > np.std(base[:, -1])


def test_growth_backtest_beats_naive_on_real_data():
    p = panel.snapshot_panel()
    met, r = longrun.growth_backtest(p, 5, first_origin=1995)
    assert met["obs"] > 1500 and met["skill_vs_naive"] > 0.05
    assert np.isfinite(met["corr"]) and met["corr"] > 0.15
    out, year = longrun.growth_forecast(p, 5)
    assert (out["p10"] < out["growth"]).all() and (out["growth"] < out["p90"]).all() and 2005 <= year <= 2013


def test_debt_dynamics_identities():
    d0, g, pi, r = 80.0, 2.0, 2.0, 4.0
    pb_star = longrun.stabilizing_balance(d0, g, pi, r)
    paths = longrun.debt_paths(d0, g, pi, r, pb_star, years=10, n=10, g_sd=0, r_sd=0, pb_sd=0, infl_sd=0)
    np.testing.assert_allclose(paths, d0, atol=1e-9)                       # на стабилизирующем балансе долг не меняется
    worse = longrun.debt_paths(d0, g, pi, r, pb_star - 2, years=10, n=10, g_sd=0, r_sd=0, pb_sd=0, infl_sd=0)
    assert (worse[:, -1] > d0).all()
    noisy = longrun.debt_paths(d0, g, pi, r, pb_star, years=10, n=3000)
    assert noisy[:, -1].std() > 5
