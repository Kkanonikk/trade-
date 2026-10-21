import numpy as np
import pandas as pd
import pytest

from tradelab import autotrade, demo, optimize, paper, risk


@pytest.fixture(scope="module")
def R():
    frames = demo.universe(["A", "B", "C", "D"], n=600)
    return risk.returns_frame(frames)


def test_var_cvar_ordering(R):
    r = risk.portfolio_returns(R, {c: 0.25 for c in R.columns})
    var95, cvar95 = risk.var_cvar(r, 0.95)
    var99, _ = risk.var_cvar(r, 0.99)
    assert 0 < var95 <= cvar95 and var99 >= var95


def test_risk_contributions_sum_to_one(R):
    rc = risk.risk_contributions(R, {"A": 0.5, "B": 0.3, "C": 0.2})
    assert rc.sum() == pytest.approx(1.0)
    assert rc["D"] == 0


@pytest.mark.parametrize("obj", ["min_vol", "max_sharpe", "risk_parity", "equal"])
def test_optimizer_constraints(R, obj):
    w = optimize.optimize(R, obj, max_weight=0.4)
    vals = np.array(list(w.values()))
    assert vals.sum() == pytest.approx(1.0)
    assert vals.min() >= -1e-9 and vals.max() <= 0.4 + 1e-6


def test_min_vol_not_worse_than_equal(R):
    _, v_min = optimize.portfolio_point(R, optimize.optimize(R, "min_vol", max_weight=1.0))
    _, v_eq = optimize.portfolio_point(R, optimize.optimize(R, "equal"))
    assert v_min <= v_eq + 1e-9


def test_paper_buy_sell_and_fees(tmp_path):
    b = paper.PaperBroker(tmp_path / "p.db", start_cash=100_000, fee_bps=10)
    b.order("SBER", "BUY", 100, 200.0)
    assert b.cash == pytest.approx(100_000 - 20_000 - 20)
    b.order("SBER", "SELL", 40, 210.0)
    assert b.positions()["SBER"][0] == 60
    assert b.equity({"SBER": 210.0}) == pytest.approx(b.cash + 60 * 210.0)


def test_paper_forbids_leverage_and_short(tmp_path):
    b = paper.PaperBroker(tmp_path / "p.db", start_cash=1_000)
    with pytest.raises(ValueError):
        b.order("X", "BUY", 100, 100.0)
    with pytest.raises(ValueError):
        b.order("X", "SELL", 1, 100.0)


def test_kill_switch_blocks_orders(tmp_path):
    b = paper.PaperBroker(tmp_path / "p.db")
    b.set_kill_switch(True)
    with pytest.raises(ValueError):
        b.order("X", "BUY", 1, 10.0)


def test_rebalance_respects_max_weight(tmp_path):
    b = paper.PaperBroker(tmp_path / "p.db", start_cash=1_000_000, fee_bps=0)
    prices = {"A": 100.0, "B": 50.0}
    b.rebalance({"A": 0.9, "B": 0.9}, prices, max_weight=0.3)
    eq = b.equity(prices)
    for t, (q, _) in b.positions().items():
        assert q * prices[t] / eq <= 0.3 + 1e-6
    b.rebalance({}, prices)  # всё в кэш
    assert b.positions() == {}


def test_target_weights():
    sig = pd.DataFrame({"ticker": list("ABCD"), "forecast": [0.01, -0.01, 0.02, 0.005],
                        "ic": [0.1, 0.2, 0.05, -0.1], "hit": 0.5, "sharpe": 0.0, "price": 1.0})
    w = autotrade.target_weights(sig, top_n=5, min_ic=0.0, max_weight=0.25)
    assert set(w) == {"A", "C"} and all(v == 0.25 for v in w.values())


# ---------------------------------------------------------------- расширенная аналитика
from tradelab import compliance, config, strategies  # noqa: E402


def test_component_var_sums_to_var(R):
    w = {c: 1 / len(R.columns) for c in R.columns}
    var, comp = risk.parametric_var(R, w, 0.95)
    assert comp.sum() == pytest.approx(var) and var > 0


def test_cov_estimators_are_psd(R):
    for cov in (risk.shrunk_cov(R), risk.ewma_cov(R)):
        assert np.linalg.eigvalsh(cov.to_numpy()).min() > -1e-12


def test_factor_decomposition_shares(R):
    d = risk.factor_decomposition(R, {c: 0.25 for c in R.columns}, k=2)
    assert d["systematic"] + d["idiosyncratic"] == pytest.approx(1.0)
    assert d["explained"].sum() == pytest.approx(1.0)
    assert d["systematic"] > 0.3  # в демо один общий фактор: он должен быть заметен


def test_active_stats_identity():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    a = risk.active_stats(r, r)
    assert a["Tracking error"] == pytest.approx(0) and a["Бета"] == pytest.approx(1.0)


def test_drawdown_and_concentration():
    r = pd.Series([0.1, -0.2, 0.05])
    assert risk.drawdown_series(r).min() == pytest.approx(-0.2)
    hhi, n_eff = risk.concentration({"A": 0.5, "B": 0.5})
    assert hhi == pytest.approx(0.5) and n_eff == pytest.approx(2.0)


def test_what_if_and_sector():
    assert risk.what_if({"A": 0.5, "B": 0.5}, {"A": -0.1}) == pytest.approx(-0.05)
    s = risk.sector_exposure({"JPM": 0.2, "V": 0.2, "XOM": 0.1}, config.sector)
    assert s["Финансы"] == pytest.approx(0.4)


def test_black_litterman_moves_toward_view(R):
    prior = optimize.black_litterman(R, {}, {})
    post = optimize.black_litterman(R, {"A": 0.5}, {"A": 0.9})
    assert post["A"] > prior["A"]
    weak = optimize.black_litterman(R, {"A": 0.5}, {"A": 0.05})
    assert weak["A"] < post["A"]


def test_mean_variance_uses_custom_mu(R):
    mu = np.array([0.0, 0.0, 0.0, 1.0])
    w = optimize.optimize(R, "mean_variance", max_weight=1.0, mu=mu, risk_aversion=1.0)
    assert w[R.columns[3]] > 0.5


def test_compliance_flags_breaches():
    lim = compliance.Limits(max_position=0.2, max_sector=0.3, min_cash=0.1, max_var95=0.02, max_turnover=0.5)
    b = compliance.check({"SBER": 0.25, "VTBR": 0.2, "GAZP": 0.5}, lim, var95=0.03, current={})
    rules = {x["Правило"] for x in b}
    assert {"Доля бумаги", "Доля сектора", "Мин. деньги", "VaR 95% (1д)", "Оборот"} <= rules
    assert compliance.check({"SBER": 0.1}, lim, var95=0.01, current={}) == []


@pytest.mark.parametrize("method", list(strategies.METHODS))
def test_strategy_backtest_runs(R, method):
    out = strategies.run(R, method, rebalance_days=40, lookback=100, top_n=2, max_weight=0.6)
    assert len(out) == len(R) - 100 and np.isfinite(out["equity"].iloc[-1])
    assert (out["turnover"] >= 0).all()


def test_strategy_costs_reduce_return(R):
    free = strategies.run(R, "momentum", 20, 100, 2, cost_bps=0)["equity"].iloc[-1]
    paid = strategies.run(R, "momentum", 20, 100, 2, cost_bps=100)["equity"].iloc[-1]
    assert paid < free
