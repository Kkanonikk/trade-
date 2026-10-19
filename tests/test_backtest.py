"""Тесты на синтетических данных (случайное блуждание) только для проверки механики кода."""
import numpy as np
import pandas as pd
import pytest

from tradelab import backtest, features, models


def synthetic(n=900, seed=0, drift=0.0, trend=0.0):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = drift + trend * r[i - 1] + rng.normal(0, 0.01)
    close = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC"),
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.integers(1000, 2000, n).astype(float),
    })


def test_target_is_next_bar_return():
    df = synthetic()
    f = features.build(df)
    expected = np.log(df["close"].iloc[11] / df["close"].iloc[10])
    assert f["target"].iloc[10] == pytest.approx(expected)


def test_no_lookahead_future_change_does_not_affect_past_predictions():
    df = synthetic()
    changed = df.copy()
    changed.loc[changed.index[-1], "close"] *= 3  # меняем только самую последнюю цену
    a = backtest.walk_forward(df, "Ridge-регрессия")
    b = backtest.walk_forward(changed, "Ridge-регрессия")
    common = a.index[:-3]
    np.testing.assert_allclose(a.loc[common, "pred"], b.loc[common, "pred"])


@pytest.mark.parametrize("name", list(models.MODELS))
def test_all_models_run(name):
    res = backtest.walk_forward(synthetic(), name)
    m = backtest.metrics(res)
    assert m["Баров вне выборки"] > 100
    assert np.isfinite(m["IC (Спирмен)"])


def test_costs_reduce_return():
    df = synthetic()
    free = backtest.metrics(backtest.walk_forward(df, "Моментум (20)", cost_bps=0))
    paid = backtest.metrics(backtest.walk_forward(df, "Моментум (20)", cost_bps=50))
    assert paid["Доходность стратегии"] < free["Доходность стратегии"]


def test_momentum_finds_planted_autocorrelation():
    df = synthetic(n=1500, trend=0.0, seed=1)
    # Явная внедрённая зависимость: доходности автокоррелированы
    trend = backtest.metrics(backtest.walk_forward(synthetic(n=1500, trend=0.3, seed=2), "Ridge-регрессия"))
    assert trend["IC (Спирмен)"] > 0.05


def test_latest_forecast():
    value, ts = backtest.latest_forecast(synthetic(), "Ridge-регрессия")
    assert np.isfinite(value)


def test_parse_bars():
    from tradelab.bcs import parse_bars
    df = parse_bars({"bars": [{"time": "2025-01-02T07:00:00Z", "open": 1, "high": 2,
                               "low": 0.5, "close": 1.5, "volume": 10}]})
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == 1.5
    assert parse_bars({"bars": []}).empty
