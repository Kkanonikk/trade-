"""Тесты глобального режима: данные, реестр источников, прогнозный движок, старение, индекс голода, горизонт кризисов."""
import warnings

import numpy as np
import pandas as pd
import pytest

from tradelab import aging, crisis, forecast, gdata, globe, hunger, panel, registry

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def w():
    return gdata.load()


# ---------------------------------------------------------------- данные и источники
def test_gdata_loads_real_indicators(w):
    assert w.index.names == ["iso3", "year"] and w.shape[1] > 100
    assert w.index.get_level_values("iso3").nunique() > 180
    assert 20 < gdata.cross_section(w, aging.MEDIAN_AGE, 2020)["JPN"] < 55


def test_every_indicator_has_a_registered_source(w):
    meta = gdata.meta()
    for k in w.columns:
        assert meta[k]["source"] in registry.SOURCES, k
    for d in registry.SOURCES.values():
        assert d["url"].startswith("http") and d["license"] and d["org"]
    assert "gapminder" in registry.cite(["gm:median_age_years"]).lower() or "ООН" in registry.cite(["gm:median_age_years"])


def test_projection_flag_for_un_series(w):
    assert gdata.meta()[aging.MEDIAN_AGE]["projection_from"] == 2024
    assert gdata.last_actual_year(aging.MEDIAN_AGE) == 2023


# ---------------------------------------------------------------- прогнозный движок
def test_forecast_beats_naive_on_trend_and_reports_honest_skill():
    rng = np.random.default_rng(0)
    ser = {f"C{i}": pd.Series(10 + 0.5 * i + 0.3 * np.arange(40) + rng.normal(0, 0.2, 40), index=range(1980, 2020)) for i in range(25)}
    bt = forecast.run_backtest(ser, horizon=5)
    assert bt.best != "naive" and bt.mase[bt.best][-1] < bt.mase["naive"][-1]
    fc = forecast.forecast_series(ser["C3"], bt, 5)
    assert len(fc) == 5 and (fc["p10"] < fc["mean"]).all() and (fc["mean"] < fc["p90"]).all()
    assert fc["mean"].iloc[-1] > ser["C3"].iloc[-1]


def test_forecast_random_walk_has_no_skill():
    rng = np.random.default_rng(1)
    ser = {f"C{i}": pd.Series(np.cumsum(rng.normal(0, 1, 40)), index=range(1980, 2020)) for i in range(40)}
    bt = forecast.run_backtest(ser, horizon=3)
    sk = forecast.skill_table(bt)
    assert abs(float(sk["Навык 3 г."].max())) < 0.15


def test_forecast_handles_short_and_flat_series():
    assert forecast.run_backtest({"A": pd.Series([1, 2, 3.0], index=[2000, 2001, 2002])}, horizon=3).mase == {}
    flat = {f"F{i}": pd.Series(np.full(30, 5.0), index=range(1990, 2020)) for i in range(5)}
    bt = forecast.run_backtest(flat, horizon=3)
    assert all(float(v.max()) < 1e-6 for v in bt.mase.values())      # плоский ряд не взрывает метрику


# ---------------------------------------------------------------- старение
def test_population_path_is_calibrated_to_un_anchors(w):
    path = aging.population_path(w)
    err = aging.path_consistency(w, path)
    assert err[2050] < 0.01 and err[2100] < 0.01
    assert path.loc["CHN", 2025] > 1.3e9 and path.loc["IND", 2100] > 1.0e9


def test_population_peaks_and_ageing_speed(w):
    peaks = aging.population_peaks(w)
    assert peaks.loc["CHN", "peak_year"] <= 2030 and peaks.loc["CHN", "change_2100"] < 0
    assert peaks.loc["NGA", "change_2100"] > 0.5
    old, _ = aging.ranking(w, 2025)
    assert old.index[0] in ("JPN", "ITA", "DEU", "HKG", "KOR", "MCO", "PRT", "BGR", "FIN", "GRC", "HRV", "SVN", "ESP")


def test_fixed_effects_estimator_recovers_known_effect():
    rng = np.random.default_rng(2)
    rows = []
    for c in range(60):
        fe = rng.normal(0, 3)
        for p in range(1980, 2010, 5):
            d = rng.normal(2, 1)
            rows.append({"iso3": f"C{c}", "period": p, "y": fe + 0.8 * d + rng.normal(0, 0.3), "d_driver": d, "log_income0": rng.normal()})
    res, info = aging._fe_ols(pd.DataFrame(rows), "y", ["d_driver", "log_income0"])
    assert abs(res.loc["d_driver", "coef"] - 0.8) < 0.15 and abs(res.loc["d_driver", "t"]) > 4
    assert info["countries"] == 60


def test_economic_effect_runs_on_real_data(w):
    fe = aging.economic_effect(w, country_fe=True)
    assert len(fe) >= 2 and fe["se"].gt(0).all()
    assert not aging.scenario_effect(w, fe, outcome="growth").empty


# ---------------------------------------------------------------- голод
def test_hunger_index_shape_and_ranking(w):
    idx, contrib = hunger.index(w, 2020)
    assert idx.between(0, 100).all() and len(idx) > 80
    assert idx.rank(ascending=False)["USA"] > idx.rank(ascending=False)["NER"] + 50     # богатая страна безопаснее бедной
    assert np.allclose(contrib.sum(axis=1), idx, atol=1e-6)


def test_hunger_validation_has_skill_but_below_persistence(w):
    v = hunger.validate(w)
    assert len(v) >= 3 and v["rho_index"].mean() > 0.4
    assert v["rho_index"].mean() < v["rho_persistence"].mean()


# ---------------------------------------------------------------- кризисы с горизонтом, карта
def test_multi_year_horizon_labels_cover_single_year():
    snap = panel.snapshot_panel()
    X1, y1, _, _ = crisis.dataset(snap, "currency", 1)
    X3, y3, _, _ = crisis.dataset(snap, "currency", 3)
    assert y3.mean() >= y1.mean()


def test_scale_color_directions():
    assert globe.scale_color(None, 0, 1, "up") is None
    assert globe.scale_color(0.0, 0, 1, "down") != globe.scale_color(1.0, 0, 1, "down")
    assert globe.scale_color(0.0, 0, 1, "up") != globe.scale_color(1.0, 0, 1, "up")
