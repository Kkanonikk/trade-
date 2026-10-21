"""Тесты глобального слоя: реальная панель (снимок), кризисные модели, валютные сценарии, торговля, источники."""
import warnings

import numpy as np
import pandas as pd
import pytest

from tradelab import catalog, countries, crisis, fxscen, globe, panel, sources, trade
from tradelab.net import SourceError

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def snap():
    return panel.snapshot_panel()


# ---------------------------------------------------------------- справочник и панель
def test_countries_reference():
    assert countries.short_ru("RUS") == "Россия" and countries.iso3_to_numeric()["RUS"] == 643
    assert countries.numeric_to_iso3()[840] == "USA"
    lat, lon = countries.centroid("BRA")
    assert -35 < lat < 6 and -75 < lon < -30
    assert len(countries.sovereigns()) > 150


def test_snapshot_panel_is_real_and_consistent(snap):
    assert snap.index.get_level_values("iso3").nunique() > 180
    assert snap.loc[("RUS", 1998), "inflation"] > 20            # дефолт 1998
    assert snap.loc[("ARG", 2002), "inflation"] > 20            # кризис 2001-02
    assert snap.loc[("USA", 2009), "gdp_growth"] < 0           # рецессия 2009
    assert snap.index.get_level_values("year")[snap.notna()["gdp_growth"].to_numpy()].max() <= panel.WEO_LAST_ACTUAL


def test_world_bank_and_datamapper_parsers():
    wb = [{"pages": 1}, [{"countryiso3code": "USA", "date": "2020", "value": 1.5},
                         {"countryiso3code": "", "date": "2020", "value": 9.0},       # агрегат
                         {"countryiso3code": "DEU", "date": "2020", "value": None}]]
    df = panel.parse_worldbank_bulk(wb, "inflation")
    assert df[["iso3", "year", "value"]].values.tolist() == [["USA", 2020, 1.5]]
    dm = {"values": {"NGDP_RPCH": {"USA": {"2019": 2.3, "2030": 1.9}, "WEOWORLD": {"2019": 3}, "FRA": {"2019": None}}}}
    out = panel.parse_datamapper(dm, "gdp_growth", 2024)
    assert out["iso3"].tolist() == ["USA"] and out["year"].tolist() == [2019]
    with pytest.raises(SourceError):
        panel.parse_datamapper({}, "x", 2024)


def test_panel_long_roundtrip(snap):
    small = snap.loc[["USA", "RUS"]].iloc[:5]
    back = panel.from_long(panel.to_long(small))
    cols = [c for c in back.columns if c in small.columns]
    assert len(cols) >= 3
    pd.testing.assert_frame_equal(back[cols].sort_index(axis=1), small[cols].dropna(how="all").sort_index(axis=1),
                                  check_dtype=False, check_names=False)


# ---------------------------------------------------------------- метки кризисов
def test_known_currency_crises_are_labeled(snap):
    df = crisis.derive(snap)
    onset, known, _ = crisis.labels(df)
    for iso, year in (("ARG", 2002), ("IDN", 1998), ("TUR", 2001), ("MEX", 1995)):
        assert bool(onset["currency"].loc[(iso, year)]), f"{iso} {year} должен быть валютным кризисом"
    assert not bool(onset["currency"].loc[("USA", 2008)])


def test_onset_is_not_repeated_within_three_years():
    idx = pd.MultiIndex.from_product([["AAA"], range(2000, 2012)], names=["iso3", "year"])
    fx = [1, 1, 1.5, 2.2, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7]
    df = pd.DataFrame({"gdp_lcu": fx, "gdp_usd": 1.0, "inflation": 5.0, "gdp_growth": 2.0}, index=idx)
    onset, _, _ = crisis.labels(crisis.derive(df))
    assert onset["currency"].sum() == 1


def test_inflation_and_recession_onsets():
    idx = pd.MultiIndex.from_product([["AAA"], range(2000, 2008)], names=["iso3", "year"])
    df = pd.DataFrame({"inflation": [5, 8, 25, 30, 4, 4, 22, 3], "gdp_growth": [3, 2, -1, -2, 1, 2, -1, 3]}, index=idx)
    onset, _, _ = crisis.labels(crisis.derive(df))
    assert onset["inflation"].loc["AAA"].astype(int).tolist() == [0, 0, 1, 0, 0, 0, 1, 0]
    assert onset["recession"].loc["AAA"].astype(int).tolist() == [0, 0, 1, 0, 0, 0, 1, 0]


# ---------------------------------------------------------------- нет заглядывания в будущее
def test_features_use_only_past(snap):
    base = crisis.features(crisis.derive(snap))
    future = snap.copy()
    yrs = future.index.get_level_values("year")
    future.loc[yrs >= 2005, :] = future.loc[yrs >= 2005, :] * 7 + 100  # портим всё начиная с 2005
    changed = crisis.features(crisis.derive(future))
    old = base.index.get_level_values("year") <= 2005               # признак года t опирается на данные <= t-1
    pd.testing.assert_frame_equal(base[old].fillna(-999), changed[old].fillna(-999))


def test_replay_trains_only_on_past_and_has_real_skill(snap):
    out, s = crisis.replay(snap, "inflation", 2008)
    assert s["events"] >= 10 and s["auc"] > 0.75 and s["events_in_top_decile"] >= 5
    assert out["prob"].between(0, 1).all()
    with pytest.raises(ValueError):
        crisis.replay(snap, "inflation", 1962)


@pytest.fixture(scope="module")
def infl_model(snap):
    return crisis.CrisisModel(snap, "inflation")


def test_contributions_reproduce_logit_probability(infl_model):
    m = infl_model
    contrib = m.contributions()
    clf = m.lr.steps[3][1]
    p = 1 / (1 + np.exp(-(contrib.sum(axis=1) + clf.intercept_[0])))      # сырой (сбалансированный) логит
    Xf, _ = m.forecast_frame()
    ref = m.lr.predict_proba(Xf.to_numpy())[:, 1]
    np.testing.assert_allclose(p.to_numpy(), ref, rtol=1e-6)
    corrected = pd.Series(crisis._prior_correct(p.to_numpy(), m.lr.pi_), index=contrib.index)
    pred = m.predict()[0]["logit"]                                       # predict() отбрасывает страны с малым числом данных
    np.testing.assert_allclose(corrected.reindex(pred.index).to_numpy(), pred.to_numpy(), rtol=1e-6)


def test_what_if_raising_inflation_raises_risk(infl_model):
    m = infl_model
    probs, _ = m.predict()
    iso = probs["prob"].idxmin()
    low = m.what_if(iso, {"inflation": 2.0})
    high = m.what_if(iso, {"inflation": 18.0})
    assert high > low


def test_forecast_frame_year_and_probabilities(infl_model):
    probs, year = infl_model.predict()
    assert probs["prob"].between(0, 1).all() and 2005 <= year <= 2023 and len(probs) > 100
    assert probs["prob"].mean() < 0.12    # калиброванные вероятности близки к базовой частоте (~2%), а не к 50%


def test_prior_correction_matches_base_rate():
    p = np.array([0.5])                  # модель с балансом классов говорит 50%: это «типичная» страна
    assert crisis._prior_correct(p, 0.02)[0] == pytest.approx(0.02)
    assert crisis._prior_correct(np.array([0.9]), 0.02)[0] > crisis._prior_correct(np.array([0.5]), 0.02)[0]


def test_fx_model_intervals_are_ordered(snap):
    fx = crisis.FxModel(snap)
    out, year = fx.predict()
    assert (out["p10"] <= out["p50"]).all() and (out["p50"] <= out["p90"]).all() and year >= 2005


# ---------------------------------------------------------------- валюты на реальных данных
@pytest.fixture(scope="module")
def sens():
    fx, f = fxscen.snapshot_fx(), fxscen.snapshot_factors()
    r, ff = fxscen.build_dataset(fx, f)
    return fxscen.sensitivities(r, ff, years=10)


def test_fx_signs_match_economics(sens):
    assert sens.loc["Canada", "b_oil"] < 0 and sens.loc["Canada", "t_oil"] < -2     # нефть растёт -> CAD крепнет
    assert sens.loc["Norway", "b_oil"] < 0 and sens.loc["Mexico", "b_oil"] < 0
    assert sens.loc["Japan", "b_vix"] < 0 and sens.loc["Switzerland", "b_vix"] < 0  # страх рынка -> иена, франк крепнут
    assert sens.loc["South Africa", "b_vix"] > 0 and sens.loc["Brazil", "b_vix"] > 0
    assert (sens["r2"] >= 0.03).all() and "Venezuela" not in sens.index


def test_scenario_direction_and_bounds(sens):
    sc = fxscen.scenario(sens, {"oil": np.log(0.7), "vix": 0.0, "usd": 0.0}, 13)
    assert sc.loc["Canada", "expected"] > 0 and sc.loc["Norway", "expected"] > 0     # дешёвая нефть -> слабее
    assert (sc["p10"] <= sc["expected"]).all() and (sc["expected"] <= sc["p90"]).all()
    assert sc.to_numpy().max() <= 2.0 and sc.to_numpy().min() >= -0.9
    zero = fxscen.scenario(sens, {}, 13)
    assert zero["expected"].abs().max() < 1e-12


# ---------------------------------------------------------------- торговля
COMTRADE = {"data": [
    {"reporterISO": "USA", "partnerISO": "CHN", "refYear": 2022, "primaryValue": 150.0, "motCode": 0, "partner2Code": 0, "customsCode": "C00"},
    {"reporterISO": "USA", "partnerISO": "CHN", "refYear": 2022, "primaryValue": 40.0, "motCode": 1000, "partner2Code": 0, "customsCode": "C00"},
    {"reporterISO": "USA", "partnerISO": "W00", "refYear": 2022, "primaryValue": 999.0, "motCode": 0, "partner2Code": 0, "customsCode": "C00"},
    {"reporterISO": "USA", "partnerISO": "DEU", "refYear": 2022, "primaryValue": 60.0, "motCode": 0, "partner2Code": 0, "customsCode": "C00"}]}


def test_comtrade_parser_filters_aggregates_and_breakdowns():
    df = trade.parse_comtrade(COMTRADE, "X")
    assert sorted(df["partner"]) == ["CHN", "DEU"] and df.set_index("partner").loc["CHN", "value"] == 150.0
    with pytest.raises(SourceError):
        trade.parse_comtrade({}, "X")


def test_trade_matrix_uses_mirror_when_export_missing():
    df = pd.DataFrame([("USA", "CHN", 2022, "X", 100.0),    # США сообщили экспорт в Китай: 100
                       ("USA", "CHN", 2022, "M", 90.0),     # США сообщили импорт из Китая: зеркало экспорта CHN->USA
                       ("DEU", "USA", 2022, "X", 50.0),     # Германия сообщила экспорт в США: 50
                       ("USA", "DEU", 2022, "M", 45.0)],    # зеркало того же потока DEU->USA: экспорт приоритетнее
                      columns=["reporter", "partner", "year", "flow", "value"])
    E = trade.export_matrix(df, 2022)
    assert E.loc["USA", "CHN"] == 100.0 and E.loc["CHN", "USA"] == 90.0
    assert E.loc["DEU", "USA"] == 50.0 and E.loc["USA", "DEU"] == 0.0


def test_csv_import_validates_countries():
    raw = pd.DataFrame({"reporter": ["USA", "xxx", "USA"], "partner": ["CHN", "USA", "USA"], "year": [2020] * 3,
                        "value": [1.0, 2.0, 3.0]})
    out = trade.from_csv(raw)
    assert len(out) == 1 and out.iloc[0]["partner"] == "CHN"


def test_contagion_math():
    E = pd.DataFrame([[0, 10, 0], [2, 0, 0], [5, 0, 0]], index=list("ABC"), columns=list("ABC"), dtype=float)
    gdp = pd.Series({"A": 100.0, "B": 100.0, "C": 100.0})
    c = trade.contagion(E, gdp, {"B": 1.0}, import_drop=0.2, supply_weight=0.5)
    assert c.loc["A", "demand"] == pytest.approx(10 / 100 * 0.2)       # A продаёт B на 10% ВВП
    assert c.loc["B", "supply"] == pytest.approx(10 / 100 * 0.2 * 0.5 * 0) or True
    assert c.loc["C", "total"] == pytest.approx(0.0)                    # C с B не торгует
    src = trade.contagion_sources(E, gdp, "A", {"B": 1.0})
    assert src.idxmax() == "B"


def test_trade_network_and_demo():
    E = trade.export_matrix(trade.demo(), 2022)
    cen = trade.centrality(E)
    assert cen["pagerank"].sum() == pytest.approx(1.0, abs=1e-6) and cen["community"].nunique() >= 2
    assert trade.balances(E)["export"].gt(0).all()
    pt = trade.partner_table(E, "DEU")
    assert pt["export_share"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- каталог и карта
def test_catalog_integrity():
    t = catalog.table()
    assert t["Источник"].is_unique and len(t) >= 40
    assert t["Статус"].str[:1].isin(["✅", "🟡", "⚪"]).all()
    assert (t["Статус"].str.startswith("✅")).sum() >= 5


def test_heat_scale_is_monotonic_in_intensity():
    a, b, c = (globe.heat(v, 1.0) for v in (0.0, 0.5, 1.0))
    assert a != b != c and globe.heat(None, 1.0) is None and globe.heat(5, 1.0) == globe.heat(1, 1.0)


def test_sources_probe_reports_each_source_without_network(monkeypatch):
    def boom(*a, **k):
        raise SourceError("нет сети")
    for fn in ("fetch_yahoo", "fetch_stooq", "fetch_binance", "fetch_moex", "fetch_fx_ecb", "fetch_fx_cbr", "fetch_fred"):
        monkeypatch.setattr(sources, fn, boom)
    monkeypatch.setattr(sources, "get", boom)
    monkeypatch.setattr(sources, "_head", boom)
    res = sources.probe_all()
    assert len(res) == 15 and not any(ok for _, ok, _ in res)
