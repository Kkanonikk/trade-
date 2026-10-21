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


def test_comparison_groups_are_valid_countries():
    from tradelab import countries, views_worldmode
    known = set(countries.table().index)
    for name, isos in views_worldmode.GROUPS.items():
        assert len(isos) >= 5 and set(isos) <= known, name


# ---------------------------------------------------------------- качество данных, досье, вопросы, устойчивость
def test_quality_flags_detect_gap_jump_and_flat():
    from tradelab import quality
    vals = np.r_[np.linspace(10, 12, 10), [60.0], np.linspace(12, 13, 5), np.full(6, 13.0)]       # 22 значения
    years = list(range(1980, 1990)) + [1990] + list(range(1996, 2001)) + list(range(2001, 2007))   # дыра 1991–1995
    s = pd.Series(vals, index=years)
    kinds = {f[0] for f in quality.series_flags(s)}
    assert {"пропуск", "скачок", "залипание"} <= kinds
    assert quality.series_flags(pd.Series(np.linspace(1, 2, 30), index=range(1990, 2020))) == []


def test_quality_report_marks_stale_series(w):
    from tradelab import quality
    r = quality.report(w, ["gm:median_age_years", "gm:adults_with_hiv_percent_age_15_49"], current_year=2026)
    st_ = r.set_index("key")["Статус"]
    assert st_["gm:adults_with_hiv_percent_age_15_49"] == "устарел"


def test_country_brief_has_sources_and_caveat(w):
    from tradelab import brief
    b = brief.build(w, "JPN", aging.population_path(w))
    assert len(b["bullets"]) >= 5 and any("Медианный возраст" in t for t, _ in b["bullets"])
    html = brief.to_html(b)
    assert "Источники" in html and "лицензия" in html and "Trade Lab" in html and brief.CAVEAT[:30] in html


def test_ask_routes_questions(w):
    from tradelab import ask
    a = gdata.available(w)
    assert ask.interpret("риск голода в Нигерии", a)["page"] == "ГОЛОД"
    assert ask.interpret("риск голода в Нигерии", a)["iso"] == "NGA"
    assert ask.interpret("где больше всего стариков", a)["page"] == "ДЕМОГРАФИЯ"
    assert ask.interpret("кризис в Турции", a)["iso"] == "TUR"
    assert ask.interpret("абракадабра", a)["page"] is None


def test_hunger_weight_robustness_and_what_if(w):
    rob = hunger.weight_robustness(w, 2020, n=15)
    assert rob.attrs["spearman_mean"] > 0.9 and (rob["p5"] <= rob["p95"]).all()
    base, worse = hunger.what_if(w, 2020, "NGA", {"gm:gdppercapita_us_inflation_adjusted": 0.4, "gm:malnutrition_weight_for_age_percent_of_children_under_5": 2.0})
    assert worse >= base
    b2, same = hunger.what_if(w, 2020, "NGA", {})
    assert abs(b2 - same) < 1e-6
    comps = hunger.COMPONENTS
    assert abs(sum(v[0] for v in comps.values()) - 1.0) < 1e-9          # веса восстановлены после проверки


def test_country_backtest_ratio_on_trend():
    s = pd.Series(10 + 0.5 * np.arange(40), index=range(1980, 2020))
    d, ratio = forecast.country_backtest(s, "trend", 5)
    assert len(d) > 10 and ratio < 0.05                    # линейный тренд почти идеален против наивного
    assert forecast.country_backtest(s.head(10), "trend", 5)[0] is None


def test_user_csv_mapping(w):
    df = pd.DataFrame({"country": ["Россия", "Germany", "USA", "Nowhere"], "year": [2020, 2020, 2020, 2020], "value": [1, 2, 3, 4]})
    ser, bad = gdata.from_user_csv(df, "user:x")
    assert set(ser.index.get_level_values("iso3")) == {"RUS", "DEU", "USA"} and bad == 1
    with pytest.raises(ValueError):
        gdata.from_user_csv(pd.DataFrame({"a": [1]}), "user:y")


def test_aging_robustness_returns_specs_and_permutation(w):
    rb = aging.robustness(w, "growth")
    assert len(rb) >= 5 and rb["p"].notna().sum() == 1
    assert rb["SE"].dropna().gt(0).all()


# ---------------------------------------------------------------- хаб открытых данных и кокпит
def test_hub_name_mapping_and_parse_thresholds():
    from tradelab import hub
    co2 = pd.read_csv(panel.SNAP / "owid_co2.csv.gz", usecols=["country", "iso_code"])
    ids = hub.name_map(co2)
    assert ids["united states"] == "USA" and ids["russia"] == "RUS" and ids["korea, rep."] == "KOR"
    ents = list(ids)[:80]
    df = pd.DataFrame({"Entity": ents * 10, "Year": np.repeat(np.arange(2001, 2011), len(ents)), "Index": np.random.default_rng(0).normal(size=len(ents) * 10)})
    long, info = hub.parse_dataset(df, "x", ids)
    assert long is not None and info[0][1] >= hub.MIN_COUNTRIES
    few = df[df["Entity"].isin(ents[:5])]
    assert hub.parse_dataset(few, "x", ids)[0] is None            # мало стран
    assert hub.classify("Democracy - Polity 5") == "Безопасность и институты" and hub.classify("Daily supply of calories") == "Питание и голод"


def test_hub_snapshot_is_loaded_with_sources(w):
    from tradelab import registry
    meta = gdata.meta()
    hubk = [k for k in w.columns if k.startswith("hub:")]
    assert len(hubk) > 300
    for k in hubk[:50]:
        m = meta[k]
        assert m["source"] == "owid_hub" and m["source_name"] and m["dataset"]
        assert registry.SOURCES["owid_hub"]["license"]
    assert (w[hubk].dropna(how="all").index.get_level_values("year") >= 1960).all()


def test_cockpit_look_through_and_overlay():
    from tradelab import cockpit
    e = cockpit.look_through({"AAPL": 0.3, "EEM": 0.2, "GLD": 0.1, "BTC-USD": 0.1, "ZZZZ": 0.1, "SBER.MOEX": 0.2})
    assert abs(e.sum() - 1.0) < 1e-9 and e["USA"] == pytest.approx(0.3) and e["CHN"] == pytest.approx(0.06) and e["RUS"] == pytest.approx(0.2)
    assert e["Без страны"] == pytest.approx(0.2) and e["н/д"] == pytest.approx(0.1)
    vals = pd.Series({"USA": 10.0, "CHN": 30.0})
    avg, cover, tbl = cockpit.weighted_overlay(e, vals)
    assert avg == pytest.approx((0.3 * 10 + 0.06 * 30) / 0.36) and 0 < cover < 1 and tbl.index[0] == "USA"


def test_cockpit_factor_exposure_recovers_known_beta():
    from tradelab import cockpit
    rng = np.random.default_rng(3)
    idx = pd.date_range("2018-01-07", periods=300, freq="W-SUN")          # недели, как их строит resample("W")
    F = pd.DataFrame({"Акции мира (S&P 500)": rng.normal(0, 0.02, 300), "Нефть Brent": rng.normal(0, 0.04, 300)}, index=idx)
    y = 0.8 * F["Акции мира (S&P 500)"] + 0.1 * F["Нефть Brent"] + rng.normal(0, 0.005, 300)
    weekly_level = pd.Series((1 + y).cumprod().to_numpy(), index=idx)
    daily = weekly_level.reindex(pd.date_range(idx[0], idx[-1], freq="D")).ffill()
    pr = daily.pct_change().dropna()                                      # дневной ряд, из которого недельная доходность восстанавливается в y
    exp = cockpit.exposures(pr, F)
    assert abs(exp.loc["Акции мира (S&P 500)", "beta"] - 0.8) < 0.1 and exp.loc["Акции мира (S&P 500)", "t"] > 10
    est, lo, hi = cockpit.scenario_pnl(exp, {"Акции мира (S&P 500)": -0.2})
    assert est == pytest.approx(-0.16, abs=0.03) and lo < est < hi
    with pytest.raises(ValueError):
        cockpit.exposures(pr.head(10), F)


# ---------------------------------------------------------------- «загрузить всё», тема
def test_load_all_runs_steps_in_parallel_and_isolates_failures(monkeypatch, tmp_path):
    from tradelab import config, loadall, openmarket, store
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "m.db")
    good = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5, tz="UTC"), "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0})

    def fake_load(ticker, start, tf, source):
        if ticker == "BAD":
            raise RuntimeError("источник недоступен")
        return good, "Yahoo Finance"
    monkeypatch.setattr(openmarket, "load", fake_load)
    steps = [(f"Котировки {t}", loadall._quotes(t, "D", 2, "Авто")) for t in ["AAA", "BBB", "BAD", "CCC", "DDD", "EEE", "FFF", "GGG"]]
    steps.append(("Падающий шаг", lambda: 1 / 0))
    seen = []
    res = loadall.run(steps, lambda f, label: seen.append(f))
    ok = {r[0]: r[1] for r in res}
    assert [r[0] for r in res] == [s[0] for s in steps]                         # порядок отчёта = порядок шагов
    assert ok["Котировки BAD"] is False and ok["Падающий шаг"] is False and sum(ok.values()) == 7
    assert seen and abs(max(seen) - 1.0) < 1e-9
    con = store.connect()
    assert {t for t in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG") if len(store.load(con, t, "D")) == 5} == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"}


def test_load_all_default_steps_cover_every_online_source():
    from tradelab import loadall
    labels = [lab for lab, _ in loadall.build_steps(["AAPL"], include_quotes=True)]
    for key in ("Мировые рынки", "World Bank", "PortWatch", "UCDP", "Comtrade", "Котировки AAPL"):
        assert any(key in lab for lab in labels), key


def test_theme_top_bar_and_tape_have_no_broker_mentions():
    from tradelab import theme
    html = theme.top_bar("Глобальный мир", "обзор", "snap", "встроенный снимок")
    assert "TRADE LAB" in html and "снимок" in html and "БКС" not in html
    tape = theme.tape_snapshot()
    assert "BRENT" in tape and "VIX" in tape and "S&amp;P" not in tape.replace("S&P", "")
    assert "БКС" not in theme.CSS
