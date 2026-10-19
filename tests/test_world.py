import json

import numpy as np
import pandas as pd
import pytest

from tradelab import globe, markets, ru_data, sources, structure, worlddata


@pytest.fixture(scope="module")
def world():
    from tradelab import demo_world
    return demo_world.world(n=700)


def test_demo_world_shapes(world):
    assert world["index"].shape[1] == len(markets.MARKETS)
    assert not world["index"].isna().any().any()
    assert set(world["rates"].columns) == set(markets.POLICY_RATES)
    assert {"Выручка", "Чистая прибыль"} <= set(world["facts"]["ACME"].concept)


def test_demo_has_asia_to_europe_lead(world):
    R = worlddata.returns(world["index"])
    L = structure.lead_lag(R[["Nikkei 225", "DAX", "S&P 500"]])
    assert L.loc["Nikkei 225", "DAX"] > L.loc["DAX", "Nikkei 225"] + 0.05


def test_demo_crisis_is_visible_in_topology_and_regimes(world):
    R = worlddata.returns(world["index"])
    topo = structure.rolling_topology(R, 60, 10)
    n = len(R)
    crisis = topo.loc[R.index[int(n * 0.64)]:R.index[int(n * 0.70)]]
    calm = topo.loc[R.index[60]:R.index[int(n * 0.5)]]
    assert crisis["absorption"].mean() > calm["absorption"].mean()
    assert crisis["mst_length"].mean() < calm["mst_length"].mean()
    state, _ = structure.regimes(R, k=3)
    assert state.loc[R.index[int(n * 0.66)]:R.index[int(n * 0.69)]].mean() > state.mean() + 0.5


def test_demo_clusters_recover_regions(world):
    R = worlddata.returns(world["index"])
    cl = pd.Series(structure.clusters(R.corr().to_numpy(), 3), index=R.columns)
    assert cl["Nikkei 225"] == cl["Hang Seng"] == cl["KOSPI"]
    assert cl["DAX"] == cl["CAC 40"] == cl["FTSE 100"]


def test_great_circle_endpoints_and_length():
    pts = globe.great_circle(40.7, -74.0, 51.5, -0.1)
    assert pts[0] == pytest.approx([-74.0, 40.7], abs=1e-6) and pts[-1] == pytest.approx([-0.1, 51.5], abs=1e-6)
    antipode = globe.great_circle(0, 0, 0, 90, n=3)
    assert antipode[1][0] == pytest.approx(45.0, abs=1e-6)


def test_countries_geometry_loaded():
    c = globe.countries()
    isos = {x["iso"] for x in c}
    assert {"USA", "RUS", "CHN", "DEU", "BRA"} <= isos and len(c) > 150
    assert all(ring[0] == ring[-1] or len(ring) > 3 for x in c for ring in x["rings"])


def test_globe_html_embeds_valid_json_and_no_external_urls():
    mk = [{"name": "A", "iso": "USA", "city": "X", "lat": 0, "lon": 0, "label": "l", "color": "#fff", "open": True},
          {"name": "B", "iso": "DEU", "city": "Y", "lat": 10, "lon": 20, "label": "l", "color": "#fff", "open": False}]
    html = globe.build(mk, [{"a": 0, "b": 1, "w": 1.0, "color": "#f90", "label": ""}], {"USA": "#123456"}, "T")
    blob = html.split("const D=")[1].split(";\nconst cv")[0]
    data = json.loads(blob)
    assert len(data["markets"]) == 2 and len(data["arcs"][0]["pts"]) == 48
    assert "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html  # офлайн


def test_color_scales():
    assert globe.diverging(0.05, 0.03) == globe.diverging(0.03, 0.03)  # насыщение
    assert globe.diverging(None, 0.03) is None
    assert globe.diverging(0.02, 0.03) != globe.diverging(-0.02, 0.03)
    assert globe.sequential(5, 0, 10) == globe.lerp_color((30, 38, 52), (255, 159, 26), 0.5)


def test_ru_source_resolution_and_moex_timeframes(monkeypatch):
    monkeypatch.setenv("BCS_REFRESH_TOKEN", "")
    assert ru_data.resolve("Авто") == "MOEX ISS"
    monkeypatch.setenv("BCS_REFRESH_TOKEN", "x")
    assert ru_data.resolve("Авто") == "БКС"
    from datetime import datetime, timezone
    with pytest.raises(ValueError):
        ru_data.load("SBER", datetime.now(timezone.utc), "M15", "MOEX ISS")


def test_moex_pagination_stops_on_short_page(monkeypatch):
    calls = []

    def fake_get(url, params=None, **kw):
        calls.append(params["start"])
        n = 500 if params["start"] == 0 else 3
        cols = ["open", "close", "high", "low", "value", "volume", "begin", "end"]
        base = pd.Timestamp("2020-01-01") + pd.Timedelta(days=params["start"])
        data = [[1, 1, 1, 1, 1, 1, str(base + pd.Timedelta(days=i)), ""] for i in range(n)]
        return {"candles": {"columns": cols, "data": data}}

    monkeypatch.setattr(sources, "get", fake_get)
    df = sources.fetch_moex("SBER", 5)
    assert calls == [0, 500] and len(df) == 503


def test_index_fetch_falls_back_between_sources(monkeypatch):
    def boom(*a, **k):
        raise sources.SourceError("down")
    ok = pd.DataFrame({"ts": pd.to_datetime(["2024-01-02"], utc=True), "open": 1, "high": 1, "low": 1,
                       "close": 1.0, "volume": 1})
    monkeypatch.setattr(sources, "fetch_yahoo", boom)
    monkeypatch.setattr(sources, "fetch_stooq", lambda *a, **k: ok)
    df, src = sources.fetch_index("S&P 500")
    assert src == "Stooq" and len(df) == 1
    monkeypatch.setattr(sources, "fetch_stooq", boom)
    with pytest.raises(sources.SourceError):
        sources.fetch_index("S&P 500")


def test_refresh_reports_failures_without_crashing(tmp_path, monkeypatch):
    from tradelab import store
    con = store.connect_world(tmp_path / "w.db")

    def boom(*a, **k):
        raise sources.SourceError("нет сети")
    for fn in ("fetch_index", "fetch_yahoo", "fetch_fx_ecb", "fetch_fx_cbr", "fetch_fred", "fetch_worldbank"):
        monkeypatch.setattr(sources, fn, boom)
    report = worlddata.refresh(con, 1)
    assert len(report) > 40 and not any(ok for _, ok, _ in report)
    assert all("нет сети" in msg for _, _, msg in report)


def test_series_roundtrip(tmp_path):
    from tradelab import store
    con = store.connect_world(tmp_path / "w.db")
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True), "key": "DFF", "value": [5.3, 5.4]})
    store.upsert_series(con, "rate", df)
    store.upsert_series(con, "rate", df)  # идемпотентно
    out = store.load_series(con, "rate")
    assert out["DFF"].tolist() == [5.3, 5.4]
