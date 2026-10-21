import json
import warnings

import numpy as np
import pandas as pd
import pytest

from tradelab import conflicts, crisis, globe, maritime, panel, store
from tradelab.net import SourceError

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- PortWatch и AIS
ARCGIS = {"features": [
    {"attributes": {"portid": "chokepoint6", "date": 1735689600000, "n_total": 110, "n_tanker": 40, "n_container": 20}},
    {"attributes": {"portid": "chokepoint6", "date": 1735776000000, "n_total": 100, "n_tanker": 35, "n_container": 22}}]}


def test_arcgis_parser_dates_and_columns():
    df = maritime.parse_arcgis(ARCGIS)
    assert df["date"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC") and len(df) == 2
    assert maritime.type_columns(df) == ["n_total", "n_tanker", "n_container"]
    with pytest.raises(SourceError):
        maritime.parse_arcgis({"error": {"message": "Invalid query"}})
    with pytest.raises(SourceError):
        maritime.parse_arcgis({})


def test_anomaly_table_detects_drop():
    dates = pd.date_range("2025-01-01", periods=120, tz="UTC")
    vals = np.r_[np.full(113, 100.0), np.full(7, 50.0)]
    daily = pd.DataFrame({"portid": "x", "date": dates, "n_tanker": vals})
    t = maritime.anomaly_table(daily, "n_tanker")
    assert t.loc["x", "recent"] == 50.0 and t.loc["x", "change"] < -0.4 and len(t.loc["x", "spark"]) == 60


DIGI_LOC = {"type": "FeatureCollection", "features": [
    {"mmsi": 230111000, "type": "Feature", "geometry": {"type": "Point", "coordinates": [24.9, 60.1]},
     "properties": {"mmsi": 230111000, "sog": 11.2, "cog": 90.0, "heading": 88}},
    {"mmsi": 230222000, "type": "Feature", "geometry": {"type": "Point", "coordinates": [21.0, 59.0]},
     "properties": {"mmsi": 230222000, "sog": 0.1, "cog": 0, "heading": 511}},
    {"mmsi": 1, "type": "Feature", "geometry": None, "properties": {}}]}
DIGI_VES = [{"mmsi": 230111000, "name": "NORDIC TANKER", "shipType": 80, "destination": "PORVOO"},
            {"mmsi": 230222000, "name": "CARGO ONE", "shipType": 70, "destination": "TALLINN"}]


def test_digitraffic_parser_and_classes():
    df = maritime.parse_digitraffic(DIGI_LOC, DIGI_VES)
    assert len(df) == 2 and df.set_index("mmsi").loc[230111000, "class"] == "Танкер"
    pts = maritime.ships_from_ais(df, "Танкер")
    assert len(pts) == 1 and pts[0]["name"] == "NORDIC TANKER" and pts[0]["dest"] == "PORVOO"
    all_pts = maritime.ships_from_ais(df, None)
    assert len(all_pts) == 2 and all_pts[1]["heading"] == 0.0           # heading 511 = неизвестно -> берётся курс


def test_aisstream_parser():
    pos = {"MessageType": "PositionReport", "MetaData": {"MMSI": 5, "ShipName": "ABC  ", "latitude": 1.2, "longitude": 103.8},
           "Message": {"PositionReport": {"Latitude": 1.25, "Longitude": 103.85, "Sog": 9.1, "Cog": 45.0, "TrueHeading": 44}}}
    st_ = {"MessageType": "ShipStaticData", "MetaData": {"MMSI": 5}, "Message": {"ShipStaticData": {"Type": 84, "Destination": "SINGAPORE "}}}
    a = maritime.parse_aisstream_message(json.dumps(pos))
    b = maritime.parse_aisstream_message(st_)
    assert a["kind"] == "pos" and a["name"] == "ABC" and a["lat"] == 1.25
    assert b["kind"] == "static" and b["ship_type"] == 84 and b["destination"] == "SINGAPORE"
    assert maritime.parse_aisstream_message({"MessageType": "Other", "Message": {}}) is None
    assert maritime.ship_class(84) == "Танкер"


def test_routes_are_continuous_and_inside_world():
    ships = maritime.demo_ships()
    assert len(ships) == len(maritime.ROUTES) and sum(r["n"] for r in ships) > 60
    for r in ships:
        pts = np.array(r["pts"])
        assert pts[:, 1].min() >= -60 and pts[:, 1].max() <= 80 and len(pts) > 20
        steps = np.abs(np.diff(pts[:, 0]))
        assert (np.minimum(steps, 360 - steps) < 40).all()               # нет скачков по долготе, кроме перехода через 180°


def test_chokepoint_reference_sane():
    for name, (lat, lon, ru) in maritime.CHOKEPOINTS.items():
        assert -90 <= lat <= 90 and -180 <= lon <= 180 and ru


# ---------------------------------------------------------------- конфликты
GED = pd.DataFrame([
    {"id": 1, "year": 2022, "type_of_violence": 1, "conflict_name": "Russia - Ukraine", "side_a": "A", "side_b": "B",
     "country": "Ukraine", "region": "Europe", "latitude": 48.0, "longitude": 37.0, "date_start": "2022-03-01", "best": 100,
     "low": 80, "high": 120, "deaths_civilians": 30},
    {"id": 2, "year": 2022, "type_of_violence": 3, "conflict_name": "X", "side_a": "A", "side_b": "B", "country": "DR Congo (Zaire)",
     "region": "Africa", "latitude": -1.7, "longitude": 29.0, "date_start": "2022-05-01", "best": 20, "low": 10, "high": 30, "deaths_civilians": 20},
    {"id": 3, "year": 2023, "type_of_violence": 1, "conflict_name": "Y", "side_a": "A", "side_b": "B", "country": "Atlantis",
     "region": "?", "latitude": 0.0, "longitude": 0.0, "date_start": "2023-01-01", "best": 5, "low": 5, "high": 5, "deaths_civilians": 0},
    {"id": 4, "year": 2023, "type_of_violence": 1, "conflict_name": "Z", "side_a": "A", "side_b": "B", "country": "Ukraine",
     "region": "Europe", "latitude": None, "longitude": 30.0, "date_start": "2023-02-01", "best": 7, "low": 7, "high": 7, "deaths_civilians": 0}])


def test_ged_parser_maps_countries_and_drops_bad_rows():
    d = conflicts.parse_ged(GED)
    assert len(d) == 3                                                   # событие без координат отброшено
    assert d.set_index("id").loc[1, "iso3"] == "UKR" and d.set_index("id").loc[2, "iso3"] == "COD"
    assert pd.isna(d.set_index("id").loc[3, "iso3"])                    # неизвестная страна не угадывается
    with pytest.raises(SourceError):
        conflicts.parse_ged(pd.DataFrame({"year": [1]}))


def test_country_year_grid_and_summary():
    d = conflicts.parse_ged(GED)
    cy = conflicts.country_year(d)
    assert cy.set_index(["iso3", "year"]).loc[("UKR", 2022), "value"] == 100 and (cy["indicator"] == "conflict_deaths").all()
    g = conflicts.grid(d, 2.0)
    assert g["deaths"].sum() == 125 and g.iloc[0]["deaths"] == 100
    s = conflicts.summary(d, [2022])
    assert s["events"] == 2 and s["deaths"] == 120 and s["civilians"] == 50


def test_iso_matching_variants():
    for name, iso in (("Russia (Soviet Union)", "RUS"), ("Myanmar (Burma)", "MMR"), ("Syria", "SYR"), ("Iran", "IRN"),
                      ("Turkey", "TUR"), ("UKR", "UKR"), ("Sudan", "SDN"), ("Afghanistan", "AFG")):
        assert conflicts.to_iso3(name) == iso, name
    assert conflicts.to_iso3("Atlantis") is None and conflicts.to_iso3(None) is None


def test_reference_and_demo_events():
    ref = conflicts.reference_table()
    assert ref["iso3"].map(lambda i: countries_ok(i)).all() and ref["lat"].between(-90, 90).all()
    ev = conflicts.demo_events()
    assert ev["iso3"].notna().all() and ev["best"].min() >= 1 and ev["year"].nunique() == 3


def countries_ok(iso):
    from tradelab import countries
    return iso in countries.table().index


def test_store_ged_roundtrip(tmp_path):
    con = store.connect_events(tmp_path / "e.db")
    d = conflicts.parse_ged(GED)
    assert store.upsert_ged(con, d) == 3
    assert store.upsert_ged(con, d) == 3                                  # идемпотентно
    back = store.load_ged(con)
    assert len(back) == 3 and back.set_index("id").loc[1, "best"] == 100 and "latitude" in back.columns


def test_conflict_onset_model_finds_planted_signal():
    """Синтетика: конфликт начинается чаще при высокой инфляции прошлого года. Проверяем, что конвейер это находит вне выборки."""
    base = panel.snapshot_panel()
    rng = np.random.default_rng(0)
    rows = []
    for iso in base.index.get_level_values("iso3").unique():
        for y in range(1989, 2014):
            infl = base["inflation"].get((iso, y - 1), np.nan)
            p = 0.01 + (0.15 if np.isfinite(infl) and infl > 15 else 0.0)
            rows.append((iso, y, "conflict_deaths", float(rng.choice([0, 200], p=[1 - p, p]))))
    pc = panel.with_conflicts(base, pd.DataFrame(rows, columns=["iso3", "year", "indicator", "value"]))
    assert "conflict" in crisis.available_kinds(pc) and "conflict" not in crisis.available_kinds(base)
    met, _ = crisis.evaluate(pc, "conflict", first_test_year=2000, last_year=2013, refit_every=4)
    assert met["events"] >= 20 and met["ensemble"]["auc"] > 0.6


def test_conflict_labels_exclude_countries_already_in_conflict():
    idx = pd.MultiIndex.from_product([["AAA"], range(1990, 1996)], names=["iso3", "year"])
    df = pd.DataFrame({"conflict_deaths": [0, 0, 100, 150, 0, 40.0]}, index=idx)
    d = crisis.derive(df)
    onset, known, in_c = crisis.labels(d)
    assert onset["conflict"].astype(int).tolist() == [0, 0, 1, 0, 0, 1]
    assert in_c["conflict"].fillna(False).astype(bool).tolist() == [False, False, False, True, True, False]


# ---------------------------------------------------------------- карта
def test_globe_embeds_all_layers_as_valid_json_offline():
    html = globe.build([], [], {}, "T", [], ships=maritime.demo_ships(),
                       ship_points=[{"lat": 1, "lon": 2, "heading": 90, "name": "S", "type": "Танкер"}],
                       chokes=[{"name": "P", "lat": 1, "lon": 2, "color": "#fff", "size": 8, "label": "l", "spark": [1, 2, 3]}],
                       conflicts=[{"name": "C", "lat": 3, "lon": 4, "deaths": 10, "label": "l"}])
    data = json.loads(html.split("const D=")[1].split(";\nconst cv")[0])
    assert len(data["ships"]) == len(maritime.ROUTES) and len(data["chokes"]) == 1 and len(data["conflicts"]) == 1
    assert "https://" not in html and len(data["shipPoints"]) == 1


def test_demo_tanker_routes_stay_on_water():
    """Баг: условные маршруты танкеров заходили на сушу (Суматра, Оман, Куба, Швеция, Япония). Теперь маршруты по морскому растру."""
    import numpy as np
    from pathlib import Path
    root = Path(maritime.__file__).resolve().parent.parent / "assets"
    z = np.load(root / "sea_mask.npz")
    shape = tuple(z["shape"])
    land = np.unpackbits(z["land"])[:shape[0] * shape[1]].reshape(shape).astype(bool)
    res = shape[1] // 360
    ships = maritime.demo_ships()
    assert len(ships) == len(maritime.ROUTES)
    for sh in ships:
        pts = sh["pts"]
        assert len(pts) >= 20, sh["name"]
        bad = [p for p in pts if land[int(np.clip((90 - p[1]) * res, 0, shape[0] - 1)), int(((p[0] + 180) * res) % shape[1])]]
        assert not bad, f"{sh['name']}: точек на суше {len(bad)}, например {bad[:2]}"
    first = {r[0]: r[3][0] for r in maritime.ROUTES}
    for sh in ships:                                    # маршрут начинается рядом с заданным портом
        lo, la = first[sh["name"]]
        assert abs(sh["pts"][0][0] - lo) < 5 and abs(sh["pts"][0][1] - la) < 5


def test_chokepoint_name_matching_and_calendar_stats():
    assert maritime.match_chokepoint("Malacca Strait") == "Strait of Malacca"
    assert maritime.match_chokepoint("Hormuz Strait") == "Strait of Hormuz"
    assert maritime.match_chokepoint("Suez Canal") == "Suez Canal" and maritime.match_chokepoint("Cape of Good Hope") == "Cape of Good Hope"
    assert maritime.match_chokepoint("Bab el-Mandeb Strait") == "Bab el-Mandeb Strait"
    assert maritime.match_chokepoint("Kiel Canal") is None
    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    s = pd.Series(100.0, index=idx)
    s[-7:] = 50.0
    st_ = maritime.choke_stats(s)
    assert st_["change"] < -0.4 and st_["recent"] == 50.0
    gappy = s.drop(s.index[20:60])                       # пропущенные дни не должны считаться нулями и сдвигать окна
    g = maritime.choke_stats(gappy)
    assert g["recent"] == 50.0 and abs(g["baseline"] - st_["baseline"]) < 1e-9
    assert maritime.choke_stats(s.head(20)) is None
