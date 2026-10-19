"""Морской трафик и танкеры из открытых источников.

Реальные данные:
  * IMF PortWatch (ArcGIS REST, без ключа): суточные проходы судов по 28 проливам и каналам и оценки объёмов (AIS);
  * Digitraffic Finland (без ключа): живые позиции судов в Балтике;
  * AISStream (бесплатный ключ, WebSocket): живые позиции в заданной области мира.
Демо: условные маршруты танкеров для анимации (позиции НЕ реальные).
Форматы написаны по документации и проверены на образцах, не вживую.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import globe
from .net import SourceError, get

PORTWATCH = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services"
TANKER_TYPES = range(80, 90)   # AIS: 80-89 танкеры
SHIP_CLASSES = {7: "Грузовое", 8: "Танкер", 6: "Пассажирское", 3: "Спец. (рыболовное, буксир и др.)", 5: "Спец.", 9: "Прочее"}


def ship_class(ship_type):
    try:
        return SHIP_CLASSES.get(int(ship_type) // 10, "Прочее")
    except (TypeError, ValueError):
        return "Неизвестно"


# ---------------------------------------------------------------- PortWatch
def parse_arcgis(payload):
    """ArcGIS REST query -> DataFrame атрибутов; поля-даты (мс с 1970) переводятся в datetime."""
    if "error" in (payload or {}):
        raise SourceError(f"ArcGIS: {payload['error'].get('message', payload['error'])}")
    feats = (payload or {}).get("features")
    if feats is None:
        raise SourceError("ArcGIS: в ответе нет features")
    df = pd.DataFrame([f.get("attributes", {}) for f in feats])
    if df.empty:
        return df
    for c in df.columns:
        if c.lower() == "date" and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], unit="ms", utc=True).dt.normalize()
    return df


def _query(service, where, out_fields="*", order=None, page=2000):
    rows, offset = [], 0
    while True:
        params = {"where": where, "outFields": out_fields, "f": "json", "resultOffset": offset, "resultRecordCount": page}
        if order:
            params["orderByFields"] = order
        payload = get(f"{PORTWATCH}/{service}/FeatureServer/0/query", params=params, as_json=True)
        df = parse_arcgis(payload)
        if df.empty:
            break
        rows.append(df)
        if not payload.get("exceededTransferLimit"):
            break
        offset += len(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fetch_chokepoint_meta():
    """Справочник проливов PortWatch: portid, название, координаты (поля подбираются без учёта регистра)."""
    df = _query("PortWatch_chokepoints_database", "1=1")
    low = {c.lower(): c for c in df.columns}
    need = {k: low.get(k) for k in ("portid", "portname", "lat", "lon")}
    if not all(need.values()):
        raise SourceError(f"PortWatch: в справочнике нет нужных полей (есть {list(df.columns)[:12]})")
    return df.rename(columns={v: k for k, v in need.items()})


def fetch_chokepoint_daily(days=365):
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    df = _query("Daily_Chokepoints_Data", f"date >= TIMESTAMP '{since} 00:00:00'", order="date")
    if df.empty:
        raise SourceError("PortWatch: нет данных за период")
    return df


def type_columns(df):
    """Колонки числа проходов по типам судов (n_tanker, n_container, ...)."""
    return [c for c in df.columns if c.lower().startswith("n_")]


def anomaly_table(daily, col="n_tanker", short=7, long=90):
    """Для каждого пролива: среднее за short дней против среднего за long дней (отклонение в %)."""
    rows = []
    for pid, g in daily.sort_values("date").groupby("portid"):
        s = g.set_index("date")[col].astype(float)
        if len(s) < long // 2:
            continue
        a, b = s.tail(short).mean(), s.tail(long).mean()
        rows.append({"portid": pid, "recent": a, "baseline": b, "change": (a / b - 1) if b > 0 else np.nan,
                     "spark": s.tail(60).tolist()})
    return pd.DataFrame(rows).set_index("portid")


# ---------------------------------------------------------------- живые позиции судов
def parse_digitraffic(locations, vessels):
    """Digitraffic (Финляндия): GeoJSON позиций + список судов -> DataFrame mmsi, lat, lon, sog, cog, name, ship_type, dest."""
    rows = []
    meta = {v["mmsi"]: v for v in (vessels or []) if "mmsi" in v}
    for f in (locations or {}).get("features", []):
        p, geom = f.get("properties") or {}, (f.get("geometry") or {}).get("coordinates")
        mmsi = f.get("mmsi") or p.get("mmsi")
        if not geom or mmsi is None:
            continue
        m = meta.get(mmsi, {})
        rows.append({"mmsi": int(mmsi), "lon": float(geom[0]), "lat": float(geom[1]), "sog": p.get("sog"), "cog": p.get("cog"),
                     "heading": p.get("heading"), "name": m.get("name"), "ship_type": m.get("shipType"),
                     "destination": m.get("destination")})
    df = pd.DataFrame(rows, columns=["mmsi", "lon", "lat", "sog", "cog", "heading", "name", "ship_type", "destination"])
    df["class"] = df["ship_type"].map(ship_class)
    return df


def fetch_digitraffic():
    base = "https://meri.digitraffic.fi/api/ais/v1"
    loc = get(f"{base}/locations", as_json=True, headers={"Digitraffic-User": "TradeLab"})
    ves = get(f"{base}/vessels", as_json=True, headers={"Digitraffic-User": "TradeLab"})
    return parse_digitraffic(loc, ves)


def parse_aisstream_message(msg):
    """Сообщение AISStream (JSON-строка или dict) -> dict с mmsi/позицией или статикой; иначе None."""
    m = json.loads(msg) if isinstance(msg, (str, bytes)) else msg
    meta, body, kind = m.get("MetaData", {}), m.get("Message", {}), m.get("MessageType")
    mmsi = meta.get("MMSI")
    if kind == "PositionReport" and "PositionReport" in body:
        pr = body["PositionReport"]
        return {"kind": "pos", "mmsi": mmsi, "lat": pr.get("Latitude", meta.get("latitude")),
                "lon": pr.get("Longitude", meta.get("longitude")), "sog": pr.get("Sog"), "cog": pr.get("Cog"),
                "heading": pr.get("TrueHeading"), "name": (meta.get("ShipName") or "").strip() or None}
    if kind == "ShipStaticData" and "ShipStaticData" in body:
        st = body["ShipStaticData"]
        return {"kind": "static", "mmsi": mmsi, "ship_type": st.get("Type"), "destination": (st.get("Destination") or "").strip() or None}
    return None


def collect_aisstream(api_key, bbox, seconds=25):
    """Подписка на AISStream на `seconds` секунд. bbox = (lat1, lon1, lat2, lon2). Нужен пакет websockets и бесплатный ключ."""
    try:
        import asyncio
        import websockets
    except ImportError as exc:
        raise SourceError("Для AISStream установи пакет websockets: pip install websockets") from exc

    async def run():
        pos, static = {}, {}
        sub = {"APIKey": api_key, "BoundingBoxes": [[[bbox[0], bbox[1]], [bbox[2], bbox[3]]]],
               "FilterMessageTypes": ["PositionReport", "ShipStaticData"]}
        async with websockets.connect("wss://stream.aisstream.io/v0/stream") as ws:
            await ws.send(json.dumps(sub))
            end = asyncio.get_event_loop().time() + seconds
            while asyncio.get_event_loop().time() < end:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                parsed = parse_aisstream_message(raw)
                if parsed and parsed["kind"] == "pos":
                    pos[parsed["mmsi"]] = parsed
                elif parsed:
                    static[parsed["mmsi"]] = parsed
        return pos, static

    pos, static = asyncio.run(run())
    rows = [{**p, "ship_type": static.get(k, {}).get("ship_type"), "destination": static.get(k, {}).get("destination")}
            for k, p in pos.items()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["class"] = df["ship_type"].map(ship_class)
    return df


# ---------------------------------------------------------------- справочник и демо-маршруты
# имя: (lat, lon, русское название)
CHOKEPOINTS = {
    "Strait of Hormuz": (26.57, 56.25, "Ормузский пролив"), "Strait of Malacca": (2.5, 101.5, "Малаккский пролив"),
    "Suez Canal": (30.5, 32.34, "Суэцкий канал"), "Bab el-Mandeb Strait": (12.6, 43.3, "Баб-эль-Мандебский пролив"),
    "Panama Canal": (9.08, -79.68, "Панамский канал"), "Gibraltar Strait": (35.95, -5.6, "Гибралтарский пролив"),
    "Bosporus Strait": (41.12, 29.07, "Босфор"), "Dardanelles Strait": (40.2, 26.4, "Дарданеллы"),
    "Dover Strait": (51.0, 1.4, "Па-де-Кале"), "Taiwan Strait": (24.5, 119.5, "Тайваньский пролив"),
    "Cape of Good Hope": (-34.4, 18.5, "Мыс Доброй Надежды"), "Lombok Strait": (-8.6, 115.7, "Пролив Ломбок"),
    "Sunda Strait": (-6.0, 105.8, "Зондский пролив"), "Danish Straits": (55.6, 12.7, "Датские проливы"),
    "Mozambique Channel": (-18.0, 41.0, "Мозамбикский пролив"), "Kerch Strait": (45.3, 36.6, "Керченский пролив"),
    "Korea Strait": (34.0, 129.0, "Корейский пролив"),
}

# Маршрут: (название, число судов в модели, цвет, точки [(lon, lat), ...] вдоль морских путей)
ROUTES = [
    ("Персидский залив → Китай", 14, "#FF9F1A", [(50.2, 26.6), (56.4, 26.5), (60.0, 23.0), (68.0, 14.0), (78.0, 5.5), (88.0, 5.0),
                                              (96.0, 5.0), (100.0, 2.5), (103.6, 1.2), (108.0, 8.0), (114.0, 17.0), (121.8, 29.5)]),
    ("Персидский залив → Индия", 8, "#FFC857", [(50.2, 26.6), (56.4, 26.5), (60.0, 23.0), (66.0, 21.5), (69.8, 22.3)]),
    ("Персидский залив → Япония", 8, "#FF9F1A", [(50.2, 26.6), (56.4, 26.5), (60.0, 23.0), (68.0, 14.0), (78.0, 5.5), (96.0, 5.0),
                                              (103.6, 1.2), (110.0, 8.0), (118.0, 20.0), (126.0, 28.0), (139.6, 35.3)]),
    ("Персидский залив → Европа (Суэц)", 9, "#FFC857", [(50.2, 26.6), (56.4, 26.5), (60.0, 22.0), (52.0, 13.0), (43.4, 12.5),
                                                      (40.0, 17.0), (35.0, 26.0), (32.55, 29.9), (32.3, 31.4), (25.0, 34.5),
                                                      (14.0, 36.0), (6.0, 37.5), (-5.6, 36.0), (-9.8, 39.0), (-9.0, 45.0), (-4.0, 49.0), (4.0, 51.9)]),
    ("Персидский залив → Европа (вокруг Африки)", 5, "#FFC857", [(50.2, 26.6), (56.4, 26.5), (60.0, 20.0), (55.0, 5.0), (48.0, -10.0),
                                                              (38.0, -26.0), (28.0, -34.5), (18.5, -35.5), (10.0, -28.0), (0.0, -5.0),
                                                              (-12.0, 12.0), (-17.5, 25.0), (-10.0, 37.0), (-9.0, 45.0), (-4.0, 49.0), (4.0, 51.9)]),
    ("США (Мексиканский залив) → Европа", 10, "#4DA3FF", [(-95.0, 29.2), (-90.0, 28.0), (-84.0, 24.5), (-80.0, 25.8), (-72.0, 30.0),
                                                         (-50.0, 40.0), (-30.0, 46.0), (-12.0, 49.0), (-4.0, 49.5), (4.0, 51.9)]),
    ("США (Мексиканский залив) → Азия (Панама)", 4, "#4DA3FF", [(-97.0, 27.8), (-90.0, 26.0), (-83.0, 22.0), (-79.8, 9.5), (-79.6, 8.9),
                                                              (-90.0, 7.0), (-130.0, 15.0), (-170.0, 25.0), (160.0, 32.0), (139.6, 35.3)]),
    ("Россия (Балтика) → Европа", 9, "#B38CFF", [(28.5, 60.2), (24.0, 59.6), (20.0, 57.5), (14.0, 55.2), (12.7, 55.7), (10.5, 57.7), (8.0, 57.5),
                                              (3.5, 53.5), (4.0, 51.9)]),
    ("Россия (Чёрное море) → Средиземноморье", 6, "#B38CFF", [(37.8, 44.7), (33.0, 43.0), (29.2, 41.3), (29.0, 41.1), (26.8, 40.4), (26.2, 39.5),
                                                          (24.0, 37.5), (18.0, 36.0), (10.0, 37.5), (6.0, 37.5), (-5.6, 36.0)]),
    ("Западная Африка → Европа", 5, "#26C281", [(7.1, 4.3), (3.0, 2.0), (-5.0, 5.0), (-17.5, 15.0), (-17.5, 25.0), (-10.0, 37.0), (-9.0, 45.0), (-4.0, 49.0), (4.0, 51.9)]),
    ("Бразилия → Китай", 6, "#26C281", [(-44.0, -23.5), (-35.0, -28.0), (-10.0, -35.0), (18.5, -36.0), (50.0, -30.0), (80.0, -15.0),
                                      (96.0, -3.0), (103.0, 1.0), (110.0, 8.0), (121.0, 29.0)]),
    ("Сингапур → Япония и Корея", 7, "#FF9F1A", [(103.8, 1.2), (110.0, 8.0), (116.0, 17.0), (121.0, 24.0), (128.0, 30.0), (135.0, 34.0), (139.6, 35.3)]),
    ("Атлантика → Средиземноморье", 4, "#4DA3FF", [(-8.0, 38.0), (-5.6, 36.0), (-2.0, 36.4), (5.0, 38.0), (12.0, 37.5), (25.0, 34.5)]),
]


def route_polyline(points, per_segment=14):
    """Плотная линия вдоль маршрута: сегменты по дуге большого круга между опорными точками [(lon, lat), ...]."""
    out = []
    for (lo1, la1), (lo2, la2) in zip(points[:-1], points[1:]):
        seg = globe.great_circle(la1, lo1, la2, lo2, n=per_segment)
        out.extend(seg if not out else seg[1:])
    return out


def demo_ships():
    """Данные для анимации в globe.build: список маршрутов с числом судов. Условные, не AIS."""
    return [{"name": n, "n": cnt, "color": col, "pts": route_polyline(pts), "demo": True} for n, cnt, col, pts in ROUTES]


def _txt(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def ships_from_ais(df, class_filter="Танкер"):
    """Живые позиции -> точки для карты: каждая точка отдельным судном без маршрута."""
    d = df.dropna(subset=["lat", "lon"])
    if class_filter and "class" in d:
        d = d[d["class"] == class_filter]
    out = []
    for r in d.to_dict("records"):
        hd, cog = r.get("heading"), r.get("cog")
        heading = float(hd) if hd is not None and pd.notna(hd) and hd != 511 else float(cog) if cog is not None and pd.notna(cog) else 0.0
        sog = r.get("sog")
        out.append({"lat": float(r["lat"]), "lon": float(r["lon"]), "heading": heading,
                    "name": _txt(r.get("name")) or str(r.get("mmsi")), "sog": float(sog) if sog is not None and pd.notna(sog) else None,
                    "dest": _txt(r.get("destination")), "type": _txt(r.get("class"))})
    return out
