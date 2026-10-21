"""Строит морские маршруты танкеров без заходов на сушу: растр суши из assets/countries.geojson, A* по морским клеткам между
опорными точками из tradelab/maritime.ROUTES, спрямление по прямой видимости. Результат: assets/sea_routes.json.
Запуск: python scripts/build_sea_routes.py   (около минуты)."""
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tradelab import maritime  # noqa: E402

RES = 4                                  # клеток на градус у точного растра (0.25°)
W, H = 360 * RES, 180 * RES

# Узкие проходы (каналы и проливы уже карты): линии (lon, lat), вдоль которых суша «вырезается»
GATES = [
    [(55.0, 26.3), (56.3, 26.5), (57.2, 26.2), (58.6, 24.8), (60.5, 22.5)],                       # Ормуз → Оманский залив
    [(101.0, 2.9), (102.5, 2.0), (103.8, 1.2), (104.4, 1.2)],                                      # Малакка → Сингапур
    [(100.0, 6.0), (99.2, 6.8), (98.0, 7.0), (95.0, 6.0)],                                          # вход с севера Суматры
    [(32.58, 29.9), (32.55, 30.5), (32.35, 31.3), (32.3, 31.5)],                                   # Суэцкий канал
    [(33.2, 28.3), (32.8, 29.3), (32.58, 29.9)],                                                    # Суэцкий залив
    [(43.0, 12.4), (43.5, 12.7), (44.5, 12.4)],                                                     # Баб-эль-Мандеб
    [(-5.0, 35.9), (-5.7, 35.95), (-6.5, 36.0)],                                                    # Гибралтар
    [(26.2, 40.0), (26.4, 40.2), (27.0, 40.4), (28.0, 40.8), (29.0, 41.0), (29.1, 41.2), (29.2, 41.3), (30.0, 41.8)],  # Дарданеллы, Мраморное, Босфор
    [(1.0, 51.1), (1.5, 51.0), (2.0, 51.0)],                                                        # Па-де-Кале
    [(10.5, 57.7), (11.0, 57.4), (12.0, 56.3), (12.65, 55.9), (12.7, 55.4), (12.9, 55.1), (13.8, 54.9)],  # Эресунн
    [(36.5, 45.3), (36.6, 45.2), (36.9, 45.1)],                                                     # Керченский пролив
    [(-79.9, 9.35), (-79.7, 9.15), (-79.55, 8.95), (-79.4, 8.8)],                                   # Панамский канал
    [(-90.0, 7.0), (-84.0, 7.0), (-79.6, 8.7)],                                                     # Панамский залив (подход)
    [(115.7, -8.3), (115.7, -8.9)],                                                                 # Ломбок
    [(105.8, -5.9), (105.8, -6.3)],                                                                 # Зондский пролив
]
ROUTE_ENDS_SNAP = 4.0                    # порт на суше сдвигается к ближайшей морской клетке не дальше, чем на столько градусов


def land_mask():
    gj = json.loads((ROOT / "assets" / "countries.geojson").read_text(encoding="utf-8"))
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    px = lambda lon, lat: ((lon + 180) * RES, (90 - lat) * RES)
    for f in gj["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"] if g["type"] == "MultiPolygon" else []
        for poly in polys:
            d.polygon([px(*p[:2]) for p in poly[0]], fill=1)
            for hole in poly[1:]:
                d.polygon([px(*p[:2]) for p in hole], fill=0)
    m = np.array(img, dtype=bool)
    img2 = Image.fromarray(m.astype(np.uint8))
    d2 = ImageDraw.Draw(img2)
    for line in GATES:
        d2.line([px(*p) for p in line], fill=0, width=2)
    return np.array(img2, dtype=bool)


def cell(lon, lat):
    return int(np.clip((90 - lat) * RES, 0, H - 1)), int(((lon + 180) * RES) % W)


def center(r, c):
    return c / RES - 180 + 0.5 / RES, 90 - r / RES - 0.5 / RES


def hav(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def snap(land, lon, lat):
    r0, c0 = cell(lon, lat)
    if not land[r0, c0]:
        return r0, c0
    best = None
    rad = int(ROUTE_ENDS_SNAP * RES)
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            r, c = r0 + dr, (c0 + dc) % W
            if 0 <= r < H and not land[r, c]:
                dist = dr * dr + dc * dc
                if best is None or dist < best[0]:
                    best = (dist, r, c)
    if best is None:
        raise ValueError(f"нет моря рядом с {lon},{lat}")
    return best[1], best[2]


def astar(land, start, goal, coarse=2):
    """A* по клеткам (шаг coarse*0.25°), 8 соседей, склейка по долготе. Возвращает список (r, c)."""
    gl = center(*goal)
    sr, sc = start
    open_ = [(0.0, sr, sc)]
    g = {(sr, sc): 0.0}
    prev = {}
    moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while open_:
        _, r, c = heapq.heappop(open_)
        if abs(r - goal[0]) <= 1 and min(abs(c - goal[1]), W - abs(c - goal[1])) <= 1:
            path = [(r, c)]
            while (r, c) in prev:
                r, c = prev[(r, c)]
                path.append((r, c))
            return [start] + path[::-1][1:] + [goal]
        lo1, la1 = center(r, c)
        for dr, dc in moves:
            nr, nc = r + dr, (c + dc) % W
            if not 0 <= nr < H or land[nr, nc]:
                continue
            lo2, la2 = center(nr, nc)
            ng = g[(r, c)] + hav(lo1, la1, lo2, la2)
            if ng < g.get((nr, nc), 1e18):
                g[(nr, nc)] = ng
                prev[(nr, nc)] = (r, c)
                heapq.heappush(open_, (ng + hav(lo2, la2, gl[0], gl[1]), nr, nc))
    raise ValueError("маршрут не найден")


def clear(land, a, b, step=0.08):
    """Нет ли суши на отрезке a-b (lon, lat) с учётом перехода через 180°."""
    dlon = (b[0] - a[0] + 180) % 360 - 180
    n = max(int(max(abs(dlon), abs(b[1] - a[1])) / step), 1)
    for i in range(n + 1):
        t = i / n
        lon, lat = a[0] + dlon * t, a[1] + (b[1] - a[1]) * t
        r, c = cell(((lon + 180) % 360) - 180, lat)
        if land[r, c]:
            return False
    return True


def simplify(land, pts):
    """Спрямление: из каждой точки идём к самой дальней, видимой по морю."""
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not clear(land, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def build_route(land, waypoints):
    full = []
    snapped = [center(*snap(land, lo, la)) for lo, la in waypoints]
    for a, b in zip(snapped[:-1], snapped[1:]):
        cells = astar(land, cell(*a), cell(*b))
        pts = [center(r, c) for r, c in cells]
        seg = simplify(land, pts)
        full.extend(seg if not full else seg[1:])
    return full


def densify(land, pts, per=0.5):
    """Равномерные точки вдоль маршрута по дуге между опорными (шаг ≈ per градуса), без выхода на сушу (иначе берём прямую линию)."""
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        dlon = (b[0] - a[0] + 180) % 360 - 180
        n = max(int(max(abs(dlon), abs(b[1] - a[1])) / per), 1)
        for i in range(1, n + 1):
            t = i / n
            out.append((((a[0] + dlon * t + 180) % 360) - 180, a[1] + (b[1] - a[1]) * t))
    return out


def main():
    land = land_mask()
    np.savez_compressed(ROOT / "assets" / "sea_mask.npz", land=np.packbits(land), shape=np.array(land.shape))
    routes = {}
    for name, cnt, col, wps in maritime.ROUTES:
        pts = densify(land, build_route(land, wps))
        routes[name] = [[round(x, 3), round(y, 3)] for x, y in pts]
        bad = sum(land[cell(*p)] for p in pts)
        print(f"{name}: точек {len(pts)}, на суше {bad}")
    (ROOT / "assets" / "sea_routes.json").write_text(json.dumps(routes, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
