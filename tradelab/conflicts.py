"""Карта вооружённых конфликтов: события UCDP GED (открытые данные), справочник очагов, демо и страновой показатель.

UCDP GED (Uppsala): события с гибелью людей, 1989+, координаты, оценки погибших; скачивается без ключа (ZIP с CSV) либо
через API (ключ-токен). Формат написан по кодбуку GED и проверен на образцах, но не вживую.
"""
import io
import zipfile

import numpy as np
import pandas as pd

from . import countries
from .net import SourceError, get

GED_URL = "https://ucdp.uu.se/downloads/ged/ged{version}-csv.zip"
GED_VERSIONS = ["251", "241", "231"]  # v25.1, v24.1, v23.1 (подбирается первая доступная)
VIOLENCE = {1: "Государственный конфликт", 2: "Негосударственный конфликт", 3: "Насилие против гражданских"}
COLS = ["id", "year", "type_of_violence", "conflict_name", "side_a", "side_b", "country", "region", "latitude",
        "longitude", "date_start", "best", "low", "high", "deaths_civilians"]

# Названия UCDP, которых нет в справочнике дословно -> ISO3
EXTRA = {"russia (soviet union)": "RUS", "russia": "RUS", "dr congo (zaire)": "COD", "congo": "COG", "myanmar (burma)": "MMR",
         "yemen (north yemen)": "YEM", "iran": "IRN", "syria": "SYR", "ivory coast": "CIV", "vietnam (north vietnam)": "VNM",
         "vietnam": "VNM", "cambodia (kampuchea)": "KHM", "zimbabwe (rhodesia)": "ZWE", "turkey": "TUR", "madagascar (malagasy)": "MDG",
         "kingdom of eswatini (swaziland)": "SWZ", "macedonia, fyr": "MKD", "serbia (yugoslavia)": "SRB", "bosnia-herzegovina": "BIH",
         "laos": "LAO", "south korea": "KOR", "north korea": "PRK", "venezuela": "VEN", "bolivia": "BOL", "tanzania": "TZA",
         "moldova": "MDA", "palestine": "PSE", "israel": "ISR", "united states of america": "USA", "united kingdom": "GBR",
         "gambia": "GMB", "cape verde": "CPV", "timor-leste": "TLS", "east timor": "TLS", "kyrgyz republic": "KGZ",
         "democratic republic of congo": "COD", "democratic republic of the congo": "COD", "south sudan": "SSD"}


def _norm(s):
    return str(s).replace("’", "'").replace(" *", "").replace(" (the)", "").strip().lower()


def _name_to_iso():
    t = countries.table()
    m = {}
    for col in ("name_en", "name_ru"):
        m.update({_norm(n): iso for iso, n in t[col].items() if isinstance(n, str)})
    m.update(EXTRA)
    return m


def to_iso3(name):
    if not isinstance(name, str):
        return None
    t = countries.table()
    if len(name) == 3 and name.upper() in t.index:      # уже ISO3
        return name.upper()
    m = _name_to_iso()
    for key in (_norm(name), _norm(name.split(" (")[0])):
        if key in m:
            return m[key]
    return None


def parse_ged(df):
    """Приводит таблицу GED/API к стандартным колонкам; события без координат отбрасываются; добавляет iso3 и date."""
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    miss = [c for c in ("year", "latitude", "longitude", "best") if c not in d]
    if miss:
        raise SourceError(f"UCDP GED: нет колонок {miss}")
    for c in COLS:
        if c not in d:
            d[c] = np.nan
    d = d.dropna(subset=["latitude", "longitude"])
    d["date"] = pd.to_datetime(d["date_start"], errors="coerce")
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    for c in ("best", "low", "high", "deaths_civilians", "latitude", "longitude"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    lookup = {n: to_iso3(n) for n in d["country"].dropna().unique()}
    d["iso3"] = d["country"].map(lookup)
    return d[COLS + ["date", "iso3"]].reset_index(drop=True)


def fetch_ged(version=None):
    """Скачивает ZIP GED с сайта UCDP (без ключа). Перебирает версии, пока одна не отдаст файл."""
    last = None
    for v in ([version] if version else GED_VERSIONS):
        try:
            import requests
            from .net import UA
            resp = requests.get(GED_URL.format(version=v), headers={"User-Agent": UA}, timeout=180)
            if resp.status_code != 200:
                last = f"v{v}: HTTP {resp.status_code}"
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                return parse_ged(pd.read_csv(zf.open(name), low_memory=False))
        except Exception as exc:
            last = f"v{v}: {exc}"
    raise SourceError(f"UCDP GED не скачан ({last})")


def fetch_ged_api(token, version="25.1", pagesize=1000, max_pages=50, **filters):
    """API UCDP (нужен бесплатный токен): заголовок x-ucdp-access-token."""
    rows, page = [], 0
    while page < max_pages:
        payload = get(f"https://ucdpapi.pcr.uu.se/api/gedevents/{version}", params={"pagesize": pagesize, "page": page, **filters},
                      headers={"x-ucdp-access-token": token}, as_json=True)
        res = payload.get("Result", [])
        if not res:
            break
        rows.extend(res)
        if page + 1 >= int(payload.get("TotalPages", 1)):
            break
        page += 1
    if not rows:
        raise SourceError("UCDP API: пустой ответ")
    return parse_ged(pd.DataFrame(rows))


def country_year(ged):
    """Погибшие по (iso3, год): лучшая оценка UCDP. Основа показателя conflict_deaths для кризисной модели."""
    g = ged.dropna(subset=["iso3"]).groupby(["iso3", "year"])["best"].sum().reset_index()
    g.columns = ["iso3", "year", "value"]
    g["indicator"] = "conflict_deaths"
    return g[["iso3", "year", "indicator", "value"]]


def grid(ged, cell=2.0, since_year=None):
    """События на сетке cell x cell градусов: число погибших и событий в ячейке (для тепловой карты)."""
    d = ged if since_year is None else ged[ged["year"] >= since_year]
    key = list(zip((d["latitude"] // cell * cell + cell / 2).round(2), (d["longitude"] // cell * cell + cell / 2).round(2)))
    g = d.assign(k=key).groupby("k").agg(deaths=("best", "sum"), events=("best", "size")).reset_index()
    g["lat"] = [k[0] for k in g["k"]]
    g["lon"] = [k[1] for k in g["k"]]
    return g.drop(columns="k").sort_values("deaths", ascending=False)


def summary(ged, years):
    d = ged[ged["year"].isin(years)]
    return {"events": int(len(d)), "deaths": float(d["best"].sum()), "countries": int(d["iso3"].nunique()),
            "civilians": float(d["deaths_civilians"].sum())}


# ---------------------------------------------------------------- справочник очагов (составлен вручную)
# Не данные событий: опорная таблица известных очагов по состоянию на 2025 год; может устареть.
REFERENCE = [
    ("Россия — Украина", "война", "UKR", 48.5, 37.5, 2014, "Межгосударственная война; полномасштабная фаза с февраля 2022"),
    ("Израиль — Газа и соседи", "война", "ISR", 31.4, 34.4, 2023, "Война после атаки октября 2023; затронуты Ливан, Йемен, Иран"),
    ("Судан", "внутренний конфликт", "SDN", 15.6, 32.5, 2023, "Армия против «Сил быстрого реагирования»"),
    ("Мьянма", "внутренний конфликт", "MMR", 21.0, 96.0, 2021, "Гражданская война после переворота"),
    ("Сахель (Мали, Буркина-Фасо, Нигер)", "повстанчество", "MLI", 14.5, -1.5, 2012, "Джихадистское повстанчество"),
    ("Йемен", "внутренний конфликт", "YEM", 15.4, 44.2, 2014, "Хуситы, правительство, коалиция; удары по судоходству в Красном море"),
    ("ДР Конго (восток)", "внутренний конфликт", "COD", -1.7, 29.2, 1996, "M23 и десятки вооружённых групп"),
    ("Сомали", "повстанчество", "SOM", 2.0, 45.3, 2006, "«Аш-Шабаб»"),
    ("Эфиопия", "внутренний конфликт", "ETH", 11.5, 38.0, 2020, "Амхара, Оромия, ранее Тыграй"),
    ("Сирия", "нестабильность", "SYR", 35.0, 38.5, 2011, "После смены власти в конце 2024 года ситуация остаётся нестабильной"),
    ("Афганистан — Пакистан", "повстанчество", "PAK", 33.0, 70.0, 2007, "ТТП и приграничные столкновения"),
    ("Гаити", "насилие банд", "HTI", 18.9, -72.3, 2021, "Банды контролируют значительную часть столицы"),
    ("Нигерия", "повстанчество", "NGA", 11.0, 8.0, 2009, "«Боко харам», бандитизм на северо-западе"),
    ("Мозамбик (Кабо-Делгаду)", "повстанчество", "MOZ", -12.5, 40.0, 2017, "Исламистское повстанчество"),
    ("Ливия", "нестабильность", "LBY", 32.0, 15.0, 2011, "Раскол между востоком и западом"),
    ("Кашмир (Индия — Пакистан)", "напряжённость", "IND", 34.0, 75.0, 1947, "Эпизодические столкновения у линии контроля"),
    ("Колумбия", "внутренний конфликт", "COL", 7.0, -73.0, 1964, "Вооружённые группировки после соглашения 2016 года"),
    ("Южный Судан", "внутренний конфликт", "SSD", 7.5, 30.0, 2013, "Политико-этнические столкновения"),
    ("Тайваньский пролив", "очаг напряжённости", "TWN", 24.5, 120.5, 1949, "Военное давление без боевых действий"),
    ("Корейский полуостров", "очаг напряжённости", "PRK", 38.0, 127.0, 1950, "Перемирие, а не мир"),
    ("Южно-Китайское море", "очаг напряжённости", "PHL", 12.0, 114.0, 2012, "Территориальные споры, инциденты на море"),
]
REF_COLUMNS = ["name", "type", "iso3", "lat", "lon", "since", "note"]


def reference_table():
    return pd.DataFrame(REFERENCE, columns=REF_COLUMNS)


def demo_events(years=(2022, 2023, 2024), seed=3):
    """Синтетические события вокруг очагов справочника (НЕ реальные данные): для демонстрации интерфейса."""
    rng = np.random.default_rng(seed)
    rows = []
    ref = reference_table()
    ref = ref[ref["type"] != "очаг напряжённости"]
    for r in ref.itertuples():
        intensity = {"война": 400, "внутренний конфликт": 120, "повстанчество": 80, "нестабильность": 30,
                     "насилие банд": 50, "напряжённость": 8}.get(r.type, 20)
        for y in years:
            n = max(int(rng.poisson(intensity * rng.uniform(0.5, 1.5))), 3)
            lat = r.lat + rng.normal(0, 1.3, n)
            lon = r.lon + rng.normal(0, 1.6, n)
            best = np.maximum(1, rng.lognormal(1.2, 1.1, n)).round()
            rows += [(len(rows) + i, y, 1, r.name, "A", "B", r.iso3, "", lat[i], lon[i],
                      f"{y}-{rng.integers(1, 13):02d}-{rng.integers(1, 28):02d}", best[i], best[i], best[i] * 1.5,
                      best[i] * rng.uniform(0, 0.6)) for i in range(n)]
    return parse_ged(pd.DataFrame(rows, columns=COLS))
