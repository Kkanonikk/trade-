"""Хаб открытых данных: сотни дополнительных показателей по странам и годам из репозитория Our World in Data (owid-datasets).

Сборка (`python scripts/update_hub.py`) отбирает наборы, где достаточно стран и лет, приводит названия стран к ISO3 и сохраняет
компактный снимок `assets/snapshot/owid_hub.csv.gz` и описание `owid_hub_meta.json` (название, первоисточник, ссылка, лицензия).
Здесь только чистые функции разбора и отбора; сеть и git используются в скрипте.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MIN_COUNTRIES = 50        # столько стран с данными должно быть у показателя
MIN_OBS = 400
MIN_LAST_YEAR = 1995      # ряд должен доходить хотя бы до этого года
MAX_COLS_PER_DATASET = 5
FIRST_YEAR = 1960          # более ранняя история не входит в снимок (экономит размер)

GROUPS = [
    ("Безопасность и институты", ("democra", "polity", "corruption", "freedom", "regime", "press", "rights", "governance", "electoral", "suffrage", "trust")),
    ("Здоровье", ("health", "mortality", "death", "disease", "life expectancy", "vaccin", "cancer", "hiv", "malaria", "suicide", "obesity", "smok", "alcohol", "dalys", "disability", "covid", "pollution")),
    ("Образование", ("education", "literacy", "school", "learning", "enrol", "student")),
    ("Неравенство и бедность", ("inequality", "gini", "poverty", "income share", "wealth", "wage", "pov")),
    ("Климат и энергия", ("energy", "co2", "carbon", "emission", "climate", "renewable", "nuclear", "oil", "coal", "gas", "electric", "temperature", "ghg", "methane")),
    ("Питание и голод", ("food", "calor", "crop", "yield", "agricult", "fertili", "cereal", "meat", "hunger", "diet", "land use", "forest", "fish")),
    ("Труд", ("work", "labor", "labour", "employment", "unemploy", "child labor", "hours")),
    ("Демография", ("population", "fertility", "birth", "marriage", "age", "urban", "migrat", "family", "divorce")),
    ("Экономика", ("gdp", "trade", "export", "import", "econom", "capital", "debt", "tax", "invest", "productiv", "price", "growth")),
    ("Технологии", ("internet", "technolog", "patent", "research", "mobile", "telephone", "computer", "r&d", "broadband")),
    ("Безопасность и институты", ("war", "conflict", "terror", "military", "homicide", "violence", "armed", "arms", "crime")),
]


def classify(title):
    t = title.lower()
    for name, kws in GROUPS:
        if any(k in t for k in kws):
            return name
    return "Прочее (OWID)"


def name_map(co2_snapshot):
    """Название страны (как в OWID) -> ISO3 по справочнику файла owid_co2 (строки с трёхбуквенным кодом, без агрегатов OWID_*)."""
    d = co2_snapshot[["country", "iso_code"]].dropna().drop_duplicates()
    d = d[d["iso_code"].str.len() == 3]
    m = {r.country.strip().lower(): r.iso_code for r in d.itertuples()}
    aliases = {"russian federation": "RUS", "korea, rep.": "KOR", "south korea": "KOR", "north korea": "PRK", "iran, islamic rep.": "IRN",
               "czech republic": "CZE", "slovak republic": "SVK", "united states of america": "USA", "viet nam": "VNM", "vietnam": "VNM",
               "turkey": "TUR", "democratic republic of the congo": "COD", "democratic republic of congo": "COD", "congo, dem. rep.": "COD",
               "congo, rep.": "COG", "cape verde": "CPV", "ivory coast": "CIV", "myanmar (burma)": "MMR", "burma": "MMR", "swaziland": "SWZ",
               "east timor": "TLS", "macedonia": "MKD", "the gambia": "GMB", "gambia, the": "GMB", "bahamas, the": "BHS", "egypt, arab rep.": "EGY",
               "hong kong sar, china": "HKG", "macao sar, china": "MAC", "laos": "LAO", "syrian arab republic": "SYR", "yemen, rep.": "YEM",
               "venezuela, rb": "VEN", "kyrgyz republic": "KGZ", "micronesia (country)": "FSM", "palestine": "PSE", "west bank and gaza": "PSE",
               "brunei darussalam": "BRN", "lao pdr": "LAO", "st. lucia": "LCA", "st. kitts and nevis": "KNA", "st. vincent and the grenadines": "VCT",
               "cote d'ivoire": "CIV", "côte d'ivoire": "CIV", "timor-leste": "TLS", "taiwan": "TWN", "kosovo": "XKX", "russia": "RUS"}
    for k, v in aliases.items():
        m.setdefault(k, v)
    return m


def parse_dataset(df, title, ids):
    """Длинная таблица (iso3, year, col, value) для столбцов набора, прошедших пороги. df: Entity, Year, значения...
    ids: словарь названий стран -> ISO3. Возвращает (long DataFrame, список (имя столбца, число стран, число наблюдений, последний год))."""
    if df.shape[1] < 3:
        return None, []
    cols = list(df.columns)
    ent, yr = cols[0], cols[1]
    d = df.copy()
    d["iso3"] = d[ent].astype(str).str.strip().str.lower().map(ids)
    d["year"] = pd.to_numeric(d[yr], errors="coerce")
    d = d.dropna(subset=["iso3", "year"])
    d["year"] = d["year"].astype(int)
    d = d[d["year"] >= FIRST_YEAR]
    out, info = [], []
    for c in cols[2:]:
        v = pd.to_numeric(d[c], errors="coerce")
        ok = d.assign(value=v).dropna(subset=["value"])
        if ok.empty:
            continue
        n_c, n_o, last = ok["iso3"].nunique(), len(ok), int(ok["year"].max())
        if n_c < MIN_COUNTRIES or n_o < MIN_OBS or last < MIN_LAST_YEAR:
            continue
        ok = ok.groupby(["iso3", "year"], as_index=False)["value"].mean()
        ok["col"] = c
        out.append(ok)
        info.append((c, n_c, n_o, last))
    if not out:
        return None, []
    keep = sorted(info, key=lambda x: -x[2])[:MAX_COLS_PER_DATASET]
    names = {k[0] for k in keep}
    return pd.concat([o for o in out if o["col"].iloc[0] in names]), keep


def slug_title(s):
    return re.sub(r"\s+", " ", s).strip()


def to_meta(pkg, col, n_c, n_o, last, group):
    """Описание показателя из datapackage.json набора OWID."""
    srcs = pkg.get("sources") or []
    first = srcs[0] if srcs else {}
    return {"label": slug_title(col)[:140], "dataset": slug_title(pkg.get("title") or pkg.get("name") or ""), "group": group,
            "source_name": slug_title(first.get("dataPublishedBy") or first.get("name") or "н/д")[:160],
            "source_link": (first.get("link") or "")[:300], "description": slug_title(pkg.get("description") or "")[:400],
            "countries": n_c, "obs": n_o, "last": last, "owid_id": pkg.get("id")}


def build(datasets_dir, co2_snapshot):
    """Собирает хаб из локальной копии репозитория owid-datasets: возвращает (long DataFrame с key, meta dict)."""
    ids = name_map(co2_snapshot)
    parts, meta = [], {}
    for pkg_path in sorted(Path(datasets_dir).glob("*/datapackage.json")):
        d = pkg_path.parent
        csv = d / f"{d.name}.csv"
        if not csv.exists():
            continue
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
            df = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        title = pkg.get("title") or d.name
        if re.search(r"gapminder|covid|eruption|earthquake|volcan", title, re.I):
            continue                      # Gapminder уже подключён отдельно, COVID задан по датам
        long, info = parse_dataset(df, title, ids)
        if long is None:
            continue
        group = classify(title)
        for c, n_c, n_o, last in info:
            key = f"hub:{pkg.get('id', d.name)}:{_stable(c)}"
            meta[key] = to_meta(pkg, c, n_c, n_o, last, group)
            sub = long[long["col"] == c]
            parts.append(sub.assign(key=key)[["iso3", "year", "key", "value"]])
    if not parts:
        return pd.DataFrame(columns=["iso3", "year", "key", "value"]), {}
    allp = pd.concat(parts, ignore_index=True)
    v = allp["value"].astype(float)
    scale = 10 ** (3 - np.floor(np.log10(np.abs(v).where(v != 0, 1))))        # 4 значащие цифры: размер файла вдвое меньше
    allp["value"] = (v * scale).round() / scale
    return allp, meta


def _stable(text):
    import zlib
    return zlib.crc32(text.encode("utf-8")) % 10**6
