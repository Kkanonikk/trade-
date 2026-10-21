"""Глобальный слой данных: единая таблица страна x год по сотням показателей из открытых источников + метаданные.

Источники: Gapminder (ООН, ВОЗ, Всемирный банк и др.), OWID (CO₂, энергия), IMF WEO и World Bank (панель снимка или живые данные).
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from . import countries, panel as panel_mod, registry

SNAP = panel_mod.SNAP


@lru_cache(maxsize=1)
def _gapminder_wide():
    gm = pd.read_csv(SNAP / "gapminder.csv.gz")
    gm = gm[gm["geo"].isin(countries.table().index)]
    gm["key"] = "gm:" + gm["indicator"]
    wide = gm.pivot_table(index=["geo", "year"], columns="key", values="value")
    wide.index.names = ["iso3", "year"]
    return wide


@lru_cache(maxsize=1)
def _owid_wide():
    frames = []
    for fname, prefix, cols in (("owid_co2", "co2", registry.OWID_CO2), ("owid_energy", "en", registry.OWID_ENERGY)):
        d = pd.read_csv(SNAP / f"{fname}.csv.gz")
        d = d[d["iso_code"].isin(countries.table().index)].rename(columns={"iso_code": "iso3"})
        d = d.set_index(["iso3", "year"])[list(cols)]
        d.columns = [f"{prefix}:{c}" for c in d.columns]
        frames.append(d)
    return pd.concat(frames, axis=1)


@lru_cache(maxsize=1)
def _panel_wide():
    p = panel_mod.snapshot_panel()
    meta = {k: v for k, v in panel_mod.INDICATORS.items()}
    p = p[[c for c in p.columns if c in meta]]
    p.columns = [f"wp:{c}" for c in p.columns]
    return p


@lru_cache(maxsize=1)
def _hub_wide():
    path = SNAP / "owid_hub.csv.gz"
    if not path.exists():
        return None
    h = pd.read_csv(path, dtype={"value": "float32", "year": "int16", "iso3": "category", "key": "category"})
    ok = set(countries.table().index)
    h = h[h["iso3"].isin(ok)]
    keys = list(h["key"].cat.categories)
    blocks = []
    for i in range(0, len(keys), 60):                 # по частям: пиковая память заметно ниже, чем у одного pivot
        sub = h[h["key"].isin(keys[i:i + 60])].drop_duplicates(["iso3", "year", "key"])
        sub = sub.assign(iso3=sub["iso3"].astype(str), key=sub["key"].astype(str))
        blocks.append(sub.set_index(["iso3", "year", "key"])["value"].unstack("key"))
    del h
    wide = blocks[0].join(blocks[1:], how="outer") if len(blocks) > 1 else blocks[0]
    wide.columns = [str(c) for c in wide.columns]
    return wide.astype("float32")


def meta():
    """Метаданные всех доступных показателей: ключ -> {label, unit, group, better, source, projection_from}."""
    m = dict(registry.all_meta())
    for std, (label, weo, wb, group) in panel_mod.INDICATORS.items():
        m[f"wp:{std}"] = {"label": label, "unit": "", "group": group, "better": "none",
                         "source": "imf_weo" if weo else "worldbank", "projection_from": None}
    return m


def load(extra=None):
    """Единая широкая таблица (iso3, year) x показатели. extra: дополнительная панель (живые WB/IMF)."""
    parts = [_gapminder_wide(), _owid_wide(), _panel_wide()]
    hubw = _hub_wide()
    if hubw is not None:
        parts.append(hubw)
    if extra is not None and len(extra):
        e = extra.copy()
        e.columns = [f"wp:{c}" for c in e.columns]
        parts.append(e)
    wide = parts[0]
    for p in parts[1:]:
        wide = wide.join(p, how="outer", rsuffix="_dup")
    wide = wide[[c for c in wide.columns if not c.endswith("_dup")]]
    return wide.sort_index()


def available(wide):
    """Показатели с данными, отсортированные по группам: DataFrame key, label, group, unit, source, countries, years."""
    m = meta()
    rows = []
    for k in wide.columns:
        if k not in m:
            continue
        s = wide[k].dropna()
        if s.empty:
            continue
        yrs = s.index.get_level_values("year")
        rows.append({"key": k, **{x: m[k][x] for x in ("label", "group", "unit", "source", "better")},
                     "countries": s.index.get_level_values("iso3").nunique(), "first": int(yrs.min()), "last": int(yrs.max())})
    return pd.DataFrame(rows).sort_values(["group", "label"]).reset_index(drop=True)


def series(wide, key, iso3):
    """Ряд страны по показателю: Series год -> значение."""
    if key not in wide.columns or iso3 not in wide.index.get_level_values("iso3"):
        return pd.Series(dtype=float)
    return wide.loc[iso3, key].dropna()


def cross_section(wide, key, year):
    """Значения всех стран за год."""
    if key not in wide.columns:
        return pd.Series(dtype=float)
    try:
        return wide.xs(year, level="year")[key].dropna()
    except KeyError:
        return pd.Series(dtype=float)


def last_actual_year(key):
    pf = meta().get(key, {}).get("projection_from")
    return (pf - 1) if pf else None


def source_line(keys):
    return registry.cite(keys)


def from_user_csv(df, name):
    """Пользовательский ряд из CSV: колонки country/iso3 (код или название), year, value. Возвращает (Series (iso3, year), число отброшенных строк)."""
    cols = {c.lower().strip(): c for c in df.columns}
    ckey = next((cols[c] for c in ("iso3", "country", "страна", "iso") if c in cols), None)
    ykey = next((cols[c] for c in ("year", "год") if c in cols), None)
    vkey = next((cols[c] for c in ("value", "значение") if c in cols), None)
    if not (ckey and ykey and vkey):
        raise ValueError("Нужны колонки: country (или iso3), year, value")
    t = countries.table()
    names = {}
    for iso in t.index:
        for n in (iso, countries.short_ru(iso), str(t.loc[iso, "name_en"])):
            names[str(n).lower().strip()] = iso
    iso = df[ckey].astype(str).str.lower().str.strip().map(names)
    out = pd.DataFrame({"iso3": iso, "year": pd.to_numeric(df[ykey], errors="coerce"), "v": pd.to_numeric(df[vkey], errors="coerce")})
    bad = int(out.isna().any(axis=1).sum())
    out = out.dropna()
    out["year"] = out["year"].astype(int)
    if out.empty:
        raise ValueError("Ни одна строка не распознана: проверь названия стран и числа")
    ser = out.groupby(["iso3", "year"])["v"].mean().rename(name)
    return ser, bad
