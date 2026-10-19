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
