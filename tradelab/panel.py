"""Страновая панель (страна x год) из открытых источников: снимок в репозитории, World Bank, IMF DataMapper.

Единые названия показателей позволяют моделям работать с тем, что есть: чем больше источников подключено,
тем больше признаков.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import countries
from .net import SourceError, get

SNAP = Path(__file__).resolve().parent.parent / "assets" / "snapshot"
WEO_LAST_ACTUAL = 2013  # зеркало WEO в снимке: дальше идут проекции, их в обучение не берём

# std -> (подпись, код WEO/DataMapper, код World Bank, группа)
INDICATORS = {
    "gdp_growth": ("Рост реального ВВП, %", "NGDP_RPCH", "NY.GDP.MKTP.KD.ZG", "Рост"),
    "inflation": ("Инфляция (CPI), % г/г", "PCPIPCH", "FP.CPI.TOTL.ZG", "Цены"),
    "unemployment": ("Безработица, %", "LUR", "SL.UEM.TOTL.ZS", "Рост"),
    "current_account": ("Счёт текущих операций, % ВВП", "BCA_NGDPD", "BN.CAB.XOKA.GD.ZS", "Внешний сектор"),
    "gov_debt": ("Госдолг, % ВВП", "GGXWDG_NGDP", "GC.DOD.TOTL.GD.ZS", "Бюджет"),
    "fiscal_balance": ("Баланс бюджета, % ВВП", "GGXCNL_NGDP", "GC.NLD.TOTL.GD.ZS", "Бюджет"),
    "savings": ("Сбережения, % ВВП", "NGSD_NGDP", "NY.GNS.ICTR.ZS", "Рост"),
    "investment": ("Инвестиции, % ВВП", "NID_NGDP", "NE.GDI.TOTL.ZS", "Рост"),
    "output_gap": ("Разрыв выпуска, % потенциала", "NGAP_NPGDP", None, "Рост"),
    "export_growth": ("Рост объёма экспорта, %", "TX_RPCH", None, "Внешний сектор"),
    "import_growth": ("Рост объёма импорта, %", "TM_RPCH", None, "Внешний сектор"),
    "gdp_usd": ("ВВП, млрд USD", "NGDPD", "NY.GDP.MKTP.CD", "Размер"),
    "gdp_lcu": ("ВВП, млрд нац. валюты", "NGDP", "NY.GDP.MKTP.CN", "Размер"),
    "population": ("Население, млн", "LP", "SP.POP.TOTL", "Размер"),
    # доступны только через World Bank API
    "reserves_months": ("Резервы, месяцев импорта", None, "FI.RES.TOTL.MO", "Внешний сектор"),
    "ext_debt_gni": ("Внешний долг, % ВНД", None, "DT.DOD.DECT.GN.ZS", "Внешний сектор"),
    "short_debt_reserves": ("Краткосрочный долг, % резервов", None, "DT.DOD.DSTC.IR.ZS", "Внешний сектор"),
    "fdi": ("Прямые инвестиции, % ВВП", None, "BX.KLT.DINV.WD.GD.ZS", "Внешний сектор"),
    "broad_money_growth": ("Рост широких денег, %", None, "FM.LBL.BMNY.ZG", "Деньги и кредит"),
    "private_credit": ("Кредит частному сектору, % ВВП", None, "FS.AST.PRVT.GD.ZS", "Деньги и кредит"),
    "official_fx": ("Курс к USD (офиц., ср. за год)", None, "PA.NUS.FCRF", "Валюта"),
    "real_interest": ("Реальная ставка, %", None, "FR.INR.RINR", "Деньги и кредит"),
    "trade_gdp": ("Торговля, % ВВП", None, "NE.TRD.GNFS.ZS", "Внешний сектор"),
    "political_stability": ("Политическая стабильность (WGI)", None, "PV.EST", "Институты"),
    "corruption_control": ("Контроль коррупции (WGI)", None, "CC.EST", "Институты"),
    "conflict_deaths": ("Погибшие в конфликтах (UCDP), чел.", None, None, "Безопасность"),
    # живые данные World Bank: демография и продовольствие
    "pop_65plus": ("Население 65+, % населения", None, "SP.POP.65UP.TO.ZS", "Демография"),
    "pop_0014": ("Население 0–14, % населения", None, "SP.POP.0014.TO.ZS", "Демография"),
    "fertility": ("Суммарный коэффициент рождаемости", None, "SP.DYN.TFRT.IN", "Демография"),
    "life_expectancy_wb": ("Продолжительность жизни (WB), лет", None, "SP.DYN.LE00.IN", "Здоровье"),
    "undernourishment": ("Недоедание, % населения", None, "SN.ITK.DEFC.ZS", "Питание и голод"),
    "food_production_index": ("Индекс производства продовольствия", None, "AG.PRD.FOOD.XD", "Питание и голод"),
    "cereal_yield": ("Урожайность зерновых, кг/га", None, "AG.YLD.CREL.KG", "Питание и голод"),
    "food_imports_share": ("Импорт продовольствия, % импорта товаров", None, "TM.VAL.FOOD.ZS.UN", "Питание и голод"),
}
LABELS = {k: v[0] for k, v in INDICATORS.items()}
WEO_TO_STD = {v[1]: k for k, v in INDICATORS.items() if v[1]}
WB_TO_STD = {v[2]: k for k, v in INDICATORS.items() if v[2]}
SCALE = {"population": 1.0}  # WEO: население в млн уже; WB: в людях (делится ниже)


def _gz(name):
    return pd.read_csv(SNAP / f"{name}.csv.gz")


def snapshot_info():
    path = SNAP / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"updated": "н/д", "files": {}}


def _wide(df, country, year, indicator, value):
    out = df.rename(columns={country: "iso3", year: "year", indicator: "ind", value: "value"})
    out["value"] = pd.to_numeric(out["value"].astype(str).str.replace(",", ""), errors="coerce")
    return out.pivot_table(index=["iso3", "year"], columns="ind", values="value")


def snapshot_panel():
    """Реальные данные из встроенного снимка: IMF WEO (факты до 2013) + World Bank (ВВП, инфляция, население до 2023)."""
    weo = _wide(_gz("weo_values"), "Country", "Year", "Indicator", "Value")
    weo = weo[weo.index.get_level_values("year") <= WEO_LAST_ACTUAL]
    weo = weo.rename(columns=WEO_TO_STD)[[c for c in weo.rename(columns=WEO_TO_STD).columns if c in INDICATORS]]
    gdp = _gz("gdp").rename(columns={"Country Code": "iso3", "Year": "year", "Value": "gdp_usd"})
    infl = _gz("inflation").rename(columns={"Country Code": "iso3", "Year": "year", "Inflation": "inflation"})
    pop = _gz("population").rename(columns={"Country Code": "iso3", "Year": "year", "Value": "population"})
    wb = (gdp.set_index(["iso3", "year"])[["gdp_usd"]] / 1e9)
    wb = wb.join(infl.set_index(["iso3", "year"])[["inflation"]], how="outer")
    wb = wb.join(pop.set_index(["iso3", "year"])[["population"]] / 1e6, how="outer")
    panel = weo.combine_first(wb)  # WEO в приоритете там, где есть, WB заполняет пропуски и свежие годы
    return panel[panel.index.get_level_values("iso3").isin(countries.table().index)].sort_index()


def parse_worldbank_bulk(payload, std):
    """Страница JSON World Bank -> DataFrame [iso3, year, value]; агрегаты (без ISO3) отбрасываются."""
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise SourceError("World Bank: пустой ответ")
    rows = [(r["countryiso3code"], int(r["date"]), r["value"]) for r in payload[1]
            if r.get("value") is not None and r.get("countryiso3code")]
    df = pd.DataFrame(rows, columns=["iso3", "year", "value"])
    df["ind"] = std
    return df


def fetch_worldbank_indicator(code, std, start=1990):
    frames, page = [], 1
    while True:
        payload = get(f"https://api.worldbank.org/v2/country/all/indicator/{code}",
                      params={"format": "json", "per_page": 20000, "page": page,
                              "date": f"{start}:{date.today().year}"}, as_json=True)
        frames.append(parse_worldbank_bulk(payload, std))
        if page >= int(payload[0].get("pages", 1)):
            break
        page += 1
    df = pd.concat(frames)
    if std == "population":
        df["value"] = df["value"] / 1e6
    if std in ("gdp_usd", "gdp_lcu"):
        df["value"] = df["value"] / 1e9
    return df


def parse_datamapper(payload, std, last_actual_year):
    """IMF DataMapper: {'values': {код: {ISO3: {'1990': v}}}} -> [iso3, year, value, ind]; проекции отбрасываются."""
    values = (payload or {}).get("values") or {}
    rows = []
    for series in values.values():
        for iso3, by_year in series.items():
            for y, v in by_year.items():
                if v is not None and int(y) <= last_actual_year and len(iso3) == 3:
                    rows.append((iso3, int(y), float(v)))
    if not rows:
        raise SourceError("IMF DataMapper: пустой ответ")
    df = pd.DataFrame(rows, columns=["iso3", "year", "value"])
    df["ind"] = std
    return df


def fetch_datamapper(code, std):
    payload = get(f"https://www.imf.org/external/datamapper/api/v1/{code}", as_json=True)
    return parse_datamapper(payload, std, date.today().year - 1)


def refresh_live(progress=None):
    """Все показатели панели из World Bank и IMF DataMapper. Возвращает (панель, отчёт [(что, ok, сообщение)])."""
    frames, report = [], []
    jobs = []
    for std, (label, weo, wb, _) in INDICATORS.items():
        if wb:
            jobs.append((f"WB · {label}", lambda wb=wb, std=std: fetch_worldbank_indicator(wb, std)))
        if weo:
            jobs.append((f"IMF · {label}", lambda weo=weo, std=std: fetch_datamapper(weo, std)))
    for i, (label, fn) in enumerate(jobs):
        try:
            df = fn()
            frames.append(df)
            report.append((label, True, f"{len(df)} значений"))
        except Exception as exc:
            report.append((label, False, str(exc)[:200]))
        if progress:
            progress((i + 1) / len(jobs), label)
    if not frames:
        return pd.DataFrame(), report
    long = pd.concat(frames)
    # World Bank в приоритете (фактические данные), IMF заполняет пропуски
    long["prio"] = long["ind"].map(lambda _: 0)
    wide = long.pivot_table(index=["iso3", "year"], columns="ind", values="value", aggfunc="first")
    return wide.sort_index(), report


def with_conflicts(panel, ged_country_year):
    """Добавляет в панель показатель conflict_deaths из таблицы UCDP (iso3, year, indicator, value)."""
    cd = ged_country_year.pivot_table(index=["iso3", "year"], columns="indicator", values="value")
    return panel.drop(columns=["conflict_deaths"], errors="ignore").join(cd, how="outer").sort_index()


def to_long(panel):
    long = panel.stack().rename("value").reset_index()
    long.columns = ["iso3", "year", "indicator", "value"]
    return long


def from_long(long):
    return long.pivot_table(index=["iso3", "year"], columns="indicator", values="value").sort_index()


def coverage(panel):
    """Доля стран с данными по каждому показателю за последние 5 лет панели."""
    last = panel.index.get_level_values("year").max()
    recent = panel[panel.index.get_level_values("year") >= last - 4]
    n = panel.index.get_level_values("iso3").nunique()
    return (recent.notna().groupby(level="iso3").any().sum() / n).rename("coverage")


