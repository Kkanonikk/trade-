"""Справочник стран: ISO-коды, названия (рус/англ), валюта, регион, координаты центра."""
from functools import lru_cache
from pathlib import Path

import pandas as pd

ASSETS = Path(__file__).resolve().parent.parent / "assets"


@lru_cache(maxsize=1)
def table():
    df = pd.read_csv(ASSETS / "countries.csv", keep_default_na=False, na_values=[""])
    df["name_ru"] = df["name_ru"].fillna(df["name_en"])
    df = df.drop_duplicates("iso3").set_index("iso3")
    return df


def name(iso3, lang="ru"):
    t = table()
    return t.loc[iso3, "name_ru" if lang == "ru" else "name_en"] if iso3 in t.index else iso3


SHORT_RU = {
    "RUS": "Россия", "USA": "США", "GBR": "Великобритания", "KOR": "Южная Корея", "PRK": "КНДР", "VEN": "Венесуэла",
    "IRN": "Иран", "SYR": "Сирия", "TZA": "Танзания", "LAO": "Лаос", "TUR": "Турция", "BOL": "Боливия",
    "MDA": "Молдова", "VNM": "Вьетнам", "COD": "ДР Конго", "COG": "Конго", "CZE": "Чехия", "MKD": "Северная Македония",
    "FSM": "Микронезия", "PSE": "Палестина", "HKG": "Гонконг", "MAC": "Макао", "TWN": "Тайвань", "SWZ": "Эсватини",
    "CIV": "Кот-д'Ивуар", "BRN": "Бруней", "KGZ": "Киргизия", "CPV": "Кабо-Верде", "VAT": "Ватикан",
}


def short_ru(iso3):
    """Короткое русское название без канцелярита ('Российская Федерация' -> 'Россия')."""
    return SHORT_RU.get(iso3) or name(iso3)


def numeric_to_iso3():
    t = table()
    return {int(n): iso for iso, n in t["numeric"].dropna().items()}


def iso3_to_numeric():
    t = table()
    return {iso: int(n) for iso, n in t["numeric"].dropna().items()}


def centroid(iso3):
    t = table()
    if iso3 in t.index and pd.notna(t.loc[iso3, "lat"]):
        return float(t.loc[iso3, "lat"]), float(t.loc[iso3, "lon"])
    return None


def sovereigns():
    """Страны с координатами (для карты и глобуса)."""
    t = table()
    return t[t["lat"].notna()].index.tolist()
