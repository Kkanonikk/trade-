"""Панель подключений: лёгкая проверка каждого онлайн-источника (доступность, задержка) и свежесть данных в локальной базе.
Проверка делает один короткий запрос к источнику; формат ответов глубже проверяет scripts/check_sources.py."""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import config, store

UA = "TradeLab/1.0 (open-data research)"
TIMEOUT = 9
# (id, название, категория, URL, параметры, нужен ключ (env) или None, проверка текста ответа или None)
PROBES = [
    ("yahoo", "Yahoo Finance", "Рынки", "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC", {"range": "1d", "interval": "1d"}, None, "chart"),
    ("stooq", "Stooq", "Рынки", "https://stooq.com/q/d/l/", {"s": "^spx", "i": "d"}, None, "Date"),
    ("binance", "Binance (крипта)", "Рынки", "https://api.binance.com/api/v3/ping", {}, None, None),
    ("moex", "MOEX ISS", "Рынки", "https://iss.moex.com/iss/engines.json", {"iss.meta": "off"}, None, "engines"),
    ("ecb", "Frankfurter / ECB", "Валюты", "https://api.frankfurter.app/latest", {}, None, "rates"),
    ("cbr", "ЦБ РФ", "Валюты", "https://www.cbr.ru/scripts/XML_daily.asp", {}, None, "ValCurs"),
    ("fred", "FRED (ставки)", "Макро", "https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": "DFF"}, None, "DATE"),
    ("worldbank", "World Bank API", "Макро", "https://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.KD.ZG", {"format": "json", "per_page": 1}, None, "value"),
    ("imf", "IMF DataMapper", "Макро", "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/USA", {}, None, "values"),
    ("comtrade", "UN Comtrade", "Торговля", "https://comtradeapi.un.org/public/v1/preview/C/A/HS",
     {"reportercode": 842, "flowCode": "X", "period": 2022, "cmdCode": "TOTAL", "partnerCode": 156}, None, "data"),
    ("portwatch", "IMF PortWatch", "Море", "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/PortWatch_chokepoints_database/FeatureServer/0/query",
     {"where": "1=1", "outFields": "portid", "f": "json", "resultRecordCount": 1}, None, "features"),
    ("digitraffic", "Digitraffic (суда, Балтика)", "Море", "https://meri.digitraffic.fi/api/ais/v1/locations", {"from": 0}, None, None),
    ("ucdp", "UCDP (конфликты)", "Конфликты", "https://ucdp.uu.se/downloads/", {}, None, None),
    ("sec", "SEC EDGAR", "Отчётность", "https://www.sec.gov/files/company_tickers.json", {}, None, "ticker"),
    ("github", "GitHub (зеркала данных)", "Данные", "https://raw.githubusercontent.com/owid/owid-datasets/master/README.md", {}, None, None),
    ("noaa", "NOAA (климат)", "Климат", "https://gml.noaa.gov/ccgg/trends/data.html", {}, None, None),
]


def probe_one(spec):
    sid, name, cat, url, params, key, need = spec
    t0 = time.time()
    try:
        headers = {"User-Agent": UA, "Digitraffic-User": "TradeLab"}
        p = dict(params)
        if sid == "digitraffic":
            p["from"] = int((datetime.now(timezone.utc) - timedelta(minutes=2)).timestamp() * 1000)
        r = requests.get(url, params=p, headers=headers, timeout=TIMEOUT, stream=True)
        body = r.raw.read(4000, decode_content=True).decode("utf-8", "ignore") if r.status_code == 200 else ""
        ms = int((time.time() - t0) * 1000)
        if r.status_code != 200:
            return {"id": sid, "Источник": name, "Категория": cat, "Статус": "ошибка", "мс": ms, "Сообщение": f"HTTP {r.status_code}"}
        if need and need.lower() not in body.lower():
            return {"id": sid, "Источник": name, "Категория": cat, "Статус": "формат?", "мс": ms, "Сообщение": f"ответ получен, но нет «{need}»: формат мог измениться"}
        return {"id": sid, "Источник": name, "Категория": cat, "Статус": "онлайн", "мс": ms, "Сообщение": "ок"}
    except Exception as exc:
        return {"id": sid, "Источник": name, "Категория": cat, "Статус": "ошибка", "мс": int((time.time() - t0) * 1000), "Сообщение": str(exc)[:140]}


def probe_all(workers=8):
    """Параллельная проверка всех источников. Возвращает DataFrame со статусами."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(probe_one, PROBES))
    return pd.DataFrame(rows)


def _q(con, sql):
    try:
        return con.execute(sql).fetchone()
    except Exception:
        return None


def local_freshness():
    """Что и до какой даты лежит в локальной базе: список {Данные, Записей, Последняя дата}."""
    rows = []
    try:
        con = store.connect_events()
    except Exception:
        return pd.DataFrame(columns=["Данные", "Записей", "Последняя дата"])
    specs = [("Котировки бумаг", "SELECT COUNT(*), MAX(ts) FROM candles"),
             ("Мировые рынки: индексы", "SELECT COUNT(*), MAX(ts) FROM series WHERE kind='index'"),
             ("Сырьё и риск-индикаторы", "SELECT COUNT(*), MAX(ts) FROM series WHERE kind='cross'"),
             ("Валюты", "SELECT COUNT(*), MAX(ts) FROM series WHERE kind='fx'"),
             ("Ставки и доходности", "SELECT COUNT(*), MAX(ts) FROM series WHERE kind IN ('rate','yield')"),
             ("Проливы (PortWatch)", "SELECT COUNT(*), MAX(ts) FROM series WHERE kind='choke'"),
             ("Макро-панель World Bank/IMF", "SELECT COUNT(*), MAX(year) FROM panel"),
             ("Торговые потоки", "SELECT COUNT(*), MAX(year) FROM trade"),
             ("События конфликтов (UCDP)", "SELECT COUNT(*), MAX(date) FROM ged")]
    for name, sql in specs:
        r = _q(con, sql)
        n, last = (r or (0, None))
        rows.append({"Данные": name, "Записей": int(n or 0), "Последняя дата": str(last)[:10] if last else "нет"})
    return pd.DataFrame(rows)


def env_status():
    return {k: bool(config.get(k)) for k in ("FRED_API_KEY", "AISSTREAM_KEY", "UCDP_TOKEN", "TRADELAB_CONTACT")}
