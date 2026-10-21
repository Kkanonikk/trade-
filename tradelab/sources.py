"""Открытые источники данных: разбор ответов (чистые функции, покрыты тестами) и загрузка.

ВАЖНО: форматы ответов написаны по документации источников и проверены тестами на образцах,
но не вживую (из среды разработки сетевой доступ закрыт). Проверка у себя: python scripts/check_sources.py
"""
import io
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from . import markets
from .net import SourceError, get

OHLC = ["ts", "open", "high", "low", "close", "volume"]


def _empty():
    return pd.DataFrame(columns=OHLC)


# ---------------------------------------------------------------- парсеры
def parse_yahoo_chart(payload):
    res = (payload.get("chart") or {}).get("result")
    if not res:
        err = (payload.get("chart") or {}).get("error")
        raise SourceError(f"Yahoo: пустой ответ ({err})")
    r = res[0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"ts": pd.to_datetime(r.get("timestamp", []), unit="s", utc=True),
                       "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
                       "close": q.get("close"), "volume": q.get("volume")})
    df["ts"] = df["ts"].dt.normalize()
    return df.dropna(subset=["close"]).drop_duplicates("ts").reset_index(drop=True)[OHLC]


def parse_stooq_csv(text):
    if not text or "Date" not in text.splitlines()[0]:
        raise SourceError("Stooq: неожиданный ответ (возможно, требуется ключ или символ не найден)")
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns=str.lower).rename(columns={"date": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if "volume" not in df:
        df["volume"] = float("nan")
    return df.dropna(subset=["close"])[OHLC].reset_index(drop=True)


def parse_binance_klines(rows):
    """Ответ /api/v3/klines: список списков [openTime, open, high, low, close, volume, closeTime, ...]."""
    if not rows:
        return _empty()
    df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for c in OHLC[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def parse_moex_candles(payload):
    block = payload.get("candles") or {}
    cols, data = block.get("columns", []), block.get("data", [])
    if not data:
        return _empty()
    df = pd.DataFrame(data, columns=cols)
    df["ts"] = pd.to_datetime(df["begin"], utc=True).dt.normalize()
    return df[OHLC].reset_index(drop=True)


def parse_frankfurter(payload):
    """{'rates': {'2024-01-02': {'EUR': 0.9, ...}}} -> long DataFrame ts, key, value."""
    rows = [(pd.Timestamp(d, tz="UTC"), cur, v) for d, rates in (payload.get("rates") or {}).items()
            for cur, v in rates.items()]
    return pd.DataFrame(rows, columns=["ts", "key", "value"])


def _num(text):
    """'89,6914' или '9 000,50' (в т.ч. с неразрывным пробелом) -> float."""
    return float(re.sub(r"[\s\u00a0]", "", text).replace(",", "."))


def parse_cbr_dynamic(text, key):
    """XML ЦБ РФ (XML_dynamic): курс рубля за 1 единицу валюты (учитывает номинал и запятую)."""
    root = ET.fromstring(text)
    rows = []
    for rec in root.findall("Record"):
        d = datetime.strptime(rec.attrib["Date"], "%d.%m.%Y")
        nominal = _num(rec.findtext("Nominal", "1"))
        value = _num(rec.findtext("Value"))
        rows.append((pd.Timestamp(d, tz="UTC"), key, value / nominal))
    return pd.DataFrame(rows, columns=["ts", "key", "value"])


def parse_fred_csv(text, key):
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise SourceError("FRED: неожиданный ответ")
    df.columns = ["ts", "value"]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' = нет данных
    df["key"] = key
    return df.dropna(subset=["value"])[["ts", "key", "value"]].reset_index(drop=True)


def parse_worldbank(payload):
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise SourceError("World Bank: пустой ответ")
    rows = [(r["countryiso3code"] or r["country"]["id"], int(r["date"]), r["value"])
            for r in payload[1] if r.get("value") is not None]
    return pd.DataFrame(rows, columns=["country", "year", "value"])


# ---------------------------------------------------------------- загрузка
def _span(years):
    end = datetime.now(timezone.utc).date()
    return end - timedelta(days=int(365.25 * years)), end


def fetch_yahoo(symbol, years=5, interval="1d"):
    """Свечи Yahoo (неофициальный открытый эндпоинт). interval: 1d, 1wk, 1mo, 60m (60m доступен только за ~2 года)."""
    start, end = _span(years)
    if interval == "60m":
        start = max(start, end - timedelta(days=700))
    payload = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                  params={"period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
                          "period2": int(datetime.combine(end, datetime.min.time()).timestamp()) + 86400,
                          "interval": interval}, as_json=True)
    df = parse_yahoo_chart(payload)
    if interval == "60m":          # parse_yahoo_chart режет время до даты; для часовых сохраняем время
        r = payload["chart"]["result"][0]
        ts = pd.to_datetime(r.get("timestamp", []), unit="s", utc=True)
        q = r["indicators"]["quote"][0]
        df = pd.DataFrame({"ts": ts, "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
                           "close": q.get("close"), "volume": q.get("volume")}).dropna(subset=["close"]).reset_index(drop=True)[OHLC]
    return df


def fetch_stooq(symbol, years=5):
    start, end = _span(years)
    text = get("https://stooq.com/q/d/l/", params={"s": symbol, "i": "d", "d1": start.strftime("%Y%m%d"),
                                                    "d2": end.strftime("%Y%m%d")})
    return parse_stooq_csv(text)


BINANCE_INTERVALS = {"D": "1d", "W": "1w", "MN": "1M", "H4": "4h", "H1": "1h", "M15": "15m"}


def fetch_binance(symbol, years=5, interval="1d"):
    """Свечи Binance (публичный API без ключа, до 1000 баров на запрос, постранично вперёд)."""
    start, end = _span(years)
    cursor = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    stop = int(datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000
    frames = []
    for _ in range(40):
        rows = get("https://api.binance.com/api/v3/klines", params={"symbol": symbol, "interval": interval, "startTime": cursor,
                                                                    "endTime": stop, "limit": 1000}, as_json=True)
        df = parse_binance_klines(rows)
        if df.empty:
            break
        frames.append(df)
        if len(rows) < 1000:
            break
        cursor = int(rows[-1][0]) + 1
    return pd.concat(frames).drop_duplicates("ts").reset_index(drop=True) if frames else _empty()


MOEX_INTERVALS = {"M1": 1, "H1": 60, "D": 24, "W": 7, "MN": 31}


def fetch_moex(ticker, years=5, board="TQBR", market="shares", interval=24, since=None):
    """Свечи с MOEX ISS (без токена, постранично по 500 строк). Для индексов: market='index', board='SNDX'."""
    start, end = _span(years)
    if since is not None:
        start = since
    frames, offset = [], 0
    while True:
        payload = get(f"https://iss.moex.com/iss/engines/stock/markets/{market}/boards/{board}/securities/"
                      f"{ticker}/candles.json",
                      params={"from": start.isoformat(), "till": end.isoformat(), "interval": interval,
                              "start": offset}, as_json=True)
        df = parse_moex_candles(payload)
        if df.empty:
            break
        frames.append(df)
        offset += len(df)
        if len(df) < 500:
            break
    return pd.concat(frames).drop_duplicates("ts").reset_index(drop=True) if frames else _empty()


def fetch_index(name, years=5):
    """Индекс биржи с цепочкой запасных источников: основной -> Stooq; для IMOEX сначала MOEX ISS."""
    iso, yahoo, stooq, *_ = markets.MARKETS[name]
    attempts = []
    if name == "IMOEX":
        attempts.append(("MOEX ISS", lambda: fetch_moex("IMOEX", years, "SNDX", "index")))
    attempts.append(("Yahoo", lambda: fetch_yahoo(yahoo, years)))
    if stooq:
        attempts.append(("Stooq", lambda: fetch_stooq(stooq, years)))
    errors = []
    for label, fn in attempts:
        try:
            df = fn()
            if len(df):
                return df, label
            errors.append(f"{label}: пусто")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise SourceError("; ".join(errors))


def fetch_fx_ecb(years=5, base="USD"):
    start, end = _span(years)
    payload = get(f"https://api.frankfurter.dev/v1/{start.isoformat()}..{end.isoformat()}",
                  params={"base": base, "symbols": ",".join(markets.FX_ECB)}, as_json=True)
    df = parse_frankfurter(payload)
    df["key"] = base + "/" + df["key"]
    return df


def fetch_fx_cbr(years=3):
    start, end = _span(years)
    frames = []
    for cur, code in markets.FX_CODES_CBR.items():
        text = get("https://www.cbr.ru/scripts/XML_dynamic.asp",
                   params={"date_req1": start.strftime("%d/%m/%Y"), "date_req2": end.strftime("%d/%m/%Y"),
                           "VAL_NM_RQ": code})
        frames.append(parse_cbr_dynamic(text, f"{cur}/RUB"))
    return pd.concat(frames, ignore_index=True)


def fetch_fred(series_id, key=None):
    text = get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id})
    return parse_fred_csv(text, key or series_id)


def fetch_worldbank(indicator, countries=None):
    countries = countries or markets.WB_COUNTRIES
    payload = get(f"https://api.worldbank.org/v2/country/{';'.join(countries)}/indicator/{indicator}",
                  params={"format": "json", "per_page": 20000, "date": f"{date.today().year - 25}:{date.today().year}"},
                  as_json=True)
    return parse_worldbank(payload)


def _head(url):
    import requests
    r = requests.head(url, headers={"User-Agent": "TradeLab"}, timeout=30, allow_redirects=True)
    if r.status_code != 200:
        raise SourceError(f"{url}: HTTP {r.status_code}")
    return int(r.headers.get("Content-Length", 0))


def _wb_probe():
    payload = get("https://api.worldbank.org/v2/country/all/indicator/FP.CPI.TOTL.ZG",
                  params={"format": "json", "per_page": 5, "date": "2022"}, as_json=True)
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise SourceError("World Bank: пустой ответ")
    return payload[1]


def probe_all():
    """Диагностика: минимальный запрос к каждому источнику. Возвращает [(источник, ok, сообщение)]."""
    checks = [
        ("Yahoo Finance (индексы, сырьё)", lambda: len(fetch_yahoo("^GSPC", 0.05))),
        ("Stooq (запасной для индексов)", lambda: len(fetch_stooq("^spx", 0.05))),
        ("Binance (крипта, публичный API)", lambda: len(fetch_binance("BTCUSDT", 0.05))),
        ("MOEX ISS (акции РФ, без токена)", lambda: len(fetch_moex("SBER", 0.05))),
        ("Frankfurter / ECB (валюты)", lambda: len(fetch_fx_ecb(0.05))),
        ("ЦБ РФ (курсы рубля)", lambda: len(fetch_fx_cbr(0.05))),
        ("FRED (ставки, доходности)", lambda: len(fetch_fred("DFF"))),
        ("World Bank (макро)", lambda: len(parse_worldbank(get(
            "https://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.KD.ZG",
            params={"format": "json", "per_page": 5}, as_json=True)))),
        ("SEC EDGAR (отчётность США)", lambda: len(get("https://www.sec.gov/files/company_tickers.json", as_json=True))),
        ("IMF PortWatch (проливы, танкеры)", lambda: len(get(
            "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/PortWatch_chokepoints_database/FeatureServer/0/query",
            params={"where": "1=1", "outFields": "*", "f": "json", "resultRecordCount": 3}, as_json=True)["features"])),
        ("Digitraffic (живые суда, Балтика)", lambda: len(get(
            "https://meri.digitraffic.fi/api/ais/v1/locations",
            params={"from": int((datetime.now(timezone.utc) - timedelta(minutes=2)).timestamp() * 1000)},
            headers={"Digitraffic-User": "TradeLab"}, as_json=True)["features"])),
        ("UCDP GED (события конфликтов)", lambda: _head("https://ucdp.uu.se/downloads/ged/ged251-csv.zip")),
        ("UN Comtrade (торговля, preview)", lambda: len(get("https://comtradeapi.un.org/public/v1/preview/C/A/HS",
            params={"reportercode": 842, "flowCode": "X", "period": 2022, "cmdCode": "TOTAL", "partnerCode": 156},
            as_json=True)["data"])),
        ("World Bank API: панель стран", lambda: len(_wb_probe())),
        ("IMF DataMapper (WEO)", lambda: len(get("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/USA", as_json=True)["values"])),
    ]
    out = []
    for name, fn in checks:
        try:
            out.append((name, True, f"OK, получено записей: {fn()}"))
        except Exception as exc:
            out.append((name, False, str(exc)[:240]))
    return out


