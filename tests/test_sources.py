"""Парсеры открытых источников на образцах ответов в формате документации.
Образцы написаны вручную по описанию форматов: это проверка логики разбора, а не живого API."""
import numpy as np
import pandas as pd
import pytest

from tradelab import fundamentals, markets, sources
from tradelab.net import SourceError

YAHOO = {"chart": {"result": [{"timestamp": [1704205800, 1704292200, 1704378600],
         "indicators": {"quote": [{"open": [1, 2, None], "high": [2, 3, None], "low": [0.5, 1.5, None],
                                   "close": [1.5, 2.5, None], "volume": [10, 20, None]}]}}], "error": None}}
STOOQ = "Date,Open,High,Low,Close,Volume\n2024-01-02,100,101,99,100.5,1000\n2024-01-03,100.5,102,100,101.5,1200\n"
MOEX = {"candles": {"columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
        "data": [[10, 11, 12, 9, 1000, 50, "2024-01-02 00:00:00", "2024-01-02 23:59:59"],
                 [11, 12, 13, 10, 1100, 60, "2024-01-03 00:00:00", "2024-01-03 23:59:59"]]}}
FRANK = {"base": "USD", "rates": {"2024-01-02": {"EUR": 0.91, "JPY": 141.0}, "2024-01-03": {"EUR": 0.92, "JPY": 142.0}}}
CBR = ('<?xml version="1.0" encoding="windows-1251"?><ValCurs ID="R01235" DateRange1="02.01.2024" DateRange2="03.01.2024">'
       '<Record Date="02.01.2024" Id="R01235"><Nominal>1</Nominal><Value>89,6914</Value></Record>'
       '<Record Date="03.01.2024" Id="R01235"><Nominal>100</Nominal><Value>9 000,50</Value></Record></ValCurs>')
FRED = "observation_date,DFF\n2024-01-01,5.33\n2024-01-02,.\n2024-01-03,5.33\n"
WB = [{"page": 1, "pages": 1, "total": 3}, [
    {"countryiso3code": "USA", "country": {"id": "US"}, "date": "2022", "value": 1.9},
    {"countryiso3code": "USA", "country": {"id": "US"}, "date": "2023", "value": None},
    {"countryiso3code": "DEU", "country": {"id": "DE"}, "date": "2022", "value": 1.8}]]


def test_yahoo_drops_null_rows_and_normalizes_dates():
    df = sources.parse_yahoo_chart(YAHOO)
    assert len(df) == 2 and df["close"].tolist() == [1.5, 2.5]
    assert (df["ts"].dt.hour == 0).all()


def test_yahoo_error_payload():
    with pytest.raises(SourceError):
        sources.parse_yahoo_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_stooq_and_bad_response():
    df = sources.parse_stooq_csv(STOOQ)
    assert df["close"].tolist() == [100.5, 101.5]
    with pytest.raises(SourceError):
        sources.parse_stooq_csv("Exceeded the daily hits limit")


def test_moex_candles():
    df = sources.parse_moex_candles(MOEX)
    assert list(df.columns) == sources.OHLC and df["close"].tolist() == [11, 12]
    assert sources.parse_moex_candles({"candles": {"columns": [], "data": []}}).empty


def test_frankfurter_long_format():
    df = sources.parse_frankfurter(FRANK)
    assert len(df) == 4 and df[df.key == "JPY"]["value"].tolist() == [141.0, 142.0]


def test_cbr_handles_comma_and_nominal():
    df = sources.parse_cbr_dynamic(CBR, "USD/RUB")
    assert df["value"].iloc[0] == pytest.approx(89.6914)
    assert df["value"].iloc[1] == pytest.approx(90.005)  # '9 000,50' / номинал 100


def test_fred_skips_missing():
    df = sources.parse_fred_csv(FRED, "DFF")
    assert len(df) == 2 and df["value"].tolist() == [5.33, 5.33]


def test_worldbank_skips_null():
    df = sources.parse_worldbank(WB)
    assert len(df) == 2 and set(df.country) == {"USA", "DEU"}
    with pytest.raises(SourceError):
        sources.parse_worldbank([{"message": "error"}])


def test_market_status_weekday_vs_weekend():
    from datetime import datetime, timezone
    tue = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)   # вторник, 11:00 в Нью-Йорке
    sat = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    assert markets.market_status("S&P 500", tue)[0] is True
    assert markets.market_status("S&P 500", sat)[0] is False
    assert markets.market_status("Nikkei 225", tue)[0] is False  # в Токио уже вечер


def test_registry_is_consistent():
    for name, (iso, yahoo, _, city, lat, lon, tz, op, cl) in markets.MARKETS.items():
        assert len(iso) == 3 and -90 <= lat <= 90 and -180 <= lon <= 180 and op < cl and yahoo


SEC = {"facts": {"us-gaap": {
    "Revenues": {"units": {"USD": [
        {"end": "2022-12-31", "val": 100.0, "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"},
        {"end": "2023-12-31", "val": 120.0, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-01"},
        {"end": "2023-03-31", "val": 28.0, "fy": 2023, "fp": "Q1", "form": "10-Q", "filed": "2023-05-01"}]}},
    "NetIncomeLoss": {"units": {"USD": [
        {"end": "2022-12-31", "val": 10.0, "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"},
        {"end": "2023-12-31", "val": 18.0, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-01"}]}},
    "StockholdersEquity": {"units": {"USD": [
        {"end": "2022-12-31", "val": 50.0, "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"},
        {"end": "2023-12-31", "val": 60.0, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-01"}]}},
    "Assets": {"units": {"USD": [
        {"end": "2023-12-31", "val": 200.0, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-01"}]}},
}}}


def test_sec_parse_only_annual_and_ratios():
    facts = fundamentals.parse_companyfacts(SEC)
    assert "10-Q" not in set(facts.form) and len(facts[facts.concept == "Выручка"]) == 2
    t = fundamentals.table(facts)
    r = fundamentals.ratios(t)
    assert r.loc[2023, "Рост выручки"] == pytest.approx(0.2)
    assert r.loc[2023, "Чистая маржа"] == pytest.approx(0.15)
    assert r.loc[2023, "ROE"] == pytest.approx(0.3)


def test_sec_later_filing_overrides_restated_value():
    payload = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"end": "2022-12-31", "val": 100.0, "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"},
        {"end": "2022-12-31", "val": 101.0, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-01"}]}}}}}
    assert fundamentals.parse_companyfacts(payload)["value"].iloc[0] == 101.0


def test_ratios_handle_missing_columns():
    r = fundamentals.ratios(pd.DataFrame({"Выручка": [1.0, 2.0]}, index=[2022, 2023]))
    assert np.isnan(r["ROE"]).all() and r["Рост выручки"].iloc[1] == pytest.approx(1.0)
