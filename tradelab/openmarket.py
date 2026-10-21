"""Котировки мирового рынка из открытых API без ключей и токенов.

Цепочка источников по типу инструмента (при сбое берётся следующий):
- акции и ETF мира: Yahoo Finance -> Stooq (тикер вида AAPL.US);
- крипта (BTC-USD): Yahoo Finance -> Binance (BTCUSDT);
- российские бумаги: суффикс .MOEX (например SBER.MOEX) -> MOEX ISS.
Форматы ответов написаны по документации и проверены тестами на образцах; живые запросы из среды разработки не проверялись:
сверить у себя можно командой `python scripts/check_sources.py`.
"""
from datetime import datetime, timedelta, timezone

from . import sources
from .net import SourceError

TIMEFRAMES = ["D", "W", "MN", "H1"]
YAHOO_INTERVAL = {"D": "1d", "W": "1wk", "MN": "1mo", "H1": "60m"}
SOURCES = ["Авто", "Yahoo Finance", "Stooq", "Binance", "MOEX ISS"]


def _binance_symbol(ticker):
    base, _, quote = ticker.partition("-")
    return f"{base}{'USDT' if quote in ('USD', 'USDT', '') else quote}"


def _stooq_symbol(ticker):
    return ticker.lower() + ("" if "." in ticker else ".us")


def attempts(ticker, tf, years, since=None):
    """Список (название источника, функция) для тикера в порядке попыток."""
    out = []
    if ticker.endswith(".MOEX"):
        if tf in sources.MOEX_INTERVALS:
            sym = ticker[:-5]
            out.append(("MOEX ISS", lambda: sources.fetch_moex(sym, years, interval=sources.MOEX_INTERVALS[tf], since=since)))
        return out
    out.append(("Yahoo Finance", lambda: sources.fetch_yahoo(ticker, years, YAHOO_INTERVAL[tf])))
    if ticker.endswith(("-USD", "-USDT")) and tf in sources.BINANCE_INTERVALS:
        out.append(("Binance", lambda: sources.fetch_binance(_binance_symbol(ticker), years, sources.BINANCE_INTERVALS[tf])))
    elif tf == "D":
        out.append(("Stooq", lambda: sources.fetch_stooq(_stooq_symbol(ticker), years)))
    return out


def load(ticker, start, tf="D", source="Авто"):
    """Свечи [ts, open, high, low, close, volume] с даты start (datetime, UTC). Возвращает (DataFrame, источник)."""
    if tf not in YAHOO_INTERVAL:
        raise ValueError(f"Таймфрейм {tf} не поддерживается: доступны {', '.join(TIMEFRAMES)}")
    years = max((datetime.now(timezone.utc) - start).days / 365.25, 0.05)
    chain = attempts(ticker, tf, years, since=start.date())
    if source != "Авто":
        chain = [a for a in chain if a[0] == source]
        if not chain:
            raise ValueError(f"Источник {source} не подходит для {ticker} на таймфрейме {tf}")
    errors = []
    for label, fn in chain:
        try:
            df = fn()
            if len(df):
                df = df[df["ts"] >= start.replace(tzinfo=timezone.utc) if start.tzinfo is None else df["ts"] >= start]
                return df.reset_index(drop=True), label
            errors.append(f"{label}: пусто")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise SourceError(f"{ticker}: " + "; ".join(errors))


def default_start(years):
    return datetime.now(timezone.utc) - timedelta(days=365 * years)
