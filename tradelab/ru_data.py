"""Загрузка свечей российских бумаг: БКС (нужен токен) или MOEX ISS (бесплатно, без токена)."""
from datetime import datetime, timedelta, timezone

from . import config, sources
from .bcs import BcsClient

SOURCES = ["Авто", "БКС", "MOEX ISS"]


def resolve(source):
    """'Авто' -> БКС при наличии токена, иначе MOEX ISS."""
    if source == "Авто":
        return "БКС" if config.get("BCS_REFRESH_TOKEN") else "MOEX ISS"
    return source


def load(ticker, start, tf="D", source="Авто", client=None):
    """Свечи [ts, open, high, low, close, volume] с даты start (datetime, UTC)."""
    src = resolve(source)
    if src == "БКС":
        client = client or BcsClient()
        return client.candles(ticker, start, timeframe=tf)
    if tf not in sources.MOEX_INTERVALS:
        raise ValueError(f"MOEX ISS не отдаёт таймфрейм {tf}: доступны {', '.join(sources.MOEX_INTERVALS)}")
    return sources.fetch_moex(ticker, interval=sources.MOEX_INTERVALS[tf], since=start.date())


def default_start(years):
    return datetime.now(timezone.utc) - timedelta(days=365 * years)
