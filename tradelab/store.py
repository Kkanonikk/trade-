"""Локальное хранилище свечей в SQLite (один файл data/market.db)."""
import sqlite3

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    ticker TEXT NOT NULL, tf TEXT NOT NULL, ts TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, tf, ts)
)
"""


def connect(path=None):
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)        # параллельная загрузка пишет из нескольких потоков
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(SCHEMA)
    return con


def upsert(con, ticker, tf, df):
    rows = [(ticker, tf, ts.isoformat(), o, h, l, c, v)
            for ts, o, h, l, c, v in
            df[["ts", "open", "high", "low", "close", "volume"]].itertuples(index=False)]
    con.executemany("INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def last_ts(con, ticker, tf):
    row = con.execute("SELECT MAX(ts) FROM candles WHERE ticker=? AND tf=?",
                      (ticker, tf)).fetchone()
    return pd.Timestamp(row[0]) if row and row[0] else None


def load(con, ticker, tf):
    df = pd.read_sql_query(
        "SELECT ts, open, high, low, close, volume FROM candles "
        "WHERE ticker=? AND tf=? ORDER BY ts", con, params=(ticker, tf))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def summary(con):
    return pd.read_sql_query(
        "SELECT ticker, tf, COUNT(*) AS bars, MIN(ts) AS first, MAX(ts) AS last "
        "FROM candles GROUP BY ticker, tf ORDER BY ticker, tf", con)


# ---------------------------------------------------------------- рядовые данные (мир, ставки, валюты, макро)
SERIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    kind TEXT NOT NULL, key TEXT NOT NULL, ts TEXT NOT NULL, value REAL,
    PRIMARY KEY (kind, key, ts)
)
"""
FACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    ticker TEXT NOT NULL, concept TEXT NOT NULL, period_end TEXT NOT NULL, value REAL, form TEXT, filed TEXT,
    PRIMARY KEY (ticker, concept, period_end)
)
"""


def connect_world(path=None):
    con = connect(path)
    con.execute(SERIES_SCHEMA)
    con.execute(FACTS_SCHEMA)
    return con


def upsert_series(con, kind, df):
    """df: колонки ts, key, value."""
    rows = [(kind, k, pd.Timestamp(ts).isoformat(), float(v)) for ts, k, v in
            df[["ts", "key", "value"]].itertuples(index=False)]
    con.executemany("INSERT OR REPLACE INTO series VALUES (?,?,?,?)", rows)
    con.commit()
    return len(rows)


def load_series(con, kind):
    """Широкая таблица: индекс ts, колонки key."""
    df = pd.read_sql_query("SELECT ts, key, value FROM series WHERE kind=? ORDER BY ts", con, params=(kind,))
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.pivot_table(index="ts", columns="key", values="value").sort_index()


def upsert_facts(con, ticker, df):
    """df: concept, period_end, value, form, filed."""
    rows = [(ticker, r.concept, str(r.period_end), float(r.value), r.form, str(r.filed))
            for r in df.itertuples(index=False)]
    con.executemany("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def load_facts(con, ticker):
    return pd.read_sql_query("SELECT concept, period_end, value, form, filed FROM facts WHERE ticker=? "
                             "ORDER BY period_end", con, params=(ticker,))


def facts_tickers(con):
    return [r[0] for r in con.execute("SELECT DISTINCT ticker FROM facts ORDER BY ticker")]


# ---------------------------------------------------------------- панель страна-год и торговые потоки
PANEL_SCHEMA = """
CREATE TABLE IF NOT EXISTS panel (iso3 TEXT NOT NULL, year INTEGER NOT NULL, indicator TEXT NOT NULL, value REAL,
    PRIMARY KEY (iso3, year, indicator))
"""
TRADE_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade (reporter TEXT NOT NULL, partner TEXT NOT NULL, year INTEGER NOT NULL,
    flow TEXT NOT NULL, value REAL, PRIMARY KEY (reporter, partner, year, flow))
"""


def connect_macro(path=None):
    con = connect_world(path)
    con.execute(PANEL_SCHEMA)
    con.execute(TRADE_SCHEMA)
    return con


def upsert_panel(con, long):
    """long: iso3, year, indicator, value."""
    rows = [(r.iso3, int(r.year), r.indicator, float(r.value)) for r in long.itertuples(index=False)
            if pd.notna(r.value)]
    con.executemany("INSERT OR REPLACE INTO panel VALUES (?,?,?,?)", rows)
    con.commit()
    return len(rows)


def load_panel_long(con):
    return pd.read_sql_query("SELECT iso3, year, indicator, value FROM panel", con)


def upsert_trade(con, df):
    """df: reporter, partner, year, flow ('X' экспорт / 'M' импорт), value (USD)."""
    rows = [(r.reporter, r.partner, int(r.year), r.flow, float(r.value)) for r in df.itertuples(index=False)
            if pd.notna(r.value)]
    con.executemany("INSERT OR REPLACE INTO trade VALUES (?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def load_trade(con):
    return pd.read_sql_query("SELECT reporter, partner, year, flow, value FROM trade", con)


# ---------------------------------------------------------------- события конфликтов (UCDP GED)
GED_SCHEMA = """
CREATE TABLE IF NOT EXISTS ged (id INTEGER PRIMARY KEY, year INTEGER, date TEXT, violence INTEGER, conflict TEXT,
    side_a TEXT, side_b TEXT, country TEXT, iso3 TEXT, lat REAL, lon REAL, best REAL, civilians REAL)
"""


def connect_events(path=None):
    con = connect_macro(path)
    con.execute(GED_SCHEMA)
    return con


def upsert_ged(con, ged):
    rows = [(int(r.id), int(r.year), str(r.date)[:10], int(r.type_of_violence) if pd.notna(r.type_of_violence) else None,
             r.conflict_name, r.side_a, r.side_b, r.country, r.iso3, float(r.latitude), float(r.longitude),
             float(r.best) if pd.notna(r.best) else 0.0, float(r.deaths_civilians) if pd.notna(r.deaths_civilians) else 0.0)
            for r in ged.itertuples(index=False) if pd.notna(r.year) and pd.notna(r.id)]
    con.executemany("INSERT OR REPLACE INTO ged VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def load_ged(con):
    df = pd.read_sql_query("SELECT * FROM ged", con)
    return df.rename(columns={"violence": "type_of_violence", "conflict": "conflict_name", "lat": "latitude",
                              "lon": "longitude", "civilians": "deaths_civilians"})
