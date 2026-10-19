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
    con = sqlite3.connect(path)
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
