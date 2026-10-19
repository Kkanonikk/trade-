"""Мировые данные: загрузка из локальной базы / демо, и обновление из открытых источников со статусом по каждому."""
import pandas as pd

from . import demo_world, fundamentals, markets, sources, store


def load(is_demo, con=None):
    """Словарь с ключами index, cross, fx, rates, yields, macro, facts (пустые DataFrame, если данных нет)."""
    if is_demo:
        return demo_world.world()
    con = con or store.connect_world()
    macro = store.load_series(con, "macro")
    rows = []
    for key in macro.columns if len(macro) else []:
        ind, iso = key.split("|")
        for ts, v in macro[key].dropna().items():
            rows.append((ind, iso, ts.year, float(v)))
    return {"index": store.load_series(con, "index"), "cross": store.load_series(con, "cross"),
            "fx": store.load_series(con, "fx"), "rates": store.load_series(con, "rate"),
            "yields": store.load_series(con, "yield"),
            "macro": pd.DataFrame(rows, columns=["indicator", "country", "year", "value"]),
            "facts": {t: store.load_facts(con, t) for t in store.facts_tickers(con)}}


def _ohlc_to_series(df, key):
    return pd.DataFrame({"ts": df["ts"], "key": key, "value": df["close"]})


def refresh(con, years=5, progress=None):
    """Обновляет все мировые источники. Возвращает список (источник, ok, сообщение). Сбой одного не мешает остальным."""
    report = []
    tasks = []

    def add(label, fn):
        tasks.append((label, fn))

    for name in markets.MARKETS:
        def f(name=name):
            df, src = sources.fetch_index(name, years)
            return store.upsert_series(con, "index", _ohlc_to_series(df, name)), src
        add(f"Индекс {name}", f)
    for name, sym in markets.CROSS_ASSETS.items():
        def f(name=name, sym=sym):
            df = sources.fetch_yahoo(sym, years)
            return store.upsert_series(con, "cross", _ohlc_to_series(df, name)), "Yahoo"
        add(f"Актив {name}", f)
    add("Валюты ECB", lambda: (store.upsert_series(con, "fx", sources.fetch_fx_ecb(years)), "Frankfurter"))
    add("Курсы ЦБ РФ", lambda: (store.upsert_series(con, "fx", sources.fetch_fx_cbr(years)), "ЦБ РФ"))
    for name, (series, _iso) in markets.POLICY_RATES.items():
        def f(name=name, series=series):
            return store.upsert_series(con, "rate", sources.fetch_fred(series, name)), "FRED"
        add(f"Ставка {name}", f)
    for name, series in markets.YIELDS.items():
        def f(name=name, series=series):
            return store.upsert_series(con, "yield", sources.fetch_fred(series, name)), "FRED"
        add(f"Доходность {name}", f)
    for ind_name, code in markets.WB_INDICATORS.items():
        def f(ind_name=ind_name, code=code):
            df = sources.fetch_worldbank(code)
            long = pd.DataFrame({"ts": pd.to_datetime(df["year"].astype(str) + "-12-31", utc=True),
                                 "key": ind_name + "|" + df["country"], "value": df["value"]})
            return store.upsert_series(con, "macro", long), "World Bank"
        add(f"Макро: {ind_name}", f)
    for i, (label, fn) in enumerate(tasks):
        try:
            n, src = fn()
            report.append((label, True, f"{src}: {n} записей"))
        except Exception as exc:
            report.append((label, False, str(exc)[:200]))
        if progress:
            progress((i + 1) / len(tasks), label)
    return report


def refresh_fundamentals(con, tickers):
    out = []
    for t in tickers:
        try:
            facts = fundamentals.fetch_sec(t)
            out.append((t, True, f"{store.upsert_facts(con, t, facts)} записей"))
        except Exception as exc:
            out.append((t, False, str(exc)[:200]))
    return out


def returns(prices):
    """Дневные доходности по ценам, выровненные по общему календарю с протягиванием пропусков (праздники)."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    return prices.sort_index().ffill(limit=3).pct_change().dropna(how="any")
