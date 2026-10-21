"""Одна операция «загрузить все онлайн-данные»: независимые шаги идут параллельно, сбой одного не мешает остальным.
Каждый шаг пишет в свою локальную базу и возвращает (сообщение); итог это таблица (шаг, ок, сообщение, секунд)."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from . import conflicts, maritime, openmarket, panel as panel_mod, store, trade, worlddata


def _quotes(ticker, tf, years, source):
    def run():
        con = store.connect()
        last = store.last_ts(con, ticker, tf)
        start = last.to_pydatetime() if last is not None else openmarket.default_start(years)
        df, src = openmarket.load(ticker, start, tf, source)
        n = store.upsert(con, ticker, tf, df) if not df.empty else 0
        return f"{n} баров ({src})"
    return run


def _world(years):
    def run():
        report = worlddata.refresh(store.connect_world(), years)
        ok = sum(1 for _, o, _ in report if o)
        bad = [f"{n}: {m}" for n, o, m in report if not o][:3]
        if ok == 0:
            raise RuntimeError("; ".join(bad) or "нет данных")
        return f"успешно {ok} из {len(report)}" + (f"; сбои: {'; '.join(bad)}" if bad else "")
    return run


def _macro():
    wide, report = panel_mod.refresh_live()
    ok = sum(1 for _, o, _ in report if o)
    if wide.empty:
        raise RuntimeError("; ".join([m for _, o, m in report if not o][:2]) or "нет данных")
    store.upsert_panel(store.connect_macro(), panel_mod.to_long(wide))
    return f"успешно {ok} из {len(report)} показателей панели"


def _portwatch():
    n_ports, n_rows = maritime.refresh_portwatch(store.connect_macro())
    return f"проливов {n_ports}, строк {n_rows}"


def _ged():
    ged = conflicts.fetch_ged()
    store.upsert_ged(store.connect_events(), ged)
    return f"событий {len(ged)}"


def _comtrade():
    year = datetime.now().year - 2
    reps = ["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "IND", "RUS", "BRA", "TUR", "KOR", "SAU"]
    df, report = trade.fetch_comtrade(reps, year)
    if len(df) == 0:
        raise RuntimeError("; ".join([m for _, o, m in report if not o][:2]) or "нет данных")
    store.upsert_trade(store.connect_macro(), df)
    return f"{year} г.: записей {len(df)}"


def build_steps(tickers, tf="D", years=4, source="Авто", include_quotes=True):
    steps = [("Мировые рынки: индексы, сырьё, валюты, ставки", _world(years)),
             ("Макро-панель World Bank + IMF", _macro),
             ("Проливы и танкеры (IMF PortWatch)", _portwatch),
             ("Конфликты (UCDP GED)", _ged),
             ("Торговые потоки (UN Comtrade)", _comtrade)]
    if include_quotes:
        steps += [(f"Котировки {t}", _quotes(t, tf, years, source)) for t in tickers]
    return steps


def run(steps, progress=None, workers=6):
    """Выполняет шаги параллельно. progress(доля, подпись). Возвращает список (шаг, ok, сообщение, секунд) в порядке шагов."""
    results = {}

    def timed(label, fn):
        t0 = time.time()
        try:
            msg = fn()
            return label, True, str(msg or "ок"), time.time() - t0
        except Exception as exc:
            return label, False, str(exc)[:240], time.time() - t0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(timed, lab, fn): lab for lab, fn in steps}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            results[r[0]] = r
            done += 1
            if progress:
                progress(done / len(steps), r[0])
    return [results[lab] for lab, _ in steps]
