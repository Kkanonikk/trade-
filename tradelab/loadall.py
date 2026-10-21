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


def run(steps, progress=None, workers=6, deadline=240.0):
    """Выполняет шаги параллельно в быстром сетевом режиме. progress(доля, подпись). deadline: общий лимит в секундах, после него
    незавершённые шаги помечаются как прерванные. Возвращает список (шаг, ok, сообщение, секунд) в порядке шагов."""
    import queue
    import threading
    from . import net

    net.fast_mode(True)
    q, results, done_q = queue.Queue(), {}, queue.Queue()
    for st_ in steps:
        q.put(st_)

    def worker():
        while True:
            try:
                label, fn = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            try:
                msg = fn()
                done_q.put((label, True, str(msg or "ок"), time.time() - t0))
            except Exception as exc:
                done_q.put((label, False, str(exc)[:240], time.time() - t0))

    for _ in range(min(workers, len(steps)) or 1):
        threading.Thread(target=worker, daemon=True).start()
    t_end = time.time() + deadline
    try:
        while len(results) < len(steps) and time.time() < t_end:
            try:
                r = done_q.get(timeout=0.5)
            except queue.Empty:
                continue
            results[r[0]] = r
            if progress:
                progress(len(results) / len(steps), r[0])
    finally:
        net.fast_mode(False)
    for label, _ in steps:
        results.setdefault(label, (label, False, "не завершён за отведённое время (источник медленный или недоступен)", deadline))
    return [results[lab] for lab, _ in steps]
