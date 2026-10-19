"""Автономный режим без браузера: по расписанию обновляет данные БКС, считает сигналы
и делает ребаланс бумажного счёта. Реальные заявки не отправляются.

    python scripts/autotrader.py --every 60        # каждые 60 минут
    python scripts/autotrader.py --once            # один проход
Остановка: Ctrl+C. Аварийный стоп из интерфейса блокирует сделки и здесь.
"""
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradelab import autotrade, config, paper, store  # noqa: E402
from tradelab.bcs import BcsClient  # noqa: E402


def step(args):
    con = store.connect()
    client = BcsClient()
    for ticker in config.DEFAULT_TICKERS:
        last = store.last_ts(con, ticker, args.tf)
        start = last.to_pydatetime() if last is not None else datetime.now(timezone.utc) - timedelta(days=365 * 4)
        df = client.candles(ticker, start, timeframe=args.tf)
        if not df.empty:
            store.upsert(con, ticker, args.tf, df)
    frames = {t: store.load(con, t, args.tf) for t in config.DEFAULT_TICKERS}
    frames = {t: f for t, f in frames.items() if len(f)}
    prices = {t: float(f["close"].iloc[-1]) for t, f in frames.items()}
    signals = autotrade.compute_signals(frames, args.model)
    weights = autotrade.target_weights(signals, args.top_n, args.min_ic, args.max_weight)
    broker = paper.PaperBroker(config.DATA_DIR / "paper.db")
    done = broker.rebalance(weights, prices, max_weight=args.max_weight, note=f"auto:{args.model}")
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] цели: {weights or 'всё в деньги'}; сделок: {len(done)}; "
          f"капитал: {broker.equity(prices):,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="D")
    ap.add_argument("--model", default="Ridge-регрессия")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--min-ic", type=float, default=0.0)
    ap.add_argument("--max-weight", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=60, help="период в минутах")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        try:
            step(args)
        except Exception as exc:
            print("ошибка прохода:", repr(exc))
        if args.once:
            return
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
