"""Бумажный (демо) брокер: сделки симулируются локально, реальные заявки не отправляются.

Правила: только long, без плеча, комиссия с каждой сделки, лимит на вес одной бумаги,
аварийный стоп (kill switch) блокирует любые сделки. Лоты MOEX не учитываются (целые акции).
"""
import math
import sqlite3
from datetime import datetime, timezone

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY CHECK (id=1), cash REAL, start_cash REAL);
CREATE TABLE IF NOT EXISTS positions (ticker TEXT PRIMARY KEY, qty INTEGER, avg_price REAL);
CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, ticker TEXT,
    side TEXT, qty INTEGER, price REAL, fee REAL, note TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, equity REAL);
CREATE TABLE IF NOT EXISTS flags (name TEXT PRIMARY KEY, value TEXT);
"""


class PaperBroker:
    def __init__(self, path, start_cash=1_000_000.0, fee_bps=5.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.executescript(SCHEMA)
        self.fee_bps = fee_bps
        if not self.con.execute("SELECT 1 FROM account").fetchone():
            self.con.execute("INSERT INTO account VALUES (1, ?, ?)", (start_cash, start_cash))
            self.con.commit()

    @property
    def cash(self):
        return self.con.execute("SELECT cash FROM account").fetchone()[0]

    @property
    def start_cash(self):
        return self.con.execute("SELECT start_cash FROM account").fetchone()[0]

    def positions(self):
        return {t: (q, p) for t, q, p in
                self.con.execute("SELECT ticker, qty, avg_price FROM positions WHERE qty>0")}

    def equity(self, prices):
        return self.cash + sum(q * prices.get(t, p) for t, (q, p) in self.positions().items())

    def kill_switch(self):
        row = self.con.execute("SELECT value FROM flags WHERE name='kill'").fetchone()
        return bool(row and row[0] == "1")

    def set_kill_switch(self, on):
        self.con.execute("INSERT OR REPLACE INTO flags VALUES ('kill', ?)", ("1" if on else "0",))
        self.con.commit()

    def order(self, ticker, side, qty, price, note=""):
        if self.kill_switch():
            raise ValueError("Аварийный стоп включён: сделки заблокированы")
        qty = int(qty)
        if qty <= 0 or price <= 0:
            raise ValueError("Некорректный объём или цена")
        fee = qty * price * self.fee_bps / 1e4
        pos = self.positions().get(ticker, (0, 0.0))
        cash = self.cash
        if side == "BUY":
            cost = qty * price + fee
            if cost > cash + 1e-9:
                raise ValueError("Недостаточно средств (плечо запрещено)")
            new_qty = pos[0] + qty
            avg = (pos[0] * pos[1] + qty * price) / new_qty
            cash -= cost
        elif side == "SELL":
            if qty > pos[0]:
                raise ValueError("Недостаточно бумаг (short запрещён)")
            new_qty, avg = pos[0] - qty, pos[1]
            cash += qty * price - fee
        else:
            raise ValueError("side должен быть BUY или SELL")
        with self.con:
            self.con.execute("UPDATE account SET cash=?", (cash,))
            self.con.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?)", (ticker, new_qty, avg))
            self.con.execute(
                "INSERT INTO orders (ts,ticker,side,qty,price,fee,note) VALUES (?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), ticker, side, qty, price, fee, note))
        return {"ticker": ticker, "side": side, "qty": qty, "price": price, "fee": fee}

    def rebalance(self, targets, prices, max_weight=0.25, min_trade_value=1000.0, note="rebalance"):
        """targets: {ticker: вес}. Сначала продажи, потом покупки. Возвращает список исполненных сделок."""
        equity = self.equity(prices)
        targets = {t: min(w, max_weight) for t, w in targets.items() if w > 0 and t in prices}
        current = {t: q for t, (q, _) in self.positions().items()}
        plan = []
        for t in set(targets) | set(current):
            price = prices.get(t)
            if not price:
                continue
            want = math.floor(targets.get(t, 0.0) * equity / price)
            diff = want - current.get(t, 0)
            if abs(diff) * price >= min_trade_value or (want == 0 and diff < 0):
                plan.append((t, diff, price))
        done = []
        for t, diff, price in sorted(plan, key=lambda x: x[1]):  # продажи (diff<0) первыми
            try:
                done.append(self.order(t, "SELL" if diff < 0 else "BUY", abs(diff), price, note))
            except ValueError as exc:
                done.append({"ticker": t, "error": str(exc)})
        self.record_equity(prices)
        return done

    def record_equity(self, prices):
        self.con.execute("INSERT OR REPLACE INTO equity VALUES (?,?)",
                         (datetime.now(timezone.utc).isoformat(timespec="seconds"), self.equity(prices)))
        self.con.commit()

    def orders(self, limit=200):
        return pd.read_sql_query("SELECT * FROM orders ORDER BY id DESC LIMIT ?", self.con, params=(limit,))

    def equity_curve(self):
        df = pd.read_sql_query("SELECT * FROM equity ORDER BY ts", self.con)
        df["ts"] = pd.to_datetime(df["ts"])
        return df

    def reset(self, start_cash=1_000_000.0):
        with self.con:
            for table in ("positions", "orders", "equity"):
                self.con.execute(f"DELETE FROM {table}")
            self.con.execute("UPDATE account SET cash=?, start_cash=?", (start_cash, start_cash))
