"""Сигналы моделей -> целевые веса -> ребаланс бумажного счёта."""
import pandas as pd

from . import backtest


def compute_signals(frames, model_name, cost_bps=5.0):
    rows = []
    for ticker, df in frames.items():
        try:
            res = backtest.walk_forward(df, model_name, cost_bps=cost_bps, refit_every=60)
            m = backtest.metrics(res)
            forecast, _ = backtest.latest_forecast(df, model_name)
        except ValueError:
            continue
        rows.append({"ticker": ticker, "forecast": forecast, "ic": m["IC (Спирмен)"],
                     "hit": m["Доля верных знаков"], "sharpe": m["Шарп"],
                     "price": float(df["close"].iloc[-1])})
    cols = ["ticker", "forecast", "ic", "hit", "sharpe", "price"]
    return pd.DataFrame(rows, columns=cols)


def target_weights(signals, top_n=5, min_ic=0.0, max_weight=0.25):
    """Берём бумаги с положительным прогнозом и IC выше порога, лучшие top_n по прогнозу,
    равные веса с ограничением max_weight; остаток остаётся в деньгах."""
    pick = signals[(signals["forecast"] > 0) & (signals["ic"] > min_ic)]
    pick = pick.sort_values("forecast", ascending=False).head(top_n)
    if pick.empty:
        return {}
    w = min(1.0 / len(pick), max_weight)
    return {t: w for t in pick["ticker"]}
