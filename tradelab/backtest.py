"""Walk-forward бэктест без заглядывания в будущее.

В момент закрытия бара t модель обучается только на строках, чья цель (доходность
t -> t+1) уже известна, то есть на строках до t-1, и даёт прогноз на t.
Позиция: long (1) если прогноз > 0, иначе вне рынка (0) или short (-1).
Комиссия берётся с каждого изменения позиции.
"""
import numpy as np
import pandas as pd

from . import features, models


def walk_forward(df, model_name, min_train=250, refit_every=20, cost_bps=5.0, allow_short=False):
    data = features.build(df).reset_index(drop=True)
    usable = data.dropna(subset=features.FEATURES).index
    if len(usable) < min_train + 30:
        raise ValueError(f"Слишком мало данных: {len(usable)} баров, нужно от {min_train + 30}")
    preds = pd.Series(np.nan, index=data.index)
    model, last_fit = None, None
    for i in usable:
        if i < usable[0] + min_train:
            continue
        if model is None or i - last_fit >= refit_every:
            train = data.loc[usable[0]:i - 1].dropna(subset=features.FEATURES + ["target"])
            model = models.create(model_name).fit(train[features.FEATURES], train["target"])
            last_fit = i
        preds[i] = model.predict(data.loc[[i], features.FEATURES])[0]
    out = data[["ts", "close", "target"]].copy()
    out["pred"] = preds
    out = out.dropna(subset=["pred"]).copy()
    if allow_short:
        out["pos"] = np.sign(out["pred"])
    else:
        out["pos"] = (out["pred"] > 0).astype(float)
    turnover = out["pos"].diff().abs().fillna(out["pos"].abs())
    out["strategy"] = out["pos"] * out["target"].fillna(0) - turnover * cost_bps / 1e4
    out["buyhold"] = out["target"].fillna(0)
    out["equity"] = np.exp(out["strategy"].cumsum())
    out["equity_bh"] = np.exp(out["buyhold"].cumsum())
    return out


def metrics(res, periods_per_year=252):
    valid = res.dropna(subset=["target"])
    ic = valid["pred"].corr(valid["target"], method="spearman")
    hit = ((valid["pred"] > 0) == (valid["target"] > 0)).mean()
    s = valid["strategy"]
    sharpe = s.mean() / s.std() * np.sqrt(periods_per_year) if s.std() > 0 else np.nan
    equity = np.exp(s.cumsum())
    drawdown = (equity / equity.cummax() - 1).min()
    return {
        "Баров вне выборки": int(len(valid)),
        "Доля верных знаков": float(hit),
        "IC (Спирмен)": float(ic),
        "Доходность стратегии": float(equity.iloc[-1] - 1),
        "Доходность buy&hold": float(np.exp(valid["buyhold"].sum()) - 1),
        "Шарп": float(sharpe),
        "Макс. просадка": float(drawdown),
        "Время в рынке": float((valid["pos"] != 0).mean()),
    }


def latest_forecast(df, model_name, min_train=250):
    """Прогноз на следующий бар по последнему закрытию (модель обучена на всей истории)."""
    data = features.build(df)
    train = data.dropna(subset=features.FEATURES + ["target"])
    if len(train) < min_train:
        raise ValueError("Мало данных для прогноза")
    model = models.create(model_name).fit(train[features.FEATURES], train["target"])
    last = data.dropna(subset=features.FEATURES).iloc[[-1]]
    return float(model.predict(last[features.FEATURES])[0]), last["ts"].iloc[0]
