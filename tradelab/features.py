"""Признаки и цель. Все признаки в строке t используют данные только до закрытия t."""
import numpy as np
import pandas as pd

FEATURES = ["ret1", "ret5", "ret20", "vol20", "rsi14", "dist_ma20", "vol_ratio"]


def build(df):
    close, volume = df["close"], df["volume"]
    out = pd.DataFrame({"ts": df["ts"], "close": close})
    logret = np.log(close).diff()
    out["ret1"] = logret
    out["ret5"] = np.log(close).diff(5)
    out["ret20"] = np.log(close).diff(20)
    out["vol20"] = logret.rolling(20).std()
    out["rsi14"] = _rsi(close, 14)
    out["dist_ma20"] = close / close.rolling(20).mean() - 1
    out["vol_ratio"] = volume / volume.rolling(20).mean()
    out["target"] = logret.shift(-1)  # доходность следующего бара
    return out.replace([np.inf, -np.inf], np.nan)


def _rsi(close, n):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down)
