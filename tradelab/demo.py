"""Синтетические котировки для демо-режима (НЕ реальные данные).
Однофакторная модель: общий «рыночный» фактор + собственный шум каждой бумаги."""
import zlib

import numpy as np
import pandas as pd


def universe(tickers, n=900, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=n)
    market = rng.normal(0.0002, 0.011, n)
    out = {}
    for i, t in enumerate(sorted(tickers)):
        r = np.random.default_rng(seed + 1 + zlib.crc32(t.encode()) % 10_000)
        beta = 0.6 + 0.8 * r.random()
        idio = r.normal(0.0, 0.008 + 0.01 * r.random(), n)
        rets = beta * market + idio
        close = (50 + 400 * r.random()) * np.exp(np.cumsum(rets))
        spread = np.abs(r.normal(0, 0.006, n))
        open_ = np.r_[close[0], close[:-1]] * (1 + r.normal(0, 0.002, n))
        out[t] = pd.DataFrame({
            "ts": idx, "open": open_,
            "high": np.maximum(open_, close) * (1 + spread),
            "low": np.minimum(open_, close) * (1 - spread),
            "close": close,
            "volume": r.integers(200_000, 5_000_000, n).astype(float),
        })
    return out
