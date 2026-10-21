"""Универсальный прогноз годовых рядов по странам с проверкой на истории.

Методы: наивный (последнее значение), тренд (МНК по последним K годам), демпфированный тренд Хольта; для положительных
рядов также их версии в логарифмах. Лучший метод выбирается по ошибке MASE (в долях ошибки наивного прогноза) на
скользящих «прошлых прогнозах» всех стран. Интервалы строятся по эмпирическим квантилям ошибок на этом же горизонте.
Если у показателя есть официальные прогнозы ООН, они показываются отдельно как справочные.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

METHODS = ["naive", "trend", "damped", "trend_log", "damped_log"]
LABELS = {"naive": "Наивный (последнее значение)", "trend": "Линейный тренд", "damped": "Демпфированный тренд (Хольт)",
          "trend_log": "Тренд в логарифмах (рост в %)", "damped_log": "Демпфированный тренд в логарифмах"}
Z80 = (0.10, 0.90)


def _clean(s):
    """Ряд по годам без пропусков в конце: берём последний непрерывный (допуская дыры до 2 лет, заполняем линейно) отрезок."""
    s = s.dropna().astype(float)
    if len(s) < 8:
        return None
    years = s.index.to_numpy(dtype=int)
    full = pd.Series(np.interp(np.arange(years[0], years[-1] + 1), years, s.to_numpy()), index=np.arange(years[0], years[-1] + 1))
    gaps = np.diff(years)
    cut = np.where(gaps > 5)[0]
    if len(cut):
        full = full.loc[years[cut[-1] + 1]:]
    return full if len(full) >= 8 else None


def _trend(y, h, k):
    k = min(k, len(y))
    x = np.arange(k)
    slope, icpt = np.polyfit(x, y[-k:], 1)
    return icpt + slope * (k - 1 + np.arange(1, h + 1))


def _holt_damped(y, h):
    """Демпфированный тренд: параметры из небольшой сетки по сумме квадратов одношаговых ошибок на последних 30 точках."""
    y = y[-40:]
    best = None
    for a in (0.2, 0.5, 0.8):
        for b in (0.1, 0.3):
            for phi in (0.8, 0.9, 0.98):
                lvl, tr, sse = y[0], 0.0, 0.0
                for t in range(1, len(y)):
                    pred = lvl + phi * tr
                    sse += (y[t] - pred) ** 2
                    new = a * y[t] + (1 - a) * pred
                    tr = b * (new - lvl) + (1 - b) * phi * tr
                    lvl = new
                if best is None or sse < best[0]:
                    best = (sse, lvl, tr, phi)
    _, lvl, tr, phi = best
    steps = np.cumsum(phi ** np.arange(1, h + 1))
    return lvl + steps * tr


def predict(y, h, method, k=10):
    y = np.asarray(y, dtype=float)
    if method.endswith("_log"):
        if (y <= 0).any():
            return None
        out = np.exp(predict(np.log(y), h, method[:-4], k))
        recent = y[-10:]
        return np.clip(out, recent.min() / 5.0, recent.max() * 5.0)   # логарифмический тренд не должен взрываться
    if method == "naive":
        return np.full(h, y[-1])
    if method == "trend":
        return _trend(y, h, k)
    if method == "damped":
        return _holt_damped(y, h)
    raise ValueError(method)


@dataclass
class Backtest:
    key: str
    horizon: int
    mase: dict = field(default_factory=dict)         # метод -> MASE по горизонтам (массив длины horizon)
    quantiles: dict = field(default_factory=dict)    # метод -> (q10[h], q90[h]) в единицах масштаба
    best: str = "naive"
    n_series: int = 0
    n_forecasts: int = 0
    coverage: float = float("nan")                    # доля попаданий в P10–P90 на последних 30% прошлых прогнозов


def _scale(y):
    """Масштаб ряда: средний модуль годового изменения, не меньше 2% среднего уровня (иначе почти плоские ряды раздувают ошибку)."""
    d = np.abs(np.diff(y))
    s = d.mean() if len(d) else 0.0
    return max(s, 0.02 * np.abs(y).mean(), 1e-9)


ERR_CLIP = 100.0  # ошибки в единицах масштаба ограничиваются, чтобы редкие взрывы не доминировали


def run_backtest(series_by_country, horizon=10, k=10, step=None, min_train=None, last_year_cap=None, methods=METHODS):
    """Скользящие прошлые прогнозы по всем странам. series_by_country: {iso3: Series (год -> значение)}."""
    errs = {m: [] for m in methods}      # список (origin_year, h-массив ошибок / масштаб)
    n_series = 0
    cleaned = {}
    for iso, s in series_by_country.items():
        if last_year_cap is not None:
            s = s[s.index <= last_year_cap]
        c = _clean(s)
        if c is not None and np.abs(c.to_numpy()).mean() > 1e-9:
            cleaned[iso] = c
    if not cleaned:
        return Backtest(key="", horizon=horizon)
    typical = float(np.median([len(c) for c in cleaned.values()]))
    if min_train is None:                       # короткие ряды (напр. 13 лет): учим на 6+ точках, шаг 1
        min_train = 12 if typical >= 30 else 6
    if step is None:
        step = 3 if typical >= 30 else 1
    for iso, c in cleaned.items():
        if len(c) < min_train + horizon:
            continue
        n_series += 1
        y, yrs = c.to_numpy(), c.index.to_numpy()
        for end in range(min_train, len(y) - horizon + 1, step):
            train, actual = y[:end], y[end:end + horizon]
            sc = _scale(train)
            for m in methods:
                p = predict(train, horizon, m, k)
                if p is not None:
                    errs[m].append((yrs[end - 1], np.clip((actual - p) / sc, -ERR_CLIP, ERR_CLIP)))
    bt = Backtest(key="", horizon=horizon, n_series=n_series)
    base = None
    for m in methods:
        if not errs[m]:
            continue
        E = np.array([e for _, e in errs[m]])
        mase = np.abs(E).mean(axis=0)
        bt.mase[m] = mase
        bt.quantiles[m] = (np.quantile(E, Z80[0], axis=0), np.quantile(E, Z80[1], axis=0))
        if m == "naive":
            base = mase
    bt.n_forecasts = len(errs.get("naive", []))
    if bt.mase:
        # выбор по средней по горизонту ошибке; ошибки уже в единицах масштаба ряда
        bt.best = min(bt.mase, key=lambda m: float(bt.mase[m].mean()))
        # проверка интервалов: квантили по ранним прошлым прогнозам, покрытие на поздних
        yrs_all = np.array([o for o, _ in errs[bt.best]])
        E = np.array([e for _, e in errs[bt.best]])
        cut = np.quantile(yrs_all, 0.7)
        early, late = E[yrs_all <= cut], E[yrs_all > cut]
        if len(early) > 30 and len(late) > 30:
            lo, hi = np.quantile(early, Z80[0], axis=0), np.quantile(early, Z80[1], axis=0)
            bt.coverage = float(((late >= lo) & (late <= hi)).mean())
    return bt


def forecast_series(s, bt, horizon=None, method=None, k=10):
    """Прогноз ряда на horizon лет: DataFrame year, mean, p10, p90 (None, если ряд слишком короткий)."""
    c = _clean(s)
    if c is None:
        return None
    method = method or bt.best
    h = horizon or bt.horizon
    p = predict(c.to_numpy(), h, method, k)
    if p is None:
        method = "naive" if method.endswith("_log") else method
        p = predict(c.to_numpy(), h, method, k)
    sc = _scale(c.to_numpy())
    q = bt.quantiles.get(method)
    if q is None:
        return None
    hh = min(h, len(q[0]))
    lo = p[:hh] + q[0][:hh] * sc
    hi = p[:hh] + q[1][:hh] * sc
    years = np.arange(c.index[-1] + 1, c.index[-1] + 1 + hh)
    return pd.DataFrame({"year": years, "mean": p[:hh], "p10": lo, "p90": hi, "method": method})


def skill_table(bt):
    """Таблица ошибок методов на горизонтах (1, 3, 5, 10 лет) и навыка к наивному прогнозу."""
    rows = []
    base = bt.mase.get("naive")
    for m, v in bt.mase.items():
        row = {"Метод": LABELS[m]}
        for hz in (1, 3, 5, 10):
            if hz <= len(v):
                row[f"MASE {hz} г."] = float(v[hz - 1])
                if base is not None:
                    row[f"Навык {hz} г."] = float(1 - v[hz - 1] / base[hz - 1])
        rows.append(row)
    return pd.DataFrame(rows)


def country_backtest(s, method, horizon, k=10, min_train=12):
    """Прошлые прогнозы метода для ОДНОГО ряда (год -> значение): таблица origin, h, факт, прогноз, ошибка, и MASE страны против наивного."""
    c = _clean(s)
    if c is None or len(c) < min_train + horizon:
        return None, float("nan")
    y, yrs = c.to_numpy(), c.index.to_numpy()
    rows, e_m, e_n = [], [], []
    for end in range(min_train, len(y) - horizon + 1):
        train = y[:end]
        p = predict(train, horizon, method, k)
        if p is None:
            return None, float("nan")
        n = predict(train, horizon, "naive", k)
        rows.append({"origin": int(yrs[end - 1]), "target": int(yrs[end + horizon - 1]), "actual": float(y[end + horizon - 1]),
                     "forecast": float(p[-1])})
        e_m.append(abs(y[end + horizon - 1] - p[-1]))
        e_n.append(abs(y[end + horizon - 1] - n[-1]))
    ratio = float(np.mean(e_m) / max(np.mean(e_n), 1e-12))
    return pd.DataFrame(rows), ratio
