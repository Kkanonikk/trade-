"""Контроль качества и свежести рядов: пропуски, скачки, «залипшие» значения, возраст последних данных.

Все проверки статистические и дают подозрения, а не вердикты: скачок может быть реальным событием (война, дефолт), а не ошибкой.
"""
from datetime import date

import numpy as np
import pandas as pd

from . import gdata

JUMP_Z = 8.0          # размер изменения в робастных сигмах (MAD), начиная с которого год считается подозрительным
JUMP_REL = 0.5        # и изменение должно превышать долю среднего уровня ряда
FLAT_RUN = 5          # столько одинаковых значений подряд считаем «залипанием» (вероятно, заполнение пропусков)
GAP_YEARS = 3         # внутренний пропуск длиннее этого числа лет


def series_flags(s):
    """Подозрения по одному ряду (год -> значение): список (тип, год, описание)."""
    s = s.dropna()
    out = []
    if len(s) < 6:
        return out
    yrs = s.index.to_numpy(int)
    gaps = np.diff(yrs)
    for i in np.where(gaps > GAP_YEARS)[0]:
        out.append(("пропуск", int(yrs[i + 1]), f"нет данных {yrs[i] + 1}–{yrs[i + 1] - 1}"))
    d = s.diff().dropna()
    mad = float(np.median(np.abs(d - d.median()))) * 1.4826
    level = float(np.abs(s).mean()) or 1.0
    if mad > 0:
        z = (d - d.median()) / mad
        for y in z.index[(z.abs() > JUMP_Z) & (d.abs() > JUMP_REL * level)]:
            out.append(("скачок", int(y), f"изменение {d[y]:+.3g} при типичном ±{mad:.3g}"))
    vals = s.to_numpy()
    run = 1
    for i in range(1, len(vals)):
        run = run + 1 if vals[i] == vals[i - 1] else 1
        if run == FLAT_RUN:
            out.append(("залипание", int(yrs[i]), f"{FLAT_RUN} одинаковых значений подряд"))
    return out


def report(w, keys=None, current_year=None):
    """Сводка по показателям: страны, последний год с достаточным покрытием, возраст данных, доли рядов с пропусками/скачками."""
    cy = current_year or date.today().year
    keys = keys or list(w.columns)
    meta = gdata.meta()
    rows = []
    for k in keys:
        wide = w[k].unstack("year")
        cap = gdata.last_actual_year(k)
        if cap:
            wide = wide.loc[:, wide.columns <= cap]
        wide = wide.dropna(how="all")
        if wide.empty:
            continue
        cnt = wide.notna().sum()
        ok = cnt[cnt >= 0.5 * cnt.max()]
        last = int(ok.index.max()) if len(ok) else int(wide.columns.max())
        n = len(wide)
        flagged = {"пропуск": 0, "скачок": 0, "залипание": 0}
        sample = wide if n <= 400 else wide.iloc[:400]
        for _, row in sample.iterrows():
            kinds = {f[0] for f in series_flags(row)}
            for kd in kinds:
                flagged[kd] += 1
        m = len(sample)
        age = cy - last
        rows.append({"key": k, "Показатель": meta[k]["label"], "Группа": meta[k]["group"], "Стран": n, "Последний год": last,
                     "Возраст, лет": age, "С пропусками": flagged["пропуск"] / m, "Со скачками": flagged["скачок"] / m,
                     "Залипание": flagged["залипание"] / m,
                     "Статус": "свежий" if age <= 1 else "приемлемо" if age <= 3 else "устарел"})
    return pd.DataFrame(rows)
