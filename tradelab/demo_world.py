"""Синтетические мировые данные для демо-режима (НЕ реальные). Нужны, чтобы изучить интерфейс без сети.
В данных заложены: общий мировой фактор, региональные факторы, эффект «Азия -> Европа -> США» через сутки,
кризисный эпизод с ростом корреляций и волатильности."""
import numpy as np
import pandas as pd

from . import markets

REGION = {"USA": "am", "CAN": "am", "BRA": "am", "MEX": "am", "GBR": "eu", "DEU": "eu", "FRA": "eu", "CHE": "eu",
          "RUS": "eu", "TUR": "eu", "ZAF": "eu", "SAU": "eu", "IND": "as", "CHN": "as", "HKG": "as", "JPN": "as",
          "KOR": "as", "SGP": "as", "AUS": "as"}


def world(n=1000, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=n)
    crisis = np.zeros(n)
    crisis[int(n * 0.62):int(n * 0.70)] = 1.0
    vol_mult = 1 + 2.2 * crisis
    glob = rng.normal(0.0003, 0.008, n) * vol_mult - 0.004 * crisis
    reg = {k: rng.normal(0, 0.005, n) * vol_mult for k in ("am", "eu", "as")}
    # Азия закрывается раньше: её сегодняшний шок частично переходит в Европу и США завтра
    carry = {"as": 0.0, "eu": 0.30, "am": 0.35}
    closes = {}
    for name, (iso, *_rest) in markets.MARKETS.items():
        r = np.random.default_rng(seed + sum(map(ord, name)))
        g = REGION[iso]
        beta = 0.5 + 0.7 * r.random()
        lead = np.r_[0, reg["as"][:-1]] * carry[g]
        ret = beta * glob + (1 + crisis) * 0.5 * reg[g] + lead + r.normal(0, 0.006, n) * (0.5 + 0.5 * vol_mult)
        closes[name] = 100 * (1 + r.random() * 20) * np.exp(np.cumsum(ret))
    index = pd.DataFrame(closes, index=idx)

    cross = {}
    for i, name in enumerate(markets.CROSS_ASSETS):
        r = np.random.default_rng(seed + 100 + i)
        cross[name] = (50 + 80 * r.random()) * np.exp(np.cumsum(r.normal(0, 0.012, n) * vol_mult + 0.3 * glob))
    cross = pd.DataFrame(cross, index=idx)

    fx = {}
    for i, cur in enumerate(markets.FX_ECB):
        r = np.random.default_rng(seed + 200 + i)
        fx[f"USD/{cur}"] = (1 + 50 * r.random()) * np.exp(np.cumsum(r.normal(0, 0.004, n)))
    for i, cur in enumerate(markets.FX_CODES_CBR):
        r = np.random.default_rng(seed + 300 + i)
        fx[f"{cur}/RUB"] = (60 + 40 * r.random()) * np.exp(np.cumsum(r.normal(0, 0.007, n)))
    fx = pd.DataFrame(fx, index=idx)

    rates = {}
    for i, (name, (series, iso)) in enumerate(markets.POLICY_RATES.items()):
        r = np.random.default_rng(seed + 400 + i)
        steps = np.cumsum(r.choice([0, 0, 0, 0.25, -0.25], n // 60 + 1))
        base = 1.0 + 8 * r.random()
        level = np.clip(base + np.repeat(steps, 60)[:n], 0.1, 25)
        rates[name] = level
    rates = pd.DataFrame(rates, index=idx)

    yields = pd.DataFrame({k: 1.5 + 3 * np.random.default_rng(seed + 500 + i).random()
                           + np.cumsum(np.random.default_rng(seed + 510 + i).normal(0, 0.02, n)) * 0.5
                           for i, k in enumerate(markets.YIELDS)}, index=idx)

    years = list(range(2000, pd.Timestamp.now().year))
    rows = []
    for ci, iso in enumerate(markets.WB_COUNTRIES):
        r = np.random.default_rng(seed + 600 + ci)
        for ind in markets.WB_INDICATORS:
            base = {"Рост ВВП, %": 2.5, "Инфляция, %": 4.0, "Безработица, %": 6.0}[ind]
            vals = base + np.cumsum(r.normal(0, 0.4, len(years))) * 0.5
            rows += [(ind, iso, y, float(v)) for y, v in zip(years, vals)]
    macro = pd.DataFrame(rows, columns=["indicator", "country", "year", "value"])

    facts = {}
    for i, t in enumerate(["ACME", "GLOBEX", "INITECH", "UMBRELLA"]):
        r = np.random.default_rng(seed + 700 + i)
        yrs = list(range(2015, pd.Timestamp.now().year))
        rev = 1000 * (1 + r.random() * 5) * np.cumprod(1 + r.normal(0.07, 0.05, len(yrs)))
        ni = rev * r.uniform(0.05, 0.2) * (1 + r.normal(0, 0.1, len(yrs)))
        assets = rev * r.uniform(1.2, 2.5)
        equity = assets * r.uniform(0.3, 0.6)
        ocf = ni * r.uniform(1.0, 1.4)
        data = {"Выручка": rev, "Операционная прибыль": ni * 1.3, "Чистая прибыль": ni, "Активы": assets,
                "Обязательства": assets - equity, "Капитал": equity, "Операционный денежный поток": ocf,
                "Капитальные затраты": ocf * r.uniform(0.3, 0.7, len(yrs))}
        facts[t] = pd.DataFrame([(c, f"{y}-12-31", float(v), "10-K", f"{y + 1}-02-15")
                                 for c, arr in data.items() for y, v in zip(yrs, arr)],
                                columns=["concept", "period_end", "value", "form", "filed"])
    return {"index": index, "cross": cross, "fx": fx, "rates": rates, "yields": yields, "macro": macro, "facts": facts}
