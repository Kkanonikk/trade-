"""Долгосрочные (5–10 лет) вероятностные оценки на открытых данных.

Это не «прогноз будущего», а оценки с явной неопределённостью и честной проверкой вне выборки:
  * акции США: реальная 10-летняя доходность по CAPE (набор Шиллера, 1871+);
  * облигации США: 10-летняя доходность по стартовой ставке;
  * портфель: Монте-Карло по блочному бутстрепу исторических лет со сдвигом средних на модельные ожидания;
  * рост ВВП стран на 5–10 лет: конвергентная регрессия с проверкой против наивных ориентиров;
  * госдолг: траектории долг/ВВП со случайными шоками роста, ставки и бюджета.
Окна доходности перекрываются, эффективное число независимых наблюдений мало: точность оценок ниже, чем кажется по R².
"""
import numpy as np
import pandas as pd

from . import panel as panel_mod

SNAP = panel_mod.SNAP


# ---------------------------------------------------------------- данные Шиллера
def load_shiller():
    """Месячные данные: цены, дивиденды, CPI, ставка 10Y, CAPE; строки без данных Шиллера отброшены.
    Возвращает (valid DataFrame, полный ряд номинального S&P 500 до последней даты)."""
    s = pd.read_csv(SNAP / "sp500.csv.gz", parse_dates=["Date"]).set_index("Date").sort_index()
    sp_full = s["SP500"].copy()
    d = s.rename(columns={"SP500": "price", "Dividend": "div", "Earnings": "earn", "Consumer Price Index": "cpi",
                          "Long Interest Rate": "gs10", "Real Price": "rp", "Real Dividend": "rd",
                          "Real Earnings": "re", "PE10": "pe10"})
    valid = d[(d["pe10"] > 0) & (d["rp"] > 0) & (d["cpi"] > 0)].copy()
    return valid, sp_full


def equity_real_tr(d):
    """Индекс реальной полной доходности акций (дивиденды реинвестируются): (RP_t + RD_t/12) / RP_{t-1}."""
    r = (d["rp"] + d["rd"] / 12.0) / d["rp"].shift(1)
    return r.fillna(1.0).cumprod()


def _par_bond_price(y_new, coupon, years=10.0):
    """Цена облигации с купоном coupon (доля, полугодовой) при доходности y_new."""
    n = int(round(years * 2))
    c, y = coupon / 2.0, y_new / 2.0
    ann = (1 - (1 + y) ** (-n)) / y
    return c * ann + (1 + y) ** (-n)


def bond_real_tr(d):
    """Индекс реальной полной доходности 10-летних облигаций (константная срочность): купон + изменение цены,
    дефлируется CPI. Приближение: каждый месяц покупается 10-летний par-бонд по текущей ставке."""
    y = d["gs10"] / 100.0
    y_prev = y.shift(1)
    price = np.array([_par_bond_price(a, b) if np.isfinite(a) and np.isfinite(b) and a > 0 else np.nan
                      for a, b in zip(y, y_prev)])
    nominal = pd.Series(price, index=d.index) + y_prev / 12.0
    nominal = nominal.fillna(1.0)
    infl = d["cpi"] / d["cpi"].shift(1)
    return (nominal / infl).fillna(1.0).cumprod()


def bond_nominal_tr(d):
    """Индекс номинальной полной доходности 10-летних облигаций (константная срочность)."""
    y = d["gs10"] / 100.0
    y_prev = y.shift(1)
    price = np.array([_par_bond_price(a, b) if np.isfinite(a) and np.isfinite(b) and a > 0 else np.nan
                      for a, b in zip(y, y_prev)])
    return (pd.Series(price, index=d.index) + y_prev / 12.0).fillna(1.0).cumprod()


def forward_returns(index, years):
    """Годовая доходность (CAGR) за следующие years лет от каждой даты; для последних дат NaN."""
    h = int(years * 12)
    return (index.shift(-h) / index) ** (1.0 / years) - 1.0


# ---------------------------------------------------------------- регрессия «стартовая оценка -> будущая доходность»
def _fit(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid.std(ddof=2))


def walk_forward(x, fwd, years, start="1920-01-01", step_months=12):
    """Honest-проверка: для даты t обучаемся только на парах (s, fwd_s), у которых s + years <= t (исход уже известен).
    Возвращает DataFrame [pred, actual, hist_mean, sigma] и метрики."""
    h = int(years * 12)
    idx = x.index
    rows = []
    for t in range(np.searchsorted(idx, pd.Timestamp(start)), len(idx), step_months):
        known = idx[: max(t - h, 0)]
        known = known[fwd.reindex(known).notna().to_numpy()]
        if len(known) < 120 or np.isnan(fwd.iloc[t]):
            continue
        beta, sigma = _fit(x.reindex(known).to_numpy(), fwd.reindex(known).to_numpy())
        rows.append((idx[t], beta[0] + beta[1] * x.iloc[t], fwd.iloc[t], float(fwd.reindex(known).mean()), sigma))
    df = pd.DataFrame(rows, columns=["date", "pred", "actual", "hist_mean", "sigma"]).set_index("date")
    if df.empty:
        raise ValueError("Недостаточно истории для проверки")
    sse_m = float(((df["actual"] - df["pred"]) ** 2).sum())
    sse_h = float(((df["actual"] - df["hist_mean"]) ** 2).sum())
    z = 1.2816
    inside = ((df["actual"] >= df["pred"] - z * df["sigma"]) & (df["actual"] <= df["pred"] + z * df["sigma"])).mean()
    dm = (df["pred"] - df["hist_mean"]).to_numpy()
    shrink = float(np.clip((dm * (df["actual"] - df["hist_mean"]).to_numpy()).sum() / max((dm ** 2).sum(), 1e-12), 0, 1))
    metrics = {"n": int(len(df)), "shrink_weight": shrink, "oos_sigma": float((df["actual"] - df["pred"]).std()),
               "oos_r2": 1 - sse_m / sse_h, "mae_model": float((df["actual"] - df["pred"]).abs().mean()),
               "mae_hist": float((df["actual"] - df["hist_mean"]).abs().mean()), "coverage_80": float(inside),
               "corr": float(np.corrcoef(df["pred"], df["actual"])[0, 1])}
    return df, metrics


class EquityBondModel:
    """Ожидаемая 10-летняя доходность акций (реальная, по CAPE) и облигаций (номинальная, по ставке) США.

    Интервалы строятся по реальным внеобразцовым ошибкам walk-forward (перекрывающиеся окна делают внутривыборочную
    сигму слишком оптимистичной). Оценка акций сжимается к историческому среднему с весом, подобранным вне выборки."""

    def __init__(self, years=10):
        self.years = years
        self.d, self.sp_full = load_shiller()
        self.eq_index = equity_real_tr(self.d)
        self.bd_index = bond_real_tr(self.d)
        self.bd_nom_index = bond_nominal_tr(self.d)
        self.eq_fwd = forward_returns(self.eq_index, years)
        self.bd_fwd = forward_returns(self.bd_index, years)          # реальная (для бутстрепа)
        self.bd_nom_fwd = forward_returns(self.bd_nom_index, years)
        self.x_eq = np.log(self.d["pe10"])
        self.x_bd = self.d["gs10"] / 100.0
        ok_e, ok_b = self.eq_fwd.notna(), self.bd_nom_fwd.notna()
        self.beta_eq, self.sigma_eq = _fit(self.x_eq[ok_e].to_numpy(), self.eq_fwd[ok_e].to_numpy())
        self.beta_bd, self.sigma_bd = _fit(self.x_bd[ok_b].to_numpy(), self.bd_nom_fwd[ok_b].to_numpy())
        self.hist_eq = float(self.eq_fwd[ok_e].mean())
        ev = self.evaluate()
        self.metrics = {k: v[1] for k, v in ev.items()}
        self.oos_sigma_eq = self.metrics["equity"]["oos_sigma"]
        self.oos_sigma_bd = self.metrics["bond"]["oos_sigma"]
        self.shrink_eq = self.metrics["equity"]["shrink_weight"]

    def evaluate(self, start="1920-01-01"):
        eq, m_eq = walk_forward(self.x_eq, self.eq_fwd, self.years, start)
        bd, m_bd = walk_forward(self.x_bd, self.bd_nom_fwd, self.years, start)
        return {"equity": (eq, m_eq), "bond": (bd, m_bd)}

    def current_cape(self, infl=0.027):
        """CAPE на последнюю дату: последняя оценка Шиллера, пересчитанная на движение цены S&P 500 с учётом
        роста 10-летней средней прибыли (экстраполяция тренда) и инфляции. Грубая оценка."""
        last = self.d.index.max()
        e10 = (self.d["rp"] / self.d["pe10"])
        g = float(np.log(e10.iloc[-1] / e10.iloc[-61]) / 5) if len(e10) > 61 else 0.02
        now = self.sp_full.index.max()
        years = (now - last).days / 365.25
        price_ratio = self.sp_full.iloc[-1] / self.d["price"].iloc[-1]
        real_price_ratio = price_ratio / (1 + infl) ** years
        cape = float(self.d["pe10"].iloc[-1] * real_price_ratio / np.exp(g * years))
        return cape, last, now

    def estimate(self, cape=None, bond_yield=None, infl=0.027):
        """Оценка следующих `years` лет: акции (реальная) и облигации (номинальная и реальная) с интервалом 10–90%.
        'raw' это чистая CAPE-модель, 'mean' её сжатие к историческому среднему."""
        if cape is None:
            cape, _, _ = self.current_cape(infl)
        if bond_yield is None:
            y = pd.read_csv(SNAP / "us10y.csv.gz")
            bond_yield = float(y["Rate"].iloc[-1]) / 100.0
        z = 1.2816
        raw = self.beta_eq[0] + self.beta_eq[1] * np.log(cape)
        eq_mu = self.hist_eq + self.shrink_eq * (raw - self.hist_eq)
        bd_nom = self.beta_bd[0] + self.beta_bd[1] * bond_yield
        bd_real = (1 + bd_nom) / (1 + infl) - 1
        return {"cape": cape, "bond_yield": bond_yield, "hist_equity": self.hist_eq,
                "equity": {"raw": raw, "mean": eq_mu, "p10": eq_mu - z * self.oos_sigma_eq, "p90": eq_mu + z * self.oos_sigma_eq},
                "bond": {"nominal": bd_nom, "mean": bd_real, "p10": bd_real - z * self.oos_sigma_bd, "p90": bd_real + z * self.oos_sigma_bd},
                "nominal_equity": (1 + eq_mu) * (1 + infl) - 1, "nominal_bond": bd_nom}

    def annual_history(self):
        """Годовые реальные доходности (акции, облигации) за все полные годы: основа бутстрепа."""
        eq = self.eq_index.resample("YE").last().pct_change().dropna()
        bd = self.bd_index.resample("YE").last().pct_change().dropna()
        return pd.concat([eq.rename("equity"), bd.rename("bond")], axis=1).dropna()


# ---------------------------------------------------------------- портфель: Монте-Карло
ASSETS = ("equity", "bond", "cash", "gold")


def simulate_portfolio(hist, mu, weights, years=10, n=5000, start=100_000.0, contribution=0.0, block=3,
                       cash_real=0.0, gold_real=0.0, gold_vol=0.16, cash_vol=0.01, seed=0, shift=None):
    """Траектории реального капитала. hist: годовые реальные доходности (equity, bond); mu: целевые средние
    {'equity','bond'}; shift: дополнительные поправки к средним (сценарий). Бутстреп блоками по `block` лет."""
    rng = np.random.default_rng(seed)
    H = hist[["equity", "bond"]].to_numpy()
    H = H - H.mean(axis=0)  # сохраняем форму распределения и связь, средние задаём сами
    target = np.array([mu["equity"], mu["bond"]]) + np.array([(shift or {}).get("equity", 0.0), (shift or {}).get("bond", 0.0)])
    T = len(H)
    paths = np.empty((n, years))
    for i in range(n):
        out = []
        while len(out) < years:
            s = rng.integers(0, T)
            out.extend(H[(s + np.arange(block)) % T])
        R = np.array(out[:years]) + target
        paths[i, :, ] = 0
        cash = cash_real + (shift or {}).get("cash", 0.0) + rng.normal(0, cash_vol, years)
        gold = gold_real + (shift or {}).get("gold", 0.0) + rng.normal(0, gold_vol, years)
        ret = np.column_stack([R[:, 0], R[:, 1], cash, gold])
        w = np.array([weights.get(a, 0.0) for a in ASSETS])
        paths[i] = ret @ (w / w.sum())
    wealth = np.empty((n, years + 1))
    wealth[:, 0] = start
    for t in range(years):
        wealth[:, t + 1] = (wealth[:, t] + (contribution if t > 0 else 0.0)) * (1 + paths[:, t])
    return wealth, paths


def summarize_wealth(wealth, start, contribution=0.0, target=None):
    years = wealth.shape[1] - 1
    invested = start + contribution * (years - 1)
    term = wealth[:, -1]
    peak = np.maximum.accumulate(wealth, axis=1)
    dd = (wealth / peak - 1).min(axis=1)
    out = {"p5": float(np.percentile(term, 5)), "p25": float(np.percentile(term, 25)),
           "p50": float(np.percentile(term, 50)), "p75": float(np.percentile(term, 75)),
           "p95": float(np.percentile(term, 95)), "prob_real_loss": float((term < invested).mean()),
           "median_drawdown": float(np.median(dd)), "p10_drawdown": float(np.percentile(dd, 10)), "invested": invested}
    if target:
        out["prob_target"] = float((term >= target).mean())
    return out


SCENARIOS = {
    "База": {"equity": 0.0, "bond": 0.0, "cash": 0.0, "gold": 0.0},
    "Стагфляция (реальные ставки вниз)": {"equity": -0.03, "bond": -0.025, "cash": -0.02, "gold": 0.03},
    "Дефляционный кризис": {"equity": -0.02, "bond": 0.015, "cash": 0.01, "gold": 0.01},
    "Продуктивный подъём": {"equity": 0.02, "bond": 0.005, "cash": 0.0, "gold": -0.01},
    "Рост ставок и сжатие оценок": {"equity": -0.02, "bond": 0.01, "cash": 0.01, "gold": -0.02},
}


# ---------------------------------------------------------------- рост ВВП стран на 5–10 лет
GROWTH_FEATURES = ["log_gdp_pc", "investment", "savings", "past_growth", "inflation_c", "unemployment", "current_account",
                   "gov_debt"]


def growth_dataset(panel, horizon):
    """Строки (страна, год t): признаки на конец t, цель = средний рост реального ВВП за t+1..t+h."""
    df = panel.copy()
    g = df.groupby(level="iso3")
    gdp_pc = (df["gdp_usd"] / df["population"]) if {"gdp_usd", "population"} <= set(df.columns) else None
    feats = pd.DataFrame(index=df.index)
    feats["log_gdp_pc"] = np.log(gdp_pc.where(gdp_pc > 0)) if gdp_pc is not None else np.nan
    for c in ("investment", "savings", "unemployment", "current_account", "gov_debt"):
        feats[c] = df[c] if c in df else np.nan
    growth = df["gdp_growth"]
    feats["past_growth"] = growth.groupby(level="iso3").transform(lambda s: s.rolling(5, min_periods=3).mean())
    feats["inflation_c"] = np.log1p(df["inflation"].clip(-5, 100) / 10) if "inflation" in df else np.nan
    fwd = growth.groupby(level="iso3").transform(lambda s: s[::-1].rolling(horizon, min_periods=horizon).mean()[::-1].shift(-1))
    return feats, fwd


def growth_backtest(panel, horizon=5, first_origin=1995, last_actual=panel_mod.WEO_LAST_ACTUAL):
    """Для каждого года-происхождения t: обучение на парах, исход которых известен к t, прогноз среднего роста на h лет.
    Сравнение с наивными ориентирами: прошлый 5-летний рост и мировое среднее."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    feats, fwd = growth_dataset(panel, horizon)
    years = feats.index.get_level_values("year")
    rows = []
    for t in range(first_origin, last_actual - horizon + 1):
        tr = (years + horizon <= t) & fwd.notna().to_numpy() & (feats.notna().sum(axis=1) >= 3).to_numpy()
        te = (years == t) & fwd.notna().to_numpy() & (feats.notna().sum(axis=1) >= 3).to_numpy()
        if tr.sum() < 200 or te.sum() == 0:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=30.0)).fit(
            feats[tr][GROWTH_FEATURES].to_numpy(), fwd[tr].clip(-10, 15).to_numpy())
        pred = model.predict(feats[te][GROWTH_FEATURES].to_numpy())
        part = pd.DataFrame({"actual": fwd[te].to_numpy(), "model": pred,
                             "naive_past": feats[te]["past_growth"].fillna(float(fwd[tr].median())).to_numpy(),
                             "world_mean": float(fwd[tr].mean())}, index=feats[te].index)
        rows.append(part)
    r = pd.concat(rows)
    out = {"obs": int(len(r)), "origins": (int(r.index.get_level_values("year").min()), int(r.index.get_level_values("year").max()))}
    for k in ("model", "naive_past", "world_mean"):
        out[f"mae_{k}"] = float((r["actual"] - r[k]).abs().mean())
    out["skill_vs_naive"] = 1 - out["mae_model"] / out["mae_naive_past"]
    out["skill_vs_world"] = 1 - out["mae_model"] / out["mae_world_mean"]
    out["corr"] = float(np.corrcoef(r["model"], r["actual"])[0, 1])
    return out, r


def growth_forecast(panel, horizon=5, year=None):
    """Прогноз среднего годового роста на h лет вперёд для всех стран по последним доступным признакам."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    feats, fwd = growth_dataset(panel, horizon)
    ok = fwd.notna() & (feats.notna().sum(axis=1) >= 3)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=30.0)).fit(
        feats[ok][GROWTH_FEATURES].to_numpy(), fwd[ok].clip(-10, 15).to_numpy())
    years = feats.index.get_level_values("year")
    year = year or int(panel["gdp_growth"].dropna().index.get_level_values("year").max())  # последний год с фактом
    cur = feats[(years == year) & (feats.notna().sum(axis=1) >= 3)].droplevel("year")
    pred = pd.Series(model.predict(cur[GROWTH_FEATURES].to_numpy()), index=cur.index)
    resid_sd = float((fwd[ok].clip(-10, 15) - model.predict(feats[ok][GROWTH_FEATURES].to_numpy())).std())
    return pd.DataFrame({"growth": pred, "p10": pred - 1.2816 * resid_sd, "p90": pred + 1.2816 * resid_sd}), year


# ---------------------------------------------------------------- госдолг
def debt_paths(d0, g_mean, infl, r_nominal, primary_balance, years=10, n=5000, g_sd=2.5, r_sd=1.0, pb_sd=1.0,
               infl_sd=1.5, seed=0):
    """Траектории долга/ВВП (% ВВП). d_{t+1} = d_t (1+r)/((1+g)(1+π)) − pb, где g реальный рост, π дефлятор, pb первичный баланс
    (% ВВП, плюс = профицит). Шоки независимые нормальные (простое допущение). Все ставки в процентах."""
    rng = np.random.default_rng(seed)
    d = np.full(n, float(d0))
    path = [d.copy()]
    for _ in range(years):
        g = (g_mean + rng.normal(0, g_sd, n)) / 100
        pi = (infl + rng.normal(0, infl_sd, n)) / 100
        r = (r_nominal + rng.normal(0, r_sd, n)) / 100
        pb = primary_balance + rng.normal(0, pb_sd, n)
        d = d * (1 + r) / ((1 + g) * (1 + pi)) - pb
        path.append(d.copy())
    return np.array(path).T


def stabilizing_balance(d0, g_mean, infl, r_nominal):
    """Первичный баланс (% ВВП), при котором долг/ВВП остаётся на уровне d0: pb* = d (r − g_nom)/(1 + g_nom)."""
    g_nom = (1 + g_mean / 100) * (1 + infl / 100) - 1
    return d0 * (r_nominal / 100 - g_nom) / (1 + g_nom)
