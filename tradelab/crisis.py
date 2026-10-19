"""Раннее предупреждение о кризисах по странам (панель страна x год).

События определяются по данным, без ручной разметки (стандарты литературы):
  * валютный кризис (Frankel-Rose): курс к USD ослаб на >=25% и на >=10 п.п. сильнее, чем годом ранее;
    начало эпизода (без кризиса в предыдущие 3 года);
  * инфляционный кризис: инфляция впервые поднялась до >=20% (годом ранее была ниже);
  * рецессия: рост реального ВВП впервые стал отрицательным (годом ранее >=0).
Признаки берутся на конец года t-1, цель: начало события в году t. Проверка только вне выборки:
для каждого тестового года модель учится на строго более ранних годах (walk-forward).
Не прогнозируются: банковские и долговые кризисы (нет открытых готовых меток).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KINDS = {"currency": "Валютный кризис", "inflation": "Инфляционный кризис", "recession": "Рецессия"}
LEVEL_COLS = ["gdp_growth", "inflation", "unemployment", "current_account", "gov_debt", "fiscal_balance", "savings",
              "investment", "output_gap", "export_growth", "import_growth", "reserves_months", "ext_debt_gni",
              "short_debt_reserves", "fdi", "broad_money_growth", "private_credit", "real_interest", "trade_gdp",
              "political_stability", "corruption_control"]
DELTA_COLS = ["inflation", "gdp_growth", "current_account", "gov_debt", "fiscal_balance", "unemployment",
              "private_credit", "reserves_months"]
FEATURE_LABELS = {
    "dep": "Ослабление валюты", "dep_lag": "Ослабление валюты (год назад)", "usd_growth": "Рост ВВП в USD",
    "log_gdp_pc": "Уровень дохода (ВВП на душу, log)", "log_pop": "Размер населения (log)",
}


def derive(panel):
    """Добавляет производные ряды: курс (из ВВП в нац. валюте и USD), ослабление валюты, рост ВВП в USD."""
    df = panel.copy()
    fx = df["official_fx"] if "official_fx" in df else pd.Series(np.nan, index=df.index)
    if {"gdp_lcu", "gdp_usd"} <= set(df.columns):
        fx = fx.fillna(df["gdp_lcu"] / df["gdp_usd"])
    df["fx"] = fx
    g = df.groupby(level="iso3")
    df["dep"] = g["fx"].transform(lambda s: s.pct_change(fill_method=None))
    df["usd_growth"] = g["gdp_usd"].transform(lambda s: s.pct_change(fill_method=None)) if "gdp_usd" in df else np.nan
    if "gdp_usd" in df and "population" in df:
        df["log_gdp_pc"] = np.log((df["gdp_usd"] / df["population"]).where(lambda s: s > 0))
    if "population" in df:
        df["log_pop"] = np.log(df["population"].where(lambda s: s > 0))
    return df.replace([np.inf, -np.inf], np.nan)


def labels(df):
    """Начала событий (bool-таблица по годам) и признак «страна в кризисном состоянии» (исключается из риск-набора)."""
    g = df.groupby(level="iso3")
    dep, dep_prev = df["dep"], g["dep"].shift(1)
    flag_c = (dep >= 0.25) & ((dep - dep_prev) >= 0.10)
    onset_c = flag_c & ~(g_shift(flag_c, 1) | g_shift(flag_c, 2) | g_shift(flag_c, 3))
    infl = df["inflation"] if "inflation" in df else pd.Series(np.nan, index=df.index)
    high = infl >= 20
    onset_i = high & (g_shift(infl, 1) < 20)
    gr = df["gdp_growth"] if "gdp_growth" in df else pd.Series(np.nan, index=df.index)
    onset_r = (gr < 0) & (g_shift(gr, 1) >= 0)
    known = {"currency": dep.notna() & dep_prev.notna(), "inflation": infl.notna() & g_shift(infl, 1).notna(),
             "recession": gr.notna() & g_shift(gr, 1).notna()}
    onset = {"currency": onset_c, "inflation": onset_i, "recession": onset_r}
    # В риск-набор не попадают страны, уже находящиеся в кризисе в предыдущем году
    in_crisis = {"currency": g_shift(flag_c, 1) | g_shift(flag_c, 2), "inflation": g_shift(infl, 1) >= 20,
                 "recession": g_shift(gr, 1) < 0}
    return onset, known, in_crisis


def g_shift(s, n):
    return s.groupby(level="iso3").shift(n)


def features(df):
    """Признаки на конец года y: уровни, изменения, ослабление валюты. Для цели года t берутся значения за t-1."""
    cols = {}
    for c in LEVEL_COLS:
        if c in df:
            cols[c] = df[c]
    g = df.groupby(level="iso3")
    for c in DELTA_COLS:
        if c in df:
            cols[f"d_{c}"] = df[c] - g[c].shift(1)
    for c in ("dep", "usd_growth", "log_gdp_pc", "log_pop"):
        if c in df:
            cols[c] = df[c]
    if "dep" in df:
        cols["dep_lag"] = g["dep"].shift(1)
    F = pd.DataFrame(cols)
    return F.groupby(level="iso3").shift(1)  # информация, известная к началу года t


class Clipper(BaseEstimator, TransformerMixin):
    """Обрезает выбросы по 2-98 перцентилям обучающей выборки (гиперинфляция не ломает модель)."""
    def fit(self, X, y=None):
        self.lo_ = np.nanpercentile(X, 2, axis=0)
        self.hi_ = np.nanpercentile(X, 98, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


def _logit():
    return make_pipeline(SimpleImputer(strategy="median"), Clipper(), StandardScaler(),
                         LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000))


def _gbm():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=2.0,
                                          min_samples_leaf=30, class_weight="balanced", random_state=0)


def dataset(panel, kind):
    """Строки (iso3, год t): признаки, метка, пригодность. Возвращает (X, y) только по риск-набору с известной меткой."""
    df = derive(panel)
    onset, known, in_crisis = labels(df)
    F = features(df)
    mask = known[kind] & ~in_crisis[kind].fillna(False).astype(bool)
    mask &= F.notna().sum(axis=1) >= max(3, int(0.3 * F.shape[1]))
    X = F[mask]
    y = onset[kind][mask].astype(int)
    keep = X.columns[X.notna().mean() >= 0.3]
    return X[keep], y, F, df


def _fit_models(X, y):
    X = X.copy()
    lr = _logit().fit(X.to_numpy(), y.to_numpy())
    gb = _gbm().fit(X.to_numpy(), y.to_numpy())
    lr.pi_ = gb.pi_ = float(np.clip(y.mean(), 1e-4, 0.5))  # частота события в обучении
    return lr, gb


def _prior_correct(p, pi):
    """Модели учатся с балансом классов (эффективная частота 50%), поэтому их числа завышены.
    Поправка на априорную частоту возвращает калиброванные вероятности (порядок стран не меняется)."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    odds = p / (1 - p) * pi / (1 - pi)
    return odds / (1 + odds)


def _score(lr, gb, X):
    p_lr = _prior_correct(lr.predict_proba(X.to_numpy())[:, 1], lr.pi_)
    p_gb = _prior_correct(gb.predict_proba(X.to_numpy())[:, 1], gb.pi_)
    return p_lr, p_gb, 0.5 * (p_lr + p_gb)


def evaluate(panel, kind, first_test_year=2000, last_year=None, refit_every=3):
    """Walk-forward: модель, обученная строго на годах до начала блока, прогнозирует блок из refit_every лет
    (переобучение раз в refit_every лет ускоряет расчёт, будущего модель не видит). Возвращает метрики и прогнозы."""
    X, y, _, _ = dataset(panel, kind)
    years = X.index.get_level_values("year")
    last_year = last_year or int(years.max())
    rows = []
    for Y in range(first_test_year, last_year + 1, refit_every):
        tr, te = years < Y, (years >= Y) & (years < Y + refit_every) & (years <= last_year)
        if y[tr].sum() < 15 or te.sum() == 0:
            continue
        lr, gb = _fit_models(X[tr], y[tr])
        p_lr, p_gb, p_en = _score(lr, gb, X[te])
        part = pd.DataFrame({"y": y[te].to_numpy(), "logit": p_lr, "gbm": p_gb, "ensemble": p_en}, index=X[te].index)
        rows.append(part)
    if not rows:
        raise ValueError("Недостаточно данных для проверки вне выборки")
    preds = pd.concat(rows)
    out = {"events": int(preds["y"].sum()), "obs": int(len(preds)), "base_rate": float(preds["y"].mean()),
           "years": (int(preds.index.get_level_values("year").min()), int(preds.index.get_level_values("year").max()))}
    for m in ("logit", "gbm", "ensemble"):
        p = preds[m]
        top = p >= p.quantile(0.9)
        out[m] = {"auc": float(roc_auc_score(preds["y"], p)), "ap": float(average_precision_score(preds["y"], p)),
                  "top_decile_capture": float(preds.loc[top, "y"].sum() / max(preds["y"].sum(), 1))}
    # Простые ориентиры: «просто смотрим на один признак»
    naive = {"currency": "dep_lag", "inflation": "inflation", "recession": "gdp_growth"}[kind]
    if naive in X.columns:
        s = X[naive].reindex(preds.index)
        sign = -1 if kind == "recession" else 1
        ok = s.notna()
        out["naive"] = {"feature": naive, "auc": float(roc_auc_score(preds.loc[ok, "y"], sign * s[ok]))}
    return out, preds


def calibration(preds, col="ensemble", bins=5):
    q = pd.qcut(preds[col].rank(method="first"), bins, labels=False)
    return preds.groupby(q).agg(pred=(col, "mean"), actual=("y", "mean"), n=("y", "size"))


class CrisisModel:
    """Обученная на всей доступной истории модель одного вида кризиса: прогноз, причины, what-if."""

    def __init__(self, panel, kind):
        self.kind = kind
        X, y, F, df = dataset(panel, kind)
        self.X, self.y, self.F, self.df = X, y, F, df
        self.lr, self.gb = _fit_models(X, y)
        self.columns = list(X.columns)
        self.last_year = int(df.index.get_level_values("year").max())
        self.median_ = X.median()

    def forecast_frame(self, year=None):
        """Признаки для прогноза на год year (по умолчанию год после последнего с данными по ключевым признакам)."""
        F = self.F[self.columns]
        years = F.index.get_level_values("year")
        if year is None:
            cnt = F.notna().sum(axis=1).groupby(level="year").mean()
            ok = cnt[cnt >= 0.5 * len(self.columns)]
            year = int(ok.index.max()) if len(ok) else int(years.max())
        return F[years == year].droplevel("year"), int(year)

    def predict(self, year=None):
        Xf, year = self.forecast_frame(year)
        Xf = Xf[Xf.notna().sum(axis=1) >= max(3, int(0.3 * len(self.columns)))]
        p_lr, p_gb, p_en = _score(self.lr, self.gb, Xf)
        return pd.DataFrame({"logit": p_lr, "gbm": p_gb, "prob": p_en}, index=Xf.index), year

    def contributions(self, year=None):
        """Вклад каждого признака в логит-оценку для всех стран прогнозного года (страны x признаки)."""
        Xf, _ = self.forecast_frame(year)
        imp, clip, sc, clf = self.lr.steps[0][1], self.lr.steps[1][1], self.lr.steps[2][1], self.lr.steps[3][1]
        Z = sc.transform(clip.transform(imp.transform(Xf.to_numpy())))
        return pd.DataFrame(Z * clf.coef_[0], index=Xf.index, columns=self.columns)

    def drivers(self, iso3, year=None, top=6):
        """Вклад признаков в логит-оценку страны относительно типичной (медианной) страны."""
        Xf, _ = self.forecast_frame(year)
        if iso3 not in Xf.index:
            return pd.DataFrame(columns=["feature", "value", "contribution"])
        row = Xf.loc[[iso3]]
        imp, clip, sc, clf = self.lr.steps[0][1], self.lr.steps[1][1], self.lr.steps[2][1], self.lr.steps[3][1]
        Z = sc.transform(clip.transform(imp.transform(row.to_numpy())))[0]
        contrib = clf.coef_[0] * Z
        out = pd.DataFrame({"feature": self.columns, "value": row.iloc[0].to_numpy(), "contribution": contrib})
        return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top)

    def what_if(self, iso3, overrides, year=None):
        """Вероятность по логит-модели при изменённых значениях признаков (сценарий)."""
        Xf, _ = self.forecast_frame(year)
        row = Xf.loc[[iso3]].copy()
        for k, v in overrides.items():
            if k in row.columns:
                row[k] = v
        return float(_prior_correct(self.lr.predict_proba(row.to_numpy())[0, 1], self.lr.pi_))

    def feature_importance(self):
        clf = self.lr.steps[3][1]
        return pd.Series(clf.coef_[0], index=self.columns).sort_values()


# ---------------------------------------------------------------- ожидаемое движение валюты
class FxModel:
    """Прогноз ослабления валюты к USD на следующий год: медиана и интервал P10-P90 (квантильный бустинг)."""

    def __init__(self, panel):
        df = derive(panel)
        F = features(df)
        y = df["dep"]
        ok = y.notna() & (F.notna().sum(axis=1) >= max(3, int(0.3 * F.shape[1])))
        X = F[ok]
        keep = X.columns[X.notna().mean() >= 0.3]
        self.columns = list(keep)
        self.X, self.y = X[keep], y[ok].clip(-0.5, 1.5)
        self.F, self.df = F, df
        self.models = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=3, learning_rate=0.05,
                                                        max_iter=150, min_samples_leaf=30, random_state=0)
                       .fit(self.X.to_numpy(), self.y.to_numpy()) for q in (0.1, 0.5, 0.9)}

    def predict(self, year=None):
        F = self.F[self.columns]
        years = F.index.get_level_values("year")
        if year is None:
            cnt = F.notna().sum(axis=1).groupby(level="year").mean()
            ok = cnt[cnt >= 0.5 * len(self.columns)]
            year = int(ok.index.max()) if len(ok) else int(years.max())
        Xf = F[years == year].droplevel("year")
        Xf = Xf[Xf.notna().sum(axis=1) >= max(3, int(0.3 * len(self.columns)))]
        out = pd.DataFrame({f"p{int(q * 100)}": m.predict(Xf.to_numpy()) for q, m in self.models.items()}, index=Xf.index)
        out[["p10", "p50", "p90"]] = np.sort(out[["p10", "p50", "p90"]].to_numpy(), axis=1)
        return out, int(year)


def evaluate_fx(panel, first_test_year=2000):
    """Walk-forward для интервала: покрытие P10-P90 (должно быть около 80%) и ошибка медианы против «ничего не меняется»."""
    df = derive(panel)
    F = features(df)
    y = df["dep"]
    ok = y.notna() & (F.notna().sum(axis=1) >= max(3, int(0.3 * F.shape[1])))
    X, y = F[ok], y[ok].clip(-0.5, 1.5)
    keep = X.columns[X.notna().mean() >= 0.3]
    X = X[keep]
    years = X.index.get_level_values("year")
    rows = []
    for Y in range(first_test_year, int(years.max()) + 1):
        tr, te = years < Y, years == Y
        if tr.sum() < 200 or te.sum() == 0:
            continue
        preds = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=3, learning_rate=0.05,
                                                  max_iter=100, min_samples_leaf=30, random_state=0)
                 .fit(X[tr].to_numpy(), y[tr].to_numpy()).predict(X[te].to_numpy()) for q in (0.1, 0.5, 0.9)}
        rows.append(pd.DataFrame({"y": y[te].to_numpy(), **{f"p{int(q * 100)}": v for q, v in preds.items()},
                                  "hist_med": float(y[tr].median())}, index=X[te].index))
    r = pd.concat(rows)
    lo, hi = np.minimum(r["p10"], r["p90"]), np.maximum(r["p10"], r["p90"])
    return {"obs": int(len(r)), "coverage_10_90": float(((r["y"] >= lo) & (r["y"] <= hi)).mean()),
            "mae_model": float((r["y"] - r["p50"]).abs().mean()), "mae_naive": float((r["y"] - r["hist_med"]).abs().mean())}, r


def replay(panel, kind, year):
    """«Машина времени»: модель, обученная только на годах < year, оценивает риск на год year; рядом факт.

    Возвращает (таблица [logit, gbm, prob, actual], сводка). В таблице только страны из риск-набора года."""
    X, y, _, _ = dataset(panel, kind)
    years = X.index.get_level_values("year")
    tr, te = years < year, years == year
    if y[tr].sum() < 15 or te.sum() == 0:
        raise ValueError(f"Для {year} года недостаточно истории или данных")
    lr, gb = _fit_models(X[tr], y[tr])
    p_lr, p_gb, p_en = _score(lr, gb, X[te])
    out = pd.DataFrame({"logit": p_lr, "gbm": p_gb, "prob": p_en, "actual": y[te].to_numpy()},
                       index=X[te].index.droplevel("year"))
    n_ev = int(out["actual"].sum())
    top = out["prob"].rank(ascending=False) <= max(1, int(round(0.1 * len(out))))
    summary = {"countries": int(len(out)), "events": n_ev,
               "events_in_top_decile": int(out.loc[top, "actual"].sum()),
               "auc": float(roc_auc_score(out["actual"], out["prob"])) if 0 < n_ev < len(out) else float("nan")}
    return out, summary
