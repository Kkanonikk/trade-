"""Торговые потоки между странами: загрузка (UN Comtrade, CSV), матрицы, сеть, каналы заражения.

Данные Comtrade: для каждой пары есть две оценки (экспорт по отчёту A и импорт по отчёту B); берём экспорт
отчитавшейся страны, а если его нет, импорт партнёра ('зеркало').
"""
import time

import networkx as nx
import numpy as np
import pandas as pd

from . import countries, panel as panel_mod
from .net import SourceError, get


# ---------------------------------------------------------------- загрузка
def parse_comtrade(payload, flow):
    """Ответ preview-API Comtrade -> reporter, partner (ISO3), year, flow, value. Агрегаты (мир, группы) отбрасываются."""
    rows = (payload or {}).get("data")
    if rows is None:
        raise SourceError("Comtrade: в ответе нет поля data")
    df = pd.json_normalize(rows)
    if df.empty:
        return pd.DataFrame(columns=["reporter", "partner", "year", "flow", "value"])
    for col, val in (("motCode", 0), ("partner2Code", 0)):  # без разбивки по видам транспорта и третьим странам
        if col in df:
            df = df[pd.to_numeric(df[col], errors="coerce").fillna(0) == val]
    if "customsCode" in df:
        df = df[df["customsCode"].astype(str).isin(["C00", "0", "nan"])]
    table = countries.table().index
    year_col = "refYear" if "refYear" in df else "period"
    out = pd.DataFrame({"reporter": df["reporterISO"], "partner": df["partnerISO"], "year": pd.to_numeric(df[year_col]),
                        "flow": flow, "value": pd.to_numeric(df["primaryValue"], errors="coerce")})
    out = out[out["partner"].isin(table) & out["reporter"].isin(table) & (out["reporter"] != out["partner"])]
    return out.groupby(["reporter", "partner", "year", "flow"], as_index=False)["value"].sum()


def fetch_comtrade_reporter(reporter_iso3, year, flow="X"):
    code = countries.iso3_to_numeric().get(reporter_iso3)
    if code is None:
        raise SourceError(f"Нет числового кода ISO для {reporter_iso3}")
    payload = get("https://comtradeapi.un.org/public/v1/preview/C/A/HS",
                  params={"reportercode": code, "flowCode": flow, "period": year, "cmdCode": "TOTAL"}, as_json=True)
    return parse_comtrade(payload, flow)


def fetch_comtrade(reporters, year, progress=None, pause=1.2):
    """Экспорт и импорт по списку стран-отчитывающихся; сбой одной страны не останавливает остальных."""
    frames, report = [], []
    jobs = [(r, f) for r in reporters for f in ("X", "M")]
    for i, (rep, flow) in enumerate(jobs):
        try:
            df = fetch_comtrade_reporter(rep, year, flow)
            frames.append(df)
            report.append((f"{rep} {flow}", True, f"{len(df)} партнёров"))
        except Exception as exc:
            report.append((f"{rep} {flow}", False, str(exc)[:160]))
        if progress:
            progress((i + 1) / len(jobs), f"{rep} {flow}")
        time.sleep(pause)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), report


def from_csv(df, reporter="reporter", partner="partner", value="value", year="year", flow=None):
    """Универсальный импорт (BACI, Comtrade-выгрузки, WITS): колонки задаются именами; страны в ISO3."""
    out = pd.DataFrame({"reporter": df[reporter].astype(str).str.upper(), "partner": df[partner].astype(str).str.upper(),
                        "year": pd.to_numeric(df[year]), "value": pd.to_numeric(df[value], errors="coerce")})
    out["flow"] = df[flow].astype(str).str.upper().str[0] if flow else "X"
    table = countries.table().index
    return out[out["reporter"].isin(table) & out["partner"].isin(table) & (out["reporter"] != out["partner"])].dropna()


# ---------------------------------------------------------------- матрицы и аналитика
def export_matrix(trade, year=None):
    """Квадратная матрица E[i, j] = экспорт i -> j (USD). Экспорт отчёта, иначе зеркальный импорт партнёра."""
    t = trade if year is None else trade[trade["year"] == year]
    exp = t[t["flow"] == "X"].groupby(["reporter", "partner"])["value"].sum()
    imp = t[t["flow"] == "M"].groupby(["partner", "reporter"])["value"].sum()  # партнёр -> отчитавшаяся: экспорт партнёра
    imp.index.names = ["reporter", "partner"]
    combined = exp.combine_first(imp)
    names = sorted(set(combined.index.get_level_values(0)) | set(combined.index.get_level_values(1)))
    return combined.unstack().reindex(index=names, columns=names).fillna(0.0)


def gdp_usd(year, panel=None):
    """ВВП в USD (не в млрд) на последний год не позже year; панель по умолчанию из снимка."""
    p = panel if panel is not None else panel_mod.snapshot_panel()
    s = p["gdp_usd"].dropna().reset_index()
    s = s[s["year"] <= year].sort_values("year").groupby("iso3")["gdp_usd"].last()
    return s * 1e9


def top_flows(E, n=30):
    s = E.stack()
    s = s[s > 0].nlargest(n)
    return pd.DataFrame({"from": s.index.get_level_values(0), "to": s.index.get_level_values(1), "value": s.values})


def partner_table(E, iso3):
    """Крупнейшие партнёры страны: экспорт, импорт, баланс и доли."""
    if iso3 not in E.index:
        return pd.DataFrame()
    out = pd.DataFrame({"export": E.loc[iso3], "import": E[iso3]})
    out = out[(out["export"] > 0) | (out["import"] > 0)].copy()
    out["balance"] = out["export"] - out["import"]
    out["export_share"] = out["export"] / out["export"].sum() if out["export"].sum() else np.nan
    out["import_share"] = out["import"] / out["import"].sum() if out["import"].sum() else np.nan
    return out.sort_values("export", ascending=False)


def balances(E):
    """Сводка по странам: экспорт, импорт, сальдо, концентрация партнёров (HHI)."""
    exp, imp = E.sum(axis=1), E.sum(axis=0)
    hhi = ((E.div(E.sum(axis=1).replace(0, np.nan), axis=0)) ** 2).sum(axis=1)
    return pd.DataFrame({"export": exp, "import": imp, "balance": exp - imp, "export_hhi": hhi})


def network(E, min_share=0.0):
    """Взвешенный ориентированный граф торговли. Центральность: PageRank по весам (кто «важен» как рынок сбыта)."""
    G = nx.DiGraph()
    thresh = E.to_numpy().sum() * min_share
    for i in E.index:
        for j in E.columns:
            if E.loc[i, j] > thresh and i != j:
                G.add_edge(i, j, weight=float(E.loc[i, j]))
    return G


def centrality(E):
    G = network(E)
    # PageRank по обратным рёбрам: важна страна, куда много экспортируют важные страны (рынок сбыта)
    pr = nx.pagerank(G.reverse(copy=True), weight="weight")
    und = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d["weight"] + (G[v][u]["weight"] if G.has_edge(v, u) else 0)
        und.add_edge(u, v, weight=w)
    comm = nx.community.greedy_modularity_communities(und, weight="weight")
    membership = {n: k for k, c in enumerate(sorted(comm, key=len, reverse=True)) for n in c}
    return pd.DataFrame({"pagerank": pd.Series(pr), "community": pd.Series(membership),
                         "out_strength": E.sum(axis=1), "in_strength": E.sum(axis=0)}).fillna(0)


def contagion(E, gdp, crisis_prob, import_drop=0.2, supply_weight=0.5):
    """Ожидаемый удар по ВВП страны i через торговлю при кризисах у партнёров (в долях ВВП).

    Канал спроса: i теряет экспорт в j: E[i,j]/GDP_i * P_j * import_drop.
    Канал поставок: i теряет импортные ресурсы из j: E[j,i]/GDP_i * P_j * import_drop * supply_weight.
    P_j: вероятность кризиса j (или шок 0..1 для сценария). Это оценка экспозиции при допущениях, а не прогноз."""
    names = [n for n in E.index if n in gdp.index]
    E2 = E.reindex(index=names, columns=names).fillna(0.0)
    P = pd.Series(crisis_prob).reindex(names).fillna(0.0)
    g = gdp.reindex(names)
    demand = (E2.to_numpy() * P.to_numpy()[None, :]).sum(axis=1) / g.to_numpy()
    supply = (E2.to_numpy().T * P.to_numpy()[None, :]).sum(axis=1) / g.to_numpy()
    out = pd.DataFrame({"demand": demand * import_drop, "supply": supply * import_drop * supply_weight}, index=names)
    out["total"] = out["demand"] + out["supply"]
    return out


def contagion_sources(E, gdp, target, crisis_prob, import_drop=0.2, supply_weight=0.5):
    """Через кого именно страна target подвержена заражению: вклад каждого партнёра в её ожидаемый удар."""
    if target not in E.index or target not in gdp.index:
        return pd.Series(dtype=float)
    P = pd.Series(crisis_prob).reindex(E.columns).fillna(0.0)
    g = gdp[target]
    demand = E.loc[target] / g * P * import_drop
    supply = E[target] / g * P * import_drop * supply_weight
    return (demand + supply).sort_values(ascending=False)


# ---------------------------------------------------------------- демо (синтетика по гравитационной модели)
def demo(year=2022, n=60, seed=5):
    """Синтетическая матрица торговли по гравитационной модели (НЕ реальные данные): T ~ GDP_i^0.9 * GDP_j^0.9 / dist."""
    rng = np.random.default_rng(seed)
    p = panel_mod.snapshot_panel()
    g = p["gdp_usd"].dropna().groupby(level="iso3").last().nlargest(n)
    names = [c for c in g.index if countries.centroid(c)]
    lat = np.radians([countries.centroid(c)[0] for c in names])
    lon = np.radians([countries.centroid(c)[1] for c in names])
    cosd = np.sin(lat)[:, None] * np.sin(lat)[None, :] + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.cos(lon[:, None] - lon[None, :])
    dist = 6371 * np.arccos(np.clip(cosd, -1, 1)) + 500
    gv = g[names].to_numpy() * 1e9
    T = gv[:, None] ** 0.9 * gv[None, :] ** 0.9 / dist
    T *= rng.lognormal(0, 0.6, T.shape)
    np.fill_diagonal(T, 0)
    T *= 0.3 * gv.sum() / T.sum()  # торговля около 30% ВВП в сумме
    rows = [(names[i], names[j], year, "X", float(T[i, j])) for i in range(len(names)) for j in range(len(names)) if i != j]
    return pd.DataFrame(rows, columns=["reporter", "partner", "year", "flow", "value"])
