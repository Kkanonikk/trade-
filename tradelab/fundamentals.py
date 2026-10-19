"""Финансовая отчётность компаний из открытого SEC EDGAR (XBRL). Только компании, подающие отчёты в SEC
(США и иностранные эмитенты). Для российских компаний открытого API нет: используйте импорт CSV."""
import pandas as pd

from . import config
from .net import SourceError, get

CONCEPTS = {
    "Выручка": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "Операционная прибыль": ["OperatingIncomeLoss"],
    "Чистая прибыль": ["NetIncomeLoss", "ProfitLoss"],
    "Активы": ["Assets"],
    "Обязательства": ["Liabilities"],
    "Капитал": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "Операционный денежный поток": ["NetCashProvidedByUsedInOperatingActivities"],
    "Капитальные затраты": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "Деньги": ["CashAndCashEquivalentsAtCarryingValue"],
    "EPS (разводнённая)": ["EarningsPerShareDiluted"],
}
ANNUAL_FORMS = {"10-K", "20-F", "40-F", "10-K/A"}


def _headers():
    contact = config.get("TRADELAB_CONTACT", "")
    return {"User-Agent": f"TradeLab research {contact}".strip(), "Accept-Encoding": "gzip, deflate"}


def parse_companyfacts(payload):
    """JSON companyfacts -> годовые значения: concept (русское имя), period_end, value, form, filed."""
    taxonomies = payload.get("facts", {})
    rows = []
    for ru_name, tags in CONCEPTS.items():
        for tag in tags:
            for tax in taxonomies.values():
                node = tax.get(tag)
                if not node:
                    continue
                for unit, entries in node.get("units", {}).items():
                    for e in entries:
                        if e.get("fp") == "FY" and e.get("form") in ANNUAL_FORMS:
                            rows.append((ru_name, e["end"], e["val"], e["form"], e.get("filed", "")))
            if any(r[0] == ru_name for r in rows):
                break  # первый подходящий тег для понятия
    df = pd.DataFrame(rows, columns=["concept", "period_end", "value", "form", "filed"])
    if df.empty:
        return df
    return (df.sort_values("filed").drop_duplicates(["concept", "period_end"], keep="last")
              .sort_values(["concept", "period_end"]).reset_index(drop=True))


def fetch_sec(ticker):
    tickers = get("https://www.sec.gov/files/company_tickers.json", headers=_headers(), as_json=True)
    cik = next((v["cik_str"] for v in tickers.values() if v["ticker"].upper() == ticker.upper()), None)
    if cik is None:
        raise SourceError(f"SEC: тикер {ticker} не найден (нужен эмитент, подающий отчёты в SEC)")
    payload = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json",
                  headers=_headers(), as_json=True)
    return parse_companyfacts(payload)


def table(facts):
    """Широкая таблица по годам: строки — годы, колонки — показатели."""
    if facts.empty:
        return pd.DataFrame()
    t = facts.pivot_table(index="period_end", columns="concept", values="value")
    t.index = pd.to_datetime(t.index).year
    return t[~t.index.duplicated(keep="last")].sort_index()


def ratios(t):
    """Основные коэффициенты по годовой таблице (где хватает данных)."""
    out = pd.DataFrame(index=t.index)
    g = lambda c: t[c] if c in t else pd.Series(float("nan"), index=t.index)  # noqa: E731
    rev, ni, op = g("Выручка"), g("Чистая прибыль"), g("Операционная прибыль")
    out["Рост выручки"] = rev.pct_change()
    out["Операционная маржа"] = op / rev
    out["Чистая маржа"] = ni / rev
    out["ROE"] = ni / g("Капитал")
    out["ROA"] = ni / g("Активы")
    out["Долг / капитал"] = g("Обязательства") / g("Капитал")
    fcf = g("Операционный денежный поток") - g("Капитальные затраты")
    out["FCF"] = fcf
    out["FCF маржа"] = fcf / rev
    return out
