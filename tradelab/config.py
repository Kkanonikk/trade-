"""Настройки из .env (необязательные адреса и контакт; ключи для открытых источников не нужны)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "market.db"

# Мировой универсум по умолчанию: тикер -> (название, сектор). Тикеры в формате Yahoo Finance.
UNIVERSE = {
    "AAPL": ("Apple", "Технологии"), "MSFT": ("Microsoft", "Технологии"), "NVDA": ("Nvidia", "Технологии"),
    "GOOGL": ("Alphabet", "Технологии"), "AMZN": ("Amazon", "Потребительский"), "META": ("Meta", "Технологии"),
    "TSLA": ("Tesla", "Потребительский"), "AVGO": ("Broadcom", "Технологии"), "TSM": ("TSMC", "Технологии"),
    "ASML": ("ASML", "Технологии"), "SAP": ("SAP", "Технологии"), "JPM": ("JPMorgan", "Финансы"),
    "V": ("Visa", "Финансы"), "BRK-B": ("Berkshire Hathaway", "Финансы"), "HSBC": ("HSBC", "Финансы"),
    "XOM": ("Exxon Mobil", "Нефть и газ"), "CVX": ("Chevron", "Нефть и газ"), "SHEL": ("Shell", "Нефть и газ"),
    "JNJ": ("Johnson & Johnson", "Здравоохранение"), "LLY": ("Eli Lilly", "Здравоохранение"), "NVO": ("Novo Nordisk", "Здравоохранение"),
    "WMT": ("Walmart", "Ритейл"), "KO": ("Coca-Cola", "Потребительский"), "NSRGY": ("Nestlé", "Потребительский"),
    "TM": ("Toyota", "Транспорт"), "BHP": ("BHP", "Металлы и добыча"), "RIO": ("Rio Tinto", "Металлы и добыча"),
    "BABA": ("Alibaba", "Технологии"), "SPY": ("S&P 500 ETF", "Фонды"), "QQQ": ("Nasdaq 100 ETF", "Фонды"),
    "EEM": ("Развивающиеся рынки ETF", "Фонды"), "GLD": ("Золото ETF", "Сырьё"), "TLT": ("Облигации США 20+ ETF", "Облигации"),
    "BTC-USD": ("Биткоин", "Крипта"), "ETH-USD": ("Эфир", "Крипта"),
}
DEFAULT_TICKERS = list(UNIVERSE)


def _load_dotenv():
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("\'"))


_load_dotenv()


def get(name, default=""):
    return os.environ.get(name, default)


def sector(ticker):
    return UNIVERSE.get(ticker, ("", "Прочее"))[1]
