"""Настройки из .env (токен и адреса API лежат только там, не в коде)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "market.db"

DEFAULT_TICKERS = [
    "SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "YDEX", "TATN",
    "SNGS", "PLZL", "CHMF", "NLMK", "MGNT", "MTSS", "MOEX", "VTBR",
    "ALRS", "PHOR", "AFLT", "T",
]
CLASS_CODE = "TQBR"


def _load_dotenv():
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def get(name, default=""):
    return os.environ.get(name, default)
