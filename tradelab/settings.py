"""Пользовательские настройки (data/settings.json) и ключи API (.env). Ключи пишутся только в локальный .env и не попадают в репозиторий."""
import json
import os

from . import config

PATH = config.DATA_DIR / "settings.json"
DEFAULTS = {
    "start_page": "ОБЗОР", "map_mode": "Карта", "as_of_year": 2021, "red_zone": 65, "forecast_horizon": 5, "rank_size": 10,
    "show_tape": True, "auto_rotate": True, "auto_refresh_on_start": False, "history_years": 4, "dim_weights": {},
    "show_demo_warnings": True, "decimals": 1,
}
KEYS = {  # переменная окружения -> (название, подсказка)
    "FRED_API_KEY": ("FRED API", "Необязательно: расширенные ряды ФРС (fred.stlouisfed.org/docs/api/api_key.html)"),
    "AISSTREAM_KEY": ("AISStream", "Бесплатный ключ для живых позиций судов (aisstream.io)"),
    "UCDP_TOKEN": ("UCDP API", "Токен для событий конфликтов через API (ucdp.uu.se)"),
    "TRADELAB_CONTACT": ("Контакт для SEC EDGAR", "Почта: SEC просит указывать контакт в запросах"),
}


def load():
    try:
        d = json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    return {**DEFAULTS, **d}


def save(values):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps({k: values[k] for k in DEFAULTS if k in values}, ensure_ascii=False, indent=1), encoding="utf-8")


def save_env(name, value):
    """Записывает/обновляет переменную в локальном .env и в окружении процесса. Пустое значение удаляет ключ."""
    path = config.ROOT / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, done = [], False
    for ln in lines:
        if ln.split("=", 1)[0].strip() == name:
            if value:
                out.append(f"{name}={value}")
            done = True
        else:
            out.append(ln)
    if not done and value:
        out.append(f"{name}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def mask(v):
    return "" if not v else ("•" * max(len(v) - 4, 0)) + v[-4:]
