"""Диагностика открытых источников: python scripts/check_sources.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradelab import sources  # noqa: E402

if __name__ == "__main__":
    bad = 0
    for name, ok, msg in sources.probe_all():
        print(("[ OK ]" if ok else "[FAIL]"), name, "-", msg)
        bad += not ok
    print(f"\nИсточников с ошибкой: {bad}. Работающие запасные каналы (Stooq, MOEX ISS) закрывают часть сбоев.")
    sys.exit(1 if bad else 0)
