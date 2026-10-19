"""Проверка подключения к БКС: токен -> access-токен -> свечи SBER за 10 дней."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradelab.bcs import BcsClient, BcsError  # noqa: E402


def main():
    try:
        client = BcsClient()
        print("1/2 получаем access-токен ...")
        client._access_token()
        print("    OK")
        print("2/2 запрашиваем свечи SBER за 10 дней ...")
        end = datetime.now(timezone.utc)
        df = client.candles("SBER", end - timedelta(days=10), end)
        print(df.tail())
        print(f"OK, получено баров: {len(df)}")
    except BcsError as exc:
        print("ОШИБКА:", exc)
        sys.exit(1)
    except Exception as exc:  # сеть, прокси, неожиданный формат ответа
        print("ОШИБКА СЕТИ ИЛИ ФОРМАТА:", repr(exc))
        sys.exit(2)


if __name__ == "__main__":
    main()
