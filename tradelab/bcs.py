"""Клиент БКС Trade API (только чтение рыночных данных).

Адреса и формат ответа написаны по документации БКС (trade-api.bcs.ru), но
не проверялись вживую из этой среды разработки. Проверить у себя:
    python scripts/check_bcs.py
Адреса вынесены в .env (BCS_AUTH_URL, BCS_API_URL), чтобы их можно было поправить.
"""
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import config

# Максимум 1000 баров за запрос, поэтому диапазон режем на окна (в днях).
CHUNK_DAYS = {"D": 900, "W": 5000, "MN": 20000, "H4": 300, "H1": 100,
              "M30": 50, "M15": 25, "M5": 8, "M1": 2}
TIMEFRAMES = list(CHUNK_DAYS)


class BcsError(RuntimeError):
    pass


class BcsClient:
    def __init__(self, refresh_token=None, client_id=None, auth_url=None, api_url=None):
        self.refresh_token = refresh_token or config.get("BCS_REFRESH_TOKEN")
        self.client_id = client_id or config.get("BCS_CLIENT_ID", "trade-api-read")
        self.auth_url = auth_url or config.get("BCS_AUTH_URL")
        self.api_url = (api_url or config.get("BCS_API_URL")).rstrip("/")
        if not self.refresh_token:
            raise BcsError("Не задан BCS_REFRESH_TOKEN (см. .env.example)")
        self._access = None
        self._expires = 0.0
        self.session = requests.Session()

    def _access_token(self):
        if self._access and time.time() < self._expires - 30:
            return self._access
        resp = self.session.post(
            self.auth_url,
            data={"client_id": self.client_id, "grant_type": "refresh_token",
                  "refresh_token": self.refresh_token},
            timeout=30,
        )
        if resp.status_code != 200:
            raise BcsError(f"Не удалось получить access-токен: {resp.status_code} {resp.text[:300]}")
        body = resp.json()
        self._access = body["access_token"]
        self._expires = time.time() + int(body.get("expires_in", 300))
        return self._access

    def _get(self, path, params, retries=3):
        for attempt in range(retries):
            resp = self.session.get(
                f"{self.api_url}/{path}", params=params, timeout=60,
                headers={"Authorization": f"Bearer {self._access_token()}"},
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 401:
                self._access = None
                continue
            if resp.status_code != 200:
                raise BcsError(f"{path}: {resp.status_code} {resp.text[:300]}")
            return resp.json()
        raise BcsError(f"{path}: запрос не удался после {retries} попыток")

    def candles_raw(self, ticker, start, end, timeframe="D", class_code=None):
        return self._get("candles-chart", {
            "classCode": class_code or config.CLASS_CODE,
            "ticker": ticker,
            "startDate": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endDate": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timeFrame": timeframe,
        })

    def candles(self, ticker, start, end=None, timeframe="D", class_code=None):
        """Свечи за период (с разбиением на окна). Колонки: ts, open, high, low, close, volume."""
        end = end or datetime.now(timezone.utc)
        step = timedelta(days=CHUNK_DAYS[timeframe])
        frames, cursor = [], start
        while cursor < end:
            window_end = min(cursor + step, end)
            payload = self.candles_raw(ticker, cursor, window_end, timeframe, class_code)
            frames.append(parse_bars(payload))
            cursor = window_end
            time.sleep(0.15)
        frames = [f for f in frames if not f.empty]
        if not frames:
            return parse_bars({"bars": []})
        return (pd.concat(frames).drop_duplicates("ts").sort_values("ts")
                .reset_index(drop=True))


def parse_bars(payload):
    bars = payload.get("bars") or []
    cols = ["ts", "open", "high", "low", "close", "volume"]
    if not bars:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(bars)
    df = df.rename(columns={"time": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df[cols]
