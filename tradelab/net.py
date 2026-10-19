"""HTTP-слой для открытых источников: единая сессия, повторы, понятные ошибки."""
import time

import requests

UA = "Mozilla/5.0 (TradeLab research tool; contact: set TRADELAB_CONTACT in .env)"


class SourceError(RuntimeError):
    pass


_session = requests.Session()


def get(url, params=None, headers=None, timeout=30, retries=3, as_json=False):
    head = {"User-Agent": UA, "Accept": "*/*"}
    head.update(headers or {})
    last = None
    for attempt in range(retries):
        try:
            resp = _session.get(url, params=params, headers=head, timeout=timeout)
        except requests.RequestException as exc:  # сеть, прокси, DNS
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
            continue
        if resp.status_code == 200:
            try:
                return resp.json() if as_json else resp.text
            except ValueError as exc:
                raise SourceError(f"{url}: ответ не разобрать как JSON ({exc})") from exc
        if resp.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {resp.status_code}"
            time.sleep(2 ** attempt)
            continue
        raise SourceError(f"{url}: HTTP {resp.status_code} {resp.text[:160]!r}")
    raise SourceError(f"{url}: не удалось получить ответ ({last})")
