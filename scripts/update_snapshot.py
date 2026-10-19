"""Обновляет встроенный снимок реальных открытых данных (assets/snapshot) из зеркал на GitHub (datasets/*).
Снимок позволяет работать без сети и без ключей. Запуск: python scripts/update_snapshot.py"""
import io
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "assets" / "snapshot"
RAW = "https://raw.githubusercontent.com/datasets"
FILES = {  # имя -> (путь в репозитории, что это)
    "weo_values": ("imf-weo/main/data/values.csv", "IMF WEO: 45 макропоказателей, страны, 1980+ (включая проекции)"),
    "weo_indicators": ("imf-weo/main/data/indicators.csv", "справочник показателей WEO"),
    "gdp": ("gdp/main/data/gdp.csv", "World Bank: ВВП, текущие USD"),
    "inflation": ("inflation/main/data/inflation-consumer.csv", "World Bank: инфляция CPI, % г/г"),
    "population": ("population/main/data/population.csv", "World Bank: население"),
    "fx_daily": ("exchange-rates/main/data/daily.csv", "ФРС США (H.10): курсы валют к USD, дневные"),
    "fx_monthly": ("exchange-rates/main/data/monthly.csv", "ФРС США (H.10): курсы валют к USD, месячные"),
    "brent": ("oil-prices/main/data/brent-daily.csv", "Нефть Brent, дневные"),
    "natgas": ("natural-gas/main/data/daily.csv", "Газ Henry Hub, дневные"),
    "gold": ("gold-prices/main/data/monthly.csv", "Золото, месячные"),
    "vix": ("finance-vix/main/data/vix-daily.csv", "Индекс VIX, дневные"),
    "sp500": ("s-and-p-500/main/data/data.csv", "S&P 500, месячные"),
    "us10y": ("bond-yields-us-10y/main/data/monthly.csv", "Доходность US 10Y, месячная"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"updated": date.today().isoformat(), "files": {}}
    for name, (path, desc) in FILES.items():
        resp = requests.get(f"{RAW}/{path}", timeout=120)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_csv(OUT / f"{name}.csv.gz", index=False, compression="gzip")
        manifest["files"][name] = {"desc": desc, "rows": len(df), "source": f"github.com/datasets/{path.split('/')[0]}"}
        print(f"{name}: {len(df)} строк")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
