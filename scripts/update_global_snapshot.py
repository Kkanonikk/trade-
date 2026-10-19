"""Обновляет глобальный снимок открытых данных (assets/snapshot): Gapminder Systema Globalis (CC BY 4.0, данные ООН,
ВОЗ, Всемирного банка и др.), Our World in Data CO2 и энергия. Запуск: python scripts/update_global_snapshot.py"""
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tradelab import registry  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "assets" / "snapshot"
GM = "https://raw.githubusercontent.com/open-numbers/ddf--gapminder--systema_globalis/master/countries-etc-datapoints"
OWID = "https://raw.githubusercontent.com/owid"


def fetch_gm(name):
    r = requests.get(f"{GM}/ddf--datapoints--{name}--by--geo--time.csv", timeout=120)
    if r.status_code != 200:
        print("пропущен (нет файла):", name, r.status_code)
        return None
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["geo", "year", "value"]
    df["indicator"] = name
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = list(registry.GAPMINDER)
    with ThreadPoolExecutor(8) as ex:
        frames = list(ex.map(fetch_gm, names))
    gm = pd.concat([f for f in frames if f is not None], ignore_index=True)
    gm["geo"] = gm["geo"].str.upper()
    gm["value"] = pd.to_numeric(gm["value"], errors="coerce")
    gm = gm.dropna(subset=["value"])
    gm.to_csv(OUT / "gapminder.csv.gz", index=False, compression="gzip")
    print("gapminder:", len(gm), "строк,", gm["indicator"].nunique(), "показателей")
    co2 = pd.read_csv(f"{OWID}/co2-data/master/owid-co2-data.csv", usecols=["country", "year", "iso_code"] + list(registry.OWID_CO2))
    co2.to_csv(OUT / "owid_co2.csv.gz", index=False, compression="gzip")
    en = pd.read_csv(f"{OWID}/energy-data/master/owid-energy-data.csv", usecols=["country", "year", "iso_code"] + list(registry.OWID_ENERGY))
    en.to_csv(OUT / "owid_energy.csv.gz", index=False, compression="gzip")
    print("owid co2:", len(co2), "| energy:", len(en))
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"files": {}}
    manifest["global_updated"] = date.today().isoformat()
    manifest["files"].update({
        "gapminder": {"desc": "Gapminder Systema Globalis: ~110 показателей по странам", "rows": len(gm), "source": "github.com/open-numbers/ddf--gapminder--systema_globalis"},
        "owid_co2": {"desc": "OWID: выбросы, температура, ВВП", "rows": len(co2), "source": "github.com/owid/co2-data"},
        "owid_energy": {"desc": "OWID: энергетика", "rows": len(en), "source": "github.com/owid/energy-data"}})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
