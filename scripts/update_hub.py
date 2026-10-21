"""Собирает хаб открытых данных (Our World in Data) в assets/snapshot: owid_hub.csv.gz и owid_hub_meta.json.
Нужен git и интернет (клонирует github.com/owid/owid-datasets, около 300 МБ, удаляется после сборки).
Запуск: python scripts/update_hub.py [путь к уже склонированному owid-datasets/datasets]"""
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tradelab import hub  # noqa: E402

SNAP = ROOT / "assets" / "snapshot"


def main():
    if len(sys.argv) > 1:
        src, tmp = Path(sys.argv[1]), None
    else:
        tmp = Path(tempfile.mkdtemp())
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/owid/owid-datasets.git", str(tmp / "r")], check=True)
        src = tmp / "r" / "datasets"
    try:
        co2 = pd.read_csv(SNAP / "owid_co2.csv.gz", usecols=["country", "iso_code"])
        long, meta = hub.build(src, co2)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    long.to_csv(SNAP / "owid_hub.csv.gz", index=False, compression="gzip")
    (SNAP / "owid_hub_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8")
    man_path = SNAP / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man.setdefault("files", {})["owid_hub"] = {"desc": f"Our World in Data: {len(meta)} показателей из {len({m['dataset'] for m in meta.values()})} наборов",
                                               "rows": int(len(long)), "source": "github.com/owid/owid-datasets"}
    man["hub_updated"] = date.today().isoformat()
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Показателей: {len(meta)}, строк: {len(long)}, размер: {(SNAP / 'owid_hub.csv.gz').stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
