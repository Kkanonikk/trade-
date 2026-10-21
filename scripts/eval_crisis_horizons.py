"""Измеряет точность кризисных моделей на горизонтах 1, 3 и 5 лет (walk-forward, снимок IMF/World Bank) и пишет
assets/crisis_horizon_eval.json. Запуск: python scripts/eval_crisis_horizons.py  (несколько минут)."""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradelab import crisis, panel  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "assets" / "crisis_horizon_eval.json"


def main():
    p = panel.snapshot_panel()
    rows = []
    for kind in crisis.available_kinds(p):
        for h in (1, 3, 5):
            try:
                met, _ = crisis.evaluate(p, kind, 1996, panel.WEO_LAST_ACTUAL, horizon=h)
            except Exception as exc:
                print(kind, h, "ошибка:", exc)
                continue
            row = {"kind": kind, "label": crisis.KINDS[kind], "horizon": h, "years": list(met["years"]), "obs": int(met["obs"]),
                   "events": int(met["events"]), "base_rate": float(met["base_rate"]), "auc": float(met["ensemble"]["auc"]),
                   "auc_naive": float(met.get("naive", {}).get("auc", float("nan"))),
                   "top_decile_capture": float(met["ensemble"]["top_decile_capture"])}
            rows.append(row)
            print(f"{kind:10s} h={h}: AUC {row['auc']:.3f} (наивный {row['auc_naive']:.3f}), верхний дециль {row['top_decile_capture']:.0%}")
    OUT.write_text(json.dumps({"measured": date.today().isoformat(), "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
