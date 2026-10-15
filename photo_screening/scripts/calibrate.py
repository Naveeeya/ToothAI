#!/usr/bin/env python3
"""Recompute confidence-band thresholds from the validation curves the notebooks saved.

    python photo_screening/scripts/calibrate.py            # every model with a saved curve
    python photo_screening/scripts/calibrate.py --dry-run  # print, don't write

Rule (validation split, box-level IoU >= 0.5), ordered by construction:
  t_low  = highest score threshold that still keeps recall >= 0.90   (screening: sensitivity first)
  t_med  = threshold with the best F1, never below t_low
  t_high = lowest threshold >= t_med whose precision is >= 0.90
If recall 0.90 is unreachable the class is marked weak and t_low falls back to t_med; the service
also applies per-type minimum keep scores (config.MIN_KEEP). The previous values are kept under
``notebook_classes`` in the same file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = Path(os.environ.get("PHOTO_WEIGHTS_DIR", PKG_DIR / "weights"))
RECALL_KEEP, PRECISION_HIGH = 0.90, 0.90


def bands(curve: list[dict]) -> dict:
    """Thresholds for one class from its validation P/R curve (list of {conf, precision, recall, f1})."""
    keep = [c["conf"] for c in curve if c["recall"] >= RECALL_KEEP]
    best = max(curve, key=lambda c: c["f1"])
    t_med = best["conf"]
    t_low = max(keep) if keep else t_med
    t_med = max(t_med, t_low)
    high = [c["conf"] for c in curve if c["precision"] >= PRECISION_HIGH and c["conf"] >= t_med]
    t_high = min(high) if high else round(min(t_med + 0.10, 0.95), 2)
    at = {c["conf"]: c for c in curve}
    return {"t_low": t_low, "t_med": t_med, "t_high": t_high, "recall_target_reached": bool(keep),
            "valid_at_t_low": {k: round(at[t_low][k], 3) for k in ("precision", "recall")} if t_low in at else None,
            "valid_best_f1": round(best["f1"], 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for path in sorted(WEIGHTS_DIR.glob("*/thresholds.json")):
        data = json.loads(path.read_text())
        curves = data.get("valid_curves") or ({"caries": data["valid_curve"]} if "valid_curve" in data else {})
        if not curves:
            print(f"{path.parent.name}: no saved validation curve, skipped")
            continue
        new = {name: bands(curve) for name, curve in curves.items()}
        for name, t in new.items():
            print(f"{path.parent.name:15s} {name:8s} low {t['t_low']:.2f}  med {t['t_med']:.2f}  high {t['t_high']:.2f}"
                  f"  recall>=0.90 {'yes' if t['recall_target_reached'] else 'NO (weak class)'}")
        if not args.dry_run:
            data.setdefault("notebook_classes", data.get("classes", {}))
            data["classes"] = new
            data["method"] = __doc__.split("Rule", 1)[1].split("If recall", 1)[0].strip()
            path.write_text(json.dumps(data, indent=2) + "\n")
    if not args.dry_run:
        print("written; restart the service to apply")


if __name__ == "__main__":
    main()
