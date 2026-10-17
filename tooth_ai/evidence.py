"""Measured performance, read from paper_eval/results so every surface quotes the harness.

Configurations (paper_eval/README.md):
  full_fix_routing        two passes (original + CLAHE) with the class-routing fix
                          == what this app runs
  single_pass             one pass, best measured configuration
  network_numbering_t005  41-class network that predicts tooth numbers directly (baseline)

Subsets: canonical_117 (all canonical test images), canonical_full_arch_67 (complete
arches), canonical_partial_50 (partial arches). Test split: 716 held-out OPGs.
"""

import csv
from functools import lru_cache
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "paper_eval" / "results"

DEPLOYED = "full_fix_routing"
BEST = "single_pass"
NETWORK = "network_numbering_t005"
CONFIG_LABELS = {
    DEPLOYED: "As deployed (two passes)",
    BEST: "Single pass",
    NETWORK: "Network numbers teeth directly",
}
SUBSET_ALL, SUBSET_COMPLETE, SUBSET_PARTIAL = "canonical_117", "canonical_full_arch_67", "canonical_partial_50"

# Values from paper_eval/results at commit c151d5b, used if the CSVs are absent.
_FALLBACK = {
    (DEPLOYED, SUBSET_ALL): dict(n_images=117, detection_recall=0.9493, jaw_acc_given_detected=0.9787,
                                 fdi_acc_given_detected=0.6493, fdi_acc_joint=0.6163),
    (DEPLOYED, SUBSET_COMPLETE): dict(n_images=67, fdi_acc_joint=0.7663, detection_recall=0.9398),
    (DEPLOYED, SUBSET_PARTIAL): dict(n_images=50, fdi_acc_joint=0.3540, detection_recall=0.9657),
    (BEST, SUBSET_ALL): dict(n_images=117, detection_recall=0.9531, jaw_acc_given_detected=0.9795,
                             fdi_acc_given_detected=0.6768, fdi_acc_joint=0.6451),
    (BEST, SUBSET_COMPLETE): dict(n_images=67, fdi_acc_joint=0.8102, detection_recall=0.9450),
    (BEST, SUBSET_PARTIAL): dict(n_images=50, fdi_acc_joint=0.3564, detection_recall=0.9674),
    (NETWORK, SUBSET_ALL): dict(n_images=117, detection_recall=0.7718, fdi_acc_joint=0.4629),
}


def _read(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


@lru_cache(maxsize=1)
def decomposition():
    rows = {(r["config"], r["subset"]): {k: _num(v) for k, v in r.items()} for r in _read("decomposition.csv")}
    return rows or _FALLBACK


def metric(config, subset, name):
    """Value (fraction) and its 95 % interval (lo, hi) or (None, None)."""
    row = decomposition().get((config, subset), {})
    return row.get(name), (row.get(f"{name}_lo"), row.get(f"{name}_hi")), int(row.get("n_images") or 0)


@lru_cache(maxsize=1)
def coco_ap():
    for r in _read("coco_eval_716.csv"):
        if r["model"] == "resnet50_9class_20k":
            return {"generic_AP_Tooth": float(r["generic_AP_Tooth"]), "segm_AP": float(r["segm_AP"]),
                    "segm_AP50": float(r["segm_AP50"])}
    return {"generic_AP_Tooth": 72.5, "segm_AP": 37.8, "segm_AP50": 65.7}


def geometry_vs_network():
    """Paired delta in end-to-end FDI accuracy, single-pass geometry minus the network
    (paper_eval/results/paired_deltas.txt). Returns (delta_pp, lo_pp, hi_pp, n)."""
    path = RESULTS / "paired_deltas.txt"
    if path.exists():
        block = path.read_text(encoding="utf-8").replace("\r", "").split(f"MINUS  {NETWORK}", 1)
        if len(block) == 2:
            for line in block[1].splitlines():
                if line.startswith("FDI accuracy (joint)"):
                    nums = line.replace("[", " ").replace("]", " ").replace(",", " ").replace("pp", " ").split()
                    vals = [x for x in nums if x.lstrip("+-").replace(".", "").isdigit()]
                    n = int(line.rsplit("n=", 1)[1])
                    return float(vals[0]), float(vals[1]), float(vals[2]), n
    return 18.22, 11.84, 24.30, 117


def pct(v, digits=1):
    return "-" if v is None else f"{v * 100:.{digits}f}%"


def deployed_summary():
    """One sentence for reports and banners, in the configuration the app runs."""
    rec, _, _ = metric(DEPLOYED, SUBSET_ALL, "detection_recall")
    jaw, _, _ = metric(DEPLOYED, SUBSET_ALL, "jaw_acc_given_detected")
    full, _, n_full = metric(DEPLOYED, SUBSET_COMPLETE, "fdi_acc_joint")
    part, _, n_part = metric(DEPLOYED, SUBSET_PARTIAL, "fdi_acc_joint")
    return dict(recall=rec, jaw=jaw, complete=full, partial=part, n_complete=n_full, n_partial=n_part)


@lru_cache(maxsize=1)
def engine_v2():
    """Engine v1 vs v2 on the test split (paper_eval/engine_lab.py -> results/engine_v2.csv), or {} if absent."""
    return {r["config"]: {k: _num(v) for k, v in r.items()} for r in _read("engine_v2.csv")}
