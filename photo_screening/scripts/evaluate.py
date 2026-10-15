#!/usr/bin/env python3
"""Measure the photo pipeline end to end and write EVAL.md (+ eval_results.json).

    python photo_screening/scripts/evaluate.py --aku "<...>/Benchmarking Dataset" --dentalai "<...>/dentalai-2"

Each model runs ONCE per image; raw detections are cached (``--cache``), and every pipeline
configuration is then scored from the cache:

* AKU (smartphone photos, caries boxes): caries AP and box P/R; image-level "any caries?"
  sensitivity/specificity; the referral rate (triage SOON/PROMPT) on photos with and without caries;
  clutter (findings per photo); the quality gate's rejection rate on real photos; AlphaDent's caries
  class alone (cross-domain).
* DentalAI (semantic masks: caries, cavity, crack, tooth): box P/R for cavity/crack/tooth at the
  service thresholds, plus validation curves for calibrate.py.

Configurations are compared on the validation split; the chosen one is reported on the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from photo_screening.app import quality, rules  # noqa: E402
from photo_screening.app.config import Settings, load_thresholds  # noqa: E402
from photo_screening.app.fusion import fuse_models, fuse_passes  # noqa: E402
from photo_screening.app.imaging import COLOUR_PASSES, decode  # noqa: E402
from photo_screening.app.models.registry import TYPE_PRIORITY, Registry  # noqa: E402
from photo_screening.app.schemas import Finding  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def visible(p: Path) -> bool:
    return not p.name.startswith("._") and "__MACOSX" not in p.parts


# ------------------------------------------------------------------ ground truth

def aku_items(split_dir: Path) -> list[tuple[Path, Path | None]]:
    imgs = sorted(p for p in (split_dir / "images").rglob("*") if visible(p) and p.suffix.lower() in IMG_EXT)
    labels = {p.stem: p for p in (split_dir / "yolo").rglob("*.txt") if visible(p)}
    return [(p, labels.get(p.stem)) for p in imgs]


def aku_gt(label: Path | None, w: int, h: int) -> dict[str, np.ndarray]:
    boxes = []
    if label is not None:
        for line in label.read_text(errors="replace").splitlines():
            v = line.split()
            if len(v) == 5:
                cx, cy, bw, bh = (float(x) for x in v[1:])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return {"caries": np.array(boxes, np.float32).reshape(-1, 4)}


DENTALAI_VALUES = {1: "caries", 2: "cavity", 3: "crack", 4: "tooth"}


def dentalai_items(split_dir: Path) -> list[tuple[Path, Path]]:
    out = []
    for m in sorted(split_dir.glob("*_mask.png")):
        if not visible(m):
            continue
        img = next((q for q in split_dir.glob(m.name[: -len("_mask.png")] + ".*")
                    if visible(q) and q.suffix.lower() in IMG_EXT), None)
        if img:
            out.append((img, m))
    return out


def dentalai_gt(mask_path: Path, w: int, h: int) -> dict[str, np.ndarray]:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    m = m[:, :, 0] if m.ndim == 3 else m
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    out = {}
    for value, name in DENTALAI_VALUES.items():
        binary = (m == value).astype(np.uint8)
        parts = _split_touching(binary) if name == "tooth" else _components(binary)
        boxes = []
        for part in parts:
            if part.sum() < 0.0002 * w * h:
                continue
            ys, xs = np.nonzero(part)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        out[name] = np.array(boxes, np.float32).reshape(-1, 4)
    return out


def _components(binary: np.ndarray) -> list[np.ndarray]:
    n, lab = cv2.connectedComponents(binary)
    return [(lab == k).astype(np.uint8) for k in range(1, n)]


def _split_touching(binary: np.ndarray) -> list[np.ndarray]:
    """Same watershed split Notebook B used to make the tooth training labels."""
    if not binary.any():
        return []
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    n, comps = cv2.connectedComponents(binary)
    markers = np.zeros(binary.shape, np.int32)
    next_id = 2
    for k in range(1, n):
        blob = comps == k
        m, cores = cv2.connectedComponents(((dist > 0.5 * dist[blob].max()) & blob).astype(np.uint8))
        for j in range(1, m):
            markers[cores == j] = next_id
            next_id += 1
    markers[cv2.dilate(binary, np.ones((3, 3), np.uint8)) == 0] = 1
    cv2.watershed(cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR), markers)
    return [((markers == k) & (binary > 0)).astype(np.uint8) for k in range(2, next_id)]


# ------------------------------------------------------------------ one model pass per image, cached

def collect(items, gt_fn, registry: Registry, settings: Settings, limit: int | None,
            colour_models: tuple[str, ...] = ("caries_aku", "alphadent")) -> list[dict]:
    rows = []
    t0 = time.time()
    for i, (img_path, label) in enumerate(items[:limit] if limit else items):
        rgb, original = decode(img_path.read_bytes(), settings.max_side)
        h, w = rgb.shape[:2]
        q = quality.assess(rgb, original, settings)
        timings, raw = {}, {}
        for model_id, det in registry.detectors.items():
            t = time.perf_counter()
            raw[model_id] = [(d.type, d.score, d.box.tolist()) for d in fuse_passes(det.predict(rgb))]
            timings[model_id] = time.perf_counter() - t
            if model_id in colour_models:          # colour-normalised passes, cached under "<model>@<pass>"
                for name, fn in COLOUR_PASSES.items():
                    raw[f"{model_id}@{name}"] = [(d.type, d.score, d.box.tolist())
                                                 for d in fuse_passes(det.predict(np.ascontiguousarray(fn(rgb))))]
        gt = {k: v.tolist() for k, v in gt_fn(label, w, h).items()}
        rows.append({"name": img_path.name, "w": w, "h": h, "gt": gt,
                     "quality_passed": q.passed,
                     "quality_failed": [c.name for c in q.checks if c.hard and not c.passed],
                     "raw": raw, "timings": timings})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} images, {(time.time() - t0) / (i + 1):.2f} s/image", flush=True)
    return rows


# ------------------------------------------------------------------ scoring

def phone_trained(model_id: str) -> bool:
    from photo_screening.app.models.registry import SPECS

    return bool(model_id in SPECS and SPECS[model_id].phone_trained)


@dataclass(frozen=True)
class Config:
    name: str
    alphadent_caries: str = "low"          # "low": standalone, capped Low · "agree": only as agreement · "off"
    keep: tuple = ()                       # per-type keep-threshold overrides, (("wear", 0.55), ...)
    drop_types: tuple = ()


CONFIGS = [
    Config("original (AlphaDent caries standalone, cavity/crack on, wear from 0.30)",
           keep=(("wear", 0.30),)),
    Config("AlphaDent caries as agreement only", alphadent_caries="agree", keep=(("wear", 0.30),)),
    Config("+ wear from 0.55", alphadent_caries="agree", keep=(("wear", 0.55),)),
    Config("service default (+ cavity/crack off)", alphadent_caries="agree", keep=(("wear", 0.55),),
           drop_types=("cavity", "crack")),
]


def findings_for(row: dict, cfg: Config, thresholds: dict, phone_trained) -> list[Finding]:
    from photo_screening.app.models.base import RawDet

    th = {k: dict(v) for k, v in thresholds.items()}
    for t, v in cfg.keep:
        th[t]["t_low"] = v
    per = {}
    for m, dets in row["raw"].items():
        if "@" in m:                               # colour-pass caches are scored by their own configs
            continue
        keep = []
        for t, s, b in dets:
            if t == "tooth" or t in cfg.drop_types:
                continue
            if m == "alphadent" and t == "caries" and cfg.alphadent_caries == "off":
                continue
            keep.append(RawDet(m, t, s, np.array(b, np.float32), None, 0, [m]))
        per[m] = keep
    fused = fuse_models(per, TYPE_PRIORITY)
    out = []
    for i, d in enumerate(fused):
        if d.score < th[d.type]["t_low"]:
            continue
        if cfg.alphadent_caries == "agree" and d.type == "caries" and d.sources == ["alphadent"] \
                and "caries_aku" in row["raw"]:
            continue
        out.append(Finding(id=i + 1, view="front", region="-", type=d.type, confidence=min(1.0, d.score),
                           band=rules.band(d.type, d.score, d.sources, th, phone_trained), sources=d.sources,
                           bbox=list(map(float, d.box))))
    return out


def match_pr(preds: list[tuple[float, list]], gts: np.ndarray, thr: float = 0.5):
    """Greedy IoU matching. Returns (tp flags in score order, scores, n_gt)."""
    from photo_screening.app.fusion import iou

    preds = sorted(preds, key=lambda p: -p[0])
    used = np.zeros(len(gts), bool)
    flags = []
    for _, b in preds:
        ious = [iou(np.array(b), g) for g in gts]
        j = int(np.argmax(ious)) if ious else -1
        ok = j >= 0 and ious[j] >= thr and not used[j]
        if ok:
            used[j] = True
        flags.append(ok)
    return flags, [p[0] for p in preds], len(gts)


def ap(scores, flags, n_gt):
    if n_gt == 0 or not scores:
        return 0.0
    o = np.argsort(-np.array(scores))
    tp = np.cumsum(np.array(flags)[o])
    fp = np.cumsum(~np.array(flags)[o])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate([[1.0], prec, [0.0]]))))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def wilson(k: int, n: int) -> tuple[float, float]:
    """95 % Wilson interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(float(c - hw), 3), round(float(c + hw), 3))


def score_aku(rows: list[dict], cfg: Config, thresholds: dict, phone_trained) -> dict:
    analysed = [r for r in rows if r["quality_passed"]]
    pos = [r for r in analysed if r["gt"]["caries"]]
    neg = [r for r in analysed if not r["gt"]["caries"]]
    all_flags, all_scores, n_gt = [], [], 0
    det_pos = ref_pos = det_neg = ref_neg = 0
    n_find, n_strong = [], []
    for r in analysed:
        fs = findings_for(r, cfg, thresholds, phone_trained)
        car = [f for f in fs if f.type == "caries"]
        flags, scores, n = match_pr([(f.confidence, f.bbox) for f in car], np.array(r["gt"]["caries"]).reshape(-1, 4))
        all_flags += flags
        all_scores += scores
        n_gt += n
        level = rules.triage(fs, analysed=True).level
        has_pos = bool(r["gt"]["caries"])
        if has_pos:
            det_pos += bool(car)
            ref_pos += level != "ROUTINE"
        else:
            det_neg += bool(car)
            ref_neg += level != "ROUTINE"
        n_find.append(len(fs))
        n_strong.append(sum(f.band != "Low" for f in fs))
    tp = sum(all_flags)
    return {
        "config": cfg.name, "n_images": len(rows), "n_analysed": len(analysed), "n_pos": len(pos), "n_neg": len(neg),
        "caries_ap50": round(ap(all_scores, all_flags, n_gt), 3),
        "caries_box_precision": round(tp / max(len(all_flags), 1), 3),
        "caries_box_recall": round(tp / max(n_gt, 1), 3),
        "image_sensitivity": round(det_pos / max(len(pos), 1), 3), "image_sensitivity_ci": wilson(det_pos, len(pos)),
        "image_specificity": round(1 - det_neg / max(len(neg), 1), 3),
        "image_specificity_ci": wilson(len(neg) - det_neg, len(neg)),
        "referral_rate_with_caries": round(ref_pos / max(len(pos), 1), 3),
        "referral_rate_with_caries_ci": wilson(ref_pos, len(pos)),
        "false_referral_rate": round(ref_neg / max(len(neg), 1), 3),
        "false_referral_rate_ci": wilson(ref_neg, len(neg)),
        "findings_per_photo": round(float(np.mean(n_find)), 2) if n_find else 0.0,
        "medium_plus_per_photo": round(float(np.mean(n_strong)), 2) if n_strong else 0.0,
    }


def score_alphadent_alone(rows: list[dict]) -> dict:
    flags, scores, n_gt, det_pos, det_neg, pos, neg = [], [], 0, 0, 0, 0, 0
    for r in rows:
        preds = [(s, b) for t, s, b in r["raw"].get("alphadent", []) if t == "caries"]
        f, sc, n = match_pr(preds, np.array(r["gt"]["caries"]).reshape(-1, 4))
        flags += f
        scores += sc
        n_gt += n
        hit = any(s >= 0.25 for s, _ in preds)
        if r["gt"]["caries"]:
            pos += 1
            det_pos += hit
        else:
            neg += 1
            det_neg += hit
    return {"ap50": round(ap(scores, flags, n_gt), 3), "image_sensitivity_at_0.25": round(det_pos / max(pos, 1), 3),
            "image_specificity_at_0.25": round(1 - det_neg / max(neg, 1), 3)}


def score_dentalai(rows: list[dict], thresholds: dict) -> dict:
    out = {}
    for t in ("tooth", "cavity", "crack"):
        flags, scores, n_gt = [], [], 0
        for r in rows:
            preds = [(s, b) for typ, s, b in r["raw"].get("teeth_dentalai", []) if typ == t]
            f, sc, n = match_pr(preds, np.array(r["gt"][t]).reshape(-1, 4))
            flags += f
            scores += sc
            n_gt += n
        curve = []
        for c in np.round(np.arange(0.05, 0.96, 0.01), 2):
            kept = [fl for fl, s in zip(flags, scores, strict=True) if s >= c]
            tp = sum(kept)
            p, rc = tp / max(len(kept), 1), tp / max(n_gt, 1)
            curve.append({"conf": float(c), "precision": p, "recall": rc, "f1": 2 * p * rc / max(p + rc, 1e-9)})
        at = next(x for x in curve if x["conf"] >= round(thresholds[t]["t_low"], 2))
        out[t] = {"n_boxes": n_gt, "ap50": round(ap(scores, flags, n_gt), 3),
                  "at_keep_threshold": {k: round(v, 3) for k, v in at.items()}, "curve": curve}
    return out


# ------------------------------------------------------------------ main

def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--aku", type=Path, help="AKU 'Benchmarking Dataset' folder (valid/ and test/ inside)")
    ap_.add_argument("--dentalai", type=Path, help="DentalAI Roboflow export folder (valid/ and test/ inside)")
    ap_.add_argument("--cache", type=Path, default=PKG / ".smoke" / "eval_cache")
    ap_.add_argument("--out", type=Path, default=PKG / "eval_results.json")
    ap_.add_argument("--limit", type=int, default=None, help="images per split (quick runs)")
    ap_.add_argument("--from-cache", type=Path, help="re-score cached AKU detections (aku_valid/test.json) "
                                                      "with the current settings; no model is run")
    args = ap_.parse_args()

    settings = Settings()
    if args.from_cache:                       # scoring only: thresholds as if every model were loaded
        registry = Registry()
        loaded = ["alphadent", "caries_aku", "teeth_dentalai"]
        thresholds = load_thresholds(settings, loaded)
        registry.statuses = {}
        results_models = json.loads((args.from_cache.parent / "eval_results_aku.json").read_text()).get("models", {}) \
            if (args.from_cache.parent / "eval_results_aku.json").exists() else {}
    else:
        registry = Registry.load(settings)
        registry.warm_up()
        thresholds = load_thresholds(settings, registry.loaded)
        results_models = registry.versions()
    args.cache.mkdir(parents=True, exist_ok=True)
    results: dict = {"models": results_models, "device": "cuda:0 (Kaggle T4)" if args.from_cache
                     else settings.resolved_device(), "thresholds": thresholds}

    def cached(name, items, gt_fn):
        path = args.cache / f"{name}.json"            # plain JSON: no code runs when a cache is read back
        if path.exists():
            return json.loads(path.read_text())
        print(f"{name}: {len(items[:args.limit] if args.limit else items)} images", flush=True)
        rows = collect(items, gt_fn, registry, settings, args.limit)
        path.write_text(json.dumps(rows))
        return rows

    if args.dentalai:
        for split in ("valid", "test"):
            rows = cached(f"dentalai_{split}", dentalai_items(args.dentalai / split),
                          lambda m, w, h: dentalai_gt(m, w, h))
            results[f"dentalai_{split}"] = score_dentalai(rows, thresholds)

    if args.aku or args.from_cache:
        configs = CONFIGS
        for split in ("valid", "test"):
            if args.from_cache:
                rows = json.loads((args.from_cache / f"aku_{split}.json").read_text())
                for r in rows:                 # re-apply the current resolution rule (w, h are <= max_side)
                    failed = [f for f in r["quality_failed"] if f != "resolution"]
                    if min(r["w"], r["h"]) < settings.min_side_px:
                        failed.append("resolution")
                    r["quality_failed"], r["quality_passed"] = failed, not failed
            else:
                rows = cached(f"aku_{split}", aku_items(args.aku / split), aku_gt)
            results[f"aku_{split}"] = {
                "configs": [score_aku(rows, c, thresholds, phone_trained) for c in configs],
                "alphadent_caries_alone": score_alphadent_alone(rows),
                "quality_gate_rejected": round(1 - np.mean([r["quality_passed"] for r in rows]), 3),
                "quality_failures": {k: int(sum(k in r["quality_failed"] for r in rows))
                                     for k in ("resolution", "blur", "exposure")},
                "seconds_per_image": {m: round(float(np.mean([r["timings"][m] for r in rows])), 3)
                                      for m in rows[0]["timings"]} if rows else {},
            }
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
