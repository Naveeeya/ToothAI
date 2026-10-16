#!/usr/bin/env python3
"""Score engine variants on cached model outputs (paper_eval/cache_outputs.py): no model runs here.

    python paper_eval/engine_lab.py --split valid                 # design on the validation split
    python paper_eval/engine_lab.py --split test --variants v1,v2 --out results/engine_v2.csv

Metrics and matching are paper_eval's (evalkit: mask IoU >= 0.5, image bootstrap, seed 20260921):
recall on full-arch images, FDI joint accuracy on canonical images, split into complete and partial
arches. Test subsets are the paper's manifests. On the validation split (no audited manifests) full arch
means >= 28 labelled teeth and partial 14-27, and FDI uses only images passing paper_eval's canonical-mapping
test (``canonical_ids``), as on test. Prints numeric ids only.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import pycocotools.mask as mask_util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import evalkit as K  # noqa: E402
from tooth_ai import engine_v2 as V2  # noqa: E402
from tooth_ai import pipeline as P  # noqa: E402

TOOTH = 0
TAU = 0.20          # the app's default tooth threshold (tooth_ai.analysis.DEFAULT_TEETH_THRESHOLD)


# ------------------------------------------------------------------ cached outputs -> detections
def teeth_from(rec, name):
    p = rec[name]
    out = []
    for b, s, c, m in zip(p["boxes"], p["scores"], p["classes"], p["masks"]):
        if c == TOOTH:
            out.append({"box": b, "score": s, "rle": m, "pass": name})
    return out


def mask_of(d):
    if "mask" not in d:
        d["mask"] = mask_util.decode({"size": d["rle"]["size"], "counts": d["rle"]["counts"].encode()}).astype(bool)
    return d["mask"]


def fuse(passes, iou_thr=0.55, penalty=0.6):
    """numpy port of pipeline.weighted_box_fusion for the tooth class (same consensus rule)."""
    flat = sorted((d for p in passes for d in p), key=lambda d: -d["score"])
    used = [False] * len(flat)
    out = []
    for i, d in enumerate(flat):
        if used[i]:
            continue
        used[i] = True
        cluster = [d]
        for j in range(i + 1, len(flat)):
            if not used[j] and P.compute_iou(d["box"], flat[j]["box"]) > iou_thr:
                used[j] = True
                cluster.append(flat[j])
        w = np.array([c["score"] for c in cluster]); w = w / w.sum()
        box = np.average(np.array([c["box"] for c in cluster]), axis=0, weights=w).tolist()
        n_pass = len({c["pass"] for c in cluster})
        score = d["score"] if n_pass >= min(2, len(passes)) else d["score"] * penalty
        out.append({"box": box, "score": score, "rle": d["rle"], "pass": "fused"})
    return out


def threshold(dets, w, tau=TAU, zone=True):
    keep = []
    for d in dets:
        cx = (d["box"][0] + d["box"][2]) / 2
        central = 0.35 * w < cx < 0.65 * w
        t = max(tau * 0.7, 0.08) if (zone and central) else tau
        if d["score"] >= t:
            k = dict(d)
            k["mask"] = mask_of(d)          # decoded once per detection; the engines carry it through
            keep.append(k)
    return keep


# ------------------------------------------------------------------ engines
def engine_v1(teeth):
    nms = P.DentalGeometricEngine.apply_nms(teeth, iou_threshold=0.4)
    up, lo = P.DentalGeometricEngine.split_jaws(nms)
    numbered = P.DentalGeometricEngine.sort_and_assign_fdi(up, "upper") + P.DentalGeometricEngine.sort_and_assign_fdi(lo, "lower")
    numbered = P.DentalGeometricEngine.map_anomalies(numbered, [])
    final, _ = P.validate_anatomy(numbered, [], anomaly_threshold=0.15)
    return final


def engine_v2(teeth, midline, **opts):
    nms = P.DentalGeometricEngine.apply_nms(teeth, iou_threshold=0.4)
    return V2.number_teeth(nms, midline=midline, **opts)


def detections(rec, inputs, zone):
    w = rec["w"]
    if inputs == "2pass":
        dets = fuse([teeth_from(rec, "orig"), teeth_from(rec, "clahe")])
    elif inputs == "1pass":
        dets = teeth_from(rec, "orig")
    elif inputs == "flip":
        dets = fuse([teeth_from(rec, "orig"), teeth_from(rec, "flip")])
    elif inputs == "big":
        dets = teeth_from(rec, "big")
    elif inputs == "orig+big":
        dets = fuse([teeth_from(rec, "orig"), teeth_from(rec, "big")])
    else:
        raise ValueError(inputs)
    return threshold(dets, w, zone=zone)


# name -> (inputs, zone relaxation, engine, midline)
VARIANTS = {
    "v1": ("2pass", True, "v1", None),                                  # what the app runs
    "v1-1pass": ("1pass", True, "v1", None),
    "v2-cleanup": ("2pass", True, "v2", "right-anchored"),
    "v2-median": ("2pass", True, "v2", "median"),
    "v2-symmetry": ("2pass", True, "v2", "symmetry"),
    "v2-sym-1pass": ("1pass", False, "v2", "symmetry"),
    "v2-sym-flip": ("flip", False, "v2", "symmetry"),
    "v2-sym-big": ("big", False, "v2", "symmetry"),
    "v2-sym-orig+big": ("orig+big", False, "v2", "symmetry"),
    "v2-sym-dedupe": ("2pass", True, "v2", "symmetry", {"dedupe": True}),
    "v2-sym-curve": ("2pass", True, "v2", "symmetry", {"jaw_split": "curve"}),
    "v2-sym-shared": ("2pass", True, "v2", "symmetry", {"shared_midline": True}),
    "v2-sym-all": ("2pass", True, "v2", "symmetry", {"dedupe": True, "jaw_split": "curve", "shared_midline": True}),
    "v2-sym-dc": ("2pass", True, "v2", "symmetry", {"dedupe": True, "jaw_split": "curve"}),
    "v2-1p-c": ("1pass", False, "v2", "symmetry", {"jaw_split": "curve"}),
    "v2-1p-dc": ("1pass", False, "v2", "symmetry", {"dedupe": True, "jaw_split": "curve"}),
    "v2-1p-all": ("1pass", False, "v2", "symmetry", {"dedupe": True, "jaw_split": "curve", "shared_midline": True}),
    "v2-1pz-dc": ("1pass", True, "v2", "symmetry", {"dedupe": True, "jaw_split": "curve"}),
    # the configuration the app ships as engine "v2" (tooth_ai.engine_v2.DEFAULTS, one pass, uniform threshold)
    "v2": ("1pass", False, "v2", "symmetry", {"dedupe": True, "jaw_split": "curve"}),
}


# ------------------------------------------------------------------ evaluation
def subsets(split, gt_teeth):
    if split == "test":
        man = K.load_manifests()
        return man["full_arch"], man["canonical"] & man["full_arch"], man["canonical"] - man["full_arch"]
    full = {i for i, g in gt_teeth.items() if len(g) >= 28}
    part = {i for i, g in gt_teeth.items() if 14 <= len(g) < 28}
    canon = canonical_ids(gt_teeth)
    return full, full & canon, part & canon


def canonical_ids(gt_teeth):
    """paper_eval's canonical-mapping test (label_quality.py), applied to another split: in each jaw the
    labels read left to right must follow the canonical class order, with no duplicate class."""
    cmap = K.load_class_map()
    order = {jaw: [c for c, _ in sorted(((c, v[1]) for c, v in cmap.items() if v[0] == jaw), key=lambda x: x[1])]
             for jaw in ("upper", "lower")}

    def subseq(seq, ref):
        it = iter(ref)
        return all(any(x == y for y in it) for x in seq)

    out = set()
    for iid, anns in gt_teeth.items():
        ids = [a["category_id"] for a in anns]
        if not ids or len(ids) != len(set(ids)) or any(c not in cmap for c in ids):
            continue
        seqs = {}
        for jaw in ("upper", "lower"):
            seqs[jaw] = [a["category_id"] for a in sorted(anns, key=lambda a: a["bbox"][0] + a["bbox"][2] / 2)
                         if cmap[a["category_id"]][0] == jaw]
        if all(subseq(seqs[j], order[j]) for j in seqs):
            out.add(iid)
    return out


def run(split, names, cache_dir):
    if split == "valid":
        K.SPLIT_DIR = K.SPLIT_DIR.replace(os.sep + "test", os.sep + "valid")
        K.SPLIT_JSON = os.path.join(K.SPLIT_DIR, "_annotations.coco.json")
    images, _, gt_teeth, _ = K.load_split()
    cmap = K.load_class_map()
    full, complete, partial = subsets(split, gt_teeth)
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".json") and not f.startswith("._"))
    per = {n: {"recall": [], "fdi": [], "complete": [], "partial": [], "fp": []} for n in names}
    for f in files:
        rec = json.load(open(os.path.join(cache_dir, f)))
        iid = rec["image_id"]
        if iid not in full | complete | partial:
            continue
        gts = gt_teeth[iid]
        gm = K.gt_masks_for(gts, rec["h"], rec["w"])
        for n in names:
            inputs, zone, eng, mid, *rest = VARIANTS[n]
            dets = detections(rec, inputs, zone)
            final = engine_v1(dets) if eng == "v1" else engine_v2(dets, mid, **(rest[0] if rest else {}))
            m = K.match_detections(gts, final, rec["h"], rec["w"], gt_masks=gm)
            hit = sum(p is not None for _, p, _ in m)
            good = sum(1 for g, p, _ in m if p is not None and g["category_id"] in cmap
                       and p.get("fdi") == cmap[g["category_id"]][2])
            if iid in full:
                per[n]["recall"].append((hit, len(gts)))
                per[n]["fp"].append((len(final) - hit, 1))
            if iid in complete:
                per[n]["complete"].append((good, len(gts)))
                per[n]["fdi"].append((good, len(gts)))
            if iid in partial:
                per[n]["partial"].append((good, len(gts)))
                per[n]["fdi"].append((good, len(gts)))
    return per


def paired_delta(a, b, n=10000):
    """Paired bootstrap of ratio(b) - ratio(a) over images (both lists aligned by image)."""
    import random
    rng = random.Random(K.SEED)
    point = sum(x for x, _ in b) / sum(y for _, y in b) - sum(x for x, _ in a) / sum(y for _, y in a)
    k, stats = len(a), []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        na = sum(a[i][0] for i in idx); da = sum(a[i][1] for i in idx)
        nb = sum(b[i][0] for i in idx); db = sum(b[i][1] for i in idx)
        if da and db:
            stats.append(nb / db - na / da)
    stats.sort()
    return point, stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["valid", "test"], required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", help="CSV under paper_eval/results (test split only)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--gap-ratio", type=float, default=None, help="override engine_v2.GAP_RATIO (sweeps)")
    args = ap.parse_args()
    if args.gap_ratio:
        V2.GAP_RATIO = args.gap_ratio
    names = args.variants.split(",")
    per = run(args.split, names, os.path.join(HERE, ".cache", args.split))
    rows = []
    for n in names:
        r = {"config": n, "split": args.split}
        for key in ("recall", "fdi", "complete", "partial"):
            v, lo, hi = K.bootstrap_ci(per[n][key], n=args.boot)
            r[key], r[key + "_lo"], r[key + "_hi"], r[key + "_n_images"] = v, lo, hi, len(per[n][key])
        r["fp_per_image"] = sum(a for a, _ in per[n]["fp"]) / max(len(per[n]["fp"]), 1)
        rows.append(r)
        print(f"{n:16s} recall {100 * r['recall']:.2f} [{100 * r['recall_lo']:.1f}, {100 * r['recall_hi']:.1f}]  "
              f"FDI {100 * r['fdi']:.1f} [{100 * r['fdi_lo']:.1f}, {100 * r['fdi_hi']:.1f}]  "
              f"complete {100 * r['complete']:.1f}  partial {100 * r['partial']:.1f}  FP/img {r['fp_per_image']:.2f}",
              flush=True)
    if len(names) == 2:
        a, b = names
        for key in ("recall", "fdi", "complete", "partial"):
            d, lo, hi = paired_delta(per[a][key], per[b][key], n=args.boot)
            line = f"paired {b} - {a} {key}: {100 * d:+.2f} pp [{100 * lo:+.2f}, {100 * hi:+.2f}] (n images {len(per[a][key])})"
            print(line)
            rows[1]["delta_" + key] = d; rows[1]["delta_" + key + "_lo"] = lo; rows[1]["delta_" + key + "_hi"] = hi
        rows[0].update({k: "" for k in rows[1] if k.startswith("delta_")})
    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
        with open(path, "w", newline="") as fh:
            for line in K.provenance_header([f"engine_lab.py on cached outputs ({args.split} split)",
                                             f"bootstrap resamples: {args.boot}"]):
                if args.split == "valid" and line.startswith("# split:"):
                    line = "# split: data/final-di-stratified/valid  (images with >= 14 labelled teeth)"
                fh.write(line + "\n")
            w = csv.DictWriter(fh, fieldnames=list(rows[-1]))
            w.writeheader(); w.writerows(rows)
        print("wrote", path)


if __name__ == "__main__":
    main()
