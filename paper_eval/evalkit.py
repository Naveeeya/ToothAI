#!/usr/bin/env python3
"""Shared evaluation machinery for the Tooth-AI paper exports.

Split out of run_eval.py so export_panels.py can reuse matching and geometry
without duplicating it.
"""
import hashlib
import json
import os
import random
import subprocess
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(ROOT, "data", "final-di-stratified", "test")
SPLIT_JSON = os.path.join(SPLIT_DIR, "_annotations.coco.json")

TEETH_MODEL = os.path.join(ROOT, "output", "resnet50_9class_20k")
ANOM_MODEL = os.path.join(ROOT, "output", "rtx4060_48k")
CASCADE_MODEL = os.path.join(ROOT, "output", "resnext101_cascade_60k")

CENTRAL_LO, CENTRAL_HI = 0.35, 0.65     # app.py:922-923
MATCH_IOU = 0.5
SEED = 20260921
N_BOOT = 10000

FDI_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_state():
    """Commit hash of the code that produced a result.

    On Kaggle there is no git repo, so `git rev-parse` fails and provenance
    would be lost exactly where it matters most. make_bundle.py stamps the
    state into BUILD_PROVENANCE.json beside this file; fall back to that.
    """
    stamp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "BUILD_PROVENANCE.json")
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        try:
            with open(stamp) as fh:
                rec = json.load(fh)
            return f"{rec['git']} (stamped at bundle build {rec['built_utc']})"
        except Exception:
            return "unknown"
    try:
        d = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                    stderr=subprocess.DEVNULL).decode().strip()
        return f"{head}{'+dirty' if d else ''}"
    except subprocess.CalledProcessError:
        return f"{head}+dirty-unknown"


def gpu_name():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()
        return out[0] if out else "unknown"
    except Exception:
        return "no-nvidia-smi"


def config_input_size(model_dir):
    """(MIN_SIZE_TEST, MAX_SIZE_TEST) read from the model's own config.yaml."""
    import re
    txt = open(os.path.join(model_dir, "config.yaml"), errors="replace").read()
    mn = re.search(r"^\s+MIN_SIZE_TEST:\s*(\d+)", txt, re.M)
    mx = re.search(r"^\s+MAX_SIZE_TEST:\s*(\d+)", txt, re.M)
    return (int(mn.group(1)) if mn else None, int(mx.group(1)) if mx else None)


def provenance_header(extra=()):
    lines = [
        "# Tooth-AI paper_eval",
        f"# split: data/final-di-stratified/test  (716 images)",
        f"# teeth model:   output/resnet50_9class_20k/model_final.pth",
        f"#   sha256:      {sha256_file(os.path.join(TEETH_MODEL, 'model_final.pth'))}",
        f"#   input size:  MIN_SIZE_TEST={config_input_size(TEETH_MODEL)[0]} "
        f"MAX_SIZE_TEST={config_input_size(TEETH_MODEL)[1]}",
        f"# git: {git_state()}",
        f"# gpu: {gpu_name()}",
        f"# seed: {SEED}   bootstrap resamples: {N_BOOT}",
        "# NOTE: recall only. With a median of 8 labelled teeth per image, an",
        "#   unlabelled but correctly detected tooth scores as a false positive,",
        "#   so precision and FP/image are NOT meaningful on this ground truth.",
    ]
    lines += [f"# {e}" for e in extra]
    return lines


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------
def load_split():
    coco = json.load(open(SPLIT_JSON))
    images = {i["id"]: i for i in coco["images"]}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    teeth = defaultdict(list)
    anoms = defaultdict(list)
    for a in coco["annotations"]:
        (teeth if 1 <= a["category_id"] <= 32 else anoms)[a["image_id"]].append(a)
    return images, names, teeth, anoms


def load_class_map():
    """class_id -> (jaw, rank, fdi), from the Task B derivation."""
    import csv
    p = os.path.join(ROOT, "paper_eval", "results", "class_to_position.csv")
    m = {}
    with open(p) as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            m[int(r["class_id"])] = (r["jaw"], int(r["position_rank"]), int(r["fdi"]))
    return m


def load_manifests():
    """The three image subsets, each justified by a different label property."""
    import csv
    p = os.path.join(ROOT, "paper_eval", "results", "label_quality.csv")
    rows = {}
    with open(p) as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            rows[int(r["image_id"])] = r
    full_arch = {i for i, r in rows.items() if r["full_arch"] == "1"}
    canonical = {i for i, r in rows.items() if r["canonical_mapping"] == "1"}
    sparse = {i for i, r in rows.items() if 1 <= int(r["labeled_teeth"]) <= 10}
    mid = {i for i, r in rows.items() if 11 <= int(r["labeled_teeth"]) <= 31}
    return {
        "all": set(rows),
        # FDI accuracy headline: consistent labels, includes missing teeth
        "canonical": canonical,
        # FDI accuracy clean-label bound: consistent labels AND complete arch
        "fdi_evaluable": canonical & full_arch,
        # jaw membership is reliable on every full-arch image (Check 1)
        "full_arch": full_arch,
        # Sparse images belong to the ANOMALY-annotation regime: 43.1% of their
        # labelled teeth carry an anomaly box, vs 0.0% of full-arch ones, and
        # none of them is a complete arch. Recall there measures something
        # different, so it is reported under its own name and excluded from
        # central-zone recall and per-tooth recall entirely.
        "sparse": sparse,
        "mid": mid,
        # Offset histogram: canonical images whose gaps are genuinely missing
        # teeth. The 5 canonical-partial images in the sparse group are dropped
        # -- their gaps are unannotated teeth, not absent ones, so an offset
        # there would measure annotation policy, not the anchoring flaw.
        "offset_subset": canonical & (full_arch | mid),
        "rows": rows,
    }


def poly_to_mask(segm, h, w):
    """COCO polygon -> boolean mask. Uses pycocotools (present on Kaggle)."""
    from pycocotools import mask as maskutil
    if isinstance(segm, list):
        rles = maskutil.frPyObjects(segm, h, w)
        rle = maskutil.merge(rles)
    elif isinstance(segm.get("counts"), list):
        rle = maskutil.frPyObjects(segm, h, w)
    else:
        rle = segm
    return maskutil.decode(rle).astype(bool)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def box_centroid_frac(bbox, width):
    x, y, w, h = bbox
    return (x + w / 2.0) / float(width)


def is_central(bbox, width):
    f = box_centroid_frac(bbox, width)
    return CENTRAL_LO < f < CENTRAL_HI


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------
def gt_masks_for(gt_anns, h, w):
    """Decode each ground-truth polygon once.

    Anything that sweeps a parameter must reuse the result: decoding is the
    most expensive step in the harness, and the ground truth does not change
    with tau. Re-decoding per tau made the projected sweep ~6 hours on its own.
    """
    return [poly_to_mask(a["segmentation"], h, w) for a in gt_anns]


def _xyxy(bbox):
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def _boxes_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def match_detections(gt_anns, preds, h, w, gt_masks=None):
    """Greedy highest-score-first matching at mask IoU >= 0.5.

    gt_anns  : COCO annotation dicts (carry 'segmentation' and 'category_id')
    preds    : dicts with 'mask' (bool array), 'box' [x1,y1,x2,y2], 'fdi'
    gt_masks : optional, from gt_masks_for(), to avoid re-decoding in a sweep
    returns  : list of (gt_ann, pred_or_None, iou), one entry per gt_ann

    A mask lies inside its own box, so two detections whose BOXES are disjoint
    cannot have overlapping masks. Screening on boxes first is therefore exact
    -- it changes no match -- and skips almost every full-resolution mask
    comparison, which is what makes the 716-image sweep finish in time.
    """
    if gt_masks is None:
        gt_masks = gt_masks_for(gt_anns, h, w)
    gt_boxes = [_xyxy(a["bbox"]) for a in gt_anns]

    order = sorted(range(len(preds)), key=lambda i: -preds[i].get("score", 0.0))
    taken = set()
    out = [(a, None, 0.0) for a in gt_anns]
    for pi in order:
        pm = preds[pi].get("mask")
        if pm is None:
            continue
        pb = preds[pi].get("box")
        best_iou, best_gi = 0.0, None
        for gi, gm in enumerate(gt_masks):
            if gi in taken:
                continue
            if pb is not None and not _boxes_overlap(gt_boxes[gi], pb):
                continue                      # exact: disjoint boxes -> IoU 0
            v = mask_iou(gm, pm)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi is not None and best_iou >= MATCH_IOU:
            taken.add(best_gi)
            out[best_gi] = (gt_anns[best_gi], preds[pi], best_iou)
    return out


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def bootstrap_ci(per_image, n=N_BOOT, seed=SEED, alpha=0.05):
    """95% CI for a ratio, resampling IMAGES (not instances).

    per_image : list of (numerator, denominator) per image
    Returns (point, lo, hi) or (nan, nan, nan) when there is no denominator.
    """
    num = sum(x for x, _ in per_image)
    den = sum(y for _, y in per_image)
    if den == 0:
        return float("nan"), float("nan"), float("nan")
    point = num / den
    if not per_image:
        return point, float("nan"), float("nan")
    rng = random.Random(seed)
    k = len(per_image)
    stats = []
    for _ in range(n):
        sn = sd = 0
        for _ in range(k):
            a, b = per_image[rng.randrange(k)]
            sn += a; sd += b
        if sd:
            stats.append(sn / sd)
    if not stats:
        return point, float("nan"), float("nan")
    stats.sort()
    lo = stats[int((alpha / 2) * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return point, lo, hi
