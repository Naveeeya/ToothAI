#!/usr/bin/env python3
"""Task B: label-quality audit of the 716-image stratified test split.

Standard library only -- runs on this Mac, no torch/Detectron2 needed.

Derives, from the data alone:
  * the empirical mapping from the 41-class schema's classes "1".."32" to arch
    positions (and thus to FDI numbers). This mapping is documented NOWHERE in
    the repo; the paper has to state it.
  * per-image label quality on the 716 split.
  * ground-truth instance counts per class.
  * the subset of images on which FDI accuracy can honestly be computed.

Nothing here is estimated or interpolated. Every number is a count.

Usage:
    python3 paper_eval/label_quality.py
"""
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT = "data/final-di-stratified/test/_annotations.coco.json"
RESULTS = os.path.join(ROOT, "paper_eval", "results")
MANIFESTS = os.path.join(ROOT, "paper_eval", "manifests")

# FDI sequences exactly as hardcoded in app.py:57-58. Index i is the i-th tooth
# left-to-right in image coordinates.
FDI_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

TOOTH_IDS = range(1, 33)          # 41-class schema: category 0 is "t" (unused)
ANOMALY_IDS = range(33, 41)


def git_state():
    """Commit hash plus dirty flag.

    `git status` is resolved separately from `git rev-parse` on purpose: this
    repo sets `*.pth filter=lfs` in .gitattributes, so on a machine without
    git-lfs installed `git status` exits 128 ("git-lfs: command not found").
    That must not cost us the commit hash, which resolves fine.
    """
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                        stderr=subprocess.DEVNULL).decode().strip()
        return f"{head}{'+dirty' if dirty else ''}"
    except subprocess.CalledProcessError:
        return f"{head}+dirty-unknown(git-lfs missing)"


def provenance(extra=()):
    lines = [
        "# Tooth-AI paper_eval -- Task B (label quality)",
        f"# split: {SPLIT}",
        "# source: ground-truth COCO annotations only; no model was run",
        f"# git: {git_state()}",
        "# generator: paper_eval/label_quality.py",
    ]
    lines.extend(f"# {e}" for e in extra)
    return lines


def load():
    path = os.path.join(ROOT, SPLIT)
    with open(path) as fh:
        coco = json.load(fh)
    images = {i["id"]: i for i in coco["images"]}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    per_image = defaultdict(list)
    for a in coco["annotations"]:
        per_image[a["image_id"]].append(a)
    return images, names, per_image


def centre(ann):
    x, y, w, h = ann["bbox"]
    return x + w / 2.0, y + h / 2.0


def derive_jaw_partition(images, per_image, full_arch_ids):
    """Which of classes 1..32 are upper? Decided by counting, not assumption.

    For each full-arch image, split its 32 teeth at the y-median of their own
    centroids (the same rule app.py:82-104 uses), then count how often each
    class lands in the upper half.
    """
    upper_votes = Counter()
    total = Counter()
    for iid in full_arch_ids:
        teeth = [a for a in per_image[iid] if a["category_id"] in TOOTH_IDS]
        ys = [centre(a)[1] for a in teeth]
        median_y = statistics.median(ys)
        for a in teeth:
            cid = a["category_id"]
            total[cid] += 1
            if centre(a)[1] < median_y:
                upper_votes[cid] += 1
    upper = {c for c in TOOTH_IDS if total[c] and upper_votes[c] / total[c] > 0.5}
    return upper, upper_votes, total


def derive_position_mapping(images, per_image, full_arch_ids, upper):
    """Modal left-to-right rank of each class within its jaw, with support."""
    ranks = defaultdict(list)
    for iid in full_arch_ids:
        im = images[iid]
        up, lo = [], []
        for a in per_image[iid]:
            if a["category_id"] not in TOOTH_IDS:
                continue
            cx = centre(a)[0] / im["width"]
            (up if a["category_id"] in upper else lo).append((cx, a["category_id"]))
        for arr in (up, lo):
            arr.sort()
            for r, (_, cid) in enumerate(arr):
                ranks[cid].append(r)
    mapping = {}
    for cid, rs in ranks.items():
        counts = Counter(rs)
        modal, support = counts.most_common(1)[0]
        mapping[cid] = {
            "rank": modal,
            "support_images": support,
            "total_images": len(rs),
            "agreement": support / len(rs) if rs else 0.0,
        }
    return mapping


def canonical_sequences(mapping, upper):
    """The canonical left-to-right class order for each jaw."""
    up = sorted((c for c in mapping if c in upper), key=lambda c: mapping[c]["rank"])
    lo = sorted((c for c in mapping if c not in upper), key=lambda c: mapping[c]["rank"])
    return up, lo


def is_subsequence(seq, ref):
    it = iter(ref)
    return all(any(x == y for y in it) for x in seq)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MANIFESTS, exist_ok=True)
    images, names, per_image = load()

    # ---- full-arch images: all 32 tooth classes present exactly once --------
    full_arch_ids = []
    for iid in images:
        teeth = [a["category_id"] for a in per_image[iid] if a["category_id"] in TOOTH_IDS]
        if len(teeth) == 32 and len(set(teeth)) == 32:
            full_arch_ids.append(iid)
    full_arch_ids.sort()

    if not full_arch_ids:
        print("FATAL: no full-arch images; mapping cannot be derived.", file=sys.stderr)
        sys.exit(1)

    upper, upper_votes, tot = derive_jaw_partition(images, per_image, full_arch_ids)
    mapping = derive_position_mapping(images, per_image, full_arch_ids, upper)
    canon_up, canon_lo = canonical_sequences(mapping, upper)

    # ---- class_to_position.csv --------------------------------------------
    p = os.path.join(RESULTS, "class_to_position.csv")
    with open(p, "w", newline="") as fh:
        for line in provenance([
            f"derived from {len(full_arch_ids)} full-arch images (all 32 classes present once)",
            "jaw assigned by y-median split, matching app.py:82-104",
            "fdi read off FDI_UPPER/FDI_LOWER in app.py:57-58 at the modal rank",
            "agreement = fraction of full-arch images giving this class this rank",
        ]):
            fh.write(line + "\n")
        w = csv.writer(fh)
        w.writerow(["class_id", "jaw", "position_rank", "fdi",
                    "support_images", "total_images", "agreement"])
        for cid in sorted(mapping, key=lambda c: (c not in upper, mapping[c]["rank"])):
            m = mapping[cid]
            jaw = "upper" if cid in upper else "lower"
            fdi = (FDI_UPPER if cid in upper else FDI_LOWER)[m["rank"]]
            w.writerow([cid, jaw, m["rank"], fdi, m["support_images"],
                        m["total_images"], f"{m['agreement']:.4f}"])

    # ---- per-image label quality ------------------------------------------
    rows = []
    evaluable = []
    canon_instances = 0
    total_instances = 0
    for iid in sorted(images):
        im = images[iid]
        teeth = [a for a in per_image[iid] if a["category_id"] in TOOTH_IDS]
        n = len(teeth)
        total_instances += n
        ids = [a["category_id"] for a in teeth]
        full = (n == 32 and len(set(ids)) == 32)
        dup = len(ids) != len(set(ids))

        up, lo = [], []
        for a in teeth:
            cx = centre(a)[0] / im["width"]
            (up if a["category_id"] in upper else lo).append((cx, a["category_id"]))
        up.sort(); lo.sort()
        seq_up = [c for _, c in up]
        seq_lo = [c for _, c in lo]
        canonical = (not dup and n > 0
                     and is_subsequence(seq_up, canon_up)
                     and is_subsequence(seq_lo, canon_lo))
        if canonical:
            canon_instances += n
        if full and canonical:
            evaluable.append(iid)
        rows.append([iid, im["file_name"], n, int(full), int(canonical), int(dup)])

    p = os.path.join(RESULTS, "label_quality.csv")
    with open(p, "w", newline="") as fh:
        for line in provenance([
            "full_arch = all 32 tooth classes present exactly once",
            "canonical = left-to-right class order in each jaw is a subsequence of the derived canonical order",
            "duplicate_class = the same tooth class appears more than once in the image",
        ]):
            fh.write(line + "\n")
        w = csv.writer(fh)
        w.writerow(["image_id", "file_name", "labeled_teeth", "full_arch",
                    "canonical_mapping", "duplicate_class"])
        w.writerows(rows)

    # ---- instance counts ---------------------------------------------------
    counts = Counter(a["category_id"] for anns in per_image.values() for a in anns)
    p = os.path.join(RESULTS, "instance_counts.csv")
    with open(p, "w", newline="") as fh:
        for line in provenance(["kind: tooth (classes 1-32) or anomaly (classes 33-40); class 0 't' is an unused supercategory"]):
            fh.write(line + "\n")
        w = csv.writer(fh)
        w.writerow(["class_id", "class_name", "kind", "instances", "images_present"])
        present = defaultdict(set)
        for iid, anns in per_image.items():
            for a in anns:
                present[a["category_id"]].add(iid)
        for cid in sorted(names):
            kind = ("tooth" if cid in TOOTH_IDS
                    else "anomaly" if cid in ANOMALY_IDS else "unused")
            w.writerow([cid, names[cid], kind, counts.get(cid, 0), len(present.get(cid, ()))])

    # ---- evaluable manifest ------------------------------------------------
    p = os.path.join(MANIFESTS, "fdi_evaluable.txt")
    with open(p, "w") as fh:
        for line in provenance([
            "image ids on which FDI accuracy may be computed: full arch AND canonical mapping",
            "one image id per line",
        ]):
            fh.write(line + "\n")
        for iid in evaluable:
            fh.write(f"{iid}\n")

    # ---- summary -----------------------------------------------------------
    per_counts = [r[2] for r in rows]
    hist = Counter(per_counts)
    print("=" * 68)
    print(f"LABEL QUALITY -- {SPLIT}")
    print("=" * 68)
    print(f"images                       : {len(images)}")
    print(f"tooth instances (classes 1-32): {total_instances}")
    print(f"  mean per image             : {statistics.mean(per_counts):.2f}")
    print(f"  median per image           : {int(statistics.median(per_counts))}")
    print(f"  min / max                  : {min(per_counts)} / {max(per_counts)}")
    print("\nlabeled teeth per image (count of images):")
    for k in sorted(hist):
        bar = "#" * min(60, hist[k])
        print(f"  {k:>2} teeth : {hist[k]:>4}  {bar}")
    print(f"\nfull-arch images (32 distinct): {len(full_arch_ids)}")
    n_canon = sum(r[4] for r in rows)
    n_dup = sum(r[5] for r in rows)
    print(f"canonical-mapping images      : {n_canon}")
    print(f"images with duplicate classes : {n_dup}")
    print(f"FDI-EVALUABLE SUBSET          : {len(evaluable)} images "
          f"({len(evaluable)/len(images):.1%} of the split)")

    agrees = [m["agreement"] for m in mapping.values()]
    print(f"\nclass->position mapping agreement over {len(full_arch_ids)} full-arch images:")
    print(f"  min {min(agrees):.1%} / median {statistics.median(agrees):.1%} / max {max(agrees):.1%}")

    print("\n" + "-" * 68)
    print("FDI ACCURACY CEILING IMPLIED BY LABEL NOISE")
    print("-" * 68)
    print("Computed as a count, not an estimate:")
    print("  ceiling = tooth instances in canonical-mapping images")
    print("            / all tooth instances in the split")
    print(f"          = {canon_instances} / {total_instances}"
          f" = {canon_instances/total_instances:.1%}")
    print("A tooth in a non-canonical image cannot be scored correct no matter")
    print("what the model predicts, because its own label disagrees with the")
    print("mapping every other image uses. That is a bound on the model, from")
    print("the labels alone.")
    print(f"\nOn the {len(evaluable)}-image evaluable subset the ceiling is 100% by")
    print("construction -- those images were selected for canonical labels. That")
    print("is the reason to restrict FDI accuracy to them, and the reason the n")
    print("must be reported next to the number.")
    print("\nwrote:")
    for f in ["results/class_to_position.csv", "results/label_quality.csv",
              "results/instance_counts.csv", "manifests/fdi_evaluable.txt"]:
        print(f"  paper_eval/{f}")


if __name__ == "__main__":
    main()
