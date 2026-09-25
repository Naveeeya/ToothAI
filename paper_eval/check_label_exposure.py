#!/usr/bin/env python3
"""How much annotation-convention noise did the 41-class model train through?

Standard library only.

Section 5.7b measures that detection-plus-geometry numbers teeth about 20
percentage points more accurately than a 41-class network trained to do it
directly. This supplies the mechanism: the class-to-position convention in the
training data is not consistent, so a model that predicts position directly
must learn through mutually contradictory examples, while a geometric engine
never learns positions at all and is structurally immune.

The canonical test is the same one label_quality.py applies to the test split,
applied here to train and valid as well.

    python3 paper_eval/check_label_exposure.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")

UPPER = {1, 4, 12, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
CANON_UP = [1, 4, 16, 20, 21, 22, 24, 25, 26, 12, 23, 27, 28, 29, 30, 31]
CANON_LO = [19, 18, 17, 15, 14, 13, 11, 10, 9, 8, 7, 6, 5, 3, 2, 32]


def centre(a):
    x, y, w, h = a["bbox"]
    return x + w / 2.0, y + h / 2.0


def analyse(path, label):
    if not os.path.exists(path):
        return None
    coco = json.load(open(path))
    imgs = {i["id"]: i for i in coco["images"]}
    per = defaultdict(list)
    for a in coco["annotations"]:
        if 1 <= a["category_id"] <= 32:
            per[a["image_id"]].append(a)

    full = [i for i in imgs
            if len(per[i]) == 32
            and len({a["category_id"] for a in per[i]}) == 32]
    canon = 0
    orderings = Counter()
    for iid in full:
        im = imgs[iid]
        up, lo = [], []
        for a in per[iid]:
            cx = centre(a)[0] / im["width"]
            (up if a["category_id"] in UPPER else lo).append((cx, a["category_id"]))
        up.sort(); lo.sort()
        u = [c for _, c in up]
        l = [c for _, c in lo]
        orderings[(tuple(u), tuple(l))] += 1
        if u == CANON_UP and l == CANON_LO:
            canon += 1
    return {
        "split": label, "images": len(imgs),
        "tooth_instances": sum(len(v) for v in per.values()),
        "full_arch_checkable": len(full),
        "canonical": canon, "non_canonical": len(full) - canon,
        "pct_non_canonical": (100.0 * (len(full) - canon) / len(full)) if full else 0.0,
        "distinct_orderings": len(orderings),
        "pct_of_split_checkable": (100.0 * len(full) / len(imgs)) if imgs else 0.0,
    }


def main():
    base = os.path.join(ROOT, "data", "final-di-stratified")
    rows = []
    for sp in ("train", "valid", "test"):
        r = analyse(os.path.join(base, sp, "_annotations.coco.json"), sp)
        if r:
            rows.append(r)
    if not rows:
        sys.exit("no annotation files found")

    print("=" * 96)
    print("LABEL-CONVENTION EXPOSURE -- data/final-di-stratified")
    print("=" * 96)
    print(f"{'split':<8}{'images':>8}{'tooth inst':>12}{'checkable':>11}"
          f"{'canonical':>11}{'NON-canon':>11}{'% non-canon':>13}{'orderings':>11}")
    print("-" * 96)
    for r in rows:
        print(f"{r['split']:<8}{r['images']:>8}{r['tooth_instances']:>12,}"
              f"{r['full_arch_checkable']:>11}{r['canonical']:>11}"
              f"{r['non_canonical']:>11}{r['pct_non_canonical']:>12.1f}%"
              f"{r['distinct_orderings']:>11}")
    print("-" * 96)

    tr = next((r for r in rows if r["split"] == "train"), None)
    if tr:
        print(f"\nThe 41-class model trained on {tr['images']:,} images carrying "
              f"{tr['tooth_instances']:,} tooth instances.")
        print(f"Of the {tr['full_arch_checkable']} training images where the "
              f"convention is checkable at all,")
        print(f"{tr['non_canonical']} ({tr['pct_non_canonical']:.1f}%) use a "
              f"NON-canonical mapping, across "
              f"{tr['distinct_orderings']} distinct orderings.")
        print()
        print("Each of those teaches a different answer for the same anatomical")
        print("position. A model predicting position directly must learn through")
        print("that. A geometric engine derives position from left-to-right order")
        print("at inference and never learns a convention, so it cannot be taught")
        print("a wrong one.")
        print()
        print(f"LOWER BOUND: only {tr['pct_of_split_checkable']:.0f}% of the "
              f"training split is full-arch and therefore checkable. The other "
              f"{tr['images'] - tr['full_arch_checkable']:,} images may be")
        print("equally inconsistent and simply invisible to this test.")

    p = os.path.join(RESULTS, "label_convention_exposure.json")
    json.dump({r["split"]: r for r in rows}, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
