#!/usr/bin/env python3
"""Are sparse tooth labels concentrated on restored/diseased teeth?

Standard library only.

If the 8-or-so teeth annotated in a sparse image are mostly the ones carrying a
filling, crown, caries or root canal, then "labelled teeth" is not a random
sample of the mouth, and recall measured against it is biased: the model is
being scored only on the teeth that happen to look abnormal. Task D then has to
report recall separately for full-arch and sparse images.

Measured as: fraction of labelled teeth whose bbox has IoU > 0 with ANY anomaly
annotation in the same image.
"""
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT = os.path.join(ROOT, "data/final-di-stratified/test/_annotations.coco.json")


def iou(a, b):
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / float(aw * ah + bw * bh - inter)


def main():
    coco = json.load(open(SPLIT))
    teeth = defaultdict(list)
    anoms = defaultdict(list)
    for a in coco["annotations"]:
        (teeth if 1 <= a["category_id"] <= 32 else anoms)[a["image_id"]].append(a)

    groups = {
        "full-arch (32 labelled)":        lambda n: n == 32,
        "sparse (1-10 labelled)":         lambda n: 1 <= n <= 10,
        "  of which  1-5":                lambda n: 1 <= n <= 5,
        "  of which  6-8":                lambda n: 6 <= n <= 8,
        "  of which 9-10":                lambda n: 9 <= n <= 10,
        "mid (11-31 labelled)":           lambda n: 11 <= n <= 31,
    }

    print("=" * 78)
    print("Do sparse labels sit on restored / diseased teeth?")
    print("  metric: labelled tooth whose bbox has IoU > 0 with ANY anomaly box")
    print("=" * 78)
    print(f"{'group':<26} {'images':>7} {'teeth':>7} {'overlapping':>12} {'fraction':>9}")
    out = {}
    for label, pred in groups.items():
        imgs = [i for i in teeth if pred(len(teeth[i]))]
        n_t = n_o = 0
        for i in imgs:
            abx = [a["bbox"] for a in anoms.get(i, ())]
            for t in teeth[i]:
                n_t += 1
                if any(iou(t["bbox"], b) > 0 for b in abx):
                    n_o += 1
        frac = n_o / n_t if n_t else float("nan")
        out[label] = (len(imgs), n_t, n_o, frac)
        print(f"{label:<26} {len(imgs):>7} {n_t:>7} {n_o:>12} {frac:>8.1%}")

    # images carrying no anomaly at all -- a sparse image with no anomaly cannot
    # be explained by "they only annotated the interesting teeth"
    print()
    for label, pred in (("full-arch", lambda n: n == 32),
                        ("sparse 1-10", lambda n: 1 <= n <= 10)):
        imgs = [i for i in teeth if pred(len(teeth[i]))]
        none = sum(1 for i in imgs if not anoms.get(i))
        print(f"  {label:<12} images with ZERO anomaly annotations: {none}/{len(imgs)}")

    fa = out["full-arch (32 labelled)"][3]
    sp = out["sparse (1-10 labelled)"][3]
    print()
    print("-" * 78)
    print(f"full-arch {fa:.1%}  vs  sparse {sp:.1%}   ratio {sp/fa:.2f}x" if fa else "")
    print("-" * 78)
    if sp > fa * 1.5:
        print("VERDICT: sparse labels ARE concentrated on abnormal teeth.")
        print("  Recall on sparse images is biased and must be reported separately.")
    elif sp > fa * 1.15:
        print("VERDICT: a moderate skew. Report recall separately to be safe.")
    else:
        print("VERDICT: no strong concentration. Sparse labelling is not")
        print("  explained by anomaly presence -- it looks like partial annotation.")


if __name__ == "__main__":
    main()
