#!/usr/bin/env python3
"""Paired per-image deltas between configurations, with bootstrap CIs.

Standard library only.

Every configuration ran on the same images in the same order, so a difference
between two of them is a PAIRED comparison. Overlapping marginal confidence
intervals neither establish nor refute a difference -- only the CI on the
per-image delta does that, and a paired CI is usually far tighter because the
image-to-image variance cancels.

    python3 paper_eval/paired_deltas.py
    python3 paper_eval/paired_deltas.py --a single_pass --b full
"""
import argparse
import csv
import os
import random
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SEED = 20260921
N_BOOT = 10000

# metric -> (numerator column, denominator column, subset column or None)
#
# "FDI accuracy" is JOINT: its denominator is every ground-truth tooth, so a
# tooth the detector never found counts as a numbering error. The two rows
# under it split that, using the number MATCHED on the same image as the
# denominator, so detection and numbering can be compared separately when the
# two sides run on different detectors.
METRICS = {
    "full-arch recall":      ("recall_hit",  "recall_tot",  "full_arch"),
    "central-zone recall":   ("central_hit", "central_tot", "full_arch"),
    "peripheral recall":     ("periph_hit",  "periph_tot",  "full_arch"),
    "FDI accuracy (joint)":  ("fdi_hit",     "fdi_tot",     "canonical"),
    "detection recall":      ("recall_hit",  "recall_tot",  "canonical"),
    "FDI | detected":        ("fdi_hit",     "recall_hit",  "canonical"),
    "jaw | detected":        ("jaw_hit",     "jaw_tot",     "canonical"),
    "jaw accuracy (93 FA)":  ("jaw_hit",     "jaw_tot",     "full_arch"),
}


def load(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path} -- run the harness with --all first")
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def ival(r, k):
    v = r.get(k, "")
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def paired(rows_a, rows_b, num, den, subset):
    """Per-image (num, den) for both configs, restricted to a subset, aligned
    on image_id. Images where either side has no denominator are dropped from
    BOTH -- keeping them would compare different populations."""
    a = {int(r["image_id"]): r for r in rows_a}
    b = {int(r["image_id"]): r for r in rows_b}
    out = []
    for iid in sorted(set(a) & set(b)):
        ra, rb = a[iid], b[iid]
        if subset and not (ra.get(subset) == "1" and rb.get(subset) == "1"):
            continue
        na, da = ival(ra, num), ival(ra, den)
        nb, db = ival(rb, num), ival(rb, den)
        if None in (na, da, nb, db) or da == 0 or db == 0:
            continue
        out.append((iid, na, da, nb, db))
    return out


def ratio(pairs, which):
    n = sum(p[1] if which == "a" else p[3] for p in pairs)
    d = sum(p[2] if which == "a" else p[4] for p in pairs)
    return (n / d) if d else float("nan")


def bootstrap_delta(pairs, n=N_BOOT, seed=SEED, alpha=0.05):
    """Resample IMAGES; each resample carries both configs' counts together.
    That pairing is the whole point -- it removes between-image variance."""
    if not pairs:
        return None
    point = ratio(pairs, "a") - ratio(pairs, "b")
    rng = random.Random(seed)
    k = len(pairs)
    stats = []
    for _ in range(n):
        an = ad = bn = bd = 0
        for _ in range(k):
            _i, na, da, nb, db = pairs[rng.randrange(k)]
            an += na; ad += da; bn += nb; bd += db
        if ad and bd:
            stats.append(an / ad - bn / bd)
    if not stats:
        return None
    stats.sort()
    lo = stats[int((alpha / 2) * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return point, lo, hi, k


def wins(pairs):
    """How many individual images favour a, b, or neither."""
    wa = wb = tie = 0
    for _i, na, da, nb, db in pairs:
        ra, rb = na / da, nb / db
        if ra > rb: wa += 1
        elif rb > ra: wb += 1
        else: tie += 1
    return wa, wb, tie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="single_pass")
    ap.add_argument("--b", default="full")
    ap.add_argument("--all-pairs", action="store_true")
    args = ap.parse_args()

    rows = load(os.path.join(RESULTS, "per_image.csv"))
    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    configs = sorted(by)
    print(f"configs present: {', '.join(configs)}")
    print(f"images per config: "
          f"{ {c: len(by[c]) for c in configs} }")

    # --all-pairs anchors on --a (default single_pass) rather than whichever
    # config sorts first, so adding a config cannot silently change what every
    # comparison is measured against.
    anchor = args.a if args.a in by else configs[0]
    comparisons = ([(args.a, args.b)] if not args.all_pairs else
                   [(anchor, c) for c in configs if c != anchor])

    n_intervals = 0
    lines = []
    for a, b in comparisons:
        if a not in by or b not in by:
            print(f"\nskip {a} vs {b}: missing config")
            continue
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"PAIRED DELTA   {a}  MINUS  {b}")
        lines.append("=" * 100)
        lines.append(f"{'metric':<22}{a[:10]:>10}{b[:10]:>10}{'delta':>10}"
                     f"{'95% interval on delta':>26}{'img w/l/t':>15}")
        lines.append("-" * 100)
        for label, (num, den, subset) in METRICS.items():
            pairs = paired(by[a], by[b], num, den, subset)
            res = bootstrap_delta(pairs)
            if res is None:
                lines.append(f"{label:<22}{'no paired data':>71}")
                continue
            d, lo, hi, k = res
            n_intervals += 1
            ra, rb = ratio(pairs, "a"), ratio(pairs, "b")
            wa, wb, tie = wins(pairs)
            lines.append(
                f"{label:<22}{ra*100:>9.1f}%{rb*100:>9.1f}%{d*100:>+9.2f}pp"
                f"  [{lo*100:>+6.2f}, {hi*100:>+6.2f}]pp"
                f"{f'{wa}/{wb}/{tie}':>13}  n={k}")
        lines.append("-" * 100)
        lines.append("  img w/l/t = images where a beat b / b beat a / tied")

    print("\n".join(lines))

    # ---- comparison accounting -------------------------------------------
    print()
    print("=" * 100)
    print("COMPARISON ACCOUNTING")
    print("=" * 100)
    print(f"  paired intervals computed in this run : {n_intervals}")
    print(f"  config pairs                          : {len(comparisons)}")
    print(f"  metrics per pair                      : {len(METRICS)}")
    print()
    print("  These intervals are not independent: the same images and the same")
    print("  detections feed every one of them, and the metrics are correlated")
    print("  (peripheral recall is a component of overall recall). No")
    print("  multiplicity correction is applied and none would be easy to")
    print("  justify here, so an interval that happens to exclude zero should")
    print("  be read as one comparison out of " + str(n_intervals) + ", not as a")
    print("  standalone test.")
    print()
    print("  Report the effect size and its interval. An interval that excludes")
    print("  zero is suggestive at this scale, not established; an interval that")
    print("  includes zero means this sample cannot separate the two, which is")
    print("  not the same as showing they are equal.")
    print("=" * 100)



if __name__ == "__main__":
    main()
