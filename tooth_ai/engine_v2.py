"""Engine v2: the geometric half of the pipeline with the flaws measured in paper_eval fixed.

Not the deployed default (``tooth_ai.evidence.DEPLOYED`` still names the v1 configuration). Every change is
measured by ``paper_eval/engine_lab.py`` on the validation split (design) and the test split (report):

1. Cleanup before numbering. v1 numbered every box, then pruned to 32 by raising a score floor in 0.02
   steps (which can overshoot, e.g. 35 -> 30) and deleted the losers of number conflicts. On the 93
   full-arch test images that removed 54 correctly detected teeth while 122 false boxes stayed. v2 caps
   at 32 boxes by score before numbering, at 16 per jaw, and never deletes a detected tooth.
2. Numbering from the dental midline. v1 walks each jaw from the patient's right third molar, so one
   missing tooth at that end shifts every later number (partial arches: 35 % FDI). v2 finds the midline
   (see ``find_midline``) and numbers outward on both sides.
3. Duplicates: a detection whose mask lies mostly inside a stronger one is dropped (box NMS misses them).
4. Jaws split along a fitted bite line instead of at the median height, which forced equal halves.
5. A consistent gap rule: v1 compared the centre distance with the current tooth's width only and counted
   ``int(gap / width) - 1`` missing teeth, which is 0 for gaps of 1.6-2.0 widths although the rule fired.
   v2 uses the mean width of the two neighbours and rounds.
"""

from __future__ import annotations

import numpy as np

from tooth_ai.pipeline import DentalGeometricEngine

MAX_TEETH, MAX_PER_JAW = 32, 16
# The configuration the app uses for engine "v2", chosen on the validation split only (paper_eval/engine_lab.py,
# variant "v2-1p-dc"): on 152 validation images FDI 61.2 % vs 47.5 % for v1, recall 94.2 % vs 93.0 %.
# The detector side of v2 (in tooth_ai.analysis): one pass, no CLAHE fusion, one tooth threshold everywhere.
DEFAULTS = {"midline": "symmetry", "jaw_split": "curve", "dedupe": True, "shared_midline": False}
GAP_RATIO = 1.6
SIDE_FDI = {("upper", "right"): [11, 12, 13, 14, 15, 16, 17, 18], ("upper", "left"): [21, 22, 23, 24, 25, 26, 27, 28],
            ("lower", "right"): [41, 42, 43, 44, 45, 46, 47, 48], ("lower", "left"): [31, 32, 33, 34, 35, 36, 37, 38]}


def _cx(t):
    return (t["box"][0] + t["box"][2]) / 2


def _w(t):
    return t["box"][2] - t["box"][0]


def suppress_contained(teeth, contain=0.6):
    """Drop a detection whose mask lies mostly (``contain``) inside a higher-scoring one: a duplicate of the
    same tooth (often a crown-only box) that box-level NMS keeps because the boxes overlap little."""
    keep = []
    for t in sorted(teeth, key=lambda t: -t["score"]):
        m = t.get("mask")
        dup = False
        if m is not None and m.any():
            area = m.sum()
            for k in keep:
                km = k.get("mask")
                if km is None or not DentalGeometricEngine.compute_iou(t["box"], k["box"]):
                    continue
                if (m & km).sum() >= contain * min(area, km.sum()):
                    dup = True
                    break
        if not dup:
            keep.append(t)
    return keep


def split_jaws_curve(teeth, iters=4):
    """Upper/lower by a fitted bite line instead of the median height (which forces equal halves).

    Start from the median split, fit a quadratic through each jaw's tooth centres, put the bite line half-way
    between the two curves and reassign; repeat. Falls back to the median split for few teeth."""
    if len(teeth) < 8:
        return DentalGeometricEngine.split_jaws(teeth)
    xs = np.array([_cx(t) for t in teeth])
    ys = np.array([(t["box"][1] + t["box"][3]) / 2 for t in teeth])
    upper = ys < np.median(ys)
    for _ in range(iters):
        if upper.sum() < 4 or (~upper).sum() < 4:
            break
        cu = np.polyfit(xs[upper], ys[upper], 2)
        cl = np.polyfit(xs[~upper], ys[~upper], 2)
        line = (np.polyval(cu, xs) + np.polyval(cl, xs)) / 2
        new = ys < line
        if (new == upper).all():
            break
        upper = new
    return [t for t, u in zip(teeth, upper) if u], [t for t, u in zip(teeth, upper) if not u]


def cleanup(teeth_nms, jaw_split="median"):
    """Cap to 32 by score, then 16 per jaw. Returns (upper, lower)."""
    keep = sorted(teeth_nms, key=lambda t: -t["score"])[:MAX_TEETH]
    upper, lower = split_jaws_curve(keep) if jaw_split == "curve" else DentalGeometricEngine.split_jaws(keep)
    top = lambda j: sorted(j, key=lambda t: -t["score"])[:MAX_PER_JAW]  # noqa: E731
    return top(upper), top(lower)


def find_midline(jaw, strategy="median", return_cost=False):
    """x of the dental midline for one jaw.

    median   : median tooth centre (exact for a complete, symmetric jaw)
    symmetry : the gap between neighbouring teeth that best mirrors tooth widths on both sides
               (widths grow from incisors to molars, so the sequences match only at the midline)
    """
    xs = sorted(_cx(t) for t in jaw)
    if not xs:
        return (None, None) if return_cost else None
    if strategy == "median" or len(xs) < 4:
        return (float(np.median(xs)), None) if return_cost else float(np.median(xs))
    teeth = sorted(jaw, key=_cx)
    widths = [_w(t) for t in teeth]
    med = float(np.median(xs))
    best, best_cost = None, None
    for k in range(1, len(teeth)):                      # split between teeth k-1 and k
        left, right = widths[:k][::-1], widths[k:]      # outward from the split on each side
        n = min(len(left), len(right), 4)               # incisors, canines, first premolars
        if n < 2:
            continue
        cost = float(np.mean([abs(a - b) / max(a, b) for a, b in zip(left[:n], right[:n])]))
        cost += 0.15 * abs(len(left) - len(right)) / len(teeth)          # mild prior towards balance
        cost += 0.10 * abs((_cx(teeth[k - 1]) + _cx(teeth[k])) / 2 - med) / max(np.mean(widths), 1)
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, (_cx(teeth[k - 1]) + _cx(teeth[k])) / 2
    best = best if best is not None else med
    return (best, best_cost) if return_cost else best


def number_jaw(jaw, jaw_type, mid_x):
    """Number outward from the midline; image left of it is the patient's right."""
    out = []
    for side in ("right", "left"):
        seq = SIDE_FDI[(jaw_type, side)]
        teeth = sorted([t for t in jaw if (_cx(t) < mid_x) == (side == "right")], key=lambda t: abs(_cx(t) - mid_x))
        idx = 0
        for i, t in enumerate(teeth):
            trace = {"jaw": jaw_type, "rank": i + 1, "jaw_count": len(jaw), "gap_ratio": None, "skipped": 0,
                     "overflow": False, "reassigned_from": None, "engine": "v2", "side": side}
            if i > 0:
                gap = abs(_cx(t) - _cx(teeth[i - 1]))
                avg = (_w(t) + _w(teeth[i - 1])) / 2
                if avg > 0:
                    trace["gap_ratio"] = gap / avg
                    if gap > GAP_RATIO * avg:
                        skip = min(int(round(gap / avg)) - 1, 3)
                        idx += skip
                        trace["skipped"] = skip
            if idx >= len(seq):
                trace["overflow"] = True
            fdi = seq[min(idx, len(seq) - 1)]
            out.append({"box": t["box"], "fdi": fdi, "score": t["score"], "quadrant": fdi // 10,
                        "mask": t.get("mask"), "findings": [], "trace": trace})
            idx += 1
    return out


def resolve_conflicts(teeth):
    """Duplicate numbers: the higher score keeps it; the other moves to the nearest free number in its
    quadrant, or keeps its number flagged as a conflict. A detected tooth is never deleted."""
    by_fdi = {}
    for t in sorted(teeth, key=lambda t: -t["score"]):
        by_fdi.setdefault(t["fdi"], []).append(t)
    taken = {f for f in by_fdi}
    for fdi, group in by_fdi.items():
        for loser in group[1:]:
            free = [f for f in range(loser["quadrant"] * 10 + 1, loser["quadrant"] * 10 + 9) if f not in taken]
            if free:
                new = min(free, key=lambda f: abs(f - fdi))
                loser["trace"]["reassigned_from"] = fdi
                loser["fdi"] = new
                taken.add(new)
            else:
                loser["trace"]["conflict"] = True
    return teeth


def number_teeth(teeth_nms, midline="symmetry", jaw_split="median", dedupe=False, shared_midline=False):
    """v2 replacement for split_jaws + sort_and_assign_fdi + recursive_prune + resolve_fdi_conflicts."""
    if dedupe:
        teeth_nms = suppress_contained(teeth_nms)
    upper, lower = cleanup(teeth_nms, jaw_split)
    if midline == "right-anchored":                      # v1 numbering, v2 cleanup (for ablations)
        teeth = (DentalGeometricEngine.sort_and_assign_fdi(upper, "upper")
                 + DentalGeometricEngine.sort_and_assign_fdi(lower, "lower"))
    else:
        mids = {name: find_midline(jaw, midline, return_cost=True) for jaw, name in ((upper, "upper"), (lower, "lower"))}
        if shared_midline and all(m[0] is not None for m in mids.values()):
            # The two midlines are almost aligned on an OPG. If they disagree by more than half a tooth, both
            # jaws use the one found with the lower mirror cost (the more symmetric, usually fuller, jaw).
            widths = [_w(t) for t in upper + lower]
            (xu, cu), (xl, cl) = mids["upper"], mids["lower"]
            if abs(xu - xl) > 0.5 * float(np.median(widths)) and cu is not None and cl is not None:
                x = xu if cu <= cl else xl
                mids = {"upper": (x, cu), "lower": (x, cl)}
        teeth = []
        for jaw, name in ((upper, "upper"), (lower, "lower")):
            if jaw:
                teeth += number_jaw(jaw, name, mids[name][0])
    return resolve_conflicts(teeth)
