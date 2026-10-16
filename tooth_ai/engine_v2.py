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
5. Numbering by position, not by count: each tooth's distance from the midline (in tooth widths) is aligned,
   in order, to a template learned on the TRAIN split (``arch_template.json``), so a missing or undetected
   tooth leaves its position empty instead of shifting the rest. (The count-based alternative, ``numbering=
   "gap"``, keeps v1's gap rule made consistent: mean neighbour width, rounded.)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from tooth_ai.pipeline import DentalGeometricEngine

MAX_TEETH, MAX_PER_JAW = 32, 16
# The configuration the app uses for engine "v2", chosen on the validation split only (paper_eval/engine_lab.py,
# variant "v2"); measured numbers are in paper_eval/results/engine_v2*.csv.
# The detector side of v2 (in tooth_ai.analysis): one pass, no CLAHE fusion, one tooth threshold everywhere.
DEFAULTS = {"midline": "symmetry", "jaw_split": "curve", "dedupe": True, "shared_midline": False,
            "numbering": "template"}
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


@lru_cache(maxsize=1)
def arch_template():
    """Median distance of each position from the midline, in tooth widths, learned on the TRAIN split
    (paper_eval/learn_template.py -> tooth_ai/arch_template.json)."""
    d = json.loads((Path(__file__).with_name("arch_template.json")).read_text())
    return {jaw: (np.array([d["median_distance_from_midline"][jaw][str(p)] for p in range(1, 9)]),
                  np.array([d["std"][jaw][str(p)] for p in range(1, 9)])) for jaw in ("upper", "lower")}


def _align(dists, mean, std):
    """Assign strictly increasing positions (0-7) to teeth ordered outward, minimising the spread-weighted
    squared distance to the template. Positions may be skipped (missing teeth). Returns (cost, positions)."""
    n, m = len(dists), len(mean)
    if n > m:
        return None
    inf = float("inf")
    cost = [[inf] * m for _ in range(n)]
    back = [[-1] * m for _ in range(n)]
    for j in range(m):
        cost[0][j] = ((dists[0] - mean[j]) / std[j]) ** 2
    for i in range(1, n):
        best, arg = inf, -1
        for j in range(m):
            if j >= 1 and cost[i - 1][j - 1] < best:
                best, arg = cost[i - 1][j - 1], j - 1
            if best < inf:
                cost[i][j] = best + ((dists[i] - mean[j]) / std[j]) ** 2
                back[i][j] = arg
    j = int(np.argmin(cost[n - 1]))
    total = cost[n - 1][j]
    pos = [j]
    for i in range(n - 1, 0, -1):
        j = back[i][j]
        pos.append(j)
    return total, pos[::-1]


def template_cost(jaw, jaw_type, mid_x):
    """Total alignment cost of both sides for a candidate midline (lower = better fit)."""
    mean, std = arch_template()[jaw_type]
    unit = float(np.median([_w(t) for t in jaw])) if jaw else 1.0
    total = 0.0
    for side in ("right", "left"):
        teeth = sorted([t for t in jaw if (_cx(t) < mid_x) == (side == "right")], key=lambda t: abs(_cx(t) - mid_x))
        if len(teeth) > 8:
            return float("inf")
        if not teeth:
            continue
        d = [abs(_cx(t) - mid_x) / max(unit, 1e-6) for t in teeth]
        total += min(r[0] for r in (_align([x * s for x in d], mean, std) for s in np.linspace(0.8, 1.2, 9)) if r)
    return total


def fit_midline(jaw, jaw_type):
    """Midline = the gap between neighbouring teeth whose two sides best fit the learned arch template."""
    teeth = sorted(jaw, key=_cx)
    if len(teeth) < 4:
        return find_midline(jaw, "median")
    cands = [(_cx(a) + _cx(b)) / 2 for a, b in zip(teeth, teeth[1:])]
    return min(cands, key=lambda x: template_cost(jaw, jaw_type, x))


def number_jaw_template(jaw, jaw_type, mid_x):
    """Number by where each tooth sits: distance from the midline in tooth widths, aligned in order to the
    learned template. A missing tooth anywhere leaves its position empty instead of shifting the rest."""
    mean, std = arch_template()[jaw_type]
    unit = float(np.median([_w(t) for t in jaw])) if jaw else 1.0
    out = []
    for side in ("right", "left"):
        seq = SIDE_FDI[(jaw_type, side)]
        teeth = sorted([t for t in jaw if (_cx(t) < mid_x) == (side == "right")], key=lambda t: abs(_cx(t) - mid_x))
        if not teeth:
            continue
        teeth, extra = teeth[:8], teeth[8:]      # more than 8 on one side: the distal extras overflow (flagged)
        d = [abs(_cx(t) - mid_x) / max(unit, 1e-6) for t in teeth]
        best = None
        for s in np.linspace(0.8, 1.2, 9):                       # per-side scale (magnification, crowding)
            r = _align([x * s for x in d], mean, std)
            if r and (best is None or r[0] < best[0]):
                best = (r[0], r[1], s)
        for i, (t, p) in enumerate(zip(teeth, best[1])):
            fdi = seq[p]
            out.append({"box": t["box"], "fdi": fdi, "score": t["score"], "quadrant": fdi // 10,
                        "mask": t.get("mask"), "findings": [],
                        "trace": {"jaw": jaw_type, "rank": i + 1, "jaw_count": len(jaw), "gap_ratio": None,
                                  "skipped": p - (best[1][i - 1] if i else -1) - 1, "overflow": False,
                                  "reassigned_from": None, "engine": "v2", "side": side,
                                  "template_distance": round(d[i] * best[2], 2)}})
        for t in extra:
            out.append({"box": t["box"], "fdi": seq[-1], "score": t["score"], "quadrant": seq[-1] // 10,
                        "mask": t.get("mask"), "findings": [],
                        "trace": {"jaw": jaw_type, "rank": None, "jaw_count": len(jaw), "gap_ratio": None,
                                  "skipped": 0, "overflow": True, "reassigned_from": None, "engine": "v2",
                                  "side": side}})
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


def number_teeth(teeth_nms, midline="symmetry", jaw_split="median", dedupe=False, shared_midline=False,
                 numbering="gap"):
    """v2 replacement for split_jaws + sort_and_assign_fdi + recursive_prune + resolve_fdi_conflicts."""
    if dedupe:
        teeth_nms = suppress_contained(teeth_nms)
    upper, lower = cleanup(teeth_nms, jaw_split)
    if midline == "right-anchored":                      # v1 numbering, v2 cleanup (for ablations)
        teeth = (DentalGeometricEngine.sort_and_assign_fdi(upper, "upper")
                 + DentalGeometricEngine.sort_and_assign_fdi(lower, "lower"))
    else:
        if midline == "fit":
            mids = {name: (fit_midline(jaw, name) if jaw else None, None) for jaw, name in ((upper, "upper"), (lower, "lower"))}
        else:
            mids = {name: find_midline(jaw, midline, return_cost=True)
                    for jaw, name in ((upper, "upper"), (lower, "lower"))}
        if shared_midline and all(m[0] is not None for m in mids.values()):
            # The two midlines are almost aligned on an OPG. If they disagree by more than half a tooth, both
            # jaws use the one found with the lower mirror cost (the more symmetric, usually fuller, jaw).
            widths = [_w(t) for t in upper + lower]
            (xu, cu), (xl, cl) = mids["upper"], mids["lower"]
            if abs(xu - xl) > 0.5 * float(np.median(widths)) and cu is not None and cl is not None:
                x = xu if cu <= cl else xl
                mids = {"upper": (x, cu), "lower": (x, cl)}
        teeth = []
        place = number_jaw_template if numbering == "template" else number_jaw
        for jaw, name in ((upper, "upper"), (lower, "lower")):
            if jaw:
                teeth += place(jaw, name, mids[name][0])
    return resolve_conflicts(teeth)
