#!/usr/bin/env python3
"""Is class->jaw reliable on the 26 non-canonical full-arch images?

AUDIT.md asserted that it was: that the same 16 class ids sit in the upper
jaw in every full-arch image, so jaw-assignment accuracy could be scored on
all 93 even though FDI accuracy could only be scored on the 67 canonical ones.

That assertion was never checked by code. It is wrong, and the decomposition
in decompose.py is what exposed it: jaw accuracy given detection is 98.96% on
the 67 canonical full-arch images and 57.2% on the other 26 -- near chance,
which no geometric jaw split plausibly produces on a complete arch.

This settles it from the ground truth alone, with no model involved. For each
full-arch image, split the annotations at the y-median of their own box
centres -- the same rule DentalGeometricEngine.split_jaws uses (app.py:82-104)
-- and ask which class ids land in the upper half.

Standard library only.

Usage:
    python3 paper_eval/check_jaw_membership.py
"""
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as K


def main():
    manifests = K.load_manifests()
    class_map = K.load_class_map()
    coco = json.load(open(K.SPLIT_JSON))
    per = collections.defaultdict(list)
    for a in coco["annotations"]:
        if 1 <= a["category_id"] <= 32:
            per[a["image_id"]].append(a)

    canon_upper = frozenset(c for c, (j, _r, _f) in class_map.items()
                            if j == "upper")

    def upper_set(iid):
        anns = per[iid]
        ys = [a["bbox"][1] + a["bbox"][3] / 2 for a in anns]
        med = statistics.median(ys)
        return frozenset(a["category_id"] for a, y in zip(anns, ys) if y < med)

    def agreement(ids):
        """Per-annotation: does the class's canonical jaw match where the box
        actually sits? This is the quantity jaw accuracy is scored against."""
        hit = tot = 0
        for iid in ids:
            anns = per[iid]
            ys = [a["bbox"][1] + a["bbox"][3] / 2 for a in anns]
            med = statistics.median(ys)
            for a, y in zip(anns, ys):
                jaw = class_map[a["category_id"]][0]
                tot += 1
                hit += int(jaw == ("upper" if y < med else "lower"))
        return hit, tot

    fa = manifests["full_arch"]
    canon = sorted(fa & manifests["canonical"])
    scram = sorted(fa - manifests["canonical"])

    print(f"canonical upper classes ({len(canon_upper)}): "
          f"{sorted(canon_upper)}\n")
    print(f"{'subset':<34}{'label jaw matches geometry':>30}")
    for label, ids in (("canonical full-arch", canon),
                       ("non-canonical full-arch", scram),
                       ("all full-arch", canon + scram)):
        h, t = agreement(ids)
        print(f"{label + f' (n={len(ids)})':<34}{h:>10}/{t:<8}{h/t*100:>9.1f}%")

    print("\ndistinct upper-jaw class sets among the non-canonical images:")
    groups = collections.Counter(upper_set(i) for i in scram)
    for us, n in groups.most_common():
        print(f"  {n:>3} image(s)  overlap with canonical upper "
              f"{len(us & canon_upper)}/{len(canon_upper)}")
        print(f"       upper = {sorted(us)}")

    ok = all(us == canon_upper for us in groups)
    print("\nVERDICT: class->jaw is "
          + ("reliable on all 93." if ok else
             "NOT reliable on all 93. " + str(sum(
                 n for us, n in groups.items() if us != canon_upper))
             + " of the 26 non-canonical full-arch images use a different jaw"
               " partition, so jaw accuracy over 93 images measures label"
               " convention as much as model behaviour. Score it on the"
               " canonical subset."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
