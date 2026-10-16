#!/usr/bin/env python3
"""Learn where each tooth position sits along the arch, from the TRAIN split only.

    python paper_eval/learn_template.py        # writes tooth_ai/arch_template.json

For every training image whose labels pass paper_eval's canonical-mapping test and whose jaw has both
central incisors, measure each tooth's horizontal distance from the dental midline (half-way between the two
central incisors) in units of that jaw's median tooth width. The template is the median distance per
position (1 = central incisor ... 8 = third molar) for each jaw. Numbers only; no image names.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_lab as L  # noqa: E402
import evalkit as K  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "tooth_ai", "arch_template.json")


def main():
    K.SPLIT_DIR = K.SPLIT_DIR.replace(os.sep + "test", os.sep + "train")
    K.SPLIT_JSON = os.path.join(K.SPLIT_DIR, "_annotations.coco.json")
    _, _, gt_teeth, _ = K.load_split()
    cmap = K.load_class_map()
    canon = L.canonical_ids(gt_teeth)
    dist = {jaw: defaultdict(list) for jaw in ("upper", "lower")}
    used = defaultdict(int)
    for iid in canon:
        anns = gt_teeth[iid]
        for jaw, (r, l) in (("upper", (11, 21)), ("lower", (41, 31))):
            teeth = [(cmap[a["category_id"]][2], a["bbox"]) for a in anns if cmap[a["category_id"]][0] == jaw]
            by = {f: b for f, b in teeth}
            if r not in by or l not in by or len(teeth) < 6:
                continue
            cx = lambda b: b[0] + b[2] / 2  # noqa: E731
            mid = (cx(by[r]) + cx(by[l])) / 2
            unit = float(np.median([b[2] for _, b in teeth]))
            if unit <= 0:
                continue
            for fdi, b in teeth:
                dist[jaw][fdi % 10].append(abs(cx(b) - mid) / unit)
            used[jaw] += 1
    template = {jaw: {str(p): round(float(np.median(v)), 4) for p, v in sorted(dist[jaw].items())}
                for jaw in dist}
    spread = {jaw: {str(p): round(float(np.std(v)), 4) for p, v in sorted(dist[jaw].items())} for jaw in dist}
    out = {"source": "train split, canonical-mapping images with both central incisors labelled",
           "images_per_jaw": dict(used), "unit": "median tooth box width of the jaw",
           "median_distance_from_midline": template, "std": spread}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
