"""Synthetic detections for tests (no Detectron2)."""

import numpy as np

from tooth_ai.demo import _Instances


def outputs(dets, h=500, w=1000):
    """dets: list of (class_id, score, (x1, y1, x2, y2))."""
    boxes = np.array([d[2] for d in dets], dtype=np.float32).reshape(-1, 4)
    classes = np.array([d[0] for d in dets], dtype=np.int64)
    scores = np.array([d[1] for d in dets], dtype=np.float32)
    masks = np.zeros((len(dets), h, w), dtype=bool)
    for i, (_, _, (x1, y1, x2, y2)) in enumerate(dets):
        masks[i, int(y1):int(y2), int(x1):int(x2)] = True
    return {"instances": _Instances(boxes, classes, scores, masks)}


def arch(n_upper=16, n_lower=16, missing=(), w=1000, tooth_w=40, gap=8):
    """Raw tooth detections laid out along two jaws; `missing` = slot indices (0-31) to drop."""
    teeth = []
    span = 16 * tooth_w + 15 * gap
    x0 = (w - span) / 2
    for jaw, (y1, y2), n in (("upper", (150, 240), n_upper), ("lower", (260, 350), n_lower)):
        for i in range(16):
            slot = i if jaw == "upper" else 16 + i
            if i >= n or slot in missing:
                continue
            x = x0 + i * (tooth_w + gap)
            # Curve each jaw slightly (a smile line) so no two centres tie at the median:
            # split_jaws uses a strict y < median, so exact ties would all fall to the lower jaw.
            dy = 0.0004 * (x + tooth_w / 2 - w / 2) ** 2 * (-1 if jaw == "upper" else 1) + 0.01 * i
            teeth.append({"box": [x, y1 + dy, x + tooth_w, y2 + dy], "score": 0.9, "mask": None,
                          "class_name": "Tooth", "class_id": 0})
    return teeth
