"""Demo mode: run the real geometric engine on cached model predictions, no Detectron2.

Enabled with TOOTH_AI_DEMO=1, or automatically when Detectron2 cannot be imported.
Samples live in assets/samples/ (built by tools/make_demo_fixtures.py). Each holds the
teeth model's own saved single-pass predictions for one validation image; this module
wraps them in a duck-typed stand-in for Detectron2's ``Instances`` so that
``tooth_ai.pipeline.process_predictions`` runs unchanged.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from tooth_ai.analysis import (DEFAULT_FINDINGS_THRESHOLD, DEFAULT_TEETH_THRESHOLD,
                               MODES, chart_from_detections)
from tooth_ai.pipeline import process_predictions

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "assets" / "samples"


def demo_requested():
    return os.environ.get("TOOTH_AI_DEMO", "").strip() not in ("", "0", "false")


def detectron2_available():
    try:
        import detectron2  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def list_samples():
    index = SAMPLES_DIR / "index.json"
    if not index.exists():
        return []
    return json.loads(index.read_text())


def sample_image_path(name):
    return SAMPLES_DIR / f"{name}.jpg"


def decode_rle(counts, height, width):
    """Decode a COCO compressed RLE string (pycocotools' rleFrString + rleDecode)."""
    s = counts.encode() if isinstance(counts, str) else counts
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = s[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    flat = np.zeros(height * width, dtype=bool)
    idx, val = 0, False
    for c in cnts:
        if val:
            flat[idx:idx + c] = True
        idx += c
        val = not val
    return flat.reshape((width, height)).T  # COCO RLE is column-major


class _Array:
    def __init__(self, a):
        self._a = a

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, a):
        self.tensor = _Array(a)


class _Instances:
    """Just enough of detectron2.structures.Instances for process_predictions."""

    def __init__(self, boxes, classes, scores, masks):
        self.pred_boxes = _Boxes(boxes)
        self.pred_classes = _Array(classes)
        self.scores = _Array(scores)
        self.pred_masks = _Array(masks)

    def to(self, device):
        return self

    def has(self, name):
        return hasattr(self, name)


@lru_cache(maxsize=8)
def _load_outputs(name):
    fx = json.loads((SAMPLES_DIR / f"{name}.json").read_text())
    h, w = fx["height"], fx["width"]
    dets = fx["detections"]
    boxes = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2],
                       d["bbox"][1] + d["bbox"][3]] for d in dets], dtype=np.float32)
    classes = np.array([d["category_id"] - 1 for d in dets], dtype=np.int64)  # COCO 1-based
    scores = np.array([d["score"] for d in dets], dtype=np.float32)
    masks = np.stack([decode_rle(d["rle"], h, w) for d in dets]) if dets else np.zeros((0, h, w), bool)
    return {"instances": _Instances(boxes, classes, scores, masks)}, (h, w)


def analyze_sample(name, mode="Full chart", teeth_threshold=DEFAULT_TEETH_THRESHOLD,
                   findings_threshold=DEFAULT_FINDINGS_THRESHOLD):
    """Same geometric pipeline as tooth_ai.analysis.analyze, on cached detections."""
    outputs, (h, w) = _load_outputs(name)
    raw_teeth, raw_anomalies = process_predictions(
        outputs, image_width=w, teeth_threshold=teeth_threshold,
        anomaly_threshold=findings_threshold, class_map=None)
    if MODES[mode]["key"] == "findings":
        raw_teeth = []
    return chart_from_detections(raw_teeth, raw_anomalies, (h, w, 3), mode,
                                 findings_threshold, timings={"Detection (cached)": 0.0},
                                 source="demo")
