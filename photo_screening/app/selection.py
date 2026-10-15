"""Which fused detections become findings: thresholds plus the rules measured on the AKU splits.

Shared by the service (``report.py``) and the evaluation script, so EVAL.md scores exactly what
users see. Every rule here is backed by a measurement recorded in EVAL.md and MODEL_CARD.md.
"""

from __future__ import annotations

import numpy as np

from photo_screening.app.config import Settings
from photo_screening.app.fusion import fuse_models, iou, overlap_of_smaller
from photo_screening.app.models.base import RawDet
from photo_screening.app.models.registry import TYPE_PRIORITY

RESTORATIONS = ("filling", "crown")
EDGE_TYPES = ("wear", "filling", "crown")       # DSLR-trained AlphaDent types; fingers sit at the frame edge


def same_spot(a: np.ndarray, b: np.ndarray, iou_thr: float = 0.3, contain_thr: float = 0.6) -> bool:
    return iou(a, b) >= iou_thr or overlap_of_smaller(a, b) >= contain_thr


def touches_edge(box: np.ndarray, size: tuple[int, int], margin: float) -> bool:
    if margin <= 0:                                  # 0 switches the rule off
        return False
    w, h = size
    mx, my = margin * w, margin * h
    return bool(box[0] <= mx or box[1] <= my or box[2] >= w - mx or box[3] >= h - my)


def agreement(d: RawDet, settings: Settings) -> bool:
    """Two weak signals on the same spot: the phone caries model below its keep threshold and
    AlphaDent caries well above its own noise. ~77 % of such spots were real decay on AKU (10/13)."""
    s = d.model_scores
    return (s.get("caries_aku", 0.0) >= settings.agree_caries_aku_min
            and s.get("alphadent", 0.0) >= settings.agree_alphadent_min)


def select(per_model: dict[str, list[RawDet]], thresholds: dict, settings: Settings, size: tuple[int, int],
           caries_model_loaded: bool) -> list[RawDet]:
    """Fuse models, then keep what clears its threshold or a measured rule."""
    hidden = {"tooth"} | (set() if settings.report_cavity_crack else {"cavity", "crack"})
    lesions = {m: [d for d in dets if d.type not in hidden] for m, dets in per_model.items()}
    keep: list[RawDet] = []
    for d in fuse_models(lesions, TYPE_PRIORITY, settings.fusion_iou):
        if d.type == "caries" and caries_model_loaded and not settings.alphadent_caries_standalone \
                and d.sources == ["alphadent"]:
            continue                             # AlphaDent alone: 2-5 % precision on phone photos
        if d.score < thresholds[d.type]["t_low"] and not (d.type == "caries" and agreement(d, settings)):
            continue
        if d.type in EDGE_TYPES and touches_edge(d.box, size, settings.edge_margin):
            continue
        keep.append(d)
    caries = [d.box for d in keep if d.type == "caries"]
    # A decay mark wins over a filling/crown mark on the same spot: on AKU 22-33 % of AlphaDent
    # "fillings" lay on labelled decay (brown lesions), and a missed lesion is the costlier error.
    return [d for d in keep if not (d.type in RESTORATIONS and any(same_spot(d.box, c) for c in caries))]

