"""Raw detections and the Ultralytics YOLO wrapper shared by every photo model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class RawDet:
    """One detection in analysed-image pixels, already mapped to the unified vocabulary."""

    model_id: str
    type: str
    score: float
    box: np.ndarray                                  # x1, y1, x2, y2
    polygon: np.ndarray | None = None                # (N, 2) mask outline, if the model segments
    pass_id: int = 0                                 # 0 = original, 1 = horizontal flip (TTA)
    sources: list[str] = field(default_factory=list)
    model_scores: dict[str, float] = field(default_factory=dict)   # fused findings: best score per model


class Detector(Protocol):
    model_id: str
    types: tuple[str, ...]

    def predict(self, rgb: np.ndarray, tta: bool = False) -> list[list[RawDet]]:
        """Detections per pass (one list per pass: original, then flipped when ``tta``)."""
        ...


class YoloDetector:
    """Ultralytics YOLO (detect or segment) with a class-name -> unified-type mapping."""

    def __init__(self, model_id: str, weights: Path, imgsz: int, device: str,
                 class_map: Callable[[str], str | None], conf: float = 0.05):
        from ultralytics import YOLO

        self.model_id = model_id
        self.model = YOLO(str(weights))
        self.imgsz, self.device, self.conf = imgsz, device, conf
        self.names: dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}
        self.class_to_type = {k: class_map(v) for k, v in self.names.items()}
        self.types = tuple(sorted({t for t in self.class_to_type.values() if t}))

    def _run(self, bgr: np.ndarray, pass_id: int, flip_w: int | None) -> list[RawDet]:
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, iou=0.6, device=self.device,
                               verbose=False, max_det=300)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        polys = r.masks.xy if r.masks is not None else [None] * len(boxes)
        out: list[RawDet] = []
        for box, score, cls, poly in zip(boxes, scores, classes, polys, strict=False):
            t = self.class_to_type.get(int(cls))
            if t is None:
                continue
            poly = None if poly is None or len(poly) < 3 else np.asarray(poly, dtype=np.float32)
            if flip_w is not None:                    # map the flipped pass back to the original frame
                box = np.array([flip_w - box[2], box[1], flip_w - box[0], box[3]])
                if poly is not None:
                    poly = np.stack([flip_w - poly[:, 0], poly[:, 1]], 1)
            out.append(RawDet(self.model_id, t, float(score), np.asarray(box, dtype=np.float32), poly, pass_id))
        return out

    def predict(self, rgb: np.ndarray, tta: bool = False) -> list[list[RawDet]]:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])   # Ultralytics treats numpy input as BGR
        passes = [self._run(bgr, 0, None)]
        if tta:
            passes.append(self._run(np.ascontiguousarray(bgr[:, ::-1]), 1, rgb.shape[1]))
        return passes


def alphadent_map(name: str) -> str | None:
    """AlphaDent classes: Abrasion, Filling, Crown, Caries (4-class); 9-class variants map by name."""
    n = name.lower()
    for key, t in (("abras", "wear"), ("fill", "filling"), ("crown", "crown"), ("caries", "caries")):
        if key in n:
            return t
    return None


def caries_aku_map(name: str) -> str | None:
    """Every AKU class is a caries label."""
    return "caries"


def dentalai_map(name: str) -> str | None:
    """Tooth, cavity and crack only. DentalAI's caries class measured AP50 0.12 on its own test split,
    far below the AKU caries model, so it is not used (see MODEL_CARD.md)."""
    n = name.lower()
    return next((t for t in ("tooth", "cavity", "crack") if t in n), None)
