"""Merge detections: TTA passes of one model, then all models, per unified finding type.

The pass-level fusion is a numpy port of the OPG pipeline's consensus box fusion
(``tooth_ai/pipeline.py:weighted_box_fusion``; ``app.py:499`` at tag ``paper-eval-v1``), which is tied
to Detectron2 ``Instances`` there: clusters at IoU > 0.55, a score-weighted box, the max score when
every pass agrees and ``max x 0.6`` when only one pass saw it. Across models there is no penalty:
agreement is kept as provenance (``sources``) and raises the confidence band in ``rules.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from photo_screening.app.models.base import RawDet


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def overlap_of_smaller(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller box's area: 1.0 when one box lies inside the other."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / smaller) if smaller > 0 else 0.0


def _clusters(dets: Sequence[RawDet], iou_thr: float, contain_thr: float | None = None) -> list[list[RawDet]]:
    """Greedy clustering by descending score, per type (as the OPG fusion does per class).

    With ``contain_thr``, a box mostly inside another also joins its cluster: models draw the same
    lesion at very different sizes (a fissure outline vs a whole-surface box), so IoU alone misses it."""
    out: list[list[RawDet]] = []
    for t in sorted({d.type for d in dets}):
        pool = sorted((d for d in dets if d.type == t), key=lambda d: -d.score)
        used = [False] * len(pool)
        for i, d in enumerate(pool):
            if used[i]:
                continue
            used[i] = True
            cluster = [d]
            for j in range(i + 1, len(pool)):
                if used[j]:
                    continue
                if iou(d.box, pool[j].box) > iou_thr or (
                        contain_thr is not None and overlap_of_smaller(d.box, pool[j].box) >= contain_thr):
                    used[j] = True
                    cluster.append(pool[j])
            out.append(cluster)
    return out


def _weighted_box(cluster: Sequence[RawDet]) -> np.ndarray:
    w = np.array([d.score for d in cluster])
    return np.average(np.stack([d.box for d in cluster]), axis=0, weights=w / w.sum()).astype(np.float32)


def fuse_passes(passes: Sequence[Sequence[RawDet]], iou_thr: float = 0.55, penalty: float = 0.6) -> list[RawDet]:
    """Consensus fusion of one model's TTA passes."""
    flat = [d for p in passes for d in p]
    if len(passes) <= 1:
        return [RawDet(d.model_id, d.type, d.score, d.box, d.polygon, 0, [d.model_id]) for d in flat]
    out = []
    for cluster in _clusters(flat, iou_thr):
        top = cluster[0]
        n_passes = len({d.pass_id for d in cluster})
        score = top.score if n_passes >= min(2, len(passes)) else top.score * penalty
        out.append(RawDet(top.model_id, top.type, score, _weighted_box(cluster), top.polygon, 0, [top.model_id]))
    return out


def fuse_models(per_model: dict[str, list[RawDet]], priority: dict[str, tuple[str, ...]],
                iou_thr: float = 0.55, contain_thr: float = 0.6) -> list[RawDet]:
    """Merge same-type detections of the same spot, within and across models.

    Two marks are the same spot when IoU > ``iou_thr`` or the smaller lies mostly (``contain_thr``)
    inside the larger. The leading model (for caries the phone-trained one) supplies the score and
    the shape: its own best box and outline, never an average of two very different shapes.
    ``sources`` lists every model that agreed.
    """
    flat = [d for dets in per_model.values() for d in dets]
    out = []
    for cluster in _clusters(flat, iou_thr, contain_thr):
        models = sorted({d.model_id for d in cluster})
        order = priority.get(cluster[0].type, ())
        lead = next((m for m in order if m in models), None)
        members = [d for d in cluster if d.model_id == lead] if lead else cluster
        best = max(members, key=lambda d: d.score)
        scores = {m: max(d.score for d in cluster if d.model_id == m) for m in models}
        out.append(RawDet(best.model_id, best.type, best.score, best.box, best.polygon, 0, models, scores))
    return sorted(out, key=lambda d: -d.score)


def merge_max(passes: Sequence[Sequence[RawDet]], iou_thr: float = 0.55, contain_thr: float = 0.6) -> list[RawDet]:
    """Union of one model's colour-normalised passes: each spot keeps its best-scoring detection.

    Unlike ``fuse_passes`` there is no single-pass penalty: the passes exist because a lesion can be
    invisible to the model under a colour cast it never saw in training (see EVAL.md)."""
    flat = [d for p in passes for d in p]
    return [max(c, key=lambda d: d.score) for c in _clusters(flat, iou_thr, contain_thr)]
