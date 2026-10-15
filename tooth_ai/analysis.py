"""One call from radiograph to chart: ``analyze()`` -> ``AnalysisResult``.

Reproduces the orchestration that used to live inline in the old app's ``main()``
(``git show paper-eval-v1:app.py``, the block under ``with st.spinner``), split so the
geometric half (``chart_from_detections``) also runs on demo detections without
Detectron2. Adds the numbering-reliability assessment and the geometry summary the UI
draws.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

from tooth_ai import evidence
from tooth_ai.evidence import pct
from tooth_ai.pipeline import (
    PATHOLOGY_CLASS_MAP,
    DentalGeometricEngine,
    clean_clinical_findings,
    compute_iou,
    process_predictions,
    run_inference,
    validate_anatomy,
)

MODES = {
    "Full chart": {
        "key": "full",
        "description": "Teeth, numbers and findings",
        "config": Path("output/resnet50_9class_20k/config.yaml"),
        "weights": Path("output/resnet50_9class_20k/model_final.pth"),
        "pathology_config": Path("output/rtx4060_48k/config.yaml"),
        "pathology_weights": Path("output/rtx4060_48k/model_final.pth"),
        "class_map": None,
    },
    "Findings only": {
        "key": "findings",
        "description": "Findings without tooth numbering",
        "config": Path("output/rtx4060_48k/config.yaml"),
        "weights": Path("output/rtx4060_48k/model_final.pth"),
        "class_map": PATHOLOGY_CLASS_MAP,
    },
}

MODEL_VERSION = "Mask R-CNN R50 (9-class, 20k) + R50 (41-class findings, 48k)"

DEFAULT_TEETH_THRESHOLD = 0.20
DEFAULT_FINDINGS_THRESHOLD = 0.15

FULL_JAW = 14          # a jaw with fewer detected teeth is treated as a partial arch
LOW_TOTAL = 10         # fewer teeth than this overall: image-quality alert


@dataclass
class Reliability:
    level: str                       # "ok" | "caution" | "alert" | "na"
    title: str
    detail: str
    reasons: list = field(default_factory=list)
    validator_confidence: float = None


@dataclass
class Geometry:
    image_size: tuple                # (height, width)
    split_y: float = None            # jaw split line (median tooth-centre y)
    midline_x: float = None          # dental midline (between 1x and 2x, else image centre)
    arch_upper: tuple = None         # quadratic coefficients y = a x^2 + b x + c
    arch_lower: tuple = None


@dataclass
class AnalysisResult:
    mode: str
    teeth: list                      # tooth dicts (box, fdi, score, quadrant, mask, findings, trace)
    findings: list                   # every finding after validation (raw anomaly dicts)
    unassigned: list                 # findings not mapped to any tooth
    geometry: Geometry
    reliability: Reliability
    timings: dict
    source: str = "model"            # "model" | "demo"

    @property
    def total_seconds(self):
        return sum(self.timings.values())

    def jaw_counts(self):
        upper = sum(1 for t in self.teeth if t["quadrant"] in (1, 2))
        return upper, len(self.teeth) - upper

    def tooth(self, fdi):
        return next((t for t in self.teeth if t["fdi"] == fdi), None)


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _fit_arch(teeth):
    if len(teeth) < 3:
        return None
    pts = np.array([_centre(t["box"]) for t in teeth])
    return tuple(float(c) for c in np.polyfit(pts[:, 0], pts[:, 1], 2))


def _geometry(teeth_nms, teeth, image_shape):
    h, w = image_shape[:2]
    geo = Geometry(image_size=(h, w))
    if len(teeth_nms) >= 2:
        geo.split_y = float(np.median([_centre(t["box"])[1] for t in teeth_nms]))
    by_fdi = {t["fdi"]: t for t in teeth}
    mids = [(_centre(by_fdi[a]["box"])[0] + _centre(by_fdi[b]["box"])[0]) / 2
            for a, b in ((11, 21), (41, 31)) if a in by_fdi and b in by_fdi]
    geo.midline_x = float(np.mean(mids)) if mids else w / 2
    geo.arch_upper = _fit_arch([t for t in teeth if t["quadrant"] in (1, 2)])
    geo.arch_lower = _fit_arch([t for t in teeth if t["quadrant"] in (3, 4)])
    return geo


def assess_reliability(teeth, mode="Full chart"):
    """Numbering reliability for one case.

    Geometric numbering walks each jaw from the patient's right third molar, so a missing
    or undetected tooth shifts every later number. As deployed, FDI accuracy measured
    76.6 % on complete arches vs 35.4 % on partial ones (tooth_ai.evidence).
    """
    if MODES[mode]["key"] == "findings":
        return Reliability("na", "Numbering not run",
                           "Findings only mode does not number teeth.")
    total = len(teeth)
    upper = sum(1 for t in teeth if t["quadrant"] in (1, 2))
    lower = total - upper
    skipped = [t for t in teeth if t.get("trace", {}).get("skipped")]
    moved = [t for t in teeth if t.get("trace", {}).get("reassigned_from")]

    confidence = None
    try:
        from fdi_validator import FDIValidator
        light = [{"fdi": t["fdi"], "box": list(t["box"]), "score": t["score"]} for t in teeth]
        _, _, confidence = FDIValidator.validate_and_correct(light)
    except Exception:  # validator is advisory; never block the result on it
        pass

    if total < LOW_TOTAL:
        return Reliability(
            "alert", "Low tooth count — check image quality",
            f"Only {total} teeth were detected. Numbering is unreliable when most of the "
            "arch is missing or the image is not a panoramic radiograph.",
            [f"{total} teeth detected"], confidence)

    reasons = []
    partial = []
    if upper < FULL_JAW:
        partial.append("upper")
        reasons.append(f"Upper jaw: {upper} teeth")
    if lower < FULL_JAW:
        partial.append("lower")
        reasons.append(f"Lower jaw: {lower} teeth")
    if skipped:
        reasons.append("Gap detected before " + ", ".join(str(t["fdi"]) for t in skipped))
    if moved:
        reasons.append("Number conflict resolved for " + ", ".join(str(t["fdi"]) for t in moved))
    if confidence is not None and confidence < 0.70:
        reasons.append(f"Spatial consistency {confidence:.0%}")

    if not reasons:
        return Reliability(
            "ok", "Complete arches — numbering reliable",
            "Both jaws are close to complete, with no gaps or conflicts. On complete arches "
            f"numbering measured {pct(evidence.deployed_summary()['complete'], 0)} FDI accuracy.",
            [], confidence)

    if partial:
        title = "Partial " + " and ".join(partial) + (" jaws" if len(partial) == 2 else " jaw")
    elif skipped:
        title = "Gap in the arch"
    else:
        title = "Numbering conflict"
    return Reliability(
        "caution", f"{title} — numbers may be shifted",
        "Numbers are assigned in order along each jaw, so one missing tooth can shift every "
        "number after it. Review the highlighted jaw before relying on the chart.",
        reasons, confidence)


def chart_from_detections(raw_teeth, raw_anomalies, image_shape, mode="Full chart",
                          findings_threshold=DEFAULT_FINDINGS_THRESHOLD, timings=None,
                          source="model"):
    """Geometric half of the pipeline: NMS -> jaws -> FDI -> mapping -> validation."""
    timings = dict(timings or {})
    t0 = time.perf_counter()
    if MODES[mode]["key"] == "findings":
        raw_anomalies = [a for a in raw_anomalies if a["class_name"] != "Tooth"]
        teeth_nms, all_teeth = [], []
    else:
        teeth_nms = DentalGeometricEngine.apply_nms(raw_teeth, iou_threshold=0.4)
        upper_jaw, lower_jaw = DentalGeometricEngine.split_jaws(teeth_nms)
        all_teeth = (DentalGeometricEngine.sort_and_assign_fdi(upper_jaw, "upper")
                     + DentalGeometricEngine.sort_and_assign_fdi(lower_jaw, "lower"))
    timings["Geometric numbering"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    all_teeth = DentalGeometricEngine.map_anomalies(all_teeth, raw_anomalies)
    # map_anomalies shares each anomaly's box list with the finding it creates, which
    # identifies the pair exactly even after renumbering.
    owner = {id(f["box"]): t for t in all_teeth for f in t["findings"]}
    all_teeth = clean_clinical_findings(all_teeth)
    all_teeth, raw_anomalies = validate_anatomy(all_teeth, raw_anomalies,
                                                anomaly_threshold=findings_threshold)
    timings["Anatomical validation"] = time.perf_counter() - t0

    # validate_anatomy can drop teeth after mapping; their findings become unassigned.
    kept = {id(t) for t in all_teeth}
    unassigned = [a for a in raw_anomalies if id(owner.get(id(a["box"]))) not in kept]
    for t in all_teeth:
        for f in t["findings"]:
            f["fdi"] = t["fdi"]
    all_teeth.sort(key=lambda t: (t["quadrant"], t["fdi"]))

    return AnalysisResult(
        mode=mode, teeth=all_teeth, findings=raw_anomalies, unassigned=unassigned,
        geometry=_geometry(teeth_nms, all_teeth, image_shape),
        reliability=assess_reliability(all_teeth, mode), timings=timings, source=source)


def analyze(image_bgr, mode, predictors, teeth_threshold=DEFAULT_TEETH_THRESHOLD,
            findings_threshold=DEFAULT_FINDINGS_THRESHOLD, on_step=None):
    """Full pipeline on one BGR image.

    predictors: {"primary": DefaultPredictor, "pathology": DefaultPredictor or None}.
    on_step: optional callback(label) so the UI can show progress.
    """
    step = on_step or (lambda label: None)
    cfg = MODES[mode]
    width = image_bgr.shape[1]
    timings = {}

    step("Detecting teeth" if cfg["key"] == "full" else "Detecting findings")
    t0 = time.perf_counter()
    outputs = run_inference(predictors["primary"], image_bgr)
    raw_teeth, raw_anomalies = process_predictions(
        outputs, image_width=width, teeth_threshold=teeth_threshold,
        anomaly_threshold=findings_threshold, class_map=cfg["class_map"])
    timings["Detection"] = time.perf_counter() - t0

    if cfg["key"] == "full" and predictors.get("pathology") is not None:
        step("Detecting findings")
        t0 = time.perf_counter()
        p_outputs = run_inference(predictors["pathology"], image_bgr)
        _, p_anomalies = process_predictions(
            p_outputs, image_width=width, teeth_threshold=0.50,
            anomaly_threshold=findings_threshold, class_map=PATHOLOGY_CLASS_MAP)
        p_anomalies = [a for a in p_anomalies if a["class_name"] != "Tooth"]
        # Merge: keep the higher-confidence one of any overlapping pair (IoU > 0.3).
        for pa in p_anomalies:
            for existing in raw_anomalies:
                if compute_iou(pa["box"], existing["box"]) > 0.3:
                    if pa["score"] > existing["score"]:
                        existing["score"] = pa["score"]
                        existing["class_name"] = pa["class_name"]
                        existing["mask"] = pa.get("mask")
                    break
            else:
                raw_anomalies.append(pa)
        timings["Findings model"] = time.perf_counter() - t0

    step("Numbering and validating")
    return chart_from_detections(raw_teeth, raw_anomalies, image_bgr.shape, mode,
                                 findings_threshold, timings)
