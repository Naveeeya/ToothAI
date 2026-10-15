"""Run the photo pipeline and assemble the ScreeningReport."""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import photo_screening
from photo_screening.app import quality, regions, render, rules, selection
from photo_screening.app.config import Settings, load_thresholds
from photo_screening.app.fusion import fuse_passes, merge_max
from photo_screening.app.imaging import COLOUR_PASSES, decode, png_b64
from photo_screening.app.models.registry import Registry
from photo_screening.app.schemas import VIEWS, Finding, ImageResult, ScreeningReport, VlmStatus

log = logging.getLogger(__name__)
SEVERITY = {t: i for i, t in enumerate(render.ORDER)}
_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), undefined=StrictUndefined,
                   autoescape=False, trim_blocks=True)


@dataclass
class Upload:
    view: str
    data: bytes
    mirrored: bool | None = None


@dataclass
class AnalysedImage:
    """Pipeline output for one photo; ``rgb`` is kept in memory for the VLM step only."""

    result: ImageResult
    rgb: np.ndarray | None


def _simplify(poly: np.ndarray | None, max_points: int = 64) -> list[list[float]] | None:
    if poly is None or len(poly) < 3:
        return None
    step = max(1, len(poly) // max_points)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in poly[::step]]


def analyse_image(up: Upload, registry: Registry, settings: Settings, thresholds: dict, first_id: int) -> AnalysedImage:
    """Quality gate, detection, fusion, thresholds, regions and rendering for one photo."""
    rgb, original = decode(up.data, settings.max_side)
    h, w = rgb.shape[:2]
    q = quality.assess(rgb, original, settings)
    if not q.passed:
        return AnalysedImage(ImageResult(view=up.view, analysed=False, width=w, height=h, quality=q), None)

    per_model = {}
    for model_id, det in registry.detectors.items():
        fused_passes = fuse_passes(det.predict(rgb, tta=settings.tta), settings.fusion_iou,
                                   settings.single_pass_penalty)
        if model_id == "caries_aku" and settings.colour_passes:
            extra = [det.predict(np.ascontiguousarray(COLOUR_PASSES[name](rgb)))[0]
                     for name in settings.colour_passes]
            fused_passes = merge_max([fused_passes, *extra], settings.fusion_iou)
        per_model[model_id] = fused_passes
    tooth_t = thresholds["tooth"]["t_low"]
    teeth = [d.box.tolist() for d in per_model.get("teeth_dentalai", []) if d.type == "tooth" and d.score >= tooth_t]
    teeth_count = len(teeth) if registry.tooth_model_loaded else None
    if teeth_count is not None:
        q = quality.assess(rgb, original, settings, teeth_detected=teeth_count)
        if not q.passed:
            return AnalysedImage(ImageResult(view=up.view, analysed=False, width=w, height=h, quality=q,
                                             teeth_detected=teeth_count), None)

    fused = selection.select(per_model, thresholds, settings, (w, h), "caries_aku" in registry.loaded)
    fused.sort(key=lambda d: (SEVERITY[d.type], -d.score))
    findings = []
    for i, d in enumerate(fused):
        region, hint = regions.describe(up.view, d.box.tolist(), (w, h), teeth, up.mirrored)
        findings.append(Finding(
            id=first_id + i, view=up.view, region=region, tooth_hint=hint, type=d.type,
            confidence=round(d.score, 3),
            band=rules.band(d.type, d.score, d.sources, thresholds, registry.phone_trained),
            sources=d.sources, bbox=[round(float(v), 1) for v in d.box], mask_polygon=_simplify(d.polygon)))
    annotated = render.annotate(rgb, findings)
    return AnalysedImage(ImageResult(view=up.view, analysed=True, width=w, height=h, quality=q, findings=findings,
                                     teeth_detected=teeth_count, annotated_png_b64=png_b64(annotated)), rgb)


def template_summary(images: Sequence[ImageResult], triage: str, index: int | None, index_band: str) -> str:
    """Deterministic plain-language summary (the default and the VLM fallback)."""
    analysed = [i for i in images if i.analysed]
    counts = Counter(f.type for i in analysed for f in i.findings)
    words = [f"{n} {rules.COUNT_PHRASES[t][n > 1]}" for t, n in sorted(counts.items(), key=lambda kv: SEVERITY[kv[0]])]
    return _env.get_template("summary.j2").render(
        n_analysed=len(analysed), n_supplied=len(images), analysed_views=[i.view for i in analysed],
        counts=words, triage=triage, index=index, index_band=index_band).strip()


def build_report(uploads: Sequence[Upload], registry: Registry, settings: Settings, use_vlm: bool = False,
                 vlm=None) -> ScreeningReport:
    """Analyse every upload (front first) and assemble the report. Never stores image bytes."""
    t0 = time.perf_counter()
    thresholds = load_thresholds(settings, registry.loaded)
    order = {v: i for i, v in enumerate(VIEWS)}
    analysed: list[AnalysedImage] = []
    next_id = 1
    for up in sorted(uploads, key=lambda u: order[u.view]):
        a = analyse_image(up, registry, settings, thresholds, next_id)
        next_id += len(a.result.findings)
        analysed.append(a)
    images = [a.result for a in analysed]
    findings = [f for i in images for f in i.findings]
    any_analysed = any(i.analysed for i in images)

    triage = rules.triage(findings, any_analysed)
    index, index_band, _ = rules.screening_index(findings, any_analysed)
    cov = rules.coverage([i.view for i in images], [i.view for i in images if i.analysed], VIEWS)
    summary = template_summary(images, triage.level, index, index_band)

    report = ScreeningReport(
        report_id=uuid.uuid4().hex[:12].upper(), created_at=datetime.now(UTC),
        app_version=photo_screening.__version__, model_versions=registry.versions(), images=images,
        screening_index=index, index_band=index_band, triage=triage, coverage=cov, summary_text=summary,
        summary_source="template", disclaimer=photo_screening.DISCLAIMER,
        vlm=VlmStatus(enabled=settings.vlm_enabled, used=False))
    if use_vlm and vlm is not None and any_analysed:
        report = vlm.enrich(report, [(a.result.view, a.rgb) for a in analysed if a.rgb is not None])
    log.info("report built", extra={"report_id": report.report_id, "n_images": len(images),
                                    "n_findings": len(findings), "triage": triage.level,
                                    "seconds": round(time.perf_counter() - t0, 2)})
    return report
