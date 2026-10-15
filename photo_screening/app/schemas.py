"""Pydantic models for the screening report (the API's response contract)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

View = Literal["front", "upper", "lower", "left", "right"]
VIEWS: tuple[str, ...] = ("front", "upper", "lower", "left", "right")
FindingType = Literal["caries", "cavity", "crack", "wear", "filling", "crown"]
Band = Literal["High", "Medium", "Low"]
TriageLevel = Literal["ROUTINE", "SOON", "PROMPT"]
ObservationType = Literal["gum_colour", "gum_swelling", "tartar", "staining", "plaque", "crowding", "other"]


class QualityCheck(BaseModel):
    name: Literal["resolution", "blur", "exposure", "mouth"]
    value: float
    threshold: str
    passed: bool
    hard: bool = Field(description="a failed hard check skips analysis of the image")
    tip: str = ""


class ImageQuality(BaseModel):
    passed: bool = Field(description="all hard checks passed")
    checks: list[QualityCheck]

    @property
    def warnings(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed and not c.hard]


class Finding(BaseModel):
    id: int = Field(description="number shown on the annotated image and in the table")
    view: View
    region: str
    tooth_hint: str | None = Field(None, description="approximate position, e.g. '2nd tooth from the midline'")
    type: FindingType
    confidence: float = Field(ge=0, le=1)
    band: Band
    sources: list[str]
    bbox: list[float] = Field(min_length=4, max_length=4, description="x1, y1, x2, y2 in analysed-image pixels")
    mask_polygon: list[list[float]] | None = None


class Observation(BaseModel):
    type: ObservationType
    text: str
    band: Band
    source: str = Field(description="vlm:<provider label>")


class ImageResult(BaseModel):
    view: View
    analysed: bool
    width: int = 0
    height: int = 0
    quality: ImageQuality
    findings: list[Finding] = []
    teeth_detected: int | None = Field(None, description="only when the tooth model is loaded")
    annotated_png_b64: str | None = None


class Triage(BaseModel):
    level: TriageLevel
    reasons: list[str]
    recommend_opg: bool


class Coverage(BaseModel):
    analysed_views: list[View]
    skipped_views: list[View] = Field(description="no photo supplied")
    failed_views: list[View] = Field(description="photo supplied but failed the quality gate")
    notes: list[str]


class VlmStatus(BaseModel):
    enabled: bool
    used: bool
    provider_label: str | None = None
    third_party: bool = False
    error: str | None = None


class ScreeningReport(BaseModel):
    report_id: str
    created_at: datetime
    app_version: str
    model_versions: dict[str, str]
    images: list[ImageResult]
    observations: list[Observation] = []
    screening_index: int | None = Field(None, ge=0, le=100, description="None when no image could be analysed")
    index_band: str
    triage: Triage
    coverage: Coverage
    summary_text: str
    summary_source: Literal["template", "vlm"]
    vlm: VlmStatus
    disclaimer: str


class ModelStatus(BaseModel):
    model_id: str
    status: Literal["loaded", "missing", "error"]
    required: bool
    file: str | None = None
    version: str | None = Field(None, description="SHA-256 prefix recorded by fetch_weights.py")
    licence: str
    attribution: str
    detail: str | None = None
    metrics: dict[str, float | str] | None = Field(None, description="measured test-split headline numbers")
