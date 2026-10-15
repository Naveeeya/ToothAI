"""Load every available photo model once at start-up; report what is loaded, missing or broken."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from photo_screening.app.config import Settings
from photo_screening.app.models.base import Detector, YoloDetector, alphadent_map, caries_aku_map, dentalai_map
from photo_screening.app.schemas import ModelStatus

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    required: bool
    phone_trained: bool
    licence: str
    attribution: str


SPECS: dict[str, ModelSpec] = {
    "alphadent": ModelSpec(
        "alphadent", True, False, "Apache-2.0",
        "AlphaDent pretrained YOLOv8x-seg (ZFTurbo et al.), github.com/ZFTurbo/AlphaDent. "
        "Trained on DSLR/macro intraoral photos."),
    "caries_aku": ModelSpec(
        "caries_aku", False, True, "Project-trained weights; training data CC BY 4.0",
        "Trained on the AKU annotated intraoral caries dataset (Zenodo, doi:10.5281/zenodo.14827784), "
        "CC BY 4.0. Smartphone photos."),
    "teeth_dentalai": ModelSpec(
        "teeth_dentalai", False, True, "Project-trained weights; training data CC BY 4.0",
        "Trained on DentalAI (tooth, caries, cavity, crack), CC BY 4.0."),
}

# Which model's score represents a fused finding of each type (first present wins).
TYPE_PRIORITY: dict[str, tuple[str, ...]] = {
    "caries": ("caries_aku", "alphadent"),
}


def _headline_metrics(model_id: str, model_dir: Path) -> dict | None:
    """Test-split headline numbers from the notebook's metrics.json, labelled with their source."""
    path = model_dir / "metrics.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text()).get("test", {})
    try:
        if model_id == "caries_aku":
            img = m["image_level"]["at_t_low"]
            return {"source": f"AKU Benchmarking test split, n={m['n_images']} images",
                    "map50_class_mean": round(m["ultralytics"]["map50"], 3),
                    "published_yolov8s_map50": m["published_baseline"]["map50"],
                    "ap50_caries_merged": round(m["merged_caries"]["ap50"], 3),
                    "image_sensitivity": round(img["sensitivity"], 3),
                    "image_specificity": round(img["specificity"], 3),
                    "image_level_threshold": img["conf"]}
        if model_id == "teeth_dentalai":
            pc = m["per_class"]
            out = {"source": "DentalAI test split, n=250 images"}
            out.update({f"box_ap50_{k}": round(v["box_ap50"], 3) for k, v in pc.items() if k != "caries"})
            return out
    except (KeyError, TypeError):
        return None
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Registry:
    """Holds loaded detectors plus a status row for every known model."""

    def __init__(self, detectors: dict[str, Detector] | None = None,
                 statuses: dict[str, ModelStatus] | None = None):
        self.detectors: dict[str, Detector] = detectors or {}
        self.statuses: dict[str, ModelStatus] = statuses or {}

    @classmethod
    def load(cls, settings: Settings) -> Registry:
        """Load whatever weights exist. Never raises for a missing or broken model."""
        reg = cls()
        manifest_path = settings.weights_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        device = settings.resolved_device()
        alphadent_path, alphadent_size = settings.alphadent_file()
        plan = {
            "alphadent": (alphadent_path, alphadent_size, alphadent_map),
            "caries_aku": (settings.weights_dir / "caries_aku" / "best.pt", settings.phone_model_imgsz, caries_aku_map),
            "teeth_dentalai": (settings.weights_dir / "teeth_dentalai" / "best.pt", settings.phone_model_imgsz,
                               dentalai_map),
        }
        for model_id, (path, imgsz, cmap) in plan.items():
            spec = SPECS[model_id]
            rel = path.relative_to(settings.weights_dir) if path.is_relative_to(settings.weights_dir) else path
            row = {"model_id": model_id, "required": spec.required, "licence": spec.licence,
                   "attribution": spec.attribution, "file": str(rel)}
            if not path.exists():
                hint = ("run scripts/fetch_weights.py" if model_id == "alphadent"
                        else "optional: import the notebook artifact zip with scripts/fetch_weights.py --import-zip")
                reg.statuses[model_id] = ModelStatus(status="missing", detail=hint, **row)
                log.info("model missing", extra={"model_id": model_id})
                continue
            try:
                entry = manifest.get(model_id, {})
                digest = entry.get("sha256") if entry.get("file") == row["file"] else None
                digest = digest or _sha256(path)
                det = YoloDetector(model_id, path, imgsz, device, cmap)
                reg.detectors[model_id] = det
                reg.statuses[model_id] = ModelStatus(
                    status="loaded", version=digest[:12], metrics=_headline_metrics(model_id, path.parent),
                    detail=f"device {device}, imgsz {imgsz}, types {', '.join(det.types)}", **row)
                log.info("model loaded", extra={"model_id": model_id, "device": device, "version": digest[:12]})
            except Exception as e:  # a corrupt or incompatible file must not take the service down
                reg.statuses[model_id] = ModelStatus(status="error", detail=f"{type(e).__name__}: {e}", **row)
                log.exception("model failed to load", extra={"model_id": model_id})
        return reg

    @property
    def loaded(self) -> list[str]:
        return list(self.detectors)

    @property
    def tooth_model_loaded(self) -> bool:
        return "tooth" in getattr(self.detectors.get("teeth_dentalai"), "types", ())

    def phone_trained(self, model_id: str) -> bool:
        spec = SPECS.get(model_id)
        return bool(spec and spec.phone_trained)

    def versions(self) -> dict[str, str]:
        """model_id -> SHA-256 prefix, for loaded models only."""
        return {k: s.version or "unknown" for k, s in self.statuses.items() if s.status == "loaded"}

    def warm_up(self) -> None:
        """Run each model once on a blank image so the first user request isn't slowed by kernel set-up."""
        import numpy as np

        blank = np.full((640, 960, 3), 200, np.uint8)
        for model_id, det in self.detectors.items():
            try:
                det.predict(blank)
            except Exception:
                log.exception("warm-up failed", extra={"model_id": model_id})

    def status_list(self) -> list[ModelStatus]:
        return [self.statuses[k] for k in SPECS if k in self.statuses]
