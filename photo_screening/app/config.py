"""Environment-driven settings (prefix ``PHOTO_``; also read from ``photo_screening/.env``)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PKG_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_DIR.parent

# Unified finding vocabulary (``tooth`` is used for localisation and the mouth check only).
FINDING_TYPES = ("caries", "cavity", "crack", "wear", "filling", "crown")
ALL_TYPES = FINDING_TYPES + ("tooth",)

# Report thresholds per unified type when no calibrated thresholds.json is available.
# t_low: keep (sensitivity-first screening); t_med / t_high: confidence bands.
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "caries": {"t_low": 0.15, "t_med": 0.35, "t_high": 0.60},
    "cavity": {"t_low": 0.25, "t_med": 0.45, "t_high": 0.65},
    "crack": {"t_low": 0.30, "t_med": 0.50, "t_high": 0.70},
    # Wear (AlphaDent "abrasion", DSLR-trained) fires on ordinary incisal edges in phone photos; from 0.55
    # it cut marks per photo from 4.0 to 3.2 on the AKU test split with no change to triage. Not validated.
    "wear": {"t_low": 0.55, "t_med": 0.65, "t_high": 0.80},
    # Restorations never raise triage and cost at most 5 index points, so they are kept from a lower
    # score (AlphaDent scores amalgam fillings in phone photos around 0.2-0.4). Uncalibrated default.
    "filling": {"t_low": 0.15, "t_med": 0.40, "t_high": 0.70},
    "crown": {"t_low": 0.15, "t_med": 0.40, "t_high": 0.70},
    "tooth": {"t_low": 0.25, "t_med": 0.45, "t_high": 0.65},
}

# Calibrated keep thresholds are never allowed below these. When a weak detector cannot reach the
# calibration recall target, the notebook falls back to a very low score (0.05); without a floor the
# report would be flooded with false findings (DentalAI cavity precision is 0.15 at 0.05).
MIN_KEEP = {"caries": 0.15, "cavity": 0.25, "crack": 0.30, "wear": 0.15, "filling": 0.15, "crown": 0.15,
            "tooth": 0.15}

ALPHADENT_FILES = {
    "4c960": ("yolov8x_AlphaDent_4_classes_960px.pt", 960),
    "9c640": ("yolov8x_AlphaDent_9_classes_640px.pt", 640),
    "9c960": ("yolov8x_AlphaDent_9_classes_960px.pt", 960),
}


class Settings(BaseSettings):
    """All runtime configuration. Override with ``PHOTO_<NAME>`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="PHOTO_", env_file=str(PKG_DIR / ".env"), extra="ignore")

    weights_dir: Path = PKG_DIR / "weights"
    samples_dir: Path = PKG_DIR / "samples"
    device: str = Field("auto", description="auto | mps | cpu | cuda | cuda:0")
    alphadent_variant: str = Field("4c960", description="4c960 | 9c640 | 9c960")
    alphadent_imgsz: int | None = Field(None, description="defaults to the variant's training size")
    phone_model_imgsz: int = 640
    tta: bool = Field(False, description="horizontal-flip test-time augmentation (doubles CPU time)")
    alphadent_caries_standalone: bool = Field(
        False, description="report caries seen only by AlphaDent when the phone-trained caries model is loaded; "
                           "off because AlphaDent caries measured AP50 0.006 on phone photos (EVAL.md)")
    report_cavity_crack: bool = Field(
        False, description="report DentalAI cavity/crack marks; off because they measured precision ~0.34 "
                           "and recall ~0.25 even on DentalAI's own test split (EVAL.md)")
    max_side: int = 1280
    max_upload_mb: int = 15
    fusion_iou: float = 0.55                      # same IoU as the OPG pipeline's box fusion
    single_pass_penalty: float = 0.6              # TTA: detections seen in one pass only (as in the OPG pipeline)
    thresholds_json: Path | None = Field(None, description="optional override file for per-type thresholds")

    # Quality gate. Blur, exposure and resolution are hard checks; mouth plausibility is a soft
    # warning unless the DentalAI tooth model is loaded.
    min_side_px: int = 400                        # smallest photo in the AKU test split: 408 px (EVAL.md)
    blur_min_var: float = 20.0
    exposure_min_mean: float = 45.0
    exposure_max_mean: float = 225.0
    exposure_max_clipped: float = 0.25

    # Optional VLM (OpenAI-compatible endpoint). Off by default; the template is always the fallback.
    vlm_enabled: bool = False
    vlm_base_url: str = "http://localhost:11434/v1"
    vlm_model: str = "qwen2.5vl:3b"
    vlm_api_key: str = ""
    vlm_timeout_s: float = 90.0
    vlm_max_images: int = Field(1, description="photos sent to the VLM (front first); local 3B models need ~40 s each")
    vlm_image_side: int = 448
    vlm_provider_label: str = "Ollama (local)"

    app_url: str = "http://localhost:8501"
    opg_page_path: str = ""          # Analyze is the default page, served at the app root

    def alphadent_file(self) -> tuple[Path, int]:
        """Path and inference size for the configured AlphaDent variant."""
        name, size = ALPHADENT_FILES.get(self.alphadent_variant, ALPHADENT_FILES["4c960"])
        return self.weights_dir / "alphadent" / name, self.alphadent_imgsz or size

    def resolved_device(self) -> str:
        """Pick ``mps`` when available, then CUDA, else CPU (unless ``device`` is set explicitly)."""
        if self.device != "auto":
            return self.device
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_thresholds(settings: Settings, loaded_models: list[str]) -> dict[str, dict[str, float]]:
    """Defaults, then calibrated thresholds.json from each loaded phone model, then an override file.

    Phone-trained models are applied in ascending priority so the preferred source for a type wins.
    """
    out = {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    contributes = {"teeth_dentalai": ("tooth", "cavity", "crack"), "caries_aku": ("caries",)}
    for model_id, types in contributes.items():
        path = settings.weights_dir / model_id / "thresholds.json"
        if model_id in loaded_models and path.exists():
            for name, t in json.loads(path.read_text()).get("classes", {}).items():
                if name in types:
                    out[name].update({k: float(v) for k, v in t.items() if k in ("t_low", "t_med", "t_high")})
    if settings.thresholds_json and Path(settings.thresholds_json).exists():
        for name, t in json.loads(Path(settings.thresholds_json).read_text()).items():
            out.setdefault(name, {}).update(t)
    for name, t in out.items():                    # floor, then keep bands ordered whatever the sources say
        t["t_low"] = max(t["t_low"], MIN_KEEP.get(name, 0.0))
        if t["t_med"] <= t["t_low"] and name in DEFAULT_THRESHOLDS:
            # calibration couldn't separate the bands (weak class): use the default Medium threshold
            t["t_med"] = max(DEFAULT_THRESHOLDS[name]["t_med"], t["t_low"])
        t["t_med"] = max(t["t_med"], t["t_low"])
        t["t_high"] = max(t["t_high"], t["t_med"])
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached)."""
    return Settings()
