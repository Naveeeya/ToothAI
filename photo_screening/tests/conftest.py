"""Shared fixtures: synthetic mouth photos and a fake model registry (no weights needed)."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from photo_screening.app.config import Settings
from photo_screening.app.models.base import RawDet
from photo_screening.app.models.registry import Registry
from photo_screening.app.schemas import ModelStatus


def mouth_rgb(w: int = 960, h: int = 640, seed: int = 0) -> np.ndarray:
    """Synthetic 'smile': pink gums/lips, a row of textured white teeth, dark mouth gap."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (205, 120, 125), np.uint8)                      # lips / gums
    cv2.ellipse(img, (w // 2, h // 2), (w * 2 // 5, h // 4), 0, 0, 360, (60, 25, 30), -1)
    for i in range(8):
        x = w // 2 - 4 * w // 20 + i * w // 20
        cv2.rectangle(img, (x, h // 2 - h // 6), (x + w // 22, h // 2 - 4), (236, 232, 222), -1)
        cv2.rectangle(img, (x, h // 2 + 4), (x + w // 22, h // 2 + h // 7), (230, 226, 214), -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def jpeg_bytes(rgb: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


class FakeDetector:
    """Returns fixed detections (in analysed-image pixels) regardless of the input."""

    def __init__(self, model_id: str, dets: list[tuple[str, float, tuple[float, float, float, float]]]):
        self.model_id = model_id
        self.dets = dets
        self.types = tuple(sorted({t for t, _, _ in dets}))

    def predict(self, rgb: np.ndarray, tta: bool = False) -> list[list[RawDet]]:
        one = [RawDet(self.model_id, t, s, np.array(b, np.float32)) for t, s, b in self.dets]
        return [one, [RawDet(d.model_id, d.type, d.score, d.box, None, 1) for d in one]] if tta else [one]


def fake_registry(detectors: dict[str, FakeDetector]) -> Registry:
    statuses = {}
    for model_id in ("alphadent", "caries_aku", "teeth_dentalai"):
        loaded = model_id in detectors
        statuses[model_id] = ModelStatus(model_id=model_id, status="loaded" if loaded else "missing",
                                         required=model_id == "alphadent", version="abc123def456" if loaded else None,
                                         licence="test", attribution="test")
    return Registry(detectors=dict(detectors), statuses=statuses)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(weights_dir=tmp_path / "weights", samples_dir=tmp_path / "samples", vlm_enabled=False,
                    device="cpu", _env_file=None)


@pytest.fixture
def mouth() -> np.ndarray:
    return mouth_rgb()
