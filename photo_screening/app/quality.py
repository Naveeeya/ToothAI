"""Photo quality gate (OpenCV only, no ML).

Hard checks (a failure skips analysis of that image): resolution, blur, exposure.
Soft check: mouth plausibility from a colour heuristic. It becomes hard only when the DentalAI
tooth model is loaded and can confirm teeth directly.
"""

from __future__ import annotations

import cv2
import numpy as np

from photo_screening.app.config import Settings
from photo_screening.app.schemas import ImageQuality, QualityCheck

TIPS = {
    "resolution": "Use the phone's main camera at full resolution, not a screenshot or a thumbnail.",
    "blur": "Hold the phone steady and tap the screen on the teeth to focus. Clean the lens.",
    "exposure_dark": "Turn the phone flash on or move to brighter light.",
    "exposure_bright": "Move away from direct light or turn the flash off; avoid glare on the teeth.",
    "mouth": "Make sure the teeth fill most of the frame: pull your lips back and move the camera closer.",
    "mouth_model": "We couldn't find teeth in this photo. Pull your lips back and fill the frame with your teeth.",
}


BLUR_SIDE = 640  # measured at a fixed size: sharp phone photos ~120+, a 1.5 px blur < 10


def _laplacian_var(gray: np.ndarray) -> float:
    s = BLUR_SIDE / max(gray.shape)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mouth_colour_score(rgb: np.ndarray) -> tuple[float, float]:
    """Fractions of enamel-like (bright, low saturation) and gum/lip-like (red-pink) pixels."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    enamel = (s < 70) & (v > 130)
    pink = ((h <= 12) | (h >= 160)) & (s > 50) & (v > 60)
    return float(enamel.mean()), float(pink.mean())


def assess(rgb: np.ndarray, original_size: tuple[int, int], settings: Settings,
           teeth_detected: int | None = None) -> ImageQuality:
    """Run all checks. ``teeth_detected`` is the DentalAI tooth count when that model is loaded."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    checks: list[QualityCheck] = []

    min_side = min(original_size)
    checks.append(QualityCheck(name="resolution", value=min_side, threshold=f">= {settings.min_side_px} px",
                               passed=min_side >= settings.min_side_px, hard=True, tip=TIPS["resolution"]))

    blur = _laplacian_var(gray)
    checks.append(QualityCheck(name="blur", value=round(blur, 1),
                               threshold=f">= {settings.blur_min_var:g} (Laplacian variance at {BLUR_SIDE} px)",
                               passed=blur >= settings.blur_min_var, hard=True, tip=TIPS["blur"]))

    mean = float(gray.mean())
    clipped = float(((gray <= 5) | (gray >= 250)).mean())
    dark = mean < settings.exposure_min_mean
    bright = mean > settings.exposure_max_mean or clipped > settings.exposure_max_clipped
    checks.append(QualityCheck(
        name="exposure", value=round(mean, 1),
        threshold=(f"mean {settings.exposure_min_mean:g}-{settings.exposure_max_mean:g}, "
                   f"clipped <= {settings.exposure_max_clipped:.0%}"),
        passed=not (dark or bright), hard=True,
        tip=TIPS["exposure_dark"] if dark else TIPS["exposure_bright"] if bright else ""))

    if teeth_detected is not None:
        checks.append(QualityCheck(name="mouth", value=teeth_detected, threshold=">= 1 tooth detected",
                                   passed=teeth_detected >= 1, hard=True, tip=TIPS["mouth_model"]))
    else:
        enamel, pink = mouth_colour_score(rgb)
        checks.append(QualityCheck(name="mouth", value=round(enamel, 3),
                                   threshold="enamel >= 2% and gum/lip >= 3% of pixels",
                                   passed=enamel >= 0.02 and pink >= 0.03, hard=False, tip=TIPS["mouth"]))

    return ImageQuality(passed=all(c.passed for c in checks if c.hard), checks=checks)
