"""Decode uploads in memory: JPG/PNG/HEIC, EXIF orientation, resize to the analysis size."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:  # HEIC support is optional at import time so tests run without libheif
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:  # pragma: no cover
    HEIC_SUPPORTED = False


class ImageDecodeError(ValueError):
    """The upload is not a readable image."""


def decode(data: bytes, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (RGB uint8 array resized so the longest side <= max_side, original (w, h)).

    Nothing is written to disk; EXIF orientation is applied so phone photos are upright.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ImageDecodeError("not a readable JPG, PNG or HEIC image") from e
    original = img.size
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return np.asarray(img), original


def grey_world(rgb: np.ndarray) -> np.ndarray:
    """Grey-world white balance: removes the warm cast of indoor lighting (training photos are flash-lit)."""
    f = rgb.astype(np.float32)
    mean = f.reshape(-1, 3).mean(0)
    return np.clip(f * (mean.mean() / np.maximum(mean, 1.0)), 0, 255).astype(np.uint8)


def clahe_l(rgb: np.ndarray) -> np.ndarray:
    """Local contrast (CLAHE on the L channel): lifts dim back teeth without shifting colour."""
    import cv2

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


COLOUR_PASSES = {"grayworld": grey_world, "clahe": clahe_l}


def png_b64(rgb: np.ndarray) -> str:
    """Base64 PNG for JSON transport."""
    import base64

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
