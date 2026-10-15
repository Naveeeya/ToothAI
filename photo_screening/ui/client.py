"""HTTP client for the Photo Screening API, plus the bundled sample photos."""

from __future__ import annotations

import io
import os
import re
from pathlib import Path

import requests
from PIL import Image, ImageOps

try:  # HEIC from iPhones; optional so the module imports without libheif
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

API_URL = os.environ.get("PHOTO_API_URL", "http://localhost:8001").rstrip("/")
PREFIX = "/v1/photo-screening"
SAMPLES_DIR = Path(os.environ.get("PHOTO_SAMPLES_DIR", Path(__file__).resolve().parents[1] / "samples"))
VIEW_KEYWORDS = (("maxillary", "upper"), ("mandibular", "lower"), ("frontal", "front"),
                 ("leftlateral", "left"), ("left_lateral", "left"), ("rightlateral", "right"),
                 ("right_lateral", "right"))


class PhotoApiError(RuntimeError):
    """A user-presentable error from the API (or the API being unreachable)."""


class PhotoApiBusy(PhotoApiError):
    """The service is analysing someone else's photos (one analysis at a time)."""


MAX_PIXELS = 60_000_000          # well above any phone camera; larger files are refused, not decoded


def clean_photo(data: bytes) -> bytes:
    """Re-encode an upload as a plain JPEG: orientation applied, every metadata block (EXIF with GPS and
    device details, XMP, ICC, comments) dropped. Raises PhotoApiError if it isn't a readable image."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise PhotoApiError("This photo is too large. Use the phone camera's normal resolution.")
            rgb = ImageOps.exif_transpose(im).convert("RGB")
    except PhotoApiError:
        raise
    except Exception as e:  # PIL raises several types for unreadable or hostile files
        raise PhotoApiError("One of the photos isn't a readable JPG, PNG or HEIC image.") from e
    clean = Image.new("RGB", rgb.size)              # a fresh image carries no info/metadata at all
    clean.paste(rgb)
    buf = io.BytesIO()
    clean.save(buf, "JPEG", quality=95)          # no exif=, icc_profile= or comment: nothing is copied
    return buf.getvalue()


def _detail(r: requests.Response) -> str:
    try:
        return str(r.json().get("detail", r.text))
    except ValueError:
        return r.text[:200]


def _get(path: str, timeout: float = 10) -> dict:
    try:
        r = requests.get(API_URL + path, timeout=timeout)
    except requests.RequestException as e:
        raise PhotoApiError(f"The photo screening service isn't reachable at {API_URL}.") from e
    if r.status_code != 200:
        raise PhotoApiError(_detail(r))
    return r.json()


def health() -> dict:
    return _get("/health", timeout=3)


def models() -> dict:
    return _get(f"{PREFIX}/models")


def analyze(photos: dict[str, tuple[str, bytes, str]], use_vlm: bool = False, mirrored: list[str] | None = None,
            timeout: float = 180) -> dict:
    """photos: view -> (filename, bytes, mime). Returns the ScreeningReport as a dict.

    Uploads are re-encoded without metadata and sent under neutral names (``front.jpg``): the user's
    filename and EXIF (location, device, time) never leave the app."""
    photos = {view: (f"{view}.jpg", clean_photo(data), "image/jpeg") for view, (_, data, _) in photos.items()}
    data = {"use_vlm": "true" if use_vlm else "false"}
    if mirrored is not None:
        data["mirrored"] = ",".join(mirrored)
    try:
        r = requests.post(f"{API_URL}{PREFIX}/analyze", files=photos, data=data, timeout=timeout)
    except requests.RequestException as e:
        raise PhotoApiError(f"The photo screening service isn't reachable at {API_URL}.") from e
    if r.status_code == 503 and r.headers.get("Retry-After"):
        raise PhotoApiBusy(_detail(r))
    if r.status_code != 200:
        raise PhotoApiError(_detail(r))
    return r.json()


def report_pdf(report: dict, timeout: float = 60) -> bytes:
    try:
        r = requests.post(f"{API_URL}{PREFIX}/report.pdf", json=report, timeout=timeout)
    except requests.RequestException as e:
        raise PhotoApiError("The PDF could not be generated: the service isn't reachable.") from e
    if r.status_code != 200:
        raise PhotoApiError(_detail(r))
    return r.content


def sample_photos() -> dict[str, tuple[str, bytes, str]]:
    """Bundled AKU sample photos mapped to view slots using ATTRIBUTION.md (one photo per view)."""
    attribution = SAMPLES_DIR / "ATTRIBUTION.md"
    if not attribution.exists():
        return {}
    out: dict[str, tuple[str, bytes, str]] = {}
    for line in attribution.read_text().splitlines():
        m = re.match(r"- (aku_sample_\d+): (.+)", line.strip())
        if not m:
            continue
        name, source = m.groups()
        key = source.lower().replace("-", "").replace(" ", "")
        view = next((v for k, v in VIEW_KEYWORDS if k.replace("_", "") in key.replace("_", "")), None)
        path = next(SAMPLES_DIR.glob(f"{name}.*"), None)
        if view and path and view not in out:
            out[view] = (path.name, path.read_bytes(), "image/jpeg")
    return out


def sample_attribution() -> str:
    path = SAMPLES_DIR / "ATTRIBUTION.md"
    return path.read_text().split("\n\n")[1] if path.exists() else ""
