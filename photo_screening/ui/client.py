"""HTTP client for the Photo Screening API, plus the bundled sample photos."""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests

API_URL = os.environ.get("PHOTO_API_URL", "http://localhost:8001").rstrip("/")
PREFIX = "/v1/photo-screening"
SAMPLES_DIR = Path(os.environ.get("PHOTO_SAMPLES_DIR", Path(__file__).resolve().parents[1] / "samples"))
VIEW_KEYWORDS = (("maxillary", "upper"), ("mandibular", "lower"), ("frontal", "front"),
                 ("leftlateral", "left"), ("left_lateral", "left"), ("rightlateral", "right"),
                 ("right_lateral", "right"))


class PhotoApiError(RuntimeError):
    """A user-presentable error from the API (or the API being unreachable)."""


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
    """photos: view -> (filename, bytes, mime). Returns the ScreeningReport as a dict."""
    data = {"use_vlm": "true" if use_vlm else "false"}
    if mirrored is not None:
        data["mirrored"] = ",".join(mirrored)
    try:
        r = requests.post(f"{API_URL}{PREFIX}/analyze", files=photos, data=data, timeout=timeout)
    except requests.RequestException as e:
        raise PhotoApiError(f"The photo screening service isn't reachable at {API_URL}.") from e
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
