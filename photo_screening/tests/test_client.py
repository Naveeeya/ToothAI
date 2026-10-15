"""The UI client strips every metadata block and the user's filename before photos leave the app."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from photo_screening.tests.conftest import mouth_rgb
from photo_screening.ui import client


def _jpeg_with_metadata(orientation: int = 6) -> bytes:
    im = Image.fromarray(mouth_rgb(320, 200))
    exif = Image.Exif()
    exif[0x0112] = orientation                          # Orientation: rotate 90 when displayed
    exif[0x010F] = "PhoneMaker"                         # Make
    exif[0x8825] = {1: "N", 2: (18.0, 31.0, 12.0)}      # GPS IFD
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif, comment=b"patient: someone")
    return buf.getvalue()


def test_clean_photo_drops_exif_gps_and_comments_and_applies_orientation():
    raw = _jpeg_with_metadata()
    assert b"PhoneMaker" in raw and b"patient" in raw
    out = client.clean_photo(raw)
    assert b"PhoneMaker" not in out and b"patient" not in out and b"Exif" not in out
    with Image.open(io.BytesIO(out)) as im:
        assert not im.getexif() and "comment" not in im.info and "icc_profile" not in im.info
        assert im.size == (200, 320)                      # rotated upright, as the phone displays it


def test_analyze_sends_neutral_filenames_and_clean_bytes(monkeypatch):
    sent = {}

    class Resp:
        status_code, headers = 200, {}

        @staticmethod
        def json():
            return {"ok": True}

    def fake_post(url, files=None, data=None, timeout=None):
        sent.update(files)
        return Resp()

    monkeypatch.setattr(client.requests, "post", fake_post)
    client.analyze({"front": ("Rahul Sharma 2026-10-15.jpg", _jpeg_with_metadata(), "image/jpeg")})
    name, data, mime = sent["front"]
    assert name == "front.jpg" and mime == "image/jpeg" and b"PhoneMaker" not in data


def test_unreadable_upload_is_a_friendly_error():
    with pytest.raises(client.PhotoApiError, match="readable"):
        client.clean_photo(b"not an image")
