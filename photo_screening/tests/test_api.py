"""API tests with a stubbed registry: no weights, no network."""

import cv2
import pytest
from fastapi.testclient import TestClient

from photo_screening.app import main
from photo_screening.app.schemas import ScreeningReport
from photo_screening.tests.conftest import FakeDetector, fake_registry, jpeg_bytes, mouth_rgb

PREFIX = "/v1/photo-screening"


def _client(settings, detectors):
    main.app.dependency_overrides[main.get_registry] = lambda: fake_registry(detectors)
    main.app.dependency_overrides[main.get_app_settings] = lambda: settings
    main.app.dependency_overrides[main.get_vlm] = lambda: None
    return TestClient(main.app)          # not used as a context manager: lifespan (real model loading) is skipped


@pytest.fixture(autouse=True)
def _reset():
    yield
    main.app.dependency_overrides.clear()


ALPHADENT = {"alphadent": FakeDetector("alphadent", [("caries", 0.7, (100, 200, 140, 240)),
                                                     ("filling", 0.6, (600, 380, 660, 420)),
                                                     ("wear", 0.2, (300, 300, 320, 320))])}   # wear below t_low


def test_analyze_returns_a_valid_report(settings):
    c = _client(settings, ALPHADENT)
    r = c.post(f"{PREFIX}/analyze", files={"front": ("f.jpg", jpeg_bytes(mouth_rgb()), "image/jpeg")})
    assert r.status_code == 200, r.text
    rep = ScreeningReport.model_validate(r.json())
    front = rep.images[0]
    assert front.analysed and front.annotated_png_b64
    assert [f.type for f in front.findings] == ["caries", "filling"]          # severity order, wear dropped
    caries = front.findings[0]
    assert caries.band == "Low" and caries.sources == ["alphadent"]            # AlphaDent-only cap
    assert caries.region == "Upper jaw, your right side"
    assert rep.triage.level == "ROUTINE" and rep.screening_index == 96        # 100 - 3 (low caries) - 1
    assert rep.summary_source == "template" and "screening aid" in rep.summary_text
    assert rep.model_versions == {"alphadent": "abc123def456"}
    assert rep.coverage.skipped_views == ["upper", "lower", "left", "right"]
    assert "not for diagnostic use" in rep.disclaimer


def test_phone_model_and_agreement_raise_the_band(settings):
    detectors = dict(ALPHADENT, caries_aku=FakeDetector("caries_aku", [("caries", 0.9, (102, 202, 142, 242))]))
    r = _client(settings, detectors).post(f"{PREFIX}/analyze",
                                          files={"front": ("f.jpg", jpeg_bytes(mouth_rgb()), "image/jpeg")})
    rep = ScreeningReport.model_validate(r.json())
    caries = next(f for f in rep.images[0].findings if f.type == "caries")
    assert caries.sources == ["alphadent", "caries_aku"] and caries.band == "High"
    assert rep.triage.level == "PROMPT" and rep.triage.recommend_opg


def test_blurred_view_is_skipped_with_retake_tip(settings):
    blurry = cv2.GaussianBlur(mouth_rgb(), (0, 0), 8)
    files = {"front": ("f.jpg", jpeg_bytes(mouth_rgb()), "image/jpeg"),
             "upper": ("u.jpg", jpeg_bytes(blurry), "image/jpeg")}
    rep = ScreeningReport.model_validate(_client(settings, ALPHADENT).post(f"{PREFIX}/analyze", files=files).json())
    upper = next(i for i in rep.images if i.view == "upper")
    assert not upper.analysed and upper.annotated_png_b64 is None
    assert rep.coverage.failed_views == ["upper"]
    assert any(c.name == "blur" and not c.passed and c.tip for c in upper.quality.checks)
    assert [f.id for i in rep.images for f in i.findings] == [1, 2]          # numbering continues per report


def test_front_photo_is_required(settings):
    r = _client(settings, ALPHADENT).post(f"{PREFIX}/analyze",
                                          files={"upper": ("u.jpg", jpeg_bytes(mouth_rgb()), "image/jpeg")})
    assert r.status_code == 422 and "front" in r.json()["detail"]


def test_unreadable_image_is_a_400(settings):
    r = _client(settings, ALPHADENT).post(f"{PREFIX}/analyze",
                                          files={"front": ("f.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_oversized_upload_is_a_413(settings):
    settings.max_upload_mb = 0
    r = _client(settings, ALPHADENT).post(f"{PREFIX}/analyze",
                                          files={"front": ("f.jpg", jpeg_bytes(mouth_rgb()), "image/jpeg")})
    assert r.status_code == 413


def test_missing_primary_model_is_a_503(settings):
    r = _client(settings, {}).post(f"{PREFIX}/analyze",
                                   files={"front": ("f.jpg", jpeg_bytes(mouth_rgb()), "image/jpeg")})
    assert r.status_code == 503


def test_models_and_health_report_missing_optional_models(settings):
    c = _client(settings, ALPHADENT)
    models = {m["model_id"]: m for m in c.get(f"{PREFIX}/models").json()["models"]}
    assert models["alphadent"]["status"] == "loaded" and models["caries_aku"]["status"] == "missing"
    assert c.get("/health").json() == {"status": "ok", "models_loaded": ["alphadent"], "required_missing": []}
    assert _client(settings, {}).get("/health").json()["status"] == "degraded"


def test_cavity_and_crack_are_off_by_default(settings):
    detectors = dict(ALPHADENT, teeth_dentalai=FakeDetector("teeth_dentalai", [("cavity", 0.9, (300, 300, 340, 340)),
                                                                                ("crack", 0.9, (500, 300, 540, 340))]))
    files = {"front": ("f.jpg", jpeg_bytes(mouth_rgb()), "image/jpeg")}
    rep = ScreeningReport.model_validate(_client(settings, detectors).post(f"{PREFIX}/analyze", files=files).json())
    assert not any(f.type in ("cavity", "crack") for f in rep.images[0].findings)
    settings.report_cavity_crack = True
    rep = ScreeningReport.model_validate(_client(settings, detectors).post(f"{PREFIX}/analyze", files=files).json())
    assert {f.type for f in rep.images[0].findings} >= {"cavity", "crack"}


def test_alphadent_only_caries_is_dropped_when_the_phone_model_is_loaded(settings):
    detectors = {"alphadent": FakeDetector("alphadent", [("caries", 0.9, (100, 200, 140, 240)),
                                                         ("caries", 0.9, (600, 380, 640, 420))]),
                 "caries_aku": FakeDetector("caries_aku", [("caries", 0.8, (101, 201, 141, 241))])}
    files = {"front": ("f.jpg", jpeg_bytes(mouth_rgb()), "image/jpeg")}
    rep = ScreeningReport.model_validate(_client(settings, detectors).post(f"{PREFIX}/analyze", files=files).json())
    # the AlphaDent-only mark is gone; the agreed one stays
    assert [f.sources for f in rep.images[0].findings] == [["alphadent", "caries_aku"]]
