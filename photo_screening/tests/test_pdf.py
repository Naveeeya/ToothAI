"""PDF report: builds from a real pipeline report, escapes every dynamic string, served by the API."""

import re

from fastapi.testclient import TestClient

from photo_screening.app import main
from photo_screening.app.pdf import build_pdf
from photo_screening.app.report import Upload, build_report
from photo_screening.app.schemas import Observation, ScreeningReport
from photo_screening.tests.conftest import FakeDetector, fake_registry, jpeg_bytes, mouth_rgb

DETS = {"alphadent": FakeDetector("alphadent", [("caries", 0.7, (100, 200, 140, 240)),
                                                ("filling", 0.6, (600, 380, 660, 420))])}


def _pages(pdf: bytes) -> int:
    return len(re.findall(rb"/Type /Page\b", pdf))


def _report(settings) -> ScreeningReport:
    ups = [Upload("front", jpeg_bytes(mouth_rgb())), Upload("lower", jpeg_bytes(mouth_rgb(seed=2)))]
    return build_report(ups, fake_registry(DETS), settings)


def test_pdf_builds_and_fits_two_pages(settings):
    pdf = build_pdf(_report(settings), settings)
    assert pdf.startswith(b"%PDF") and 1 <= _pages(pdf) <= 2


def test_markup_in_dynamic_text_is_escaped(settings):
    rep = _report(settings)
    hostile = "<img src='/etc/hosts'/> <font color='red'>x</font> & <b>unclosed"
    f = rep.images[0].findings[0].model_copy(update={"region": hostile, "tooth_hint": "<br/>"})
    images = [rep.images[0].model_copy(update={"findings": [f]})] + rep.images[1:]
    rep = rep.model_copy(update={
        "images": images, "summary_text": hostile, "report_id": "<ID&>",
        "observations": [Observation(type="other", text=hostile, band="Low", source="vlm:<x>")],
        "triage": rep.triage.model_copy(update={"reasons": [hostile]}),
    })
    pdf = build_pdf(rep, settings)                        # unescaped, ReportLab would raise or fetch the image
    assert pdf.startswith(b"%PDF")


def test_report_pdf_endpoint_accepts_report_json(settings):
    main.app.dependency_overrides[main.get_registry] = lambda: fake_registry(DETS)
    main.app.dependency_overrides[main.get_app_settings] = lambda: settings
    main.app.dependency_overrides[main.get_vlm] = lambda: None
    try:
        c = TestClient(main.app)
        rep = c.post("/v1/photo-screening/analyze",
                     files={"front": ("f.jpg", jpeg_bytes(mouth_rgb()), "image/jpeg")}).json()
        r = c.post("/v1/photo-screening/report.pdf", json=rep)
        assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
        assert r.content.startswith(b"%PDF") and rep["report_id"] in r.headers["content-disposition"]
        assert c.post("/v1/photo-screening/report.pdf", json={"nope": 1}).status_code == 422
    finally:
        main.app.dependency_overrides.clear()
