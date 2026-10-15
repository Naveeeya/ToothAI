"""VLM adapter: contract filtering and graceful fallback (httpx MockTransport, no network)."""

import json
from datetime import UTC, datetime

import httpx
import pytest

from photo_screening.app.schemas import (
    Coverage,
    Finding,
    ImageQuality,
    ImageResult,
    ScreeningReport,
    Triage,
    VlmStatus,
)
from photo_screening.app.vlm import Vlm, build_vlm, is_third_party
from photo_screening.tests.conftest import mouth_rgb


def _report(types=("caries",), index=92) -> ScreeningReport:
    findings = [Finding(id=i + 1, view="front", region="Upper jaw, front", type=t, confidence=0.5, band="Medium",
                        sources=["caries_aku"], bbox=[0, 0, 10, 10]) for i, t in enumerate(types)]
    return ScreeningReport(
        report_id="TEST", created_at=datetime.now(UTC), app_version="0", model_versions={},
        images=[ImageResult(view="front", analysed=True, quality=ImageQuality(passed=True, checks=[]),
                            findings=findings)],
        screening_index=index, index_band="Low concern",
        triage=Triage(level="SOON", reasons=["x"], recommend_opg=True),
        coverage=Coverage(analysed_views=["front"], skipped_views=[], failed_views=[], notes=[]),
        summary_text=f"Template summary. Screening index {index}/100.", summary_source="template",
        vlm=VlmStatus(enabled=True, used=False), disclaimer="d")


def _vlm(settings, replies):
    """replies: list of (status, body) returned in order; body dict is wrapped as a chat completion."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        status, body = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(body, Exception):
            raise body
        content = body if isinstance(body, str) else json.dumps(body)
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})

    settings.vlm_enabled = True
    return Vlm(settings, transport=httpx.MockTransport(handler)), calls


GOOD = {"observations": [{"type": "gum_colour", "view": "front", "text": "Gums near the front teeth look redder.",
                          "band": "High"}],
        "summary": "We looked at your front photo and marked one area of possible decay. Screening index 92/100."}


def test_valid_reply_adds_observations_and_rephrases(settings):
    vlm, calls = _vlm(settings, [(200, GOOD)])
    out = vlm.enrich(_report(), [("front", mouth_rgb())])
    assert out.vlm.used and out.summary_source == "vlm"
    assert out.observations[0].band == "Medium"                       # High is capped for advisory notes
    assert out.observations[0].source == "vlm:Ollama (local)"
    sent = calls[0]
    assert sent["response_format"] == {"type": "json_object"} and sent["model"] == settings.vlm_model
    assert any(p.get("type") == "image_url" for p in sent["messages"][1]["content"])


def test_invalid_json_is_retried_once(settings):
    vlm, calls = _vlm(settings, [(200, "not json"), (200, GOOD)])
    out = vlm.enrich(_report(), [("front", mouth_rgb())])
    assert len(calls) == 2 and out.vlm.used


def test_invalid_twice_falls_back_to_template(settings):
    vlm, calls = _vlm(settings, [(200, "{bad"), (200, {"observations": "nope"})])
    rep = _report()
    out = vlm.enrich(rep, [("front", mouth_rgb())])
    assert len(calls) == 2
    assert out.summary_source == "template" and out.summary_text == rep.summary_text
    assert not out.vlm.used and out.vlm.error and out.observations == []


@pytest.mark.parametrize("exc", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_unreachable_server_does_not_block(settings, exc):
    vlm, calls = _vlm(settings, [(200, exc)])
    out = vlm.enrich(_report(), [("front", mouth_rgb())])
    assert len(calls) == 1 and out.summary_source == "template" and not out.vlm.used


def test_http_error_falls_back(settings):
    vlm, _ = _vlm(settings, [(500, GOOD)])
    assert vlm.enrich(_report(), [("front", mouth_rgb())]).summary_source == "template"


def test_observations_about_tooth_findings_or_diagnoses_are_dropped(settings):
    reply = {"observations": [
        {"type": "other", "view": "front", "text": "There is a cavity on the molar.", "band": "Low"},
        {"type": "gum_swelling", "view": "front", "text": "This looks like gingivitis.", "band": "Low"},
        {"type": "tartar", "view": "upper", "text": "Yellow deposits.", "band": "Low"},  # view not sent
        {"type": "staining", "view": "front", "text": "Brown staining between the lower front teeth.",
         "band": "Low"},
    ], "summary": GOOD["summary"]}
    vlm, _ = _vlm(settings, [(200, reply)])
    out = vlm.enrich(_report(), [("front", mouth_rgb())])
    assert [o.type for o in out.observations] == ["staining"]


@pytest.mark.parametrize("summary,ok", [
    (GOOD["summary"], True),
    ("You have gum disease. Screening index 92/100.", False),                  # diagnostic wording
    ("We marked possible decay and a cavity. Screening index 92/100.", False),  # adds a cavity
    ("We marked one area of possible decay.", False),                           # dropped the index
])
def test_summary_acceptance(summary, ok):
    assert Vlm.accept_summary(summary, _report()) is ok


def test_third_party_detection_and_disabled_mode(settings):
    assert not is_third_party("http://localhost:11434/v1")
    assert not is_third_party("http://ollama:11434/v1")
    assert is_third_party("https://generativelanguage.googleapis.com/v1beta/openai/")
    settings.vlm_enabled = False
    assert build_vlm(settings) is None


def test_echoed_or_unknown_types_are_inferred_and_health_judgements_dropped(settings):
    reply = {"observations": [
        {"type": "gum_colour|tartar|plaque|staining", "view": "front", "text": "Yellow tartar near the lower teeth."},
        {"type": "gums", "view": "front", "text": "Gums appear healthy with no visible plaque."},
        {"type": "misc", "view": "front", "text": "Slight brown staining on the front teeth."},
    ], "summary": GOOD["summary"]}
    vlm, _ = _vlm(settings, [(200, reply)])
    out = vlm.enrich(_report(), [("front", mouth_rgb()), ("lower", mouth_rgb(seed=3))])
    assert [o.type for o in out.observations] == ["tartar", "staining"]


def test_only_the_front_photo_is_sent_by_default(settings):
    vlm, calls = _vlm(settings, [(200, GOOD)])
    vlm.enrich(_report(), [("lower", mouth_rgb()), ("front", mouth_rgb(seed=4))])
    parts = calls[0]["messages"][1]["content"]
    assert [p["text"] for p in parts if p["type"] == "text"][1:] == ["Photo: front view"]
    assert sum(p["type"] == "image_url" for p in parts) == 1
