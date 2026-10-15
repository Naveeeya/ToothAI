"""Optional vision-language observations through any OpenAI-compatible endpoint.

Default: local Ollama (``qwen2.5vl:3b``) in JSON mode. Gemini works through Google's
OpenAI-compatible endpoint. The contract: the deterministic findings are passed in as JSON; the
model may only add soft-tissue/hygiene observations and rephrase the summary. Anything that adds,
removes or names a tooth finding, or uses diagnostic language, is dropped in code. Invalid output is
retried once, then the template summary stands. Failures never block the report.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Literal
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, model_validator

from photo_screening.app.config import Settings
from photo_screening.app.schemas import Observation, ScreeningReport, VlmStatus

log = logging.getLogger(__name__)

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "ollama", "host.docker.internal"}
OBS_TYPES = ("gum_colour", "gum_swelling", "tartar", "staining", "plaque", "crowding", "other")
# Tooth findings belong to the detection models only.
FINDING_WORDS = re.compile(r"\b(caries|carious|decay\w*|cavit\w*|fillings?|crowns?|restor\w*|amalgam|cracks?|"
                           r"fractur\w*|abrasion|wear)\b", re.I)
DIAGNOSTIC_WORDS = re.compile(r"\b(diagnos\w*|you have|you've got|disease|periodontitis|gingivitis|infect\w*|"
                              r"abscess\w*|cancer\w*|lesion\w*|tumou?r\w*|confirm\w*|definitely|certainly|"
                              r"healthy|unhealthy|normal|abnormal|inflam\w*)\b", re.I)

SYSTEM_PROMPT = """You assist a dental photo SCREENING tool. You are not a dentist and you never diagnose.
Tooth findings (decay, cavities, cracks, fillings, crowns, wear) are decided by other models and are
given to you as JSON. You must NOT add, remove, confirm, dispute or mention any tooth finding.

Your only tasks:
1. "observations": up to 5 short, plain-language notes about what is VISIBLE on gums and hygiene,
   each with "type" (exactly ONE of these words: gum_colour, gum_swelling, tartar, staining, plaque,
   crowding, other),
   "view" (the photo it refers to), "text" (max 25 words, describing only what you can actually see,
   with where in the mouth) and "band" (Low or Medium). Only report something that is clearly visible in
   THIS photo; if nothing is clear, return an empty list. Do not copy wording from these instructions.
2. "summary": rephrase the given summary in warm, plain English (max 90 words). Keep every number and
   recommendation exactly as given. Do not add new facts.

Describe what is visible; never judge health. Never use the words healthy, normal, inflamed, diagnose,
disease, gingivitis, periodontitis, infection, lesion or "you have".
Reply with one JSON object only: {"observations": [...], "summary": "..."}"""


def _word_type(word: str) -> str:
    """Map a finding word in free text to the unified type it refers to."""
    w = word.lower()
    for prefix, t in (("cari", "caries"), ("decay", "caries"), ("cavit", "cavity"), ("fill", "filling"),
                      ("restor", "filling"), ("amalgam", "filling"), ("crown", "crown"), ("crack", "crack"),
                      ("fractur", "crack"), ("abras", "wear"), ("wear", "wear")):
        if w.startswith(prefix):
            return t
    return w


_TYPE_HINTS = (("swell", "gum_swelling"), ("puff", "gum_swelling"), ("gum", "gum_colour"), ("tartar", "tartar"),
               ("calculus", "tartar"), ("plaque", "plaque"), ("stain", "staining"), ("crowd", "crowding"))


class _Obs(BaseModel):
    type: Literal["gum_colour", "gum_swelling", "tartar", "staining", "plaque", "crowding", "other"]

    @model_validator(mode="before")
    @classmethod
    def _normalise_type(cls, data):
        """Small local models sometimes echo the option list ('gum_colour|tartar') or invent a type:
        infer it from the type and text instead of discarding the whole reply."""
        if isinstance(data, dict) and data.get("type") not in OBS_TYPES:
            text, typ = str(data.get("text", "")).lower(), str(data.get("type", "")).lower()
            guess = next((t for k, t in _TYPE_HINTS if k in text), None)        # the text is the better signal
            if guess is None and "|" not in typ:
                guess = next((t for k, t in _TYPE_HINTS if k in typ), None)
            data = {**data, "type": guess or "other"}
        return data
    view: str = ""
    text: str = Field(min_length=3, max_length=300)
    band: Literal["Low", "Medium", "High"] = "Low"


class _VlmOut(BaseModel):
    observations: list[_Obs] = []
    summary: str = Field("", max_length=1200)


class VlmError(RuntimeError):
    pass


def is_third_party(base_url: str) -> bool:
    """True when images would leave this machine (anything but a local/compose host)."""
    return (urlparse(base_url).hostname or "").lower() not in LOCAL_HOSTS


def _jpeg_uri(rgb: np.ndarray, max_side: int = 448) -> str:
    img = Image.fromarray(rgb)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Vlm:
    """OpenAI-compatible chat-completions client with the screening prompt contract."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        headers = {"Authorization": f"Bearer {settings.vlm_api_key}"} if settings.vlm_api_key else {}
        self.client = httpx.Client(base_url=settings.vlm_base_url.rstrip("/") + "/", headers=headers,
                                   timeout=settings.vlm_timeout_s, transport=transport)

    def status(self, used: bool = False, error: str | None = None) -> VlmStatus:
        return VlmStatus(enabled=True, used=used, provider_label=self.settings.vlm_provider_label,
                         third_party=is_third_party(self.settings.vlm_base_url), error=error)

    def _call(self, messages: list[dict]) -> _VlmOut:
        r = self.client.post("chat/completions", json={
            "model": self.settings.vlm_model, "messages": messages, "temperature": 0,
            "response_format": {"type": "json_object"}})
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        try:
            return _VlmOut.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            raise VlmError(f"invalid JSON from model: {type(e).__name__}") from e

    def _messages(self, report: ScreeningReport, images: list[tuple[str, np.ndarray]]) -> list[dict]:
        facts = {
            "findings": [{"id": f.id, "view": f.view, "type": f.type, "region": f.region, "band": f.band}
                         for i in report.images for f in i.findings],
            "triage": report.triage.level, "screening_index": report.screening_index,
            "summary_to_rephrase": report.summary_text,
        }
        parts: list[dict] = [{"type": "text", "text": "Deterministic results (do not change):\n" + json.dumps(facts)}]
        ordered = sorted(images, key=lambda vi: vi[0] != "front")[: max(1, self.settings.vlm_max_images)]
        for view, rgb in ordered:
            parts.append({"type": "text", "text": f"Photo: {view} view"})
            parts.append({"type": "image_url", "image_url": {"url": _jpeg_uri(rgb, self.settings.vlm_image_side)}})
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": parts}]

    @staticmethod
    def accept_summary(text: str, report: ScreeningReport) -> bool:
        """Reject rephrasings that diagnose, mention a finding type the report doesn't contain,
        or drop the screening index."""
        if not text or DIAGNOSTIC_WORDS.search(text):
            return False
        present = {f.type for i in report.images for f in i.findings}
        mentioned = {_word_type(m.group(0)) for m in FINDING_WORDS.finditer(text)}
        if mentioned - present:
            return False
        return report.screening_index is None or str(report.screening_index) in text

    def enrich(self, report: ScreeningReport, images: list[tuple[str, np.ndarray]]) -> ScreeningReport:
        """Return a copy with observations and (maybe) a rephrased summary; never raises."""
        messages = self._messages(report, images)
        out, error = None, None
        for attempt in (1, 2):
            try:
                out = self._call(messages)
                break
            except (VlmError, httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                error = f"{type(e).__name__}: {e}"[:200]
                log.warning("vlm call failed", extra={"attempt": attempt, "error": error})
                if isinstance(e, httpx.TimeoutException | httpx.ConnectError):
                    break                                  # don't wait twice for an unreachable server
        if out is None:
            return report.model_copy(update={"vlm": self.status(used=False, error=error)})

        views = {v for v, _ in sorted(images, key=lambda vi: vi[0] != "front")[: max(1, self.settings.vlm_max_images)]}
        label = f"vlm:{self.settings.vlm_provider_label}"
        observations = [
            Observation(type=o.type, text=o.text.strip(), band="Medium" if o.band == "High" else o.band, source=label)
            for o in out.observations[:5]
            if not FINDING_WORDS.search(o.text) and not DIAGNOSTIC_WORDS.search(o.text)
            and (not o.view or o.view in views)
        ]
        update: dict = {"observations": observations, "vlm": self.status(used=True)}
        if self.accept_summary(out.summary.strip(), report):
            update.update(summary_text=out.summary.strip(), summary_source="vlm")
        return report.model_copy(update=update)


def build_vlm(settings: Settings) -> Vlm | None:
    """A client when PHOTO_VLM_ENABLED is set, else None (template mode, fully offline)."""
    return Vlm(settings) if settings.vlm_enabled else None
