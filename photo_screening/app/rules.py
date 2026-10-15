"""Deterministic screening rules: confidence bands, triage and the screening index.

Every finding in the report is decided here; the optional VLM may only rephrase the summary and add
soft-tissue observations. The formula is documented in MODEL_CARD.md and tested in tests/test_rules.py.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from photo_screening.app.schemas import Band, Coverage, Finding, Triage

URGENT_TYPES = ("cavity", "crack")
RESTORATION_TYPES = ("filling", "crown")

# Screening index: 100 minus deductions, clamped to 0..100.
DEDUCTIONS = {
    "cavity": 20,
    "crack": 20,
    "caries": {"High": 15, "Medium": 8, "Low": 3},
    "wear": 3,
    "filling": 1,
    "crown": 1,
}
CAPS = {"caries": 40, "wear": 3, "restorations": 5}            # total deduction per group
INDEX_BANDS = ((85, "Low concern"), (60, "Some concern"), (0, "Higher concern"))

LABELS = {"caries": "possible decay", "cavity": "possible cavity", "crack": "possible crack",
          "wear": "tooth wear", "filling": "filling", "crown": "crown"}
COUNT_PHRASES = {  # (one, many) for summaries
    "caries": ("area of possible decay", "areas of possible decay"),
    "cavity": ("possible cavity", "possible cavities"),
    "crack": ("possible crack", "possible cracks"),
    "wear": ("area of tooth wear", "areas of tooth wear"),
    "filling": ("filling", "fillings"),
    "crown": ("crown", "crowns"),
}


def band(type_: str, score: float, sources: Sequence[str], thresholds: dict[str, dict[str, float]],
         phone_trained: Callable[[str], bool]) -> Band:
    """High needs a high score AND either two agreeing models or a phone-trained model.

    Caries seen only by AlphaDent (DSLR-trained, weakest class) is capped at Low.
    """
    t = thresholds[type_]
    if type_ == "caries" and set(sources) == {"alphadent"}:
        return "Low"
    if score >= t["t_high"] and (len(set(sources)) >= 2 or any(phone_trained(s) for s in sources)):
        return "High"
    if score >= t["t_med"]:
        return "Medium"
    return "Low"


def _group(findings: Sequence[Finding]) -> list[str]:
    """One line per finding type: '3 areas of possible decay (#5, #6, #9), low confidence'."""
    lines = []
    rank = {"High": 0, "Medium": 1, "Low": 2}
    present = [t for t in ("cavity", "crack", "caries") if any(f.type == t for f in findings)]
    # strongest evidence first: the finding that decided the triage level leads the list
    present.sort(key=lambda t: min(rank[f.band] for f in findings if f.type == t))
    for t in present:
        fs = sorted((f for f in findings if f.type == t), key=lambda f: rank[f.band])
        ids = ", ".join(f"#{f.id}" for f in fs[:5]) + (f" and {len(fs) - 5} more" if len(fs) > 5 else "")
        bands = sorted({f.band for f in fs}, key=["High", "Medium", "Low"].index)
        lines.append(f"{len(fs)} {COUNT_PHRASES[t][len(fs) > 1]} ({ids}), {'/'.join(bands).lower()} confidence")
    return [line[0].upper() + line[1:] for line in lines]


def triage(findings: Sequence[Finding], analysed: bool) -> Triage:
    """PROMPT: cavity/crack at Medium+ or High caries. SOON: Medium caries, or 2+ Low decay/cavity/crack.

    Low-band cavity/crack does not trigger PROMPT on its own: the current cavity/crack detector is weak
    (DentalAI AP50 0.22/0.30), so a single low-score mark would otherwise send most users to a dentist
    "promptly" (decision recorded in MODEL_CARD.md)."""
    if not analysed:
        return Triage(level="ROUTINE", recommend_opg=False,
                      reasons=["No photo could be analysed. Retake the photos using the tips shown."])
    urgent = [f for f in findings if f.type in URGENT_TYPES and f.band != "Low"]
    caries = [f for f in findings if f.type == "caries"]
    high = [f for f in caries if f.band == "High"]
    medium = [f for f in caries if f.band == "Medium"]
    low = [f for f in findings if f.band == "Low" and (f.type == "caries" or f.type in URGENT_TYPES)]
    concerning = [f for f in findings if f.type == "caries" or f.type in URGENT_TYPES]
    if urgent or high:
        return Triage(level="PROMPT", recommend_opg=True, reasons=_group(concerning) + [
            "See a dentist soon and ask about a panoramic X-ray (OPG) to check what photos cannot show."])
    if medium or len(low) >= 2:
        return Triage(level="SOON", recommend_opg=True, reasons=_group(concerning) + [
            "Book a dental check-up; an OPG X-ray can confirm or rule out decay between teeth."])
    if low:
        reasons = _group(low)
    elif findings:
        reasons = ["Only restorations or wear were seen."]
    else:
        reasons = ["Nothing of concern was seen in these photos."]
    return Triage(level="ROUTINE", recommend_opg=False, reasons=reasons + ["Keep up regular dental check-ups."])


def screening_index(findings: Sequence[Finding], analysed: bool) -> tuple[int | None, str, dict[str, int]]:
    """Return (index, band label, deduction breakdown). ``None`` when nothing could be analysed."""
    if not analysed:
        return None, "Not available", {}
    caries = sum(DEDUCTIONS["caries"][f.band] for f in findings if f.type == "caries")
    breakdown = {
        "cavity_crack": sum(DEDUCTIONS[f.type] for f in findings if f.type in URGENT_TYPES),
        "caries": min(caries, CAPS["caries"]),
        "wear": min(sum(DEDUCTIONS["wear"] for f in findings if f.type == "wear"), CAPS["wear"]),
        "restorations": min(sum(DEDUCTIONS[f.type] for f in findings if f.type in RESTORATION_TYPES),
                            CAPS["restorations"]),
    }
    index = max(0, min(100, 100 - sum(breakdown.values())))
    label = next(name for floor, name in INDEX_BANDS if index >= floor)
    return index, label, breakdown


def coverage(supplied: Sequence[str], analysed: Sequence[str], all_views: Sequence[str]) -> Coverage:
    """Which views were analysed, failed or not supplied, plus what photos can never show."""
    failed = [v for v in supplied if v not in analysed]
    skipped = [v for v in all_views if v not in supplied]
    notes = []
    if skipped:
        notes.append(f"No photo for: {', '.join(skipped)}. Findings on those surfaces could be missed.")
    if failed:
        notes.append(f"Not analysed because of photo quality: {', '.join(failed)}.")
    notes.append("Photos cannot show decay between teeth, below the gum line, inside the tooth or in the bone. "
                 "A panoramic X-ray (OPG) covers these.")
    return Coverage(analysed_views=list(analysed), skipped_views=skipped, failed_views=failed, notes=notes)
