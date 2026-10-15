import pytest

from photo_screening.app.config import DEFAULT_THRESHOLDS
from photo_screening.app.rules import band, coverage, screening_index, triage
from photo_screening.app.schemas import VIEWS, Finding

PHONE = {"caries_aku", "teeth_dentalai"}


def F(type_, b="Low", i=1, view="front"):
    return Finding(id=i, view=view, region="Upper jaw, front", type=type_, confidence=0.5, band=b,
                   sources=["alphadent"], bbox=[0, 0, 10, 10])


# ---------------------------------------------------------------- bands
@pytest.mark.parametrize("type_,score,sources,expected", [
    ("caries", 0.95, ["alphadent"], "Low"),                      # AlphaDent-only caries is capped at Low
    ("caries", 0.95, ["caries_aku"], "High"),                    # phone-trained model can reach High alone
    ("caries", 0.95, ["alphadent", "caries_aku"], "High"),
    ("caries", 0.40, ["caries_aku"], "Medium"),
    ("caries", 0.20, ["caries_aku"], "Low"),
    ("wear", 0.90, ["alphadent"], "Medium"),                     # high score but one DSLR model: no High
    ("filling", 0.90, ["alphadent", "teeth_dentalai"], "High"),  # two agreeing models
    ("cavity", 0.70, ["teeth_dentalai"], "High"),
])
def test_band(type_, score, sources, expected):
    assert band(type_, score, sources, DEFAULT_THRESHOLDS, lambda m: m in PHONE) == expected


# ---------------------------------------------------------------- triage
@pytest.mark.parametrize("findings,level,opg", [
    ([], "ROUTINE", False),
    ([F("filling"), F("crown"), F("wear")], "ROUTINE", False),
    ([F("caries", "Low")], "ROUTINE", False),                    # one low caries alone
    ([F("caries", "Low"), F("caries", "Low", 2)], "SOON", True),
    ([F("caries", "Medium")], "SOON", True),
    ([F("caries", "High")], "PROMPT", True),
    ([F("cavity", "Medium")], "PROMPT", True),                   # cavity at Medium+
    ([F("crack", "High"), F("filling")], "PROMPT", True),        # crack at Medium+
    ([F("cavity", "Low")], "ROUTINE", False),                    # one low cavity alone: not PROMPT
    ([F("crack", "Low"), F("caries", "Low", 2)], "SOON", True),  # low crack counts toward SOON
])
def test_triage(findings, level, opg):
    t = triage(findings, analysed=True)
    assert (t.level, t.recommend_opg) == (level, opg)
    assert t.reasons


def test_triage_groups_reasons_per_type():
    t = triage([F("caries", "Low", i) for i in range(1, 9)], analysed=True)
    assert t.reasons[0].startswith("8 areas of possible decay (#1, #2, #3, #4, #5 and 3 more)")


def test_nothing_analysed():
    t = triage([], analysed=False)
    assert t.level == "ROUTINE" and "Retake" in t.reasons[0]
    assert screening_index([], analysed=False) == (None, "Not available", {})


# ---------------------------------------------------------------- screening index
@pytest.mark.parametrize("findings,index,label", [
    ([], 100, "Low concern"),
    ([F("cavity")], 80, "Some concern"),
    ([F("crack"), F("cavity")], 60, "Some concern"),
    ([F("caries", "High")], 85, "Low concern"),
    ([F("caries", "Medium")], 92, "Low concern"),
    ([F("caries", "Low")], 97, "Low concern"),
    ([F("caries", "High")] * 4, 60, "Some concern"),               # 4 x 15 = 60, capped at 40
    ([F("wear")] * 5, 97, "Low concern"),                          # wear total capped at 3
    ([F("filling")] * 9 + [F("crown")] * 3, 95, "Low concern"),    # restorations capped at 5
    ([F("cavity")] * 3 + [F("caries", "High")] * 3, 0, "Higher concern"),  # 60 + 40 -> clamp at 0
    ([F("cavity"), F("caries", "Medium"), F("wear"), F("filling")], 68, "Some concern"),
])
def test_screening_index(findings, index, label):
    got, got_label, breakdown = screening_index(findings, analysed=True)
    assert (got, got_label) == (index, label)
    assert breakdown["caries"] <= 40 and breakdown["wear"] <= 3 and breakdown["restorations"] <= 5


def test_index_band_edges():
    assert screening_index([F("caries", "High")], True)[:2] == (85, "Low concern")
    assert screening_index([F("caries", "High"), F("filling")], True)[:2] == (84, "Some concern")
    assert screening_index([F("cavity"), F("cavity")], True)[:2] == (60, "Some concern")
    assert screening_index([F("cavity"), F("cavity"), F("filling")], True)[:2] == (59, "Higher concern")


def test_coverage_mentions_what_photos_cannot_show():
    c = coverage(["front", "upper"], ["front"], VIEWS)
    assert c.analysed_views == ["front"] and c.failed_views == ["upper"]
    assert c.skipped_views == ["lower", "left", "right"]
    assert any("panoramic X-ray" in n for n in c.notes)


def test_reasons_lead_with_the_strongest_evidence():
    t = triage([F("cavity", "Low", 1), F("caries", "High", 2)], analysed=True)
    assert t.level == "PROMPT" and t.reasons[0].startswith("1 area of possible decay (#2)")
