import numpy as np
import pytest

from photo_screening.app.fusion import fuse_models, fuse_passes, iou
from photo_screening.app.models.base import RawDet
from photo_screening.app.models.registry import TYPE_PRIORITY
from photo_screening.app.regions import describe


def D(model, type_, score, box, pass_id=0):
    return RawDet(model, type_, score, np.array(box, np.float32), None, pass_id)


def test_iou():
    assert iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == pytest.approx(1)
    assert iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0
    assert iou(np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])) == pytest.approx(1 / 3)


def test_single_pass_is_untouched():
    out = fuse_passes([[D("alphadent", "caries", 0.8, [0, 0, 10, 10])]])
    assert out[0].score == 0.8 and out[0].sources == ["alphadent"]


def test_tta_consensus_keeps_score_and_single_pass_is_penalised():
    original = [D("a", "caries", 0.8, [0, 0, 10, 10]), D("a", "caries", 0.5, [50, 50, 60, 60])]
    flipped = [D("a", "caries", 0.7, [1, 0, 11, 10], 1)]
    out = sorted(fuse_passes([original, flipped]), key=lambda d: -d.score)
    assert out[0].score == pytest.approx(0.8)                     # seen in both passes: max score
    assert out[1].score == pytest.approx(0.5 * 0.6)               # one pass only: x 0.6
    assert len(out) == 2


def test_models_merge_with_provenance_and_priority_score():
    per_model = {"alphadent": [D("alphadent", "caries", 0.9, [0, 0, 10, 10])],
                 "caries_aku": [D("caries_aku", "caries", 0.6, [1, 1, 11, 11])]}
    out = fuse_models(per_model, TYPE_PRIORITY)
    assert len(out) == 1
    assert out[0].sources == ["alphadent", "caries_aku"]
    assert out[0].score == pytest.approx(0.6)                     # phone model's score represents caries


def test_a_mark_inside_another_models_box_is_the_same_spot():
    per_model = {"alphadent": [D("alphadent", "caries", 0.76, [40, 40, 50, 50])],          # small fissure mark
                 "caries_aku": [D("caries_aku", "caries", 0.9, [0, 0, 100, 100])]}           # whole-surface box
    out = fuse_models(per_model, TYPE_PRIORITY)
    assert len(out) == 1 and out[0].sources == ["alphadent", "caries_aku"]
    assert out[0].score == pytest.approx(0.9) and list(out[0].box) == [0, 0, 100, 100]   # lead model's own box


def test_different_types_never_merge():
    per_model = {"alphadent": [D("alphadent", "caries", 0.9, [0, 0, 10, 10]),
                               D("alphadent", "filling", 0.8, [0, 0, 10, 10])]}
    assert {d.type for d in fuse_models(per_model, TYPE_PRIORITY)} == {"caries", "filling"}


# ---------------------------------------------------------------- regions (person's own left/right)
@pytest.mark.parametrize("view,box,mirrored,expected", [
    ("front", [10, 10, 60, 60], None, "Upper jaw, your right side"),     # image-left = their right
    ("front", [940, 10, 990, 60], None, "Upper jaw, your left side"),
    ("front", [480, 600, 520, 640], None, "Lower jaw, front"),
    ("front", [10, 10, 60, 60], True, "Upper jaw, your left side"),      # selfie/mirror flips
    ("upper", [10, 400, 60, 450], None, "Upper jaw, your left side"),    # occlusal views: mirror by default
    ("lower", [480, 100, 520, 140], None, "Lower jaw, middle"),
    ("right", [900, 100, 950, 150], None, "Upper jaw, front teeth on your right side"),
    ("left", [900, 600, 950, 650], None, "Lower jaw, back teeth on your left side"),
])
def test_regions(view, box, mirrored, expected):
    region, hint = describe(view, box, (1000, 700), mirrored=mirrored)
    assert region == expected
    assert hint is None                                            # no tooth model, no tooth hint


def test_tooth_hint_counts_from_the_middle():
    teeth = [[100 + 100 * i, 100, 180 + 100 * i, 300] for i in range(8)]          # upper row
    teeth += [[100 + 100 * i, 400, 180 + 100 * i, 600] for i in range(8)]         # lower row
    region, hint = describe("front", [520, 150, 560, 200], (1000, 700), teeth)    # on the 5th tooth
    assert region.startswith("Upper jaw")
    assert hint == "about the 1st tooth from the middle"
    _, hint = describe("front", [120, 150, 160, 200], (1000, 700), teeth)         # outermost tooth
    assert hint == "about the 4th tooth from the middle"
