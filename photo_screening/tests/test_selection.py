"""Selection rules measured on the AKU splits (app/selection.py)."""

from __future__ import annotations

import numpy as np

from photo_screening.app.config import DEFAULT_THRESHOLDS, Settings
from photo_screening.app.fusion import merge_max
from photo_screening.app.imaging import clahe_l, grey_world
from photo_screening.app.models.base import RawDet
from photo_screening.app.selection import select

SIZE = (1000, 1200)
TH = {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()} | {"caries": {"t_low": 0.48, "t_med": 0.55, "t_high": 0.74}}


def det(model, type_, score, box):
    return RawDet(model, type_, score, np.array(box, np.float32), None, 0, [model])


def run(per_model, caries_loaded=True, **overrides):
    return select(per_model, TH, Settings(**overrides), SIZE, caries_loaded)


def test_weak_agreement_on_one_spot_is_kept_and_either_alone_is_not():
    aku = det("caries_aku", "caries", 0.24, (100, 300, 260, 440))
    ad = det("alphadent", "caries", 0.58, (120, 320, 200, 400))
    kept = run({"caries_aku": [aku], "alphadent": [ad]})
    assert [(d.type, sorted(d.sources)) for d in kept] == [("caries", ["alphadent", "caries_aku"])]
    assert kept[0].score == 0.24                        # the phone model's score, so the band stays Low
    assert run({"caries_aku": [aku], "alphadent": []}) == []
    assert run({"caries_aku": [], "alphadent": [ad]}) == []      # AlphaDent alone: 2-5 % precision


def test_weak_alphadent_does_not_rescue():
    aku = det("caries_aku", "caries", 0.24, (100, 300, 260, 440))
    ad = det("alphadent", "caries", 0.35, (120, 320, 200, 400))
    assert run({"caries_aku": [aku], "alphadent": [ad]}) == []


def test_restorations_need_055_and_skip_the_frame_edge():
    low = det("alphadent", "crown", 0.49, (300, 300, 400, 400))     # a thumbnail read as a crown
    edge = det("alphadent", "filling", 0.61, (0, 250, 120, 320))       # a fingertip at the frame edge
    ok = det("alphadent", "filling", 0.68, (500, 300, 580, 400))
    kept = run({"alphadent": [low, edge, ok]})
    assert [d.score for d in kept] == [0.68]
    assert len(run({"alphadent": [edge]}, edge_margin=0.0)) == 1


def test_decay_wins_over_a_filling_on_the_same_spot():
    caries = det("caries_aku", "caries", 0.77, (660, 850, 850, 1020))
    filling = det("alphadent", "filling", 0.85, (714, 900, 780, 985))   # brown lesion read as a filling
    other = det("alphadent", "filling", 0.70, (100, 100, 180, 180))
    kept = run({"caries_aku": [caries], "alphadent": [filling, other]})
    assert sorted((d.type, round(d.score, 2)) for d in kept) == [("caries", 0.77), ("filling", 0.70)]
    assert len(run({"alphadent": [filling]})) == 1            # the filling alone is kept


def test_colour_passes_keep_the_best_score_per_spot():
    orig = [det("caries_aku", "caries", 0.24, (50, 720, 210, 860))]
    gw = [det("caries_aku", "caries", 0.77, (52, 722, 208, 858)),
          det("caries_aku", "caries", 0.5, (600, 100, 700, 200))]
    merged = merge_max([orig, gw])
    assert sorted(round(d.score, 2) for d in merged) == [0.5, 0.77]


def test_colour_normalisation_is_shape_preserving_and_neutralises_a_cast():
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[...] = (200, 140, 100)                           # warm indoor cast
    rgb[10:30, 10:50] = (230, 200, 160)
    gw = grey_world(rgb)
    assert gw.shape == rgb.shape and gw.dtype == np.uint8
    means = gw.reshape(-1, 3).mean(0)
    assert means.max() - means.min() < 3
    assert clahe_l(rgb).shape == rgb.shape
