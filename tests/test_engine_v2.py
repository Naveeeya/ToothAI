"""Engine v2 (tooth_ai/engine_v2.py) on synthetic arches: the v1 failure modes measured in paper_eval."""

import copy

from helpers import arch

from tooth_ai import engine_v2 as V2
from tooth_ai.pipeline import FDI_LOWER, FDI_UPPER, DentalGeometricEngine as E, resolve_fdi_conflicts, \
    recursive_prune_teeth

SLOTS = FDI_UPPER + FDI_LOWER          # slot index (0-31) -> FDI, image left to right per jaw


def v1(teeth):
    nms = E.apply_nms(copy.deepcopy(teeth), iou_threshold=0.4)
    up, lo = E.split_jaws(nms)
    numbered = E.sort_and_assign_fdi(up, "upper") + E.sort_and_assign_fdi(lo, "lower")
    return resolve_fdi_conflicts(recursive_prune_teeth(numbered, max_teeth=32))


def v2(teeth, **opts):
    return V2.number_teeth(E.apply_nms(copy.deepcopy(teeth), iou_threshold=0.4), **opts)


def by_x(result, jaw):
    """FDI numbers of one jaw, image left to right."""
    q = (1, 2) if jaw == "upper" else (3, 4)
    return [t["fdi"] for t in sorted(result, key=lambda t: t["box"][0]) if t["quadrant"] in q]


def test_complete_arch_matches_v1():
    teeth = arch()
    assert sorted(t["fdi"] for t in v2(teeth)) == sorted(SLOTS)
    assert by_x(v2(teeth), "upper") == by_x(v1(teeth), "upper") == FDI_UPPER


def test_missing_third_molar_does_not_shift_the_jaw():
    teeth = arch(missing=(0,))                        # 18 absent
    assert by_x(v1(teeth), "upper")[0] == 18          # v1: 17 is called 18, and every number shifts
    assert by_x(v2(teeth), "upper") == FDI_UPPER[1:]  # v2: 17 ... 28, correct


def test_extra_low_score_boxes_never_cost_a_real_tooth():
    teeth = arch()
    x0 = min(t["box"][0] for t in teeth)
    for y in (150, 260):                              # two weak false boxes beyond the patient's right end
        teeth.append({"box": [x0 - 60, y, x0 - 20, y + 90], "score": 0.25, "mask": None,
                      "class_name": "Tooth", "class_id": 0})
    out = v2(teeth)
    assert len(out) == 32 and all(t["score"] == 0.9 for t in out)       # the false boxes are capped away
    assert by_x(out, "upper") == FDI_UPPER and by_x(out, "lower") == FDI_LOWER
    assert by_x(v1(teeth), "upper") != FDI_UPPER                         # v1 numbered before pruning


def test_gap_skips_the_missing_number():
    teeth = arch(missing=(3,))                        # 15 absent, its neighbours keep their places
    assert by_x(v2(teeth), "upper") == [18, 17, 16, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]


def test_conflicts_never_delete_a_detected_tooth():
    teeth = [{"box": [0, 0, 10, 10], "fdi": 11, "score": 0.9, "quadrant": 1, "trace": {}},
             {"box": [20, 0, 30, 10], "fdi": 11, "score": 0.5, "quadrant": 1, "trace": {}}]
    out = V2.resolve_conflicts(teeth)
    assert len(out) == 2 and sorted(t["fdi"] for t in out) == [11, 12]


def test_options_run():
    teeth = arch(missing=(0, 16))
    for opts in ({"midline": "median"}, {"jaw_split": "curve"}, {"shared_midline": True}, {"dedupe": True}):
        assert len(v2(teeth, **opts)) == 30


def realistic_arch(missing=(), unit=40.0, w=1000):
    """Teeth placed where the learned template puts them (incisors narrow, molars wide), as on a real OPG."""
    teeth = []
    for jaw, (y1, y2) in (("upper", (150, 240)), ("lower", (260, 350))):
        mean, _ = V2.arch_template()[jaw]
        for side, sign, seq in (("right", -1, V2.SIDE_FDI[(jaw, "right")]), ("left", 1, V2.SIDE_FDI[(jaw, "left")])):
            for pos, fdi in enumerate(seq):
                if fdi in missing:
                    continue
                cx = w / 2 + sign * mean[pos] * unit
                half = unit * (0.4 if pos < 3 else 0.5)
                dy = 0.0003 * (cx - w / 2) ** 2 * (-1 if jaw == "upper" else 1)
                teeth.append({"box": [cx - half, y1 + dy, cx + half, y2 + dy], "score": 0.9, "mask": None,
                              "class_name": "Tooth", "class_id": 0, "true": fdi})
    return teeth


def test_shipped_defaults_number_realistic_arches():
    """The configuration the app ships (template numbering, curved jaw split, dedupe)."""
    for missing in ((), (18, 48), (16,), (38, 28, 36)):
        teeth = realistic_arch(missing)
        out = v2(teeth, **V2.DEFAULTS)
        truth = {tuple(t["box"]): t["true"] for t in teeth}
        got = {tuple(t["box"]): t["fdi"] for t in out}
        assert got == truth, missing


def test_template_is_learned_and_monotonic():
    for jaw in ("upper", "lower"):
        mean, std = V2.arch_template()[jaw]
        assert len(mean) == 8 and all(b > a for a, b in zip(mean, mean[1:])) and (std > 0).all()
