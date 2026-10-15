from tooth_ai.analysis import chart_from_detections

from helpers import arch


def _level(teeth):
    return chart_from_detections(teeth, [], (500, 1000, 3)).reliability


def test_complete_arches_ok():
    rel = _level(arch())
    assert rel.level == "ok", rel


def test_partial_jaw_caution():
    rel = _level(arch(missing=(20, 21, 22)))
    assert rel.level == "caution" and "lower" in rel.title.lower()


def test_sparse_alert():
    rel = _level(arch(n_upper=4, n_lower=4))
    assert rel.level == "alert"


def test_findings_only_mode_is_na():
    res = chart_from_detections([], [], (500, 1000, 3), mode="Findings only")
    assert res.reliability.level == "na" and res.teeth == []
