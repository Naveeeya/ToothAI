"""tooth_ai.pipeline must number teeth exactly like the audited copy in paper_eval/pipeline.py
(with fix_cls_routing=True, the configuration this app now runs)."""

import copy
import importlib.util
from pathlib import Path

import pytest

import tooth_ai.pipeline as new

_spec = importlib.util.spec_from_file_location(
    "paper_eval_pipeline", Path(__file__).resolve().parents[1] / "paper_eval" / "pipeline.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)
from tooth_ai.demo import _load_outputs, list_samples


def _chart(mod, outputs, width, **kw):
    teeth, anomalies = mod.process_predictions(outputs, image_width=width, teeth_threshold=0.2,
                                               anomaly_threshold=0.15, **kw)
    eng = mod.DentalGeometricEngine
    nms = eng.apply_nms(teeth, iou_threshold=0.4)
    up, lo = eng.split_jaws(nms)
    chart = eng.sort_and_assign_fdi(up, "upper") + eng.sort_and_assign_fdi(lo, "lower")
    chart = eng.map_anomalies(chart, anomalies)
    chart = mod.clean_clinical_findings(chart)
    chart, anomalies = mod.validate_anatomy(chart, anomalies, anomaly_threshold=0.15)
    return sorted((t["fdi"], tuple(round(v, 2) for v in t["box"]), round(t["score"], 4),
                   tuple(sorted(f["type"] for f in t["findings"]))) for t in chart)


@pytest.mark.parametrize("name", [s["name"] for s in list_samples()])
def test_same_chart_as_paper_eval(name):
    outputs, (h, w) = _load_outputs(name)
    a = _chart(ref, copy.deepcopy(outputs), w, fix_cls_routing=True)
    b = _chart(new, copy.deepcopy(outputs), w)
    assert a == b and len(a) > 0
