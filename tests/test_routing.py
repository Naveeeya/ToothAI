from tooth_ai.pipeline import CLASSES, PATHOLOGY_CLASS_MAP, process_predictions

from helpers import outputs


def test_nine_class_findings_are_not_teeth():
    # ids 1-8 are findings in the 9-class schema; the old cls_id <= 32 branch made them teeth
    dets = [(0, 0.9, (100, 100, 140, 200))] + [(i, 0.9, (200 + 50 * i, 100, 240 + 50 * i, 200)) for i in range(1, 9)]
    teeth, findings = process_predictions(outputs(dets), image_width=1000, teeth_threshold=0.2,
                                          anomaly_threshold=0.15)
    assert len(teeth) == 1
    assert {f["class_name"] for f in findings} == set(CLASSES[1:])


def test_41_class_tooth_ids_still_route_to_teeth():
    dets = [(i, 0.9, (20 * i, 100, 20 * i + 15, 200)) for i in range(0, 33)] + [(33, 0.9, (900, 100, 940, 200))]
    teeth, findings = process_predictions(outputs(dets), image_width=1000, teeth_threshold=0.2,
                                          anomaly_threshold=0.15, class_map=PATHOLOGY_CLASS_MAP)
    assert len(teeth) == 33
    assert [f["class_name"] for f in findings] == ["Caries"]
