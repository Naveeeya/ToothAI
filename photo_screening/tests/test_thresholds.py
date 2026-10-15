import json

from photo_screening.app.config import DEFAULT_THRESHOLDS, MIN_KEEP, load_thresholds


def _write(settings, model_id, classes):
    d = settings.weights_dir / model_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "thresholds.json").write_text(json.dumps({"classes": classes}))


def test_defaults_without_calibration(settings):
    assert load_thresholds(settings, ["alphadent"])["caries"] == DEFAULT_THRESHOLDS["caries"]


def test_weak_class_is_floored_and_bands_fall_back(settings):
    _write(settings, "teeth_dentalai", {"cavity": {"t_low": 0.05, "t_med": 0.05, "t_high": 0.64},
                                        "caries": {"t_low": 0.01, "t_med": 0.01, "t_high": 0.2}})
    t = load_thresholds(settings, ["alphadent", "teeth_dentalai"])
    assert t["cavity"]["t_low"] == MIN_KEEP["cavity"]
    assert t["cavity"]["t_med"] == DEFAULT_THRESHOLDS["cavity"]["t_med"]
    assert t["cavity"]["t_high"] == 0.64
    assert t["caries"] == DEFAULT_THRESHOLDS["caries"]          # DentalAI never sets caries thresholds


def test_caries_thresholds_come_from_the_aku_model_only_when_loaded(settings):
    _write(settings, "caries_aku", {"caries": {"t_low": 0.48, "t_med": 0.55, "t_high": 0.74}})
    assert load_thresholds(settings, ["alphadent", "caries_aku"])["caries"] == {"t_low": 0.48, "t_med": 0.55,
                                                                                 "t_high": 0.74}
    assert load_thresholds(settings, ["alphadent"])["caries"] == DEFAULT_THRESHOLDS["caries"]


def test_bands_are_always_ordered(settings):
    _write(settings, "caries_aku", {"caries": {"t_low": 0.63, "t_med": 0.63, "t_high": 0.56}})
    t = load_thresholds(settings, ["caries_aku"])["caries"]
    assert t["t_low"] <= t["t_med"] <= t["t_high"]
