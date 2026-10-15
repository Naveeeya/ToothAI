import cv2
import numpy as np

from photo_screening.app.quality import assess
from photo_screening.tests.conftest import mouth_rgb


def _check(q, name):
    return next(c for c in q.checks if c.name == name)


def test_good_photo_passes(settings):
    q = assess(mouth_rgb(), (3000, 2000), settings)
    assert q.passed
    assert all(c.passed for c in q.checks)


def test_blur_is_a_hard_fail(settings):
    blurred = cv2.GaussianBlur(mouth_rgb(), (0, 0), 6)
    q = assess(blurred, (3000, 2000), settings)
    assert not q.passed
    blur = _check(q, "blur")
    assert not blur.passed and blur.hard and "steady" in blur.tip


def test_dark_and_blown_out_photos_fail_exposure(settings):
    dark = (mouth_rgb() * 0.12).astype(np.uint8)
    bright = np.full((640, 960, 3), 254, np.uint8)
    q_dark, q_bright = assess(dark, (3000, 2000), settings), assess(bright, (3000, 2000), settings)
    assert not _check(q_dark, "exposure").passed and "flash" in _check(q_dark, "exposure").tip
    assert not _check(q_bright, "exposure").passed and "glare" in _check(q_bright, "exposure").tip


def test_low_resolution_uses_original_size(settings):
    q = assess(mouth_rgb(), (400, 300), settings)
    assert not _check(q, "resolution").passed and not q.passed


def test_mouth_check_is_soft_without_tooth_model(settings):
    # a textured grey-blue wall: no enamel-white, no gum-pink
    texture = np.random.default_rng(1).integers(-40, 40, (640, 960, 1))
    wall = np.clip(np.array([110, 118, 128]) + texture, 0, 255).astype(np.uint8)
    q = assess(wall, (3000, 2000), settings)
    mouth = _check(q, "mouth")
    assert not mouth.passed and not mouth.hard
    assert q.passed                                   # soft warning only
    assert mouth in q.warnings


def test_mouth_check_is_hard_with_tooth_model(settings):
    q = assess(mouth_rgb(), (3000, 2000), settings, teeth_detected=0)
    mouth = _check(q, "mouth")
    assert mouth.hard and not mouth.passed and not q.passed
    assert assess(mouth_rgb(), (3000, 2000), settings, teeth_detected=6).passed
