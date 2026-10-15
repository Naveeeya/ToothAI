"""Turn a box into a plain-language mouth region for each photo view.

Sides are always the person's own left/right. A direct photo of the face shows their right on the
image's left; a photo taken through a mirror is flipped. Occlusal (upper/lower) views are assumed to
be mirror shots unless told otherwise. No FDI numbers are ever derived from photos.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_MIRRORED = {"front": False, "upper": True, "lower": True, "left": False, "right": False}
ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]


def _centre(box: Sequence[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _person_side(x_frac: float, mirrored: bool) -> str:
    """Third of the image -> 'your right' / 'the middle' / 'your left'."""
    if 1 / 3 <= x_frac <= 2 / 3:
        return "middle"
    image_left = x_frac < 1 / 3
    return "your right" if image_left != mirrored else "your left"


def _jaw_split(teeth: Sequence[Sequence[float]], height: int) -> float:
    ys = [_centre(t)[1] for t in teeth]
    return float(np.median(ys)) if len(ys) >= 4 else height / 2


def describe(view: str, box: Sequence[float], size: tuple[int, int], teeth: Sequence[Sequence[float]] = (),
             mirrored: bool | None = None) -> tuple[str, str | None]:
    """Return (region, tooth_hint). ``size`` is (width, height); ``teeth`` are tooth boxes if known."""
    w, h = size
    mirrored = DEFAULT_MIRRORED.get(view, False) if mirrored is None else mirrored
    cx, cy = _centre(box)
    if teeth:
        x0, x1 = min(t[0] for t in teeth), max(t[2] for t in teeth)
        span = max(x1 - x0, 1.0)
        x_frac = (cx - x0) / span
    else:
        x0, span, x_frac = 0.0, float(w), cx / w

    if view in ("upper", "lower"):
        side = _person_side(x_frac, mirrored)
        jaw = "Upper" if view == "upper" else "Lower"
        return f"{jaw} jaw, {'middle' if side == 'middle' else side + ' side'}", None

    jaw = "Upper" if cy < _jaw_split(teeth, h) else "Lower"
    if view in ("left", "right"):
        # Seen from the person's right, their front teeth are on the image's right (and vice versa).
        front_on_image_right = (view == "right") != mirrored
        front = (x_frac > 0.5) == front_on_image_right
        return f"{jaw} jaw, {'front' if front else 'back'} teeth on your {view} side", None

    side = _person_side(x_frac, mirrored)
    region = f"{jaw} jaw, {'front' if side == 'middle' else side + ' side'}"
    return region, _tooth_hint(box, teeth, jaw, h)


def _tooth_hint(box: Sequence[float], teeth: Sequence[Sequence[float]], jaw: str, height: int) -> str | None:
    """'about the 2nd tooth from the middle' when the finding sits on a detected tooth (front view)."""
    if len(teeth) < 4:
        return None
    split = _jaw_split(teeth, height)
    same_jaw = [t for t in teeth if (_centre(t)[1] < split) == (jaw == "Upper")]
    cx, cy = _centre(box)
    host = next((t for t in same_jaw if t[0] <= cx <= t[2] and t[1] <= cy <= t[3]), None)
    if host is None or len(same_jaw) < 2:
        return None
    mid = float(np.median([_centre(t)[0] for t in same_jaw]))
    hx = _centre(host)[0]
    same_side = sorted(abs(_centre(t)[0] - mid) for t in same_jaw if (_centre(t)[0] >= mid) == (hx >= mid))
    rank = same_side.index(abs(hx - mid))
    return f"about the {ORDINALS[min(rank, len(ORDINALS) - 1)]} tooth from the middle" if rank < 8 else None
