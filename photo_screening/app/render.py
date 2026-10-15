"""Annotated photo: semi-transparent outlines, numbered markers matching the report table, a legend."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from photo_screening.app.rules import LABELS
from photo_screening.app.schemas import Finding
from photo_screening.palette import COLORS, ORDER  # shared with the Streamlit page
from tooth_ai.ui import theme

URGENT = ("caries", "cavity", "crack")              # filled on the photo; wear and restorations are outlines


@lru_cache(maxsize=8)
def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    name = "PlusJakartaSans-Bold.ttf" if bold else "PlusJakartaSans-Regular.ttf"
    try:
        return ImageFont.truetype(str(theme.font_path(name)), size)
    except OSError:
        return ImageFont.load_default(size)


def annotate(rgb: np.ndarray, findings: Sequence[Finding]) -> np.ndarray:
    """Return an RGB image with every finding drawn and numbered, plus a legend strip below.

    Medium/High findings get a filled outline and a full-size number; Low ones only a thin outline and
    a smaller, lighter number, so the photo shows what matters first. Numbers match the report table.
    """
    base = Image.fromarray(rgb).convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke = max(2, round(w / 450))
    for f in sorted(findings, key=lambda f: (f.band != "Low", f.confidence)):   # strong ones drawn last, on top
        rgb_c = theme.hex_to_rgb(COLORS[f.type])
        low = f.band == "Low"
        filled = not low and f.type in URGENT        # wear and restorations: outline only, never filled
        line = (*rgb_c, 170 if low else 255)
        width = max(1, stroke - 1) if low else stroke
        if f.mask_polygon and len(f.mask_polygon) >= 3:
            draw.polygon([tuple(p) for p in f.mask_polygon], fill=(*rgb_c, 60) if filled else None, outline=line,
                         width=width)
        else:
            draw.rectangle(f.bbox, outline=line, width=width)
    img = Image.alpha_composite(base, overlay)

    draw = ImageDraw.Draw(img)
    r_strong = max(10, round(w / 85))
    placed: list[tuple[float, float, int]] = []
    for f in sorted(findings, key=lambda f: f.band == "Low"):                     # strong numbers placed first
        low = f.band == "Low"
        r = round(r_strong * 0.75) if low else r_strong
        x, y = _anchor(f, r)
        x, y = _free_spot(x, y, r, placed, w, h)
        placed.append((x, y, r))
        rgb_c = theme.hex_to_rgb(COLORS[f.type])
        fill = (*rgb_c, 200) if low else (*rgb_c, 255)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(255, 255, 255, 255), width=2)
        draw.text((x, y), str(f.id), fill=(255, 255, 255, 255), font=_font(round(r * 1.1)), anchor="mm")

    return np.asarray(_with_legend(img.convert("RGB"), findings))


def _anchor(f: Finding, r: int) -> tuple[float, float]:
    """Marker sits just above the top of the outline (or the box), so it never covers the finding."""
    if f.mask_polygon and len(f.mask_polygon) >= 3:
        x, y = min(f.mask_polygon, key=lambda p: p[1])
    else:
        x, y = (f.bbox[0] + f.bbox[2]) / 2, f.bbox[1]
    return x, y - r * 0.9


def _free_spot(x: float, y: float, r: int, placed: list, w: int, h: int) -> tuple[float, float]:
    """Nudge a marker sideways (then up) until it doesn't overlap a marker already placed."""
    def clamp(px, py):
        return min(max(px, r + 1), w - r - 1), min(max(py, r + 1), h - r - 1)

    for dx, dy in ((0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0), (0, -1), (1, -1), (-1, -1), (3, 0), (-3, 0)):
        cx, cy = clamp(x + dx * 2.1 * r, y + dy * 2.1 * r)
        if all((cx - px) ** 2 + (cy - py) ** 2 >= (r + pr + 2) ** 2 for px, py, pr in placed):
            return cx, cy
    return clamp(x, y)


def _with_legend(img: Image.Image, findings: Sequence[Finding]) -> Image.Image:
    w, h = img.size
    bar = max(34, round(w / 26))
    out = Image.new("RGB", (w, h + bar), theme.WHITE)
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    font = _font(round(bar * 0.42), bold=False)
    present = [t for t in ORDER if any(f.type == t for f in findings)]
    x, cy = round(bar * 0.4), h + bar // 2
    if not present:
        d.text((x, cy), "No findings marked in this photo", fill=theme.hex_to_rgb(theme.INK_500), font=font,
               anchor="lm")
        return out
    sw = round(bar * 0.38)
    for t in present:
        d.rectangle((x, cy - sw // 2, x + sw, cy + sw // 2), fill=theme.hex_to_rgb(COLORS[t]))
        label = f"{LABELS[t].capitalize()} ({sum(f.type == t for f in findings)})"
        d.text((x + sw + 6, cy), label, fill=theme.hex_to_rgb(theme.INK_900), font=font, anchor="lm")
        x += sw + 6 + round(d.textlength(label, font=font)) + round(bar * 0.6)
    return out
