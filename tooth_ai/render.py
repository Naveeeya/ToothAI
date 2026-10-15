"""Overlay rendering for the viewer, the annotated PNG and the PDF.

Replaces the old ``draw_detections``: separate layers (masks, numbers, findings,
geometry), two colour modes (by finding, by quadrant), a selection highlight, and
labels set in JetBrains Mono instead of OpenCV's Hershey font.
"""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tooth_ai.notation import label as tooth_label
from tooth_ai.ui import theme

LAYERS = ("Masks", "Numbers", "Findings", "Geometry")
DEFAULT_LAYERS = ("Masks", "Numbers", "Findings")


@lru_cache(maxsize=16)
def _font(size, name="JetBrainsMono-Medium.ttf"):
    try:
        return ImageFont.truetype(str(theme.font_path(name)), size)
    except OSError:
        return ImageFont.load_default()


def _blend(base, mask, rgb, alpha):
    if mask is None or alpha <= 0:
        return
    m = mask.astype(bool)
    if m.shape != base.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (base.shape[1], base.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    color = np.array(rgb, dtype=np.float32)
    base[m] = base[m] * (1 - alpha) + color * alpha


def _outline(img, mask, rgb, thickness):
    if mask is None:
        return
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, tuple(int(c) for c in rgb), thickness, cv2.LINE_AA)


def _box_mask(box, shape):
    m = np.zeros(shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    m[max(y1, 0):y2, max(x1, 0):x2] = 1
    return m


def _tooth_color(tooth, color_by):
    if color_by == "Quadrant":
        return theme.hex_to_rgb(theme.QUADRANT_COLORS.get(tooth["quadrant"], theme.TOOTH_FILL))
    return theme.hex_to_rgb(theme.TOOTH_FILL)


def finding_rgb(kind):
    return theme.hex_to_rgb(theme.FINDING_COLORS.get(kind, theme.MISSING))


def render_overlay(image_rgb, result, layers=DEFAULT_LAYERS, color_by="Finding",
                   opacity=0.35, notation="FDI", selected=None):
    """Return an annotated RGB uint8 image for ``result`` (an AnalysisResult)."""
    layers = set(layers)
    h, w = image_rgb.shape[:2]
    unit = max(w / 1000.0, 0.6)          # line widths and type scale with the image
    canvas = image_rgb.astype(np.float32).copy()

    if "Masks" in layers:
        for t in result.teeth:
            _blend(canvas, t.get("mask"), _tooth_color(t, color_by), opacity)
    if "Findings" in layers:
        for f in result.findings:
            mask = f.get("mask") if f.get("mask") is not None else _box_mask(f["box"], canvas.shape)
            _blend(canvas, mask, finding_rgb(f["class_name"]), min(opacity + 0.25, 0.85))

    img = np.clip(canvas, 0, 255).astype(np.uint8)

    if "Masks" in layers:
        for t in result.teeth:
            if t.get("mask") is not None:
                _outline(img, t["mask"], (235, 240, 245), max(1, int(unit)))
    if "Findings" in layers:
        for f in result.findings:
            mask = f.get("mask") if f.get("mask") is not None else _box_mask(f["box"], img.shape)
            _outline(img, mask, finding_rgb(f["class_name"]), max(1, int(round(1.5 * unit))))

    if "Geometry" in layers:
        _draw_geometry(img, result, unit)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    if selected is not None:
        t = result.tooth(selected)
        if t is not None:
            _brackets(draw, t["box"], unit)
    if "Numbers" in layers:
        _draw_numbers(draw, result, unit, notation, show_findings="Findings" in layers)
    return np.array(pil)


def _draw_geometry(img, result, unit):
    geo = result.geometry
    h, w = img.shape[:2]
    line = (198, 208, 218)
    th = max(1, int(round(1.2 * unit)))

    def dashed(p0, p1, dash=int(14 * unit) or 8):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        n = max(int(np.linalg.norm(p1 - p0) / dash), 1)
        for i in range(0, n, 2):
            a = p0 + (p1 - p0) * i / n
            b = p0 + (p1 - p0) * min(i + 1, n) / n
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), line, th, cv2.LINE_AA)

    if geo.split_y is not None:
        dashed((0, geo.split_y), (w, geo.split_y))
    if geo.midline_x is not None:
        dashed((geo.midline_x, 0), (geo.midline_x, h))
    for coeffs, jaw in ((geo.arch_upper, (1, 2)), (geo.arch_lower, (3, 4))):
        xs_teeth = [(t["box"][0] + t["box"][2]) / 2 for t in result.teeth if t["quadrant"] in jaw]
        if coeffs is None or not xs_teeth:
            continue
        xs = np.linspace(min(xs_teeth), max(xs_teeth), 120)
        ys = np.polyval(coeffs, xs)
        pts = np.stack([xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (255, 255, 255), max(1, int(round(2 * unit))), cv2.LINE_AA)
        for t in result.teeth:
            if t["quadrant"] in jaw:
                cx = (t["box"][0] + t["box"][2]) / 2
                cy = float(np.polyval(coeffs, cx))
                cv2.circle(img, (int(cx), int(cy)), max(2, int(3 * unit)), (255, 255, 255), -1, cv2.LINE_AA)


def _brackets(draw, box, unit):
    x1, y1, x2, y2 = box
    pad = 4 * unit
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    arm = max(min(x2 - x1, y2 - y1) * 0.28, 8 * unit)
    width = max(2, int(round(3 * unit)))
    white = (255, 255, 255)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        draw.line([(cx, cy + dy * arm), (cx, cy), (cx + dx * arm, cy)], fill=white, width=width)


def _draw_numbers(draw, result, unit, notation, show_findings):
    size = max(10, int(round(15 * unit)))
    font = _font(size)
    split = result.geometry.split_y
    for t in result.teeth:
        text = tooth_label(t["fdi"], notation)
        x1, y1, x2, y2 = t["box"]
        cx = (x1 + x2) / 2
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        padx, pady = 0.45 * size, 0.22 * size
        bw, bh = tw + 2 * padx, th + 2 * pady
        upper = t["quadrant"] in (1, 2) if split is None else (y1 + y2) / 2 < split
        # Upper-jaw labels above the tooth, lower-jaw labels below: outside the crowns.
        by = y1 - bh - 3 * unit if upper else y2 + 3 * unit
        by = min(max(by, 0), draw.im.size[1] - bh)
        bx = cx - bw / 2
        uncertain = t["score"] < theme.LOW_SCORE
        outline = theme.hex_to_rgb(theme.UNCERTAIN) if uncertain else (230, 236, 242)
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=0.3 * size,
                               fill=theme.hex_to_rgb(theme.INK_900), outline=outline,
                               width=max(1, int(round(1.4 * unit))))
        draw.text((bx + padx, by + pady - 0.1 * size), text, font=font, fill=(255, 255, 255))
        if show_findings and t["findings"]:
            r = 0.22 * size
            for i, f in enumerate(t["findings"][:3]):
                dx = bx + bw + r + 2 * unit + i * (2.3 * r)
                dy = by + bh / 2
                draw.ellipse([dx - r, dy - r, dx + r, dy + r], fill=finding_rgb(f["type"]),
                             outline=(255, 255, 255), width=1)


def tooth_centres(result):
    return [((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2) for t in result.teeth]
