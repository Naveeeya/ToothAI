#!/usr/bin/env python3
"""Build the ToothAI brand assets into assets/brand/ from Brand Guidelines v1.0 (Oct 2026).

The mark (tooth outline + spark) uses the brand pack's exact 64-unit paths. The spark's gap is cut
with an SVG mask instead of a background-coloured stroke, so every file works on any background.
The wordmark is converted to outlines from Plus Jakarta Sans ExtraBold (static/fonts/), tracked
-4 % as in the guidelines, so the SVGs render identically everywhere. Re-run after any change:

    ~/.venvs/photo/bin/python tools/build_brand.py

Needs: fonttools and skia-python (PNG rendering).
"""

from __future__ import annotations

from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "brand"
FONTS = ROOT / "static" / "fonts"

# Brand Guidelines v1.0, "Color".
NAVY = "#0E2236"          # Radiograph Navy: primary ink, dark grounds
SPARK = "#2D6FDB"         # Spark Blue: the spark, "AI", actions (the only accent)
SPARK_LIGHT = "#7FA9F5"   # Spark Light: the spark on dark
ICE = "#EAF1FC"           # tints, surfaces
PAPER = "#F7F9FB"         # backgrounds
SLATE = "#5B6B78"         # secondary text
WHITE = "#FFFFFF"

TOOTH = ("M32 12c-4-4-10-6-15-4-6 2-8 9-7 16 1 6 4 10 5 16 1 7 2 16 7 17 4 1 5-8 6-13 1-3 2-4 4-4s3 1 4 4"
         "c1 5 2 14 6 13 5-1 6-10 7-17 1-6 4-10 5-16 1-7-1-14-7-16-5-2-11 0-15 4z")
SPARK_PATH = ("M49 0c1.6 8.4 3.2 10 11.6 11.6C52.2 13.2 50.6 14.8 49 23.2 47.4 14.8 45.8 13.2 37.4 11.6 "
              "45.8 10 47.4 8.4 49 0z")
STROKE = 4.5


def mark(tooth=NAVY, spark=SPARK, uid="m", x=0.0, y=0.0, size=64.0) -> str:
    """The mark on a 64-unit grid, placed at (x, y) and scaled to `size`."""
    s = size / 64
    return (f'<g transform="translate({x:.2f} {y:.2f}) scale({s:.5f})">'
            f'<defs><mask id="{uid}" maskUnits="userSpaceOnUse" x="-8" y="-8" width="80" height="80">'
            f'<rect x="-8" y="-8" width="80" height="80" fill="#fff"/>'
            f'<path d="{SPARK_PATH}" fill="#000" stroke="#000" stroke-width="{STROKE}" stroke-linejoin="round"/>'
            f'</mask></defs>'
            f'<path d="{TOOTH}" fill="none" stroke="{tooth}" stroke-width="{STROKE}" stroke-linejoin="round" '
            f'mask="url(#{uid})"/>'
            f'<path d="{SPARK_PATH}" fill="{spark}"/></g>')


def svg(body: str, w: float, h: float, title: str = "ToothAI") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" '
            f'height="{h:.0f}" role="img" aria-label="{title}"><title>{title}</title>{body}</svg>\n')


class Type:
    """Outlines text from a static TTF (no font dependency in the SVGs)."""

    def __init__(self, file: str):
        self.font = TTFont(FONTS / file)
        self.upm = self.font["head"].unitsPerEm
        self.cmap = self.font.getBestCmap()
        self.glyphs = self.font.getGlyphSet()
        self.hmtx = self.font["hmtx"]
        self.cap = self.font["OS/2"].sCapHeight / self.upm

    def width(self, text: str, size: float, tracking: float = 0.0) -> float:
        adv = sum(self.hmtx[self.cmap[ord(c)]][0] for c in text) / self.upm * size
        return adv + tracking * size * (len(text) - 1)

    def paths(self, text: str, size: float, x0: float, baseline: float, colors, tracking: float = 0.0) -> str:
        scale, x, out = size / self.upm, x0, []
        for ch, color in zip(text, colors, strict=True):
            name = self.cmap[ord(ch)]
            pen = SVGPathPen(self.glyphs)
            self.glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x, baseline)))
            if d := pen.getCommands():
                out.append(f'<path d="{d}" fill="{color}"/>')
            x += self.hmtx[name][0] * scale + tracking * size
        return "".join(out)

    def ink_bounds(self, text: str, size: float, tracking: float = 0.0):
        """(left, right) ink extent of the text at x0 = 0."""
        scale, x, lo, hi = size / self.upm, 0.0, None, None
        for ch in text:
            name = self.cmap[ord(ch)]
            bp = BoundsPen(self.glyphs)
            self.glyphs[name].draw(bp)
            if bp.bounds:
                xmin, _, xmax, _ = bp.bounds
                lo = x + xmin * scale if lo is None else lo
                hi = x + xmax * scale
            x += self.hmtx[name][0] * scale + tracking * size
        return lo or 0.0, hi or x


WORD = Type("PlusJakartaSans-ExtraBold.ttf")
TEXT = Type("PlusJakartaSans-Medium.ttf")
TRACK = -0.04                     # guidelines: wordmark letter-spacing -.04em


def wordmark(size, x0, baseline, tooth_color, ai_color) -> str:
    return WORD.paths("ToothAI", size, x0, baseline, [tooth_color] * 5 + [ai_color] * 2, TRACK)


def horizontal(size=60.0, ink=NAVY, ai=SPARK, spark=SPARK, uid="h"):
    """Primary horizontal lockup: mark 1.4x the type size, gap 0.3x (guidelines: 84 / 18 / 60)."""
    m, gap = 1.4 * size, 0.3 * size
    lo, hi = WORD.ink_bounds("ToothAI", size, TRACK)
    h = m
    baseline = h / 2 + WORD.cap * size / 2
    x0 = m + gap - lo
    w = m + gap + (hi - lo)
    return svg(mark(ink, spark, uid, 0, 0, m) + wordmark(size, x0, baseline, ink, ai), w + 1, h)


def stacked(size=38.0, ink=NAVY, ai=SPARK, spark=SPARK, uid="s"):
    """Stacked lockup: mark 84 above the wordmark at 38, gap 12 (guidelines)."""
    m, gap = 84 * size / 38, 12 * size / 38
    lo, hi = WORD.ink_bounds("ToothAI", size, TRACK)
    tw = hi - lo
    w = max(m, tw)
    h = m + gap + WORD.cap * size
    return svg(mark(ink, spark, uid, (w - m) / 2, 0, m) + wordmark(size, (w - tw) / 2 - lo, h, ink, ai), w + 1, h + 1)


def app_icon(px=160, ground=NAVY, tooth=WHITE, spark=SPARK_LIGHT, uid="i"):
    """App icon: rounded square (radius 38/160), mark at 100/160 (guidelines)."""
    r, m = 38 / 160 * px, 100 / 160 * px
    return svg(f'<rect width="{px}" height="{px}" rx="{r:.2f}" fill="{ground}"/>'
               + mark(tooth, spark, uid, (px - m) / 2, (px - m) / 2, m), px, px)


def favicon(px=48, uid="f"):
    """Favicon: 48 px tile, radius 12, mark 30 (guidelines)."""
    r, m = 12 / 48 * px, 30 / 48 * px
    return svg(f'<rect width="{px}" height="{px}" rx="{r:.2f}" fill="{NAVY}"/>'
               + mark(WHITE, SPARK_LIGHT, uid, (px - m) / 2, (px - m) / 2, m), px, px)


def og_card():
    """Link-preview card: the lockup alone, centred on navy (no tagline: previews are small)."""
    w, h = 1200, 630
    size = 120
    m, gap = 1.4 * size, 0.3 * size
    lo, hi = WORD.ink_bounds("ToothAI", size, TRACK)
    block = m + gap + hi - lo
    bx, by = (w - block) / 2, (h - m) / 2
    body = (f'<rect width="{w}" height="{h}" fill="{NAVY}"/>'
            + mark(WHITE, SPARK_LIGHT, "og", bx, by, m)
            + wordmark(size, bx + m + gap - lo, by + m / 2 + WORD.cap * size / 2, WHITE, SPARK_LIGHT))
    return svg(body, w, h)


def tokens_css() -> str:
    return """/* ToothAI design tokens (Brand Guidelines v1.0). Mirror of tooth_ai/ui/theme.py; keep in sync. */
:root {
  --navy: #0E2236;          /* Radiograph Navy: primary ink, dark grounds */
  --spark: #2D6FDB;         /* Spark Blue: the only accent (spark, "AI", actions) */
  --spark-light: #7FA9F5;   /* the spark on dark */
  --ice: #EAF1FC;           /* tints, surfaces */
  --paper: #F7F9FB;         /* backgrounds */
  --slate: #5B6B78;         /* secondary text */
  --ink-700: #33475B;       /* body text on paper */
  --line: #E1E6EC;          /* hairlines */
  --border: #C9D4E0;        /* borders */
  --white: #FFFFFF;
  --viewer-950: #0A0E13;

  /* semantic: viewer + charts only, always paired with a label */
  --finding-caries: #E5484D;
  --finding-retained-root: #C2410C;
  --finding-filling: #3B82F6;
  --finding-crown: #1D4ED8;
  --finding-endo: #0F9F8F;
  --finding-implant: #64748B;
  --finding-post: #94A3B8;
  --status-ok: #16A34A;
  --status-uncertain: #CA8A04;
  --status-alert: #C2410C;
  --status-missing: #9AA5B1;

  --font-sans: "Plus Jakarta Sans", system-ui, sans-serif;   /* 800 headlines (-3 %), 500/400 text */
  --font-mono: "JetBrains Mono", ui-monospace, monospace;    /* findings, measurements, labels */
  --radius: 12px;
}
"""


def render_png(svg_text: str, path: Path, width: int) -> None:
    import skia

    stream = skia.MemoryStream(svg_text.encode(), True)   # copy: the bytes must outlive the parse
    dom = skia.SVGDOM.MakeFromStream(stream)
    vw, vh = dom.containerSize().width(), dom.containerSize().height()
    height = round(width * vh / vw)
    surface = skia.Surface(width, height)
    with surface as canvas:
        canvas.clear(skia.ColorTRANSPARENT)
        canvas.scale(width / vw, height / vh)
        dom.render(canvas)
    surface.makeImageSnapshot().save(str(path), skia.kPNG)


WEB_SUBSET = list(range(0x20, 0x7F)) + list(range(0xA0, 0x100)) + [
    0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2026, 0x2192, 0x2264, 0x2265, 0x2212]


def subset_fonts() -> None:
    """Small Latin subsets of the brand fonts, embedded into SVG charts (an <img> can't use page fonts)."""
    from fontTools import subset

    out = FONTS / "web"
    out.mkdir(exist_ok=True)
    for name in ("PlusJakartaSans-Regular", "PlusJakartaSans-Bold", "JetBrainsMono-Medium"):
        opts = subset.Options()
        opts.layout_features = ["kern", "liga"]
        opts.name_IDs = ["*"]
        font = TTFont(FONTS / f"{name}.ttf")
        s = subset.Subsetter(opts)
        s.populate(unicodes=WEB_SUBSET)
        s.subset(font)
        font.save(out / f"{name}-latin.ttf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subset_fonts()
    files = {
        "tooth-ai-mark.svg": svg(mark(), 64, 64),
        "tooth-ai-mark-reversed.svg": svg(mark(WHITE, SPARK_LIGHT, "r"), 64, 64),
        "tooth-ai-mark-mono-dark.svg": svg(mark(NAVY, NAVY, "md"), 64, 64),
        "tooth-ai-mark-mono-light.svg": svg(mark(WHITE, WHITE, "ml"), 64, 64),
        "tooth-ai-favicon.svg": favicon(),
        "tooth-ai-app-icon.svg": app_icon(),
        "tooth-ai-app-icon-blue.svg": app_icon(ground=SPARK, tooth=WHITE, spark=WHITE, uid="ib"),
        "tooth-ai-app-icon-ice.svg": app_icon(ground=ICE, tooth=NAVY, spark=SPARK, uid="ii"),
        "tooth-ai-lockup-horizontal.svg": horizontal(),
        "tooth-ai-lockup-horizontal-reversed.svg": horizontal(ink=WHITE, ai=SPARK_LIGHT, spark=SPARK_LIGHT, uid="hr"),
        "tooth-ai-lockup-horizontal-white.svg": horizontal(ink=WHITE, ai=WHITE, spark=WHITE, uid="hw"),
        "tooth-ai-lockup-stacked.svg": stacked(),
        "og-1200x630.svg": og_card(),
    }
    for name, content in files.items():
        (OUT / name).write_text(content, encoding="utf-8")
    (OUT / "tokens.css").write_text(tokens_css(), encoding="utf-8")

    render_png(files["tooth-ai-favicon.svg"], OUT / "favicon-32.png", 32)
    render_png(files["tooth-ai-app-icon.svg"], OUT / "icon-192.png", 192)
    render_png(files["tooth-ai-app-icon.svg"], OUT / "icon-512.png", 512)
    render_png(files["tooth-ai-lockup-horizontal.svg"], OUT / "tooth-ai-lockup-horizontal.png", 1200)
    render_png(files["tooth-ai-lockup-horizontal-reversed.svg"], OUT / "tooth-ai-lockup-horizontal-reversed.png", 1200)
    render_png(files["og-1200x630.svg"], OUT / "og-1200x630.png", 1200)
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir() if not p.name.startswith("._"))))


if __name__ == "__main__":
    main()
