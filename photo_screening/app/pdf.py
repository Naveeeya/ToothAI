"""Photo screening PDF, in the same visual language as the OPG report (tooth_ai/report_pdf.py).

Reuses that module's fonts, styles, colours, table style and numbered-page canvas; only the header
text and a model-versions line differ. Every dynamic string is XML-escaped before it goes into a
ReportLab Paragraph (VLM text and user-visible regions included).
"""

from __future__ import annotations

import base64
import io
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import photo_screening
from photo_screening.app.config import Settings
from photo_screening.app.render import COLORS
from photo_screening.app.rules import INDEX_BANDS, LABELS
from photo_screening.app.schemas import ImageResult, ScreeningReport
from tooth_ai.report_pdf import (
    BONE_50,
    CONTENT_W,
    HEADER_H,
    INK_500,
    INK_700,
    INK_900,
    MARGIN,
    SLATE_100,
    SLATE_300,
    _numbered_canvas,
    _register_fonts,
    _styles,
    _table,
)
from tooth_ai.ui import theme

PAGE_W, PAGE_H = A4
LEVEL_COLORS = theme.LEVEL_COLORS
W = CONTENT_W - 12   # usable width inside the frame's 6 pt padding, so tables align with text
TRIAGE_COPY = {
    "ROUTINE": ("ok", "Routine: keep up regular check-ups"),
    "SOON": ("caution", "Book a dental check-up soon"),
    "PROMPT": ("alert", "See a dentist promptly"),
}
VIEW_NAMES = {"front": "Front", "upper": "Upper teeth", "lower": "Lower teeth", "left": "Left side",
              "right": "Right side"}


def _e(text: object) -> str:
    return escape(str(text))


def _photo_canvas(fonts: dict, case_line: str, versions_line: str):
    base = _numbered_canvas(fonts, case_line)

    class PhotoCanvas(base):
        def _decorate(self, total):
            super()._decorate(total)
            self.setFont(fonts["mono"], 6.2)
            self.setFillColor(INK_500)
            self.drawString(MARGIN, 5.5 * mm, versions_line[:150])

        def _header(self):
            self.setFillColor(INK_900)
            self.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, stroke=0, fill=1)
            self.setFillColor(colors.HexColor(theme.SPARK))
            self.rect(0, PAGE_H - HEADER_H - 1.2, PAGE_W, 1.2, stroke=0, fill=1)
            lockup = theme.BRAND / "tooth-ai-lockup-horizontal-reversed.png"
            if lockup.exists():
                ir = ImageReader(str(lockup))
                iw, ih = ir.getSize()
                h = 9 * mm
                self.drawImage(ir, MARGIN, PAGE_H - HEADER_H + (HEADER_H - h) / 2, width=h * iw / ih, height=h,
                               mask="auto")
            self.setFillColor(colors.white)
            self.setFont(fonts["medium"], 11)
            self.drawRightString(PAGE_W - MARGIN, PAGE_H - HEADER_H / 2 + 1, "Photo screening report")
            self.setFont(fonts["sans"], 7.2)
            self.setFillColor(SLATE_300)
            self.drawRightString(PAGE_W - MARGIN, PAGE_H - HEADER_H / 2 - 9,
                                 "Screening aid, not a diagnosis · Smartphone intraoral photos")

    return PhotoCanvas


def _index_bar(index: int | None, fonts: dict, width: float = 70 * mm) -> Drawing:
    """0-100 bar with the three bands and a marker at the index."""
    d = Drawing(width, 13 * mm)
    band_cols = {"Higher concern": theme.ALERT, "Some concern": theme.UNCERTAIN, "Low concern": theme.OK}
    edges = sorted((floor, name) for floor, name in INDEX_BANDS)
    for i, (floor, name) in enumerate(edges):
        top = edges[i + 1][0] if i + 1 < len(edges) else 100
        d.add(Rect(width * floor / 100, 5 * mm, width * (top - floor) / 100, 2.6 * mm,
                   fillColor=colors.HexColor(band_cols[name]), strokeColor=None))
    for v in (0, 60, 85, 100):
        d.add(String(width * v / 100, 1 * mm, str(v), fontName=fonts["mono"], fontSize=6, fillColor=INK_500,
                     textAnchor="middle" if 0 < v < 100 else ("start" if v == 0 else "end")))
    if index is not None:
        x = width * index / 100
        d.add(Rect(x - 0.8, 4 * mm, 1.6, 4.6 * mm, fillColor=INK_900, strokeColor=None))
        d.add(String(x, 9.6 * mm, str(index), fontName=fonts["bold"], fontSize=8, fillColor=INK_900,
                     textAnchor="middle"))
    return d


def _qr(url: str, size: float = 22 * mm) -> Drawing:
    w = QrCodeWidget(url)
    x1, y1, x2, y2 = w.getBounds()
    d = Drawing(size, size, transform=[size / (x2 - x1), 0, 0, size / (y2 - y1), 0, 0])
    d.add(w)
    return d


def _photo(b64: str, max_w: float, max_h: float) -> Image:
    img = PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img.thumbnail((1400, 1400))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=86)
    buf.seek(0)
    scale = min(max_w / img.width, max_h / img.height)
    return Image(buf, width=img.width * scale, height=img.height * scale)


def _view_block(im: ImageResult, st: dict, fonts: dict) -> list:
    title = Paragraph(_e(VIEW_NAMES[im.view]) + (" view" if im.view == "front" else ""), st["h3"])
    if not im.analysed:
        tips = [c for c in im.quality.checks if c.hard and not c.passed]
        text = "Not analysed: " + "; ".join(f"{_e(c.name)} check failed. {_e(c.tip)}" for c in tips)
        return [KeepTogether([title, Paragraph(text, st["small"]), Spacer(1, 3 * mm)])]
    photo_w = 74 * mm
    photo = _photo(im.annotated_png_b64, photo_w, 64 * mm)
    rows = [["#", "Finding", "Where", "Confidence"]]
    for f in im.findings:
        where = f.region + (f" ({f.tooth_hint})" if f.tooth_hint else "")
        rows.append([Paragraph(f"<font color='{COLORS[f.type]}' name='{fonts['mono_medium']}'>{f.id}</font>",
                               st["cellmono"]),
                     Paragraph(_e(LABELS[f.type].capitalize()), st["cell"]), Paragraph(_e(where), st["cell"]),
                     Paragraph(_e(f.band), st["cell"])])
    side_w = W - photo_w - 4 * mm
    if len(rows) > 1:
        table = _table(rows, [8 * mm, 25 * mm, side_w - 56 * mm, 23 * mm], fonts)
        table.setStyle(TableStyle([("TOPPADDING", (0, 1), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 1), (-1, -1), 1.6)]))
    else:
        table = Paragraph("No findings marked in this photo.", st["body"])
    warnings = [c for c in im.quality.checks if not c.hard and not c.passed]
    side = [table] + ([Spacer(1, 2 * mm)] + [Paragraph("Photo note: " + _e(w.tip), st["small"]) for w in warnings]
                      if warnings else [])
    grid = Table([[photo, side]], colWidths=[photo_w + 4 * mm, side_w])
    grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return [KeepTogether([title, grid]), Spacer(1, 4 * mm)]


def build_pdf(report: ScreeningReport, settings: Settings | None = None) -> bytes:
    """Return the screening report as PDF bytes."""
    settings = settings or Settings()
    fonts = _register_fonts()
    st = _styles(fonts)
    created = report.created_at.astimezone()
    versions = ", ".join(f"{k} {v}" for k, v in report.model_versions.items()) or "none"
    case_line = f"Report {report.report_id}"
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=HEADER_H + 8 * mm,
                            bottomMargin=28 * mm, title=f"ToothAI photo screening report {report.report_id}",
                            author="ToothAI", subject="Photo screening report")
    story: list = []

    meta = [["Report", report.report_id, "Date", created.strftime("%d %b %Y, %H:%M")],
            ["Photos analysed", ", ".join(report.coverage.analysed_views) or "none", "App", report.app_version]]
    mt = Table([[Paragraph(f"<font color='{theme.INK_500}'>{_e(a)}</font>", st["small"]), Paragraph(_e(b), st["cell"]),
                 Paragraph(f"<font color='{theme.INK_500}'>{_e(c)}</font>", st["small"]), Paragraph(_e(d), st["cell"])]
                for a, b, c, d in meta], colWidths=[28 * mm, 70 * mm, 14 * mm, W - 112 * mm])
    mt.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, SLATE_100),
                            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story += [mt, Spacer(1, 4 * mm)]

    level, title = TRIAGE_COPY[report.triage.level]
    reasons = "".join(f"<br/>• {_e(r)}" for r in report.triage.reasons)
    tri = Paragraph(f"<font name='{fonts['bold']}'>{_e(title)}</font>{reasons}", st["body"])
    idx = [Paragraph(f"<font name='{fonts['bold']}'>Screening index</font> · {_e(report.index_band)}", st["body"]),
           _index_bar(report.screening_index, fonts),
           Paragraph("100 = nothing marked. Not a health score; see the method note.", st["small"])]
    box = Table([[tri, idx]], colWidths=[W - 76 * mm, 76 * mm])
    box.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, 0), 3, colors.HexColor(LEVEL_COLORS[level])),
                             ("BACKGROUND", (0, 0), (-1, -1), BONE_50), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [box, Spacer(1, 3 * mm)]
    source = (f" <font color='{theme.INK_500}'>(rephrased by AI, advisory)</font>"
              if report.summary_source == "vlm" else "")
    story += [Paragraph(_e(report.summary_text) + source, st["body"]), Spacer(1, 4 * mm)]

    for im in report.images:
        story += _view_block(im, st, fonts)

    if report.observations:
        rows = [["Observation", "Type", "Confidence"]]
        rows += [[Paragraph(_e(o.text), st["cell"]), Paragraph(_e(o.type.replace("_", " ")), st["cell"]),
                  Paragraph(_e(o.band), st["cell"])] for o in report.observations]
        label = report.vlm.provider_label or "AI model"
        story += [KeepTogether([
            Paragraph("AI observations (advisory)", st["h2"]),
            Paragraph(f"Generated by {_e(label)} from the photos. Not checked by the detection models; "
                      "use as prompts for discussion with a dentist only.", st["small"]),
            Spacer(1, 1.5 * mm), _table(rows, [W - 60 * mm, 38 * mm, 22 * mm], fonts, header_bg=INK_700)])]

    steps = [
        "Show this report to a dentist; photos can miss problems and can mark harmless stains.",
        "Ask about a panoramic X-ray (OPG). ToothAI can analyse it on the Analyse page of the app.",
        "Retake any photo marked 'Not analysed' using the tips shown.",
    ]
    next_steps = [Paragraph("What photos cannot show", st["h2"])]
    next_steps += [Paragraph("• " + _e(n), st["body"]) for n in report.coverage.notes]
    next_steps += [Paragraph("Next steps", st["h2"])] + [Paragraph(f"{i}. {_e(s)}", st["body"])
                                                         for i, s in enumerate(steps, 1)]
    url = f"{settings.app_url.rstrip('/')}/{settings.opg_page_path.lstrip('/')}"
    qr_cell = [_qr(url), Paragraph(_e(url), st["small"])]
    nt = Table([[next_steps, qr_cell]], colWidths=[W - 40 * mm, 40 * mm])
    nt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [nt]

    story += [KeepTogether([
        Paragraph("Method", st["h2"]),
        Paragraph("Detection models mark visible findings; fixed rules (not AI) set the confidence bands, the "
                  "triage level and the screening index. Index = 100 minus deductions: possible cavity or crack "
                  "20 each (when cavity and crack reporting is on); possible decay 15/8/3 by confidence "
                  "(at most 40 in total); wear 3 in total; fillings "
                  "and crowns 1 each (at most 5). Decay marked only by the DSLR-trained AlphaDent model is "
                  "always low confidence. Details and measured accuracy: photo_screening/MODEL_CARD.md and "
                  "EVAL.md.", st["body"]),
        Paragraph("Model versions: " + _e(versions), st["small"]),
        Spacer(1, 3 * mm),
        Paragraph(f"<font name='{fonts['bold']}'>{_e(report.disclaimer)}</font>", st["body"]),
    ])]

    doc.build(story, canvasmaker=_photo_canvas(fonts, case_line, f"Models: {versions} · "
                                               f"{photo_screening.PRODUCT_NAME} {report.app_version}"))
    return buf.getvalue()
