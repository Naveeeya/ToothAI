"""ToothAI PDF report (ReportLab, A4, Plus Jakarta Sans + JetBrains Mono embedded).

Page 1: header band, case details, reliability, annotated radiograph (aspect preserved),
legend, dental chart, partner strip. Then: per-tooth table, unassigned findings,
method, limitations, disclaimer. Every page: footer with page n of m.
"""

import io
import re
from datetime import datetime
from functools import lru_cache
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

import tooth_ai
from tooth_ai import evidence
from tooth_ai.analysis import MODEL_VERSION
from tooth_ai.evidence import pct
from tooth_ai.notation import label as tooth_label, tooth_name
from tooth_ai.report_md import region
from tooth_ai.ui import theme
from tooth_ai.ui.theme import FDI_ROW_LOWER, FDI_ROW_UPPER, PARTNER_LOGOS, SEVERITY

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
HEADER_H = 18 * mm

INK_900 = colors.HexColor(theme.INK_900)
INK_700 = colors.HexColor(theme.INK_700)
INK_500 = colors.HexColor(theme.INK_500)
SLATE_300 = colors.HexColor(theme.SLATE_300)
SLATE_100 = colors.HexColor(theme.SLATE_100)
BONE_50 = colors.HexColor(theme.BONE_50)
LEVEL_COLORS = theme.LEVEL_COLORS


@lru_cache(maxsize=1)
def _register_fonts():
    names = {"Jakarta": "PlusJakartaSans-Regular.ttf", "Jakarta-Medium": "PlusJakartaSans-Medium.ttf",
             "Jakarta-Bold": "PlusJakartaSans-Bold.ttf", "JetBrainsMono": "JetBrainsMono-Regular.ttf",
             "JetBrainsMono-Medium": "JetBrainsMono-Medium.ttf"}
    try:
        for name, file in names.items():
            pdfmetrics.registerFont(TTFont(name, str(theme.font_path(file))))
        return {"sans": "Jakarta", "medium": "Jakarta-Medium", "bold": "Jakarta-Bold",
                "mono": "JetBrainsMono", "mono_medium": "JetBrainsMono-Medium"}
    except Exception:  # fonts missing: fall back to the PDF base fonts
        return {"sans": "Helvetica", "medium": "Helvetica", "bold": "Helvetica-Bold",
                "mono": "Courier", "mono_medium": "Courier-Bold"}


def _styles(f):
    base = dict(fontName=f["sans"], fontSize=9, leading=12.5, textColor=INK_900, alignment=TA_LEFT)
    return {
        "body": ParagraphStyle("body", **base),
        "small": ParagraphStyle("small", **{**base, "fontSize": 7.8, "leading": 10.5, "textColor": INK_500}),
        "h2": ParagraphStyle("h2", **{**base, "fontName": f["bold"], "fontSize": 11.5, "leading": 15,
                                      "spaceBefore": 8, "spaceAfter": 4}),
        "h3": ParagraphStyle("h3", **{**base, "fontName": f["bold"], "fontSize": 9.5, "leading": 13,
                                      "spaceBefore": 6, "spaceAfter": 2}),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 8.4, "leading": 11}),
        "cellmono": ParagraphStyle("cellmono", **{**base, "fontName": f["mono"], "fontSize": 8.4, "leading": 11}),
    }


def safe_case_id(case_id) -> str:
    """Letters, digits, space, '_', '.', '-'; at most 40 characters. Everything else is dropped."""
    return re.sub(r"[^A-Za-z0-9 _.\-]", "", str(case_id or ""))[:40].strip()


def _png(rgb_array, max_width=2000):
    img = PILImage.fromarray(rgb_array)
    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    buf.seek(0)
    return buf, img.width, img.height


def _numbered_canvas(fonts, case_line):
    class NumberedCanvas(rl_canvas.Canvas):
        """Two-pass canvas so the footer can say 'Page n of m'."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pages = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._pages)
            for state in self._pages:
                self.__dict__.update(state)
                self._decorate(total)
                super().showPage()
            super().save()

        def _decorate(self, total):
            page = self._pageNumber
            self.setFont(fonts["sans"], 7.2)
            self.setFillColor(INK_500)
            self.drawString(MARGIN, 9 * mm,
                            f"{tooth_ai.PRODUCT_NAME} · Research prototype. Not for diagnostic use")
            self.drawRightString(PAGE_W - MARGIN, 9 * mm, f"{case_line} · Page {page} of {total}")
            self.setStrokeColor(SLATE_300)
            self.setLineWidth(0.5)
            self.line(MARGIN, 12.5 * mm, PAGE_W - MARGIN, 12.5 * mm)
            if page == 1:
                self._header()
                self._partners()

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
                self.drawImage(ir, MARGIN, PAGE_H - HEADER_H + (HEADER_H - h) / 2,
                               width=h * iw / ih, height=h, mask="auto")
            self.setFillColor(colors.white)
            self.setFont(fonts["medium"], 11)
            self.drawRightString(PAGE_W - MARGIN, PAGE_H - HEADER_H / 2 + 1, "Panoramic radiograph report")
            self.setFont(fonts["sans"], 7.2)
            self.setFillColor(SLATE_300)
            self.drawRightString(PAGE_W - MARGIN, PAGE_H - HEADER_H / 2 - 9,
                                 "Detection · Segmentation · Numbering · Anatomical mapping")

        def _partners(self):
            y = 15 * mm
            h = 9 * mm
            logos = [theme.ASSETS / f for f, _ in PARTNER_LOGOS if (theme.ASSETS / f).exists()]
            readers = [ImageReader(str(p)) for p in logos]
            widths = [h * r.getSize()[0] / r.getSize()[1] for r in readers]
            gap = 5 * mm
            total = sum(widths) + gap * (len(widths) - 1)
            x = PAGE_W - MARGIN - total
            for r, w in zip(readers, widths):
                self.drawImage(r, x, y, width=w, height=h, mask="auto")
                x += w + gap
            self.setFont(fonts["sans"], 6.8)
            self.setFillColor(INK_500)
            self.drawString(MARGIN, y + h / 2 + 2, "Developed under the C-CAMP Inter-Institutional")
            self.drawString(MARGIN, y + h / 2 - 6, "Biomedical Innovations Programme (AICTE–C-CAMP IBIP)")

    return NumberedCanvas


def _odontogram(result, notation, fonts, width=CONTENT_W):
    present = {t["fdi"]: t for t in result.teeth}
    cell = (width - 10) / 16
    d = Drawing(width, 34 * mm)
    h = d.height
    d.add(Line(0, h / 2, width, h / 2, strokeColor=SLATE_300, strokeWidth=0.5))
    d.add(Line(width / 2, 4, width / 2, h - 4, strokeColor=SLATE_300, strokeWidth=0.5,
               strokeDashArray=[2, 2]))
    for fdis, upper in ((FDI_ROW_UPPER, True), (FDI_ROW_LOWER, False)):
        for i, fdi in enumerate(fdis):
            x = i * cell + (10 if i >= 8 else 0)
            t = present.get(fdi)
            cw, ch = cell * 0.72, 9 * mm
            cx = x + (cell - cw) / 2
            cy = h / 2 + 2 if upper else h / 2 - 2 - ch
            if t is None:
                fill, stroke, dash = colors.HexColor("#F2F4F6"), colors.HexColor(theme.MISSING), [2, 1.5]
            else:
                kinds = [f["type"] for f in t["findings"]]
                top = next((k for k in SEVERITY if k in kinds), None)
                fill = colors.HexColor(theme.FINDING_COLORS[top]) if top else BONE_50
                low = t["score"] < theme.LOW_SCORE
                stroke = colors.HexColor(theme.UNCERTAIN) if low else INK_900
                dash = [2, 1.5] if low else None
            d.add(Rect(cx, cy, cw, ch, rx=2.5, ry=2.5, fillColor=fill, strokeColor=stroke,
                       strokeWidth=0.8, strokeDashArray=dash))
            if t is None:
                mx, my = cx + cw / 2, cy + ch / 2
                d.add(Line(mx - 3, my - 3, mx + 3, my + 3, strokeColor=colors.HexColor(theme.MISSING), strokeWidth=0.8))
                d.add(Line(mx - 3, my + 3, mx + 3, my - 3, strokeColor=colors.HexColor(theme.MISSING), strokeWidth=0.8))
            ly = cy + ch + 3 if upper else cy - 9
            d.add(String(x + cell / 2, ly, tooth_label(fdi, notation), fontName=fonts["mono"], fontSize=7,
                         fillColor=INK_900 if t is not None else colors.HexColor(theme.MISSING),
                         textAnchor="middle"))
    d.add(String(0, h - 8, "Patient's right", fontName=fonts["sans"], fontSize=6.5, fillColor=INK_500))
    d.add(String(width, h - 8, "Patient's left", fontName=fonts["sans"], fontSize=6.5, fillColor=INK_500,
                 textAnchor="end"))
    return d


def _legend(result, fonts):
    kinds = {f["type"] for t in result.teeth for f in t["findings"]} | {a["class_name"] for a in result.findings}
    items = [("Tooth", theme.TOOTH_FILL)] + [(k, theme.FINDING_COLORS[k]) for k in SEVERITY if k in kinds]
    items.append(("Not detected", theme.MISSING))
    d = Drawing(CONTENT_W, 12)
    x = 0
    for name, col in items:
        d.add(Rect(x, 2, 7, 7, fillColor=colors.HexColor(col), strokeColor=None))
        d.add(String(x + 10, 3, name, fontName=fonts["sans"], fontSize=7, fillColor=INK_500))
        x += 14 + pdfmetrics.stringWidth(name, fonts["sans"], 7) + 8
    return d


def _table(rows, widths, fonts, header_bg=INK_900):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), fonts["medium"]),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BONE_50]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, SLATE_300),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
    ]))
    return t


def _method_text(result):
    if getattr(result, "engine", "v1") == "v2":
        return ("A Mask R-CNN network segments every tooth and finding in one pass. The improved numbering engine "
                "removes duplicate detections, keeps at most 32 teeth before numbering (never dropping a detected "
                "one), splits the jaws along a fitted bite line, finds the dental midline from the symmetry of "
                "tooth widths and numbers each tooth by its distance from the midline, matched in order to an arch "
                "layout learned from training images; a missing tooth leaves its position empty. Findings are "
                "attached to the tooth they overlap, and caries next to metal restorations is down-weighted.")
    return ("A Mask R-CNN network segments every tooth and finding on the original image and on a "
            "contrast-enhanced (CLAHE) copy; the two passes are fused. A geometric engine then "
            "splits the jaws, orders the teeth along each arch and assigns FDI numbers, skipping "
            "numbers where the gap implies a missing tooth. Findings are attached to the tooth they "
            "overlap, and anatomical rules remove duplicate numbers and down-weight caries next to "
            "metal restorations.")


def _limitations_text(result, ev):
    tail = (" Finding (caries, restoration) accuracy has not been validated. "
            "Region labels refer to the image: its left side is the patient's right.")
    v2 = evidence.engine_v2().get("v2") if getattr(result, "engine", "v1") == "v2" else None
    if v2:
        return (f"Measured on held-out panoramic radiographs with the improved engine: tooth detection recall "
                f"{pct(v2['recall'])}; FDI numbering {pct(v2['complete'])} accurate on complete arches and "
                f"{pct(v2['partial'])} on partial arches, where a missing or undetected tooth can still shift a "
                "number. The improved engine is a beta, not the published configuration." + tail)
    return (f"Measured on 716 held-out panoramic radiographs in the configuration this report "
            f"uses: tooth detection recall {pct(ev['recall'])} and jaw assignment {pct(ev['jaw'])}. "
            f"FDI numbering was {pct(ev['complete'])} accurate on complete arches but "
            f"{pct(ev['partial'])} on partial arches, because one missing tooth shifts every later "
            "number in that jaw." + tail)


def build_pdf(result, annotated_rgb, case_id="", notation="FDI", created=None):
    """Return the PDF as bytes."""
    fonts = _register_fonts()
    st = _styles(fonts)
    # Case IDs are typed by the user: allow-list the characters (they reach the footer and PDF metadata),
    # then escape wherever one is placed in markup.
    case_id = safe_case_id(case_id)
    created = created or datetime.now()
    case_line = f"Case {case_id}" if case_id else created.strftime("%d %b %Y %H:%M")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=HEADER_H + 8 * mm, bottomMargin=28 * mm,
                            title=f"{tooth_ai.PRODUCT_NAME} report {case_id}".strip(),
                            author=tooth_ai.PRODUCT_NAME, subject="Panoramic radiograph report")
    h, w = result.geometry.image_size
    ev = evidence.deployed_summary()
    story = []

    # Case details
    upper, lower = result.jaw_counts()
    meta = [
        ["Case", case_id or "-", "Date", created.strftime("%d %b %Y, %H:%M")],
        ["Mode", result.mode, "Notation", notation],
        ["Teeth detected", f"{len(result.teeth)} of 32 (upper {upper}, lower {lower})",
         "Image", f"{w} × {h} px"],
        ["Model", MODEL_VERSION if result.source == "model" else "Demo: cached model detections",
         "Findings", str(len(result.findings))],
    ]
    mt = Table([[Paragraph(f"<font color='{theme.INK_500}'>{escape(str(a))}</font>", st["small"]), Paragraph(escape(str(b)), st["cell"]),
                 Paragraph(f"<font color='{theme.INK_500}'>{escape(str(c))}</font>", st["small"]), Paragraph(escape(str(e)), st["cell"])]
                for a, b, c, e in meta],
               colWidths=[24 * mm, 66 * mm, 20 * mm, CONTENT_W - 110 * mm])
    mt.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, SLATE_100),
                            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story += [mt, Spacer(1, 4 * mm)]

    # Reliability
    rel = result.reliability
    reasons = "".join(f"<br/>• {escape(str(r))}" for r in rel.reasons)
    rb = Table([[Paragraph(f"<font name='{fonts['bold']}'>{escape(str(rel.title))}</font><br/>"
                           f"{escape(str(rel.detail))}{reasons}", st["body"])]],
               colWidths=[CONTENT_W])
    rb.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, 0), 3, colors.HexColor(LEVEL_COLORS[rel.level])),
                            ("BACKGROUND", (0, 0), (-1, -1), BONE_50),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [rb, Spacer(1, 4 * mm)]

    # Radiograph, aspect ratio preserved
    img_buf, iw, ih = _png(annotated_rgb)
    img_h = CONTENT_W * ih / iw
    story += [Image(img_buf, width=CONTENT_W, height=img_h), Spacer(1, 1.5 * mm), _legend(result, fonts),
              Spacer(1, 3 * mm)]
    if result.teeth:
        story += [Paragraph("Dental chart", st["h3"]), _odontogram(result, notation, fonts)]

    # Per-tooth table
    if result.teeth:
        story += [PageBreak(), Paragraph(escape(f"Teeth ({len(result.teeth)} of 32 detected)"), st["h2"])]
        rows = [["Tooth", "Name", "Findings", "Score"]]
        for t in result.teeth:
            f = ", ".join(x["type"] + (" (near metal)" if x.get("near_metal") else "") for x in t["findings"]) or "-"
            rows.append([Paragraph(escape(str(tooth_label(t["fdi"], notation))), st["cellmono"]),
                         Paragraph(escape(str(tooth_name(t["fdi"]))), st["cell"]), Paragraph(escape(f), st["cell"]),
                         Paragraph(escape(f"{t['score']:.2f}"), st["cellmono"])])
        story.append(_table(rows, [16 * mm, 62 * mm, CONTENT_W - 96 * mm, 18 * mm], fonts))

    if result.unassigned:
        story += [Paragraph("Findings not mapped to a tooth", st["h2"])]
        rows = [["Finding", "Region", "Score"]]
        rows += [[Paragraph(escape(str(a["class_name"])), st["cell"]), Paragraph(escape(region(a["box"], w)), st["cell"]),
                  Paragraph(escape(f"{a['score']:.2f}"), st["cellmono"])] for a in result.unassigned]
        story.append(_table(rows, [60 * mm, CONTENT_W - 78 * mm, 18 * mm], fonts, header_bg=INK_700))

    story += [KeepTogether([
        Paragraph("Method", st["h2"]),
        Paragraph(f"<i>{escape(tooth_ai.FULL_TITLE)}</i>.", st["body"]), Spacer(1, 2 * mm),
        Paragraph(escape(_method_text(result)), st["body"]),
        Paragraph("Limitations", st["h2"]),
        Paragraph(escape(_limitations_text(result, ev)), st["body"]),
        Spacer(1, 4 * mm),
        Paragraph(f"<font name='{fonts['bold']}'>{escape(tooth_ai.DISCLAIMER)}</font>", st["body"]),
    ])]

    doc.build(story, canvasmaker=_numbered_canvas(fonts, case_line))
    return buf.getvalue()
