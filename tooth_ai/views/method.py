"""How it works: the pipeline, and why numbering is geometric."""

import streamlit as st

import tooth_ai
from tooth_ai.ui import components as ui
from tooth_ai.ui import theme

STAGES = [
    ("Radiograph", "Panoramic X-ray (OPG)", "input"),
    ("Contrast pass", "Original + CLAHE copy, fused", "learn"),
    ("Instance segmentation", "Mask R-CNN: every tooth and finding", "learn"),
    ("Geometric engine", "Jaw split, arch order, gap-aware FDI", "geo"),
    ("Anatomical rules", "One tooth per number, ≤ 32 teeth, metal flare", "geo"),
    ("Chart and report", "Findings mapped to teeth", "out"),
]


def _diagram():
    box_w, box_h, gap = 172, 74, 22
    w = 8 + len(STAGES) * box_w + (len(STAGES) - 1) * gap
    fills = {"input": theme.WHITE, "learn": theme.SLATE_100, "geo": theme.ICE, "out": theme.WHITE}
    strokes = {"input": theme.SLATE_300, "learn": theme.INK_500, "geo": theme.SPARK, "out": theme.SLATE_300}
    parts = [f'<svg viewBox="0 0 {w} 150" xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="ToothAI pipeline" font-family="Plus Jakarta Sans, sans-serif">',
             f'<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{theme.INK_500}"/></marker></defs>']
    x = 4
    for i, (title, sub, kind) in enumerate(STAGES):
        parts.append(f'<rect x="{x}" y="34" width="{box_w}" height="{box_h}" rx="10" fill="{fills[kind]}" '
                     f'stroke="{strokes[kind]}" stroke-width="1.4"/>')
        parts.append(f'<text x="{x + 12}" y="62" font-size="13.5" font-weight="600" fill="{theme.INK_900}">{title}</text>')
        words, line, lines = sub.split(), "", []
        for wd in words:
            if len(line + " " + wd) > 24:
                lines.append(line)
                line = wd
            else:
                line = (line + " " + wd).strip()
        lines.append(line)
        for j, ln in enumerate(lines[:2]):
            parts.append(f'<text x="{x + 12}" y="{82 + j * 15}" font-size="11.5" fill="{theme.INK_500}">{ui.esc(ln)}</text>')
        if i < len(STAGES) - 1:
            parts.append(f'<line x1="{x + box_w + 3}" y1="71" x2="{x + box_w + gap - 3}" y2="71" '
                         f'stroke="{theme.INK_500}" stroke-width="1.4" marker-end="url(#ar)"/>')
        x += box_w + gap
    # group labels
    lx = 4 + box_w + gap
    parts.append(f'<text x="{lx}" y="22" font-size="11" fill="{theme.INK_500}" font-family="JetBrains Mono, ui-monospace, monospace">DEEP LEARNING</text>')
    gx = 4 + 3 * (box_w + gap)
    parts.append(f'<text x="{gx}" y="22" font-size="11" fill="{theme.SPARK}" font-family="JetBrains Mono, ui-monospace, monospace">GEOMETRIC REASONING</text>')
    parts.append(f'<text x="{gx}" y="134" font-size="11" fill="{theme.INK_500}">No learned numbering: every number '
                 f'can be explained.</text>')
    parts.append("</svg>")
    ui.svg_image("".join(parts), "ToothAI pipeline: deep learning detects, geometric reasoning numbers",
                 padding="1rem")


def render():
    ui.page_header("How it works", tooth_ai.FULL_TITLE)
    _diagram()
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Deep learning finds the teeth")
        st.markdown(
            "A Mask R-CNN network (ResNet-50 backbone) segments every tooth as one generic class, plus eight "
            "kinds of finding: caries, crowns, fillings, implants, metal posts, retained roots, and root canal "
            "fillings and obturations. It runs twice, on the original image and on a contrast-enhanced copy "
            "(CLAHE), and the two sets of detections are fused. A second model contributes findings in Full "
            "chart mode.")
        st.subheader("Geometry numbers them")
        st.markdown(
            "1. **Clean up.** Overlapping detections are suppressed with distance-aware NMS, which handles "
            "tightly packed molars.\n"
            "2. **Split the jaws** at the median height of the tooth centres.\n"
            "3. **Order each jaw** from the patient's right to left along the arch.\n"
            "4. **Assign FDI numbers** in that order, starting at 18 / 48. A gap wider than 1.6× a tooth's "
            "width skips numbers for the missing teeth.\n"
            "5. **Apply anatomical rules.** No number is used twice, a mouth has at most 32 teeth, and caries "
            "beside metal restorations is down-weighted because metal causes flare.\n"
            "6. **Map findings** to the tooth they overlap.")
    with b:
        st.subheader("Why geometry, not a numbering network")
        st.markdown(
            "Public panoramic datasets don't agree on how to number teeth. In the training data, about "
            "**29 % of checkable images follow one of 116 inconsistent labelling conventions**, such as mirrored "
            "quadrants or shifted positions. A network that learns tooth numbers learns that noise too. The "
            "geometric engine learns no convention, so it can't absorb it, and measured end to end it "
            "numbers more teeth correctly.")
        st.subheader("Why it can explain itself")
        st.markdown(
            "Every number comes from a short chain of rules: jaw, position in the arch, gaps. On the Analyse "
            "page, select a tooth to read that chain under **Why this number?**, or turn on the **Geometry** "
            "layer to see the jaw split line, the midline and the fitted arch curves.")
        st.subheader("Where it fails")
        st.markdown(
            "Counting along the arch means one missing or undetected tooth shifts every number after it in "
            "that jaw. ToothAI flags partial arches on every case so the dentist knows where to look. See "
            "**Performance** for the measured effect.")
    ui.footer()
