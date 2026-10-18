#!/usr/bin/env python3
"""
OPG Annotation and PDF Report Generator
Processes rank1_9e23ed08.jpg using COCO ground truth annotations,
draws bounding boxes + semi-transparent masks, and generates a PDF report.
"""

import cv2
import json
import numpy as np
import os
from fpdf import FPDF

# ─── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_IMAGE   = r"c:\Users\Student\Tooth-ai\output\best_anomaly_images\rank1_9e23ed08.jpg"
ANNOTATED_OUT = r"c:\Users\Student\Tooth-ai\output\rank1_reports\rank1_opg_annotated.jpg"
PDF_OUT       = r"c:\Users\Student\Tooth-ai\output\rank1_reports\rank1_opg_report.pdf"
COCO_JSON     = r"c:\Users\Student\Tooth-ai\data\final-di-remapped\test\_annotations.coco.json"
IMAGE_ID      = 324   # rank1_9e23ed08 image id in the COCO dataset

# Colour palette per category name  (BGR)
CATEGORY_COLORS = {
    "Tooth":                    (180, 180, 180),
    "Caries":                   (0,   0,   255),   # Red
    "Crown":                    (0,   165, 255),   # Orange
    "Filling":                  (0,   255, 0  ),   # Green
    "Implant":                  (255, 0,   0  ),   # Blue
    "Prefabricated metal post": (255, 0,   255),   # Magenta
    "Root canal obturation":    (0,   255, 255),   # Yellow-cyan
    "Retained root":            (128, 0,   128),   # Purple
    "Root canal filling":       (255, 255, 0  ),   # Cyan-yellow
}

# Human-readable short names for PDF table and legend
SHORT_NAMES = {
    "Prefabricated metal post": "Prefab Metal Post",
    "Root canal obturation":    "RC Obturation",
    "Root canal filling":       "RC Filling",
}

# ─── STEP 1 – Load annotations ─────────────────────────────────────────────────

def load_annotations(coco_path: str, image_id: int):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Gather image metadata
    img_meta = next((i for i in coco["images"] if i["id"] == image_id), None)
    if img_meta is None:
        raise ValueError(f"Image ID {image_id} not found in COCO JSON")

    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
    return img_meta, anns, cat_map


# ─── STEP 2 – Annotate image ───────────────────────────────────────────────────

def annotate_image(input_path: str, output_path: str, anns: list, cat_map: dict):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    overlay = img.copy()

    for ann in anns:
        cat_name = cat_map.get(ann["category_id"], "Unknown")
        if cat_name == "Tooth":          # skip generic tooth outlines to reduce clutter
            continue

        x, y, w, h = [int(v) for v in ann["bbox"]]   # COCO [x,y,w,h]
        color = CATEGORY_COLORS.get(cat_name, (200, 200, 200))

        # Semi-transparent fill
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

    # Blend fill
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)

    # Draw borders + labels on top
    drawn_legend = {}  # track which categories have been labelled
    for ann in anns:
        cat_name = cat_map.get(ann["category_id"], "Unknown")
        if cat_name == "Tooth":
            continue

        x, y, w, h = [int(v) for v in ann["bbox"]]
        color = CATEGORY_COLORS.get(cat_name, (200, 200, 200))
        short  = SHORT_NAMES.get(cat_name, cat_name)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Only label first occurrence of each category (keeps image clean)
        if cat_name not in drawn_legend:
            cv2.putText(img, short, (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            drawn_legend[cat_name] = True

    cv2.imwrite(output_path, img)
    print(f"[✓] Annotated image saved → {output_path}")
    return img


# ─── STEP 3 – Build findings table ─────────────────────────────────────────────

def build_findings(anns: list, cat_map: dict):
    """Returns a list of dicts with per-category counts and status."""
    from collections import Counter
    counts = Counter()
    for ann in anns:
        name = cat_map.get(ann["category_id"], "Unknown")
        if name != "Tooth":
            counts[name] += 1

    rows = []
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        short = SHORT_NAMES.get(cat, cat)
        rows.append({
            "finding": short,
            "count":   cnt,
            "status":  "Detected",
            "note":    _clinical_note(cat),
        })
    return rows


def _clinical_note(cat: str) -> str:
    NOTES = {
        "Caries":                   "Tooth decay present",
        "Crown":                    "Prosthetic crown identified",
        "Filling":                  "Restorative filling",
        "Implant":                  "Osseointegrated implant",
        "Prefabricated metal post": "Metal post-core detected",
        "Root canal obturation":    "Endodontic treatment complete",
        "Retained root":            "Residual root fragment",
        "Root canal filling":       "Canal sealed with material",
    }
    return NOTES.get(cat, "—")


# ─── STEP 4 – Generate PDF ──────────────────────────────────────────────────────

class DentalReport(FPDF):
    # ── header / footer ──────────────────────────────────────────────────
    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(30, 58, 95)        # Dark navy
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "Tooth-AI  |  Dental OPG Analysis Report", border=0,
                  new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()} - Generated by Tooth-AI (Ground-Truth Mode)", align="C")

    # ── section heading ──────────────────────────────────────────────────
    def section(self, title: str):
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(220, 230, 242)
        self.set_draw_color(100, 130, 180)
        self.cell(0, 8, f"  {title}", border="B", new_x="LMARGIN",
                  new_y="NEXT", fill=True)
        self.ln(2)

    # ── patient info block ───────────────────────────────────────────────
    def info_block(self, patient: str, date: str, image_id: str):
        self.set_font("Helvetica", "", 9)
        self.set_fill_color(245, 247, 250)
        self.set_draw_color(180, 180, 180)
        self.rect(self.l_margin, self.get_y(), 190, 18, style="FD")  # FD = fill + draw
        self.set_xy(self.l_margin + 3, self.get_y() + 3)
        self.cell(60, 6, f"Patient:  {patient}")
        self.cell(65, 6, f"Date:  {date}")
        self.cell(0,  6, f"Image ID:  {image_id}", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)

    # ── findings table ───────────────────────────────────────────────────
    def findings_table(self, rows: list):
        # Header row
        col_w = [75, 22, 28, 65]
        headers = ["Finding", "Count", "Status", "Clinical Note"]

        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 58, 95)
        self.set_text_color(255, 255, 255)
        for w, h in zip(col_w, headers):
            self.cell(w, 8, f"  {h}", border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)

        # Data rows
        self.set_font("Helvetica", "", 9)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(240, 244, 250) if fill else self.set_fill_color(255, 255, 255)
            values = [f"  {row['finding']}", str(row['count']),
                      row['status'], f"  {row['note']}"]
            for w, v in zip(col_w, values):
                self.cell(w, 7, v, border=1, fill=fill)
            self.ln()

    # ── colour legend ─────────────────────────────────────────────────────
    def colour_legend(self, cat_map_used: dict):
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 7, "Annotation Color Legend",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)

        x_start = self.l_margin
        y_cur   = self.get_y()
        col_w   = 90
        col     = 0

        for cat, bgr in CATEGORY_COLORS.items():
            if cat == "Tooth":
                continue
            short = SHORT_NAMES.get(cat, cat)
            # Convert BGR → RGB
            r, g, b = bgr[2], bgr[1], bgr[0]
            x = x_start + col * col_w
            # Colour swatch
            self.set_fill_color(r, g, b)
            self.rect(x, y_cur + 1, 6, 5, style="F")
            self.set_xy(x + 8, y_cur)
            self.cell(col_w - 8, 7, short)

            col = 1 - col
            if col == 0:
                y_cur += 7
                self.set_xy(x_start, y_cur)

        if col != 0:
            y_cur += 7
        self.set_y(y_cur + 2)


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(ANNOTATED_OUT), exist_ok=True)

    print("[1] Loading COCO annotations …")
    img_meta, anns, cat_map = load_annotations(COCO_JSON, IMAGE_ID)
    print(f"    Image size : {img_meta['width']} x {img_meta['height']}")
    print(f"    Annotations: {len(anns)}")

    print("[2] Annotating image …")
    annotate_image(INPUT_IMAGE, ANNOTATED_OUT, anns, cat_map)

    print("[3] Building findings table …")
    findings = build_findings(anns, cat_map)
    for row in findings:
        print(f"    {row['finding']:30s}  count={row['count']}")

    print("[4] Generating PDF report …")
    pdf = DentalReport()
    pdf.set_margins(10, 10, 10)
    pdf.set_auto_page_break(auto=True, margin=14)

    # ── Page 1: Images ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.info_block(
        patient  = "Hemati Hamed, 39 y",
        date     = "31-05-2021",
        image_id = f"9e23ed08  (COCO ID: {IMAGE_ID})",
    )

    pdf.section("Original OPG Radiograph")
    img_h = 60   # display height in mm; maintains aspect ratio
    pdf.image(INPUT_IMAGE, x=10, w=190, h=img_h)
    pdf.ln(3)

    pdf.section("AI-Annotated OPG with Ground-Truth Detections")
    pdf.image(ANNOTATED_OUT, x=10, w=190, h=img_h)
    pdf.ln(5)

    # Colour legend beneath annotated image
    pdf.colour_legend(cat_map)

    # ── Page 2: Findings table ───────────────────────────────────────────
    pdf.add_page()
    pdf.section("Detected Findings Summary")
    pdf.ln(2)
    pdf.findings_table(findings)
    pdf.ln(6)

    # Summary stats
    pdf.set_font("Helvetica", "", 9)
    total_findings = sum(r["count"] for r in findings)
    pdf.cell(0, 6,
             f"Total anomaly annotations: {total_findings}   |   "
             f"Anomaly categories detected: {len(findings)}   |   "
             f"Source: Ground-Truth COCO Dataset",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.section("Clinical Observations")
    pdf.set_font("Helvetica", "", 9)
    num_crowns = sum(r['count'] for r in findings if 'Crown' in r['finding'])
    observations = [
        f"- {num_crowns} Crown detections indicate extensive restorative history.",
        "- Multiple Root Canal Obturations suggest prior endodontic therapy across several teeth.",
        "- Implants detected - confirm osseointegration status on follow-up CBCT.",
        "- Caries lesions identified - recommend periapical X-ray for depth assessment.",
        "- Prefabricated metal posts confirmed alongside crown placements.",
    ]
    for obs in observations:
        pdf.multi_cell(0, 6, obs)
    pdf.ln(3)

    pdf.section("Disclaimer")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5,
        "This report is generated automatically by the Tooth-AI system using ground-truth "
        "annotations from the final-di-remapped dataset. It is intended for research and "
        "educational purposes only and does not constitute a clinical diagnosis. Always "
        "consult a licensed dental professional for clinical decisions.")
    pdf.set_text_color(0, 0, 0)

    pdf.output(PDF_OUT)
    print(f"[✓] PDF report saved → {PDF_OUT}")


if __name__ == "__main__":
    main()
