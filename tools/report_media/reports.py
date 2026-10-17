#!/usr/bin/env python3
"""Build example OPG PDF reports from the bundled de-identified samples (demo detections, engine v2).

    python tools/report_media/reports.py report_media

Writes 02_reports/opg_report_<name>_v2.pdf and, when PyMuPDF is importable, a 150 dpi PNG per page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tooth_ai import demo  # noqa: E402
from tooth_ai.render import render_overlay  # noqa: E402
from tooth_ai.report_pdf import build_pdf  # noqa: E402

CASES = [("sample_02", "restorations", "Sample 02"), ("sample_03", "partial_arches", "Sample 03")]


def main():
    out = Path(sys.argv[1]) / "02_reports"
    out.mkdir(parents=True, exist_ok=True)
    try:
        import fitz
    except ImportError:
        fitz = None
    for sample, name, case_id in CASES:
        result = demo.analyze_sample(sample, mode="Full chart", engine="v2")
        image = np.array(Image.open(demo.sample_image_path(sample)).convert("RGB"))
        overlay = render_overlay(image, result, layers=("Masks", "Numbers", "Findings"),
                                 color_by="Finding", opacity=0.35, notation="FDI")
        pdf = out / f"opg_report_{name}_v2.pdf"
        pdf.write_bytes(build_pdf(result, overlay, case_id=case_id, notation="FDI"))
        print(pdf)
        if fitz:
            for old in out.glob(f"{pdf.stem}_page*.png"):
                old.unlink()
            with fitz.open(pdf) as doc:
                for i, page in enumerate(doc, 1):
                    page.get_pixmap(dpi=150).save(out / f"{pdf.stem}_page{i}.png")


if __name__ == "__main__":
    main()
