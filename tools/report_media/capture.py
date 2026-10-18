#!/usr/bin/env python3
"""Capture the report's diagrams and app screenshots with a local headless Chromium.

    python tools/report_media/capture.py diagrams report_media
    python tools/report_media/capture.py app report_media        # needs the app (:8502) and photo API (:8001)
    TOOTHAI_APP_URL=https://toothai.duckdns.org python tools/report_media/capture.py app report_media   # live site

Screenshots use only bundled de-identified sample OPGs and the CC BY AKU sample photos.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdp import Browser  # noqa: E402

HERE = Path(__file__).resolve().parent
APP = os.environ.get("TOOTHAI_APP_URL", "http://localhost:8502").rstrip("/")
FIGURES = ["architecture", "opg-engine", "photo-pipeline", "deployment", "evaluation"]


async def diagrams(out: Path):
    async with Browser(width=1300, height=900) as b:
        await b.send("Page.navigate", url=(HERE / "diagrams.html").as_uri())
        await asyncio.sleep(2.5)
        for i, fid in enumerate(FIGURES, 1):
            p = await b.element_shot(f"document.getElementById('{fid}')", out / "03_diagrams" / f"diagram_{i}_{fid}.png",
                                     pad=0)
            print("diagram", p.name)


async def app(out: Path):
    s = out / "01_screenshots"
    async with Browser(width=1440, height=900) as b:
        # --- Analyse: the three bundled sample cases, improved engine (the default)
        await b.goto(APP + "/", wait_text="Sample cases")
        await b.shot(s / "opg_00_landing.png", full=False)
        cases = [("Complete dentition", "opg_01_complete_dentition"),
                 ("Complete arches with restorations", "opg_02_restorations"),
                 ("Partial arches with restorations", "opg_03_partial_arches")]
        for label, name in cases:
            await b.click(label, exact=False)
            await b.idle(wait_text="Teeth", timeout=120)
            await b.shot(s / f"{name}_v2_full.png")
            await b.shot(s / f"{name}_v2_top.png", full=False)
        await b.element_shot("document.querySelector('[data-testid=\"stPlotlyChart\"]')",
                             s / "opg_03_partial_arches_v2_viewer.png", pad=4)
        # the partial case with each engine (the reliability panel explains the difference)
        await b.click("Published")
        await b.idle(wait_text="Teeth")
        await b.shot(s / "opg_03_partial_arches_v1_top.png", full=False)
        await b.click("Improved")
        # layers: colour by finding, geometry overlay
        for layer, name in (("Geometry", "opg_04_geometry_layer"),):
            try:
                await b.click(layer)
                await b.shot(s / f"{name}.png", full=False)
                await b.click(layer)
            except LookupError:
                print("skipped", name)
        # --- Photo screening with the AKU sample photos
        await b.goto(APP + "/photo-screening", wait_text="Photo screening")
        await b.shot(s / "photo_00_landing.png", full=False)
        await b.tick("Process my photos")
        await b.click("Use sample photos")
        await b.click("Analyse photos")
        await b.idle(wait_text="Screening index", timeout=600)
        await b.shot(s / "photo_01_results_full.png")
        await b.shot(s / "photo_01_results_top.png", full=False)
        for tab in ("Lower teeth", "Upper teeth", "Front", "Left side"):
            try:
                await b.tab(tab)
                await b.element_shot("[...document.querySelectorAll('[role=tabpanel]')].find(e => e.offsetParent)",
                                     s / f"photo_02_view_{tab.split()[0].lower()}.png")
            except LookupError:
                print("skipped tab", tab)
        # --- Information pages
        for path, name in (("how-it-works", "page_how_it_works"), ("performance", "page_performance"),
                           ("about", "page_about")):
            await b.goto(f"{APP}/{path}", wait_text="ToothAI")
            await b.shot(s / f"{name}.png")
        # --- Phone layout
        await b.viewport(390, 844, mobile=True)
        b.width, b.height = 390, 844
        await b.goto(APP + "/", wait_text="Analyse")
        await b.shot(s / "mobile_analyse.png", full=False)
        await b.goto(APP + "/photo-screening", wait_text="Photo screening")
        await b.shot(s / "mobile_photo_screening.png", full=False)
    print("screenshots in", s)


def main():
    what, out = sys.argv[1], Path(sys.argv[2])
    asyncio.run(diagrams(out) if what == "diagrams" else app(out))


if __name__ == "__main__":
    main()
