#!/usr/bin/env python3
"""Brand Streamlit's HTML shell (run once at image build time).

    python docker/brand_shell.py https://<site>

Streamlit serves a fixed index.html before the app script runs: the tab says "Streamlit", the favicon is
Streamlit's, and shared links preview as "Streamlit" with no image. This rewrites the title and favicon and
adds a description, theme colour and Open Graph / Twitter tags pointing at the brand image, which is copied
into ./static (served at /app/static/ because server.enableStaticServing is on).
"""

from __future__ import annotations

import html
import shutil
import sys
from pathlib import Path

import streamlit

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "assets" / "brand"
TITLE = "ToothAI"
DESCRIPTION = ("Tooth segmentation, FDI numbering and findings on panoramic dental X-rays, plus a smartphone photo "
               "screening aid. Research prototype, not for diagnostic use.")


def main(site: str) -> None:
    site = site.rstrip("/")
    shell = Path(streamlit.__file__).parent / "static"
    index = shell / "index.html"
    page = index.read_text(encoding="utf-8")
    if "<title>Streamlit</title>" not in page:
        raise SystemExit("unexpected Streamlit index.html: <title>Streamlit</title> not found")

    shutil.copyfile(BRAND / "favicon-32.png", shell / "favicon.png")
    (ROOT / "static").mkdir(exist_ok=True)
    shutil.copyfile(BRAND / "og-1200x630.png", ROOT / "static" / "og.png")
    shutil.copyfile(BRAND / "icon-192.png", ROOT / "static" / "icon-192.png")

    t, d, image = html.escape(TITLE), html.escape(DESCRIPTION), html.escape(f"{site}/app/static/og.png")
    meta = f"""<title>{t}</title>
    <meta name="description" content="{d}" />
    <meta name="theme-color" content="#0E2236" />
    <link rel="apple-touch-icon" href="./app/static/icon-192.png" />
    <meta property="og:type" content="website" />
    <meta property="og:site_name" content="{t}" />
    <meta property="og:title" content="{t}" />
    <meta property="og:description" content="{d}" />
    <meta property="og:url" content="{html.escape(site)}/" />
    <meta property="og:image" content="{image}" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta property="og:image:alt" content="ToothAI logo" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="{t}" />
    <meta name="twitter:description" content="{d}" />
    <meta name="twitter:image" content="{image}" />"""
    index.write_text(page.replace("<title>Streamlit</title>", meta, 1), encoding="utf-8")
    print(f"branded {index} for {site}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501")
