"""ToothAI design tokens, from Brand Guidelines v1.0 (Oct 2026).

Mirrored in assets/brand/tokens.css and .streamlit/config.toml. Ratio: about 60 navy/paper,
30 ice/slate, 10 spark. Spark Blue is the only accent; use it sparingly so it keeps signalling insight.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
BRAND = ASSETS / "brand"
FONTS = ROOT / "static" / "fonts"   # served by Streamlit static serving and embedded in PDFs

# Brand palette (guidelines names)
NAVY = "#0E2236"          # Radiograph Navy: primary ink, dark grounds
SPARK = "#2D6FDB"         # Spark Blue: the spark, "AI", actions
SPARK_LIGHT = "#7FA9F5"   # Spark Light: the spark on dark
ICE = "#EAF1FC"           # tints, surfaces
PAPER = "#F7F9FB"         # backgrounds
SLATE = "#5B6B78"         # secondary text
LINE = "#E1E6EC"          # hairlines

# Working names used across the app (kept so every import keeps working)
INK_900 = NAVY
INK_700 = "#33475B"
INK_500 = SLATE
SLATE_300 = "#C9D4E0"
SLATE_100 = "#E9EDF1"
BONE_50 = PAPER
WHITE = "#FFFFFF"
VIEWER_950 = "#0A0E13"
SAFFRON_500 = SPARK       # the accent (formerly saffron): never inside the viewer or the chart

# Semantic colours: viewer and odontogram only, always paired with a label or pattern.
FINDING_COLORS = {
    "Caries": "#E5484D",
    "Retained root": "#C2410C",
    "Filling": "#3B82F6",
    "Crown": "#1D4ED8",
    "Root canal filling": "#0F9F8F",
    "Root canal obturation": "#0F9F8F",
    "Implant": "#64748B",
    "Prefabricated metal post": "#94A3B8",
}
FINDING_GROUPS = {
    "Caries": "Pathology", "Retained root": "Pathology",
    "Filling": "Restoration", "Crown": "Restoration",
    "Root canal filling": "Endodontic", "Root canal obturation": "Endodontic",
    "Implant": "Prosthetic", "Prefabricated metal post": "Prosthetic",
}
TOOTH_FILL = "#8FB8DE"          # neutral tooth tint in "colour by finding" mode
UNCERTAIN = "#CA8A04"
MISSING = "#9AA5B1"
OK = "#16A34A"
ALERT = "#C2410C"
# Status banners (reliability, triage); mirrored in styles.css .ta-banner
LEVEL_COLORS = {"ok": OK, "caution": UNCERTAIN, "alert": ALERT, "na": MISSING}
QUADRANT_COLORS = {1: "#7FB3D5", 2: "#9BC79B", 3: "#E3B679", 4: "#C9A0C9"}

LOW_SCORE = 0.50                 # tooth detections below this are shown as uncertain


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def font_path(name="PlusJakartaSans-Regular.ttf"):
    return FONTS / name


PARTNER_LOGOS = (
    ("logo_aicte.png", "AICTE"),
    ("logo_bvu.png", "Bharati Vidyapeeth"),
    ("logo_ccamp.png", "C-CAMP"),
    ("logo_cibip.png", "C-CAMP Inter-Institutional Biomedical Innovations Programme"),
)
FDI_ROW_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_ROW_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
SEVERITY = ["Caries", "Retained root", "Root canal filling", "Root canal obturation",
            "Crown", "Filling", "Implant", "Prefabricated metal post"]
