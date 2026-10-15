"""Finding colours shared by the service's renderer and the Streamlit page (no third-party imports)."""

from tooth_ai.ui import theme

# Caries, filling and crown reuse the OPG viewer's semantic colours (tooth_ai/ui/theme.py).
COLORS = {
    "caries": theme.FINDING_COLORS["Caries"],
    "cavity": "#C2410C",
    "crack": "#7C3AED",
    "wear": "#0F9F8F",
    "filling": theme.FINDING_COLORS["Filling"],
    "crown": theme.FINDING_COLORS["Crown"],
}
ORDER = ("cavity", "crack", "caries", "wear", "filling", "crown")      # severity, most urgent first
