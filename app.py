#!/usr/bin/env python3
"""ToothAI — Streamlit entry point.

    streamlit run app.py                 # full mode (torch + detectron2 + weights in output/)
    TOOTH_AI_DEMO=1 streamlit run app.py # demo mode: bundled sample cases, no Detectron2

Pages live in tooth_ai/views/, the pipeline in tooth_ai/pipeline.py. The previous
single-file app is kept as app_legacy.py (and at git tag paper-eval-v1).
"""

import streamlit as st

from tooth_ai.ui import components as ui
from tooth_ai.ui import theme
from tooth_ai.views import about, analyze, method, performance

st.set_page_config(
    page_title="ToothAI",
    page_icon=str(theme.BRAND / "favicon-32.png"),
    layout="wide",
    initial_sidebar_state="auto",  # expanded on desktop, collapsed on phones
    menu_items={"About": "ToothAI — research prototype. Not for diagnostic use."},
)
st.logo(str(theme.BRAND / "tooth-ai-lockup-horizontal.svg"), size="large",
        icon_image=str(theme.BRAND / "tooth-ai-mark.svg"))
ui.inject_css()

page = st.navigation(
    [
        st.Page(analyze.render, title="Analyse", icon=":material/radiology:", url_path="analyze", default=True),
        st.Page(method.render, title="How it works", icon=":material/account_tree:", url_path="how-it-works"),
        st.Page(performance.render, title="Performance", icon=":material/monitoring:", url_path="performance"),
        st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
    ],
    position="top",
)
page.run()
