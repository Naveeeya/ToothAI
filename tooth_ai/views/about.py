"""About: the project, institutions, data, citation."""

import streamlit as st

import tooth_ai
from tooth_ai.ui import components as ui


def render():
    ui.page_header("About ToothAI", tooth_ai.TAGLINE)
    st.html(f'<p style="font-size:1.05rem;max-width:56rem;color:#33475B;margin:.2rem 0 1.2rem">'
            f'<b>{ui.esc(tooth_ai.FULL_TITLE)}</b></p>')
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.subheader("What it is")
        st.markdown(
            "ToothAI reads a panoramic dental radiograph (OPG) and returns a mask for every tooth, its FDI "
            "number, the findings on it and a dental chart, with a PDF report. Detection is learned; numbering "
            "is geometric, so every number can be explained and every case carries a reliability flag.\n\n"
            "**Photo screening** (beta) is the low-cost front door: smartphone photos of the teeth are "
            "checked for quality and screened for visible decay, wear and restorations, with a plain triage "
            "and a recommendation for an OPG when something is seen. It is a screening aid, not a diagnosis.\n\n"
            "It is built for teaching and research at dental colleges, and as a second read that a dentist "
            "reviews. It is not a medical device.")
        st.subheader("Data")
        st.markdown(
            "Trained and evaluated on public panoramic radiograph datasets, including the orthopantomogram "
            "teeth segmentation and numbering dataset of Adnan *et al.*, *Data in Brief* (2024), used under "
            "CC BY-NC 4.0 (non-commercial). Sample cases in this app are de-identified validation images. "
            "Photo screening uses AlphaDent weights (Apache-2.0) and models trained on the AKU intraoral caries "
            "and DentalAI datasets (CC BY 4.0).")
        st.subheader("Cite")
        st.code(f"{tooth_ai.FULL_TITLE}. Bharati Vidyapeeth (Deemed to be University), Pune, 2026. "
                f"Software version {tooth_ai.__version__}.", language=None, wrap_lines=True)
    with b:
        st.subheader("Institutions")
        st.markdown("\n".join(f"- {i}" for i in tooth_ai.INSTITUTIONS))
        st.markdown(tooth_ai.PROGRAMME)
        ui.partner_strip()
        st.subheader("Disclaimer")
        st.warning(tooth_ai.DISCLAIMER, icon=":material/health_and_safety:")
    ui.footer(logos=False)            # the partner strip is already shown above
