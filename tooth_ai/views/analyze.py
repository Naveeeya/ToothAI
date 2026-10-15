"""Analyze: upload or pick a radiograph, view the overlay, chart and findings, export."""

import base64
import hashlib
import io
import traceback
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from tooth_ai.analysis import (DEFAULT_FINDINGS_THRESHOLD, DEFAULT_TEETH_THRESHOLD, MODEL_VERSION,
                               MODES, analyze)
from tooth_ai.demo import analyze_sample, demo_requested, detectron2_available, list_samples, sample_image_path
from tooth_ai.notation import label as tooth_label, tooth_name
from tooth_ai.render import DEFAULT_LAYERS, LAYERS, render_overlay
from tooth_ai.report_md import region, to_json, to_markdown
from tooth_ai.ui import components as ui
from tooth_ai.ui import theme

VIEWER_MAX_W = 1600
MAX_CACHED = 6


# ------------------------------------------------------------------ model loading

@st.cache_resource(show_spinner=False)
def load_model(config_path, weights_path):
    """Detectron2 predictor (cached for the life of the server process)."""
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    device, device_name = "cpu", "CPU"
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.zeros(1, device="cuda")
            device, device_name = "cuda", torch.cuda.get_device_name(0)
        except RuntimeError:
            device_name = "CPU (CUDA unavailable)"
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # low; filtered downstream
    return DefaultPredictor(cfg), device_name


def _missing_weights(mode):
    cfg = MODES[mode]
    paths = [cfg["config"], cfg["weights"]] + [cfg[k] for k in ("pathology_config", "pathology_weights") if k in cfg]
    return [str(p) for p in paths if not p.exists()]


def _predictors(mode):
    cfg = MODES[mode]
    primary, device = load_model(cfg["config"], cfg["weights"])
    pathology = None
    if "pathology_config" in cfg:
        pathology, _ = load_model(cfg["pathology_config"], cfg["pathology_weights"])
    return {"primary": primary, "pathology": pathology}, device


# ------------------------------------------------------------------ helpers

def _image_uri(rgb, max_w=VIEWER_MAX_W):
    h, w = rgb.shape[:2]
    scale = 1.0
    img = Image.fromarray(rgb)
    if w > max_w:
        scale = w / max_w
        img = img.resize((max_w, round(h / scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), scale


def viewer_figure(rgb, result, notation, height):
    h, w = rgb.shape[:2]
    uri, scale = _image_uri(rgb)
    fig = go.Figure(go.Image(source=uri, x0=0, y0=0, dx=scale, dy=scale, hoverinfo="skip"))
    if result.teeth:
        xs = [(t["box"][0] + t["box"][2]) / 2 for t in result.teeth]
        ys = [(t["box"][1] + t["box"][3]) / 2 for t in result.teeth]
        text = []
        for t in result.teeth:
            f = ", ".join(dict.fromkeys(x["type"] for x in t["findings"])) or "No findings"
            text.append(f"<b>{tooth_label(t['fdi'], notation)}</b> · {tooth_name(t['fdi'])}"
                        f"<br>Detection score {t['score']:.2f}<br>{f}")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", customdata=[t["fdi"] for t in result.teeth],
            marker=dict(size=26, color="rgba(255,255,255,0.001)", line=dict(width=0)),
            hovertext=text, hoverinfo="text", hoverlabel=dict(bgcolor=theme.INK_900, font_color="white",
                                                             font_family="Plus Jakarta Sans"),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
            showlegend=False))
    fig.update_xaxes(visible=False, range=[0, w], constrain="domain")
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x", constrain="domain")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      dragmode="zoom",  # pan mode suppresses click-to-select in plotly.js
                      paper_bgcolor=theme.VIEWER_950, plot_bgcolor=theme.VIEWER_950,
                      clickmode="event+select", hovermode="closest")
    return fig


def _cache_put(key, value):
    cache = st.session_state.setdefault("ta_results", {})
    cache[key] = value
    while len(cache) > MAX_CACHED:
        cache.pop(next(iter(cache)))


def _select(fdi):
    st.session_state["ta_selected"] = fdi


def _consume_events(result):
    """Selection can come from the viewer, the teeth table or the picker; newest wins."""
    seen = st.session_state.setdefault("ta_seen", {})
    ev = st.session_state.get("ta_viewer")
    pts = []
    if ev:
        sel = ev.get("selection") if isinstance(ev, dict) else getattr(ev, "selection", None)
        pts = (sel or {}).get("points", []) if sel else []
    sig = tuple(p.get("customdata") for p in pts)
    if pts and sig != seen.get("viewer"):
        _select(int(pts[0]["customdata"]))
    seen["viewer"] = sig

    tbl = st.session_state.get("ta_teeth_table")
    rows = []
    if tbl:
        sel = tbl.get("selection") if isinstance(tbl, dict) else getattr(tbl, "selection", None)
        rows = (sel or {}).get("rows", []) if sel else []
    sig = tuple(rows)
    if rows and sig != seen.get("table") and rows[0] < len(result.teeth):
        _select(result.teeth[rows[0]]["fdi"])
    seen["table"] = sig


def _default_selection(result):
    fdis = [t["fdi"] for t in result.teeth]
    sel = st.session_state.get("ta_selected")
    if sel in fdis:
        return sel
    with_findings = [t["fdi"] for t in result.teeth if t["findings"]]
    sel = (with_findings or fdis or [None])[0]
    st.session_state["ta_selected"] = sel
    return sel


def _teeth_frame(result, notation):
    return pd.DataFrame([{
        "Tooth": tooth_label(t["fdi"], notation),
        "Name": tooth_name(t["fdi"]),
        "Findings": ", ".join(dict.fromkeys(f["type"] for f in t["findings"])) or "—",
        "Score": round(t["score"], 3),
        "Numbering": ("gap skip" if t["trace"].get("skipped") else
                      "conflict moved" if t["trace"].get("reassigned_from") else "in sequence"),
    } for t in result.teeth])


def _findings_frame(result, notation):
    rows = []
    for t in result.teeth:
        for f in t["findings"]:
            rows.append({"Finding": f["type"], "Tooth": tooth_label(t["fdi"], notation),
                         "Score": round(f["score"], 3),
                         "Note": "near metal, lower confidence" if f.get("near_metal") else ""})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ page

def render():
    demo = demo_requested() or not detectron2_available()
    samples = list_samples()

    # ---------------- sidebar: the case
    with st.sidebar:
        st.html('<div class="ta-eyebrow">Case</div>')
        upload = st.file_uploader(
            "Panoramic radiograph (OPG)", type=["jpg", "jpeg", "png", "bmp"], key="ta_upload",
            disabled=demo,
            help="Demo mode analyses the bundled sample cases only." if demo else
                 "JPG, PNG or BMP. Images are processed in memory and not stored.")
        sample = None
        if samples:
            sample = st.pills("Sample cases", [s["name"] for s in samples], key="ta_sample",
                              format_func=lambda n: next(s["description"] for s in samples if s["name"] == n),
                              help="De-identified validation images with the model's own cached detections.")
        case_id = st.text_input("Case ID", key="ta_case", max_chars=40,
                                placeholder="Optional. No patient names.")
        mode = st.segmented_control("Analysis", list(MODES), default="Full chart", key="ta_mode",
                                    help="Full chart numbers every tooth and maps findings to it. "
                                         "Findings only skips numbering.") or "Full chart"
        with st.expander("Detection thresholds", icon=":material/tune:"):
            teeth_t = st.slider("Tooth score", 0.10, 0.50, DEFAULT_TEETH_THRESHOLD, 0.05, key="ta_tt",
                                help="Minimum detection score for a tooth. Lower finds more teeth "
                                     "and more false ones.")
            find_t = st.slider("Finding score", 0.10, 0.80, DEFAULT_FINDINGS_THRESHOLD, 0.05, key="ta_ft",
                               help="Minimum detection score for a finding.")
        notation = st.segmented_control("Tooth notation", ["FDI", "Universal"], default="FDI",
                                        key="ta_notation") or "FDI"
        st.caption(f"{'Demo mode · cached detections' if demo else MODEL_VERSION}")

    ui.page_header("Analyse", "Detection · segmentation · numbering · anatomical mapping",
                   badge="Research prototype", demo=demo)

    # ---------------- input
    image_rgb, source_key = None, None
    if upload is not None and not demo:
        data = upload.getvalue()
        try:
            image_rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:
            st.error("This file could not be read as an image. Use JPG, PNG or BMP.", icon=":material/broken_image:")
            ui.footer()
            return
        source_key = "upload:" + hashlib.sha1(data).hexdigest()
    elif sample:
        image_rgb = cv2.cvtColor(cv2.imread(str(sample_image_path(sample))), cv2.COLOR_BGR2RGB)
        source_key = "sample:" + sample

    if image_rgb is None:
        ui.empty_state()
        if demo:
            st.info("Demo mode: Detectron2 or the model weights aren't available here, so you can open the "
                    "sample cases on the left. They run through the real numbering engine.",
                    icon=":material/info:")
        ui.footer()
        return

    if not demo:
        missing = _missing_weights(mode)
        if missing:
            with st.container(border=True):
                st.markdown("**Model weights not found**")
                st.markdown("ToothAI expects these files, relative to the folder you launched from:")
                st.code("\n".join(missing))
                st.caption("Run `streamlit run app.py` from the repository root, or start with "
                           "`TOOTH_AI_DEMO=1` to use the sample cases.")
            ui.footer()
            return

    h, w = image_rgb.shape[:2]
    if w / h < 1.6:
        st.warning("This doesn't look like a panoramic radiograph (it's not wide enough). "
                   "ToothAI is built for OPGs; results on other images are not meaningful.",
                   icon=":material/warning:")

    # ---------------- analysis (cached per image + settings; display changes never re-run it)
    key = (source_key, mode, round(teeth_t, 2), round(find_t, 2))
    result = st.session_state.get("ta_results", {}).get(key)
    if result is None:
        if sample and source_key.startswith("sample:"):
            result = analyze_sample(sample, mode, teeth_t, find_t)
        else:
            status = st.status("Analysing radiograph", expanded=True)
            try:
                status.write("Loading models")
                predictors, _ = _predictors(mode)
                status.write("Enhancing contrast")
                result = analyze(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), mode, predictors,
                                 teeth_t, find_t, on_step=status.write)
                status.update(label=f"Analysis complete in {result.total_seconds:.1f} s",
                              state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Analysis failed", state="error", expanded=True)
                st.error("The analysis could not finish. Check the image and try again.",
                         icon=":material/error:")
                with st.expander("Technical details"):
                    st.code("".join(traceback.format_exception(exc)))
                ui.footer()
                return
        _cache_put(key, result)

    _consume_events(result)
    selected = _default_selection(result)

    # ---------------- layout
    left, right = st.columns([7, 5], gap="medium")
    with left:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            view = st.segmented_control("View", ["Overlay", "Original", "Compare"], default="Overlay",
                                        key="ta_view", label_visibility="collapsed") or "Overlay"
            layers = st.pills("Layers", LAYERS, selection_mode="multi", default=list(DEFAULT_LAYERS),
                              key="ta_layers", label_visibility="collapsed")
            st.space("stretch")
            display = st.popover("Display", icon=":material/palette:")
        with display:
            color_by = st.segmented_control("Colour teeth by", ["Finding", "Quadrant"], default="Finding",
                                            key="ta_colorby") or "Finding"
            opacity = st.slider("Mask opacity", 0.1, 0.8, 0.35, 0.05, key="ta_opacity")

        overlay = render_overlay(image_rgb, result, layers=layers or [], color_by=color_by,
                                 opacity=opacity, notation=notation, selected=selected)
        if view == "Compare":
            a, b = st.columns(2, gap="small")
            a.image(image_rgb, caption="Original", width="stretch")
            b.image(overlay, caption="ToothAI", width="stretch")
        else:
            shown = overlay if view == "Overlay" else image_rgb
            st.plotly_chart(viewer_figure(shown, result, notation, height=int(min(720, 820 * h / w))),
                            key="ta_viewer", on_select="rerun", selection_mode="points",
                            config={"displaylogo": False, "scrollZoom": False, "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage", "autoScale2d"]})
        kinds = {f["type"] for t in result.teeth for f in t["findings"]} | {a["class_name"] for a in result.findings}
        ui.legend(kinds, color_by)
        st.caption("Drag to zoom into a region, double-click to reset, use the hand tool to pan. Click a tooth to inspect it."
                   + ("  Demo: the teeth model's cached single-pass detections on a validation image."
                      if result.source == "demo" else ""))

    with right:
        ui.reliability_banner(result.reliability)
        upper, lower = result.jaw_counts()
        ui.kpi_row([
            (f"{len(result.teeth)}<small> / 32</small>", "Teeth detected"),
            (f"{len(result.findings)}", "Findings"),
            (f"{upper}<small>/</small>{lower}", "Upper / lower jaw"),
            (f"{result.total_seconds:.1f}<small> s</small>" if result.source == "model" else "—",
             "Analysis time"),
        ])
        if result.teeth:
            ui.odontogram_svg(result, selected, notation)
            fdis = [t["fdi"] for t in result.teeth]
            st.session_state["ta_pick"] = selected
            st.selectbox("Inspect tooth", fdis, key="ta_pick",
                         format_func=lambda f: f"{tooth_label(f, notation)} · {tooth_name(f)}",
                         on_change=lambda: _select(st.session_state["ta_pick"]))
            tooth = result.tooth(selected)
            if tooth:
                with st.container(border=True):
                    ui.tooth_detail(tooth, result, notation)
        elif MODES[mode]["key"] == "findings":
            st.info("Findings only mode: teeth are not numbered, so findings are listed by image region.",
                    icon=":material/info:")

    # ---------------- tables
    t_teeth, t_find, t_un = st.tabs([f"Teeth ({len(result.teeth)})",
                                     f"Findings on teeth ({sum(len(t['findings']) for t in result.teeth)})",
                                     f"Not mapped to a tooth ({len(result.unassigned)})"])
    with t_teeth:
        if result.teeth:
            st.dataframe(_teeth_frame(result, notation), hide_index=True, key="ta_teeth_table",
                         on_select="rerun", selection_mode="single-row",
                         column_config={"Score": st.column_config.ProgressColumn(
                             "Detection score", min_value=0.0, max_value=1.0, format="%.2f")})
        else:
            st.caption("No teeth in this analysis.")
    with t_find:
        df = _findings_frame(result, notation)
        if len(df):
            st.dataframe(df, hide_index=True, column_config={"Score": st.column_config.ProgressColumn(
                "Detection score", min_value=0.0, max_value=1.0, format="%.2f")})
        else:
            st.caption("No findings mapped to teeth.")
    with t_un:
        if result.unassigned:
            st.dataframe(pd.DataFrame([{"Finding": a["class_name"], "Region": region(a["box"], w),
                                        "Score": round(a["score"], 3)} for a in result.unassigned]),
                         hide_index=True)
        else:
            st.caption("Every finding was mapped to a tooth.")

    # ---------------- exports
    st.html('<div class="ta-eyebrow" style="margin-top:1rem">Report</div>')
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    base = f"ToothAI_{(case_id or 'case').strip().replace(' ', '-')}_{stamp}"

    def pdf_bytes():
        from tooth_ai.report_pdf import build_pdf
        export = render_overlay(image_rgb, result, layers=("Masks", "Numbers", "Findings"),
                                color_by="Finding", opacity=0.35, notation=notation)
        return build_pdf(result, export, case_id=case_id, notation=notation)

    def png_bytes():
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, "PNG")
        return buf.getvalue()

    b1, b2, b3, b4, _ = st.columns([1.3, 1, 1.3, 1.2, 3])
    b1.download_button("PDF report", pdf_bytes, f"{base}.pdf", "application/pdf", type="primary",
                       icon=":material/picture_as_pdf:", on_click="ignore", width="stretch")
    b2.download_button("JSON", lambda: to_json(result, case_id), f"{base}.json", "application/json",
                       icon=":material/data_object:", on_click="ignore", width="stretch")
    b3.download_button("Annotated PNG", png_bytes, f"{base}.png", "image/png",
                       icon=":material/image:", on_click="ignore", width="stretch")
    b4.download_button("Markdown", lambda: to_markdown(result, case_id, notation=notation), f"{base}.md",
                       "text/markdown", icon=":material/description:", on_click="ignore", width="stretch")
    ui.footer()
