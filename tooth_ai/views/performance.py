"""Performance: measured numbers only, read live from paper_eval/results."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tooth_ai import evidence as ev
from tooth_ai.evidence import pct
from tooth_ai.ui import components as ui
from tooth_ai.ui import theme

COMPLETE, PARTIAL, BASELINE = theme.SPARK, theme.SPARK_LIGHT, "#A9B8C8"  # brand: one hue in two shades + neutral


def _stat(value, ci, label, n):
    ci_txt = f"95% CI {pct(ci[0])} – {pct(ci[1])}" if ci and ci[0] is not None else "&nbsp;"
    st.html(f'<div class="ta-stat"><div class="v">{value}</div><div class="ci">{ci_txt}</div>'
            f'<div class="l">{ui.esc(label)}</div><div class="n">{ui.esc(n)}</div></div>')


def _bars(rows, height):
    """rows: list of (category, series, value, lo, hi, colour). Horizontal bars with CI whiskers."""
    fig = go.Figure()
    for series in dict.fromkeys(r[1] for r in rows):
        sub = [r for r in rows if r[1] == series]
        fig.add_trace(go.Bar(
            y=[r[0] for r in sub], x=[r[2] * 100 for r in sub], name=series, orientation="h",
            marker=dict(color=[r[5] for r in sub], cornerradius=4, line=dict(color="white", width=2)),
            error_x=dict(type="data", symmetric=False, color=theme.INK_500, thickness=1.4, width=5,
                         array=[(r[4] - r[2]) * 100 if r[4] is not None else 0 for r in sub],
                         arrayminus=[(r[2] - r[3]) * 100 if r[3] is not None else 0 for r in sub]),
            # Labels sit inside the bar's start so they never collide with the CI whisker.
            text=[f"{r[2] * 100:.1f}%" for r in sub], textposition="inside", insidetextanchor="start",
            textfont=dict(color=["white" if r[5] == COMPLETE else theme.INK_900 for r in sub],
                          family="JetBrains Mono", size=12),
            hovertemplate=[f"<b>{r[0]}</b> · {series}<br>{r[2] * 100:.1f}%"
                           + (f" (95% CI {r[3] * 100:.1f}–{r[4] * 100:.1f}%)" if r[3] is not None else "")
                           + "<extra></extra>" for r in sub]))
    fig.update_layout(
        barmode="group", bargap=0.35, bargroupgap=0.08, height=height,
        margin=dict(l=8, r=40, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color=theme.INK_500)),
        font=dict(family="Plus Jakarta Sans", color=theme.INK_900), hoverlabel=dict(font_family="Plus Jakarta Sans"))
    fig.update_xaxes(range=[0, 105], ticksuffix="%", gridcolor=theme.SLATE_100, zeroline=False,
                     tickfont=dict(color=theme.INK_500))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=theme.INK_900))
    return fig


def render():
    ui.page_header("Performance", "Measured on held-out radiographs. Every number has a sample size "
                   "and a 95% confidence interval.", badge="paper_eval")

    s = ev.deployed_summary()
    ap = ev.coco_ap()
    v2 = ev.engine_v2()
    new = v2.get("v2")

    def ci(r, key):
        return (r[key + "_lo"], r[key + "_hi"])

    if new:
        st.html('<div class="ta-eyebrow">As deployed in this app (improved engine, one pass)</div>')
        c = st.columns(4)
        with c[0]:
            _stat(pct(new["recall"]), ci(new, "recall"), "Tooth detection recall · full arches",
                  f"{new['recall_n_images']:.0f} test images")
        with c[1]:
            _stat(pct(new["fdi"]), ci(new, "fdi"), "FDI number correct · all", f"{new['fdi_n_images']:.0f} images")
        with c[2]:
            _stat(pct(new["complete"]), ci(new, "complete"), "FDI number correct · complete arches",
                  f"{new['complete_n_images']:.0f} images")
        with c[3]:
            _stat(pct(new["partial"]), ci(new, "partial"), "FDI number correct · partial arches",
                  f"{new['partial_n_images']:.0f} images")
    else:   # engine_v2.csv absent: the published engine's numbers
        rec, rec_ci, n = ev.metric(ev.DEPLOYED, ev.SUBSET_ALL, "detection_recall")
        jaw, jaw_ci, _ = ev.metric(ev.DEPLOYED, ev.SUBSET_ALL, "jaw_acc_given_detected")
        full, full_ci, n_full = ev.metric(ev.DEPLOYED, ev.SUBSET_COMPLETE, "fdi_acc_joint")
        part, part_ci, n_part = ev.metric(ev.DEPLOYED, ev.SUBSET_PARTIAL, "fdi_acc_joint")
        st.html('<div class="ta-eyebrow">Published engine (two passes)</div>')
        c = st.columns(4)
        with c[0]:
            _stat(pct(rec), rec_ci, "Tooth detection recall", f"{n} test images")
        with c[1]:
            _stat(pct(jaw), jaw_ci, "Correct jaw, given detected", f"{n} test images")
        with c[2]:
            _stat(pct(full), full_ci, "FDI number correct · complete arches", f"{n_full} images")
        with c[3]:
            _stat(pct(part), part_ci, "FDI number correct · partial arches", f"{n_part} images")

    st.markdown("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Numbering depends on arch completeness")
        rows = []
        if new:
            for key, name, col in (("complete", "Complete arches", COMPLETE), ("partial", "Partial arches", PARTIAL)):
                rows.append(("Improved engine (default)", name, new[key], *ci(new, key), col))
        for cfg in (ev.DEPLOYED, ev.BEST):
            for subset, name, col in ((ev.SUBSET_COMPLETE, "Complete arches", COMPLETE),
                                      (ev.SUBSET_PARTIAL, "Partial arches", PARTIAL)):
                v, (lo, hi), _ = ev.metric(cfg, subset, "fdi_acc_joint")
                if v is not None:
                    rows.append((ev.CONFIG_LABELS[cfg], name, v, lo, hi, col))
        st.plotly_chart(_bars(rows, 330), config={"displayModeBar": False}, key="perf_arch")
        st.caption("End-to-end FDI accuracy: a tooth counts only if it is detected and numbered correctly. "
                   "The published engine numbers teeth in order along each jaw, so one missing tooth shifts "
                   "every number after it; the improved engine numbers by position from the midline, which "
                   "holds up far better on partial arches. ToothAI flags partial arches on every case.")
    with right:
        st.subheader("Geometry vs a numbering network")
        d, lo, hi, n_pair = ev.geometry_vs_network()
        g, g_ci, _ = ev.metric(ev.BEST, ev.SUBSET_ALL, "fdi_acc_joint")
        nn, nn_ci, _ = ev.metric(ev.NETWORK, ev.SUBSET_ALL, "fdi_acc_joint")
        rows = [("Geometric reasoning", "End-to-end FDI accuracy", g, *g_ci, COMPLETE),
                ("Network numbers directly", "End-to-end FDI accuracy", nn, *nn_ci, BASELINE)]
        fig = _bars(rows, 190)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, config={"displayModeBar": False}, key="perf_geo")
        st.caption(f"Single-pass geometric numbering against a 41-class network that predicts tooth numbers "
                   f"itself: **{d:+.1f} pp** (paired 95% CI {lo:+.1f} to {hi:+.1f}, n = {n_pair}). Most of "
                   "the gap comes from detection: the generic tooth detector finds more teeth.")

    if "v1" in v2 and "v2" in v2:
        st.subheader("Improved vs published engine")
        a, b = v2["v1"], v2["v2"]
        rows = []
        for key, label in (("recall", "Tooth detection recall (full arches)"), ("fdi", "FDI accuracy, all canonical"),
                           ("complete", "FDI accuracy, complete arches"), ("partial", "FDI accuracy, partial arches")):
            rows.append({"Measure": label,
                         "Published engine": f"{pct(a[key])} [{pct(a[key + '_lo'])}, {pct(a[key + '_hi'])}]",
                         "Improved (default)": f"{pct(b[key])} [{pct(b[key + '_lo'])}, {pct(b[key + '_hi'])}]"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Same test images, paired by image; the improved engine was designed on the validation split "
                   "only. It runs one pass, removes duplicate boxes, splits the jaws along a fitted bite line, "
                   "numbers by position from the midline and never drops a detected tooth. The Analyse page "
                   "uses it by default; the published engine is still available under Numbering. "
                   "Source: `paper_eval/results/engine_v2.csv`.")

    st.subheader("Detection and segmentation")
    st.dataframe(pd.DataFrame([
        {"Measure": "AP, generic tooth class (segmentation)", "Value": f"{ap['generic_AP_Tooth']:.1f}"},
        {"Measure": "Mask AP, all 9 classes", "Value": f"{ap['segm_AP']:.1f}"},
        {"Measure": "Mask AP50, all 9 classes", "Value": f"{ap['segm_AP50']:.1f}"},
        {"Measure": "Inference time per image, two passes (Tesla T4)", "Value": "0.57 – 0.61 s"},
        {"Measure": "Inference time per image, single pass (Tesla T4)", "Value": "0.23 s"},
    ]), hide_index=True)

    with st.expander("Test set and method", icon=":material/science:", expanded=False):
        st.markdown(
            "- **716 held-out panoramic radiographs** from a stratified split, with no image shared with "
            "the teeth model's training data.\n"
            "- Numbering is scored on the **117 canonical images** whose labels follow one consistent "
            "numbering convention (67 complete arches, 50 partial).\n"
            "- Intervals are 10,000-resample bootstrap 95% CIs; comparisons are paired by image.\n"
            "- Only **recall** is reported for detection. Most test images label only some of their teeth, "
            "so precision and false positives per image are not meaningful on this ground truth.\n"
            "- Source: `paper_eval/` (harness, audit and raw CSVs).")
    with st.expander("Known limitations", icon=":material/report:", expanded=True):
        st.markdown(
            (f"- **Partial arches.** With the improved engine, FDI accuracy is {pct(new['complete'])} on complete "
             f"arches and {pct(new['partial'])} on partial ones (published engine: {pct(s['complete'])} and "
             f"{pct(s['partial'])}). Review the flagged jaw.\n" if new else
             f"- **Partial arches.** FDI accuracy drops from {pct(s['complete'])} to {pct(s['partial'])} "
             "when teeth are missing. Review the flagged jaw.\n")
            + "- **Findings are not validated.** Caries, restoration and endodontic detections have no "
            "measured accuracy yet; treat them as prompts for review.\n"
            "- **Adult permanent dentition only.** Primary and mixed dentition, supernumerary teeth and "
            "implants that replace teeth are not numbered correctly.\n"
            "- **One data source.** Images come from public datasets; performance on other machines and "
            "populations is unmeasured.")
    ui.footer()
