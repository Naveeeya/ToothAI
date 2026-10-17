"""Report charts, drawn fresh from the measured results files (nothing typed in by hand).

    python tools/report_media/charts.py report_media/04_charts

Sources: paper_eval/results/{engine_v2.csv, engine_v2_design_valid.csv, v1_tooth_losses.csv,
per_tooth_recall.csv, threshold_sweep.csv}
and photo_screening/eval_results_aku.json. Brand palette and fonts from tooth_ai/ui/theme.py.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "paper_eval" / "results"
NAVY, SPARK, SPARK_LIGHT, SLATE, LINE, PAPER = "#0E2236", "#2D6FDB", "#7FA9F5", "#5B6B78", "#E1E6EC", "#FFFFFF"
V1 = "#A9B8C8"
WARN, ALERT, OK = "#CA8A04", "#C2410C", "#16A34A"

for f in ("PlusJakartaSans-Regular.ttf", "PlusJakartaSans-Medium.ttf", "PlusJakartaSans-Bold.ttf",
          "PlusJakartaSans-ExtraBold.ttf", "JetBrainsMono-Regular.ttf"):
    font_manager.fontManager.addfont(str(ROOT / "static" / "fonts" / f))
plt.rcParams.update({
    "font.family": "Plus Jakarta Sans", "font.size": 11, "axes.edgecolor": LINE, "axes.labelcolor": NAVY,
    "axes.titleweight": "bold", "axes.titlesize": 14, "axes.titlecolor": NAVY, "xtick.color": SLATE,
    "ytick.color": SLATE, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": LINE, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": PAPER,
    "axes.facecolor": PAPER, "savefig.dpi": 220, "savefig.bbox": "tight", "legend.frameon": False,
})


def rows(name):
    with open(RES / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def note(fig, text):
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8.5, color=SLATE, wrap=True)


def engine_v1_v2(out):
    r = {x["config"]: {k: float(v) for k, v in x.items() if v not in ("", None) and k != "config" and k != "split"}
         for x in rows("engine_v2.csv")}
    keys = [("recall", "Tooth detection\nrecall"), ("fdi", "FDI numbering\n(all 117)"),
            ("complete", "FDI, complete\narches (67)"), ("partial", "FDI, partial\narches (50)")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.36
    for i, (k, label) in enumerate(keys):
        for j, (cfg, col, name) in enumerate((("v1", V1, "Published engine (v1)"), ("v2", SPARK, "Improved engine (v2)"))):
            v, lo, hi = r[cfg][k] * 100, r[cfg][k + "_lo"] * 100, r[cfg][k + "_hi"] * 100
            x = i + (j - 0.5) * w
            ax.bar(x, v, w * 0.92, color=col, label=name if i == 0 else None)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=NAVY, capsize=3, lw=1)
            ax.text(x, hi + 1.5, f"{v:.1f}", ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
    ax.set_xticks(range(len(keys)), [k[1] for k in keys])
    ax.set_ylim(0, 110); ax.set_ylabel("%")
    ax.set_title("Numbering engine: published vs improved (test split)", loc="left", pad=28)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0), fontsize=10)
    note(fig, "716 held-out OPGs; recall on the 93 full-arch images, FDI on the 117 canonical images. Bars: point "
              "estimate; whiskers: 95 % bootstrap CI (10,000 resamples). Source: paper_eval/results/engine_v2.csv")
    fig.savefig(out / "fig_engine_v1_vs_v2.png"); plt.close(fig)


def engine_deltas(out):
    v2 = next(x for x in rows("engine_v2.csv") if x["config"] == "v2")
    keys = [("recall", "Tooth detection recall"), ("fdi", "FDI numbering, all"), ("complete", "FDI, complete arches"),
            ("partial", "FDI, partial arches")]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (k, label) in enumerate(keys[::-1]):
        d, lo, hi = (float(v2[f"delta_{k}{s}"]) * 100 for s in ("", "_lo", "_hi"))
        ax.plot([lo, hi], [i, i], color=SPARK, lw=3, solid_capstyle="round")
        ax.plot(d, i, "o", color=NAVY, ms=7)
        ax.text(hi + 1, i, f"+{d:.1f} pp  [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=10, color=NAVY)
    ax.axvline(0, color=SLATE, lw=1)
    ax.set_yticks(range(len(keys)), [k[1] for k in keys[::-1]])
    ax.set_xlim(-2, 62); ax.set_xlabel("v2 minus v1, percentage points (paired by image)")
    ax.set_title("Improvement from engine v2, paired 95 % CIs", loc="left")
    note(fig, "Paired bootstrap over images (same images, same detections, 10,000 resamples). Source: "
              "paper_eval/results/engine_v2.csv")
    fig.savefig(out / "fig_engine_paired_deltas.png"); plt.close(fig)


def tooth_losses(out):
    data = rows("v1_tooth_losses.csv")
    by_stage = defaultdict(int)
    by_type = defaultdict(lambda: defaultdict(int))
    for x in data:
        if x["stage"] == "kept":
            continue
        by_stage[x["stage"]] += int(x["count"])
        by_type[x["tooth_type"]][x["stage"]] += int(x["count"])
    groups = [("Never detected\nby the model", ["seen"], SLATE),
              ("Deleted by the cleanup\n(32-tooth pruning, conflicts)", ["pruned", "final"], ALERT),
              ("Fusion, threshold\nor NMS", ["fused", "threshold", "nms"], V1)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    vals = [sum(by_stage[s] for s in g[1]) for g in groups]
    a1.barh(range(3), vals, color=[g[2] for g in groups])
    for i, v in enumerate(vals):
        a1.text(v + 2, i, f"{v}", va="center", fontweight="bold", color=NAVY)
    a1.set_yticks(range(3), [g[0] for g in groups]); a1.invert_yaxis(); a1.set_xlim(0, max(vals) * 1.25)
    a1.set_xlabel("missed teeth (of 2,976 labelled)")
    a1.set_title("Where the published engine loses teeth", loc="left")
    order = ["central incisor", "lateral incisor", "canine", "first premolar", "second premolar", "first molar",
             "second molar", "third molar"]
    left = [0] * len(order)
    for name, stages, col in groups:
        v = [sum(by_type[t][s] for s in stages) for t in order]
        a2.bar(range(len(order)), v, bottom=left, color=col, label=name.replace("\n", " "))
        left = [a + b for a, b in zip(left, v)]
    short = ["Incisor 1", "Incisor 2", "Canine", "Premolar 1", "Premolar 2", "Molar 1", "Molar 2", "Molar 3"]
    a2.set_xticks(range(len(order)), [x.replace(" ", "\n") for x in short], fontsize=9)
    a2.set_ylabel("missed teeth"); a2.set_title("…by tooth type", loc="left")
    a2.legend(fontsize=8.5, loc="upper left")
    note(fig, "Every labelled tooth on the 93 full-arch test images traced through the deployed pipeline "
              "(full_fix_routing). Source: paper_eval/results/v1_tooth_losses.csv (paper_eval/trace_losses.py)")
    fig.tight_layout(); fig.savefig(out / "fig_v1_where_teeth_are_lost.png"); plt.close(fig)


def per_tooth(out):
    r = {int(x["fdi"]): float(x["recall"]) for x in rows("per_tooth_recall.csv") if x["config"] == "single_pass"}
    up = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
    lo = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
    fig, ax = plt.subplots(figsize=(11, 2.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("t", [ALERT, WARN, "#EAF1FC", SPARK])
    norm = matplotlib.colors.Normalize(0.70, 1.0)
    for row, seq in ((1, up), (0, lo)):
        for i, f in enumerate(seq):
            v = r.get(f)
            ax.add_patch(plt.Rectangle((i + (0.3 if i >= 8 else 0), row), 0.9, 0.85, color=cmap(norm(v)), lw=0))
            ax.text(i + 0.45 + (0.3 if i >= 8 else 0), row + 0.55, str(f), ha="center", fontsize=9, color=NAVY,
                    family="JetBrains Mono")
            ax.text(i + 0.45 + (0.3 if i >= 8 else 0), row + 0.22, f"{v * 100:.0f}", ha="center", fontsize=8.5,
                    color=NAVY, fontweight="bold")
    ax.set_xlim(-0.2, 16.6); ax.set_ylim(-0.2, 2.05); ax.axis("off")
    ax.text(0, 1.95, "Patient's right", fontsize=9, color=SLATE)
    ax.text(16.5, 1.95, "Patient's left", fontsize=9, color=SLATE, ha="right")
    ax.set_title("Tooth detection recall by position (single pass, %)", loc="left")
    note(fig, "93 full-arch test images. Second molars (17, 37) are the weakest: they overlap their neighbours "
              "and are often cut by the image edge. Source: paper_eval/results/per_tooth_recall.csv")
    fig.savefig(out / "fig_per_tooth_recall.png"); plt.close(fig)


def threshold_sweep(out):
    r = rows("threshold_sweep.csv")
    tau = [float(x["tau"]) for x in r]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(tau, [float(x["central_recall"]) * 100 for x in r], "-o", color=SPARK, label="Central zone (35–65 % of width)")
    ax.plot(tau, [float(x["peripheral_recall"]) * 100 for x in r], "-o", color=SLATE, label="Periphery (molars)")
    ax.axvline(0.20, color=LINE, lw=2, zorder=0); ax.text(0.205, 93.2, "app default", fontsize=8.5, color=SLATE)
    ax.set_xlabel("tooth score threshold"); ax.set_ylabel("recall, %"); ax.set_ylim(93, 99.5)
    ax.set_title("The central zone is the easy zone", loc="left"); ax.legend(loc="lower right")
    note(fig, "Published engine, 93 full-arch test images. Recall in the periphery rises with the threshold "
              "because fewer extra boxes reach the cleanup. Source: paper_eval/results/threshold_sweep.csv")
    fig.savefig(out / "fig_threshold_central_vs_periphery.png"); plt.close(fig)


def photo(out):
    r = json.loads((ROOT / "photo_screening" / "eval_results_aku.json").read_text())
    cfgs = r["aku_valid"]["configs"]
    short = {"original (AlphaDent caries standalone, cavity/crack on, wear from 0.30)": "Original",
             "AlphaDent caries as agreement only": "AlphaDent caries\nas agreement",
             "+ wear from 0.55": "+ wear from 0.55", "+ cavity/crack off": "+ cavity/crack off",
             "+ colour pass: grey world": "+ grey-world\n(shipped)", "+ colour pass: CLAHE": "+ CLAHE",
             "+ colour passes: grey world and CLAHE": "+ grey world\n+ CLAHE"}
    keep = [c for c in cfgs if c["config"] in short or c["config"].startswith("+ selection rules")]
    labels = [short.get(c["config"], "+ selection\nrules") for c in keep]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = range(len(keep)); w = 0.38
    ax.bar([i - w / 2 for i in x], [c["referral_rate_with_caries"] * 100 for c in keep], w, color=SPARK,
           label="Photos with caries referred (higher is better)")
    ax.bar([i + w / 2 for i in x], [c["false_referral_rate"] * 100 for c in keep], w, color=ALERT,
           label="Caries-free photos referred (lower is better)")
    for i, c in enumerate(keep):
        ax.text(i - w / 2, c["referral_rate_with_caries"] * 100 + 1.2, f"{c['referral_rate_with_caries'] * 100:.0f}",
                ha="center", fontsize=8.5, color=NAVY)
        ax.text(i + w / 2, c["false_referral_rate"] * 100 + 1.2, f"{c['false_referral_rate'] * 100:.1f}", ha="center",
                fontsize=8.5, color=NAVY)
    ax.set_xticks(list(x), labels, fontsize=8.5); ax.set_ylabel("%"); ax.set_ylim(0, 112)
    ax.set_title("Photo screening: each rule measured on the validation split", loc="left", pad=28)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0), fontsize=10)
    note(fig, "AKU intraoral caries dataset, official validation split (configurations chosen here, then reported "
              "on test). Source: photo_screening/eval_results_aku.json")
    fig.savefig(out / "fig_photo_screening_configurations.png"); plt.close(fig)

    t = next(c for c in r["aku_test"]["configs"] if c["config"] == "+ colour pass: grey world")
    items = [("Photos with caries\nreferred", "referral_rate_with_caries", SPARK),
             ("Caries-free photos\nreferred", "false_referral_rate", ALERT),
             ("Image-level\nsensitivity", "image_sensitivity", SPARK_LIGHT),
             ("Image-level\nspecificity", "image_specificity", SPARK_LIGHT)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (label, k, col) in enumerate(items):
        v = t[k] * 100; lo, hi = (c * 100 for c in t[k + "_ci"])
        ax.bar(i, v, 0.6, color=col); ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=NAVY, capsize=3, lw=1)
        ax.text(i, hi + 2, f"{v:.1f} %", ha="center", fontweight="bold", color=NAVY, fontsize=10)
    ax.set_xticks(range(4), [i[0] for i in items]); ax.set_ylim(0, 112); ax.set_ylabel("%")
    ax.set_title("Photo screening on the AKU test split (service defaults)", loc="left")
    note(fig, f"{t['n_images']} smartphone photos ({t['n_pos']} with caries, {t['n_neg']} without); 95 % Wilson "
              "intervals. Source: photo_screening/eval_results_aku.json")
    fig.savefig(out / "fig_photo_screening_test_results.png"); plt.close(fig)


def not_adopted(out):
    """Engine design choices measured on the validation split (results/engine_v2_design_valid.csv)."""
    r = {x["config"]: float(x["fdi"]) * 100 for x in rows("engine_v2_design_valid.csv")}
    labels = {"v1": ("Published engine (v1)", ALERT), "v1-1pass": ("v1, single pass", V1),
              "v2-cleanup": ("+ cleanup before numbering (two passes)", V1),
              "v2-symmetry": ("+ midline numbering", V1), "v2-sym-curve": ("+ bite-line jaw split", V1),
              "v2-gap": ("single pass + duplicates removed (count-based)", SPARK_LIGHT),
              "v2": ("+ template numbering = shipped v2", SPARK),
              "v2-tpl-fit": ("v2 + template-fitted midline (not adopted)", V1),
              "v2+41": ("v2 + 41-class teeth rescue (not adopted)", V1),
              "v2-sym-big": ("larger input, 1000 px (not adopted)", ALERT)}
    items = [(labels[k][0], r[k], labels[k][1]) for k in labels if k in r]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, (label, v, col) in enumerate(items[::-1]):
        ax.barh(i, v, color=col); ax.text(v + 0.8, i, f"{v:.1f} %", va="center", fontsize=9.5, color=NAVY)
    ax.set_yticks(range(len(items)), [i[0] for i in items[::-1]], fontsize=9); ax.set_xlim(0, 100)
    ax.set_xlabel("FDI accuracy, canonical validation images (%)")
    ax.set_title("How engine v2 was built (validation split)", loc="left")
    note(fig, "109 canonical validation images, in the order the steps were tried. The two 'not adopted' "
              "additions gain under 0.5 pp (paired CIs include zero), so the simpler v2 was kept. "
              "Source: paper_eval/results/engine_v2_design_valid.csv (paper_eval/engine_lab.py)")
    fig.savefig(out / "fig_engine_design_choices.png"); plt.close(fig)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "report_media" / "04_charts")
    out.mkdir(parents=True, exist_ok=True)
    for f in (engine_v1_v2, engine_deltas, tooth_losses, per_tooth, threshold_sweep, photo, not_adopted):
        f(out)
        print("drew", f.__name__)


if __name__ == "__main__":
    main()
