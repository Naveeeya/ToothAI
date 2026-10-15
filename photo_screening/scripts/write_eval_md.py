#!/usr/bin/env python3
"""Write photo_screening/EVAL.md from measured results only.

    python photo_screening/scripts/write_eval_md.py --aku eval_results_aku.json --dentalai eval_dentalai.json

Inputs: the notebooks' metrics.json (weights/<model>/), evaluate.py's results for AKU and DentalAI.
Every number in EVAL.md comes from these files; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def ci(pair) -> str:
    return f"[{100 * pair[0]:.1f}, {100 * pair[1]:.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aku", type=Path, required=True)
    ap.add_argument("--dentalai", type=Path, required=True)
    ap.add_argument("--chosen", default="recommended", help="name of the configuration the service uses")
    ap.add_argument("--out", type=Path, default=PKG / "EVAL.md")
    args = ap.parse_args()
    aku = json.loads(args.aku.read_text())
    dai = json.loads(args.dentalai.read_text())
    nb_a = json.loads((PKG / "weights/caries_aku/metrics.json").read_text())
    nb_b = json.loads((PKG / "weights/teeth_dentalai/metrics.json").read_text())
    versions = aku.get("models", {})

    L = [f"# Photo screening: measured results\n\nGenerated {date.today():%d %B %Y} by `scripts/write_eval_md.py` "
         "from measured files only. Every number below names the data split it comes from. Model versions "
         f"(SHA-256 prefix): {', '.join(f'`{k}` {v}' for k, v in versions.items())}.\n"]

    # ---- end-to-end on phone photos
    test = aku["aku_test"]
    chosen = next((c for c in test["configs"] if c["config"] == args.chosen), test["configs"][-1])
    L.append("## End to end on smartphone photos (AKU official test split)\n")
    L.append(f"The service as configured (`{chosen['config']}`), run on {chosen['n_images']} test photos "
             f"({chosen['n_pos']} with caries, {chosen['n_neg']} without; {chosen['n_images'] - chosen['n_analysed']} "
             "rejected by the quality gate). 95 % Wilson intervals in brackets.\n")
    L.append("| Measure | Value |\n| --- | --- |")
    ref = pct(chosen["referral_rate_with_caries"])
    L.append(f"| Photos with caries that get a check-up or dentist referral | {ref} "
             f"{ci(chosen['referral_rate_with_caries_ci'])} |")
    L.append(f"| Caries-free photos referred anyway (false referral) | {pct(chosen['false_referral_rate'])} "
             f"{ci(chosen['false_referral_rate_ci'])} |")
    sens, spec = chosen["image_sensitivity"], chosen["image_specificity"]
    L.append(f"| Image-level caries sensitivity | {pct(sens)} {ci(chosen['image_sensitivity_ci'])} |")
    L.append(f"| Image-level caries specificity | {pct(spec)} {ci(chosen['image_specificity_ci'])} |")
    L.append(f"| Caries AP@50 / box precision / box recall | {chosen['caries_ap50']:.3f} / "
             f"{chosen['caries_box_precision']:.3f} / {chosen['caries_box_recall']:.3f} |")
    L.append(f"| Findings per photo (all) / Medium or High | {chosen['findings_per_photo']} / "
             f"{chosen['medium_plus_per_photo']} |\n")

    L.append(f"In plain terms: of 100 people whose photos show decay, about "
             f"{round(100 * chosen['referral_rate_with_caries'])} are told to book a check-up or see a dentist; of 100 "
             f"whose photos show none, about {round(100 * chosen['false_referral_rate'])} are told to anyway. "
             "Decay between teeth or below the gum line can't be seen in photos at all, which is why the report "
             "recommends an OPG.\n")
    L.append("**Configurations compared on the validation split** (chosen there, then reported on test above):\n")
    L.append("| Configuration | Referral with caries | False referral | Specificity | Findings per photo |")
    L.append("| --- | --- | --- | --- | --- |")
    for c in aku["aku_valid"]["configs"]:
        L.append(f"| {c['config']} | {pct(c['referral_rate_with_caries'])} | {pct(c['false_referral_rate'])} | "
                 f"{pct(c['image_specificity'])} | {c['findings_per_photo']} |")
    L.append("")

    # ---- per model
    t = nb_a["test"]
    alone = test["alphadent_caries_alone"]
    L.append("## Per model\n")
    L.append("| Model | Measure | Value | Split |\n| --- | --- | --- | --- |")
    L.append(f"| `caries_aku` | mAP@50, mean of `d`/`D` classes | {t['ultralytics']['map50']:.3f} (published YOLOv8s "
             f"{t['published_baseline']['map50']}) | AKU test, {t['n_images']} images |")
    L.append(f"| `caries_aku` | AP@50, classes merged | {t['merged_caries']['ap50']:.3f} | same |")
    L.append(f"| `alphadent` | Caries AP@50 on phone photos (cross-domain) | {alone['ap50']:.3f} | same |")
    a_sens = pct(alone["image_sensitivity_at_0.25"])
    L.append(f"| `alphadent` | Caries image sensitivity / specificity at 0.25 | {a_sens} / "
             f"{pct(alone['image_specificity_at_0.25'])} | same |")
    for cls in ("tooth", "cavity", "crack"):
        d = dai["dentalai_test"][cls]
        k = d["at_keep_threshold"]
        L.append(f"| `teeth_dentalai` | {cls}: AP@50; precision / recall at keep {k['conf']} | {d['ap50']:.3f}; "
                 f"{k['precision']:.2f} / {k['recall']:.2f} | DentalAI test, {d['n_boxes']} boxes |")
    mask_ap = nb_b["test"]["per_class"]["tooth"]["mask_ap50"]
    L.append(f"| `teeth_dentalai` | tooth mask AP@50 (notebook) | {mask_ap:.3f} "
             "| DentalAI test |\n")

    # ---- quality gate and speed
    L.append("## Quality gate and speed\n")
    L.append(f"- Real AKU test photos rejected by the hard checks: {pct(test['quality_gate_rejected'])} "
             f"({', '.join(f'{k} {v}' for k, v in test['quality_failures'].items())}).")
    L.append(f"- Inference per photo ({aku.get('device')}): "
             + ", ".join(f"`{m}` {s:.2f} s" for m, s in test["seconds_per_image"].items()) + ".\n")

    L.append("## Limitations\n")
    L.append("- AKU labels caries only, so wear, fillings and crowns from AlphaDent are not validated on phone photos.")
    L.append("- DentalAI tooth instances come from semantic masks split by watershed; tooth numbers are approximate.")
    L.append("- Cavity and crack: too weak (or too few labels) to report; off by default.")
    L.append("- All photos come from two public datasets; real users' phones, lighting and ages will differ.")
    args.out.write_text("\n".join(L) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
