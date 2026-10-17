"""Markdown and JSON exports of an AnalysisResult."""

import json
from datetime import datetime

import tooth_ai
from tooth_ai.notation import FDI_TO_UNIVERSAL, QUADRANT_NAMES, tooth_name


def region(box, image_width):
    """Which third of the image a box sits in. The image's left is the patient's right.

    (The old app divided by a hard-coded 1000 instead of the image width.)
    """
    x = (box[0] + box[2]) / 2 / image_width
    if x < 1 / 3:
        return "Image left (patient's right)"
    return "Centre" if x < 2 / 3 else "Image right (patient's left)"


def to_dict(result, case_id="", created=None):
    created = created or datetime.now()
    h, w = result.geometry.image_size
    return {
        "product": tooth_ai.PRODUCT_NAME, "version": tooth_ai.__version__,
        "case_id": case_id or None, "created": created.isoformat(timespec="seconds"),
        "mode": result.mode, "source": result.source,
        "image": {"width": w, "height": h},
        "reliability": {"level": result.reliability.level, "title": result.reliability.title,
                        "reasons": result.reliability.reasons},
        "teeth": [{
            "fdi": t["fdi"], "universal": FDI_TO_UNIVERSAL[t["fdi"]], "name": tooth_name(t["fdi"]),
            "quadrant": t["quadrant"], "score": round(t["score"], 4),
            "box": [round(float(v), 1) for v in t["box"]],
            "findings": [{"type": f["type"], "score": round(f["score"], 4),
                          "near_metal": bool(f.get("near_metal"))} for f in t["findings"]],
            "numbering": {k: v for k, v in (t.get("trace") or {}).items()},
        } for t in result.teeth],
        "unassigned_findings": [{
            "type": a["class_name"], "score": round(a["score"], 4),
            "box": [round(float(v), 1) for v in a["box"]], "region": region(a["box"], w),
        } for a in result.unassigned],
        "timings_s": {k: round(v, 3) for k, v in result.timings.items()},
        "disclaimer": tooth_ai.DISCLAIMER,
    }


def to_json(result, case_id="", created=None):
    return json.dumps(to_dict(result, case_id, created), indent=2)


def to_markdown(result, case_id="", created=None, notation="FDI"):
    d = to_dict(result, case_id, created)
    num = (lambda t: t["universal"]) if notation == "Universal" else (lambda t: t["fdi"])
    lines = [
        f"# {tooth_ai.PRODUCT_NAME}: panoramic radiograph report", "",
        f"- Case: {case_id or '-'}",
        f"- Date: {d['created'].replace('T', ' ')}",
        f"- Mode: {result.mode}",
        f"- Notation: {notation}",
        f"- Image: {d['image']['width']} × {d['image']['height']} px", "",
        f"**{result.reliability.title}.** {result.reliability.detail}", "",
    ]
    if result.reliability.reasons:
        lines += [f"- {r}" for r in result.reliability.reasons] + [""]
    if d["teeth"]:
        lines += [f"## Teeth ({len(d['teeth'])} of 32 detected)", ""]
        for q in (1, 2, 3, 4):
            rows = [t for t in d["teeth"] if t["quadrant"] == q]
            if not rows:
                continue
            lines += [f"### {QUADRANT_NAMES[q]}", "", "| Tooth | Name | Findings | Score |",
                      "|---|---|---|---|"]
            for t in rows:
                f = ", ".join(x["type"] + (" (near metal)" if x["near_metal"] else "")
                              for x in t["findings"]) or "-"
                lines.append(f"| {num(t)} | {t['name']} | {f} | {t['score']:.2f} |")
            lines.append("")
    if d["unassigned_findings"]:
        lines += ["## Findings not mapped to a tooth", "", "| Finding | Region | Score |", "|---|---|---|"]
        lines += [f"| {a['type']} | {a['region']} | {a['score']:.2f} |" for a in d["unassigned_findings"]]
        lines.append("")
    lines += ["---", f"*{tooth_ai.DISCLAIMER}*", ""]
    return "\n".join(lines)
