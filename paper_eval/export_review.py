#!/usr/bin/env python3
"""Task B follow-up: render canonical-partial images for clinical review.

Standard library only. Produces self-contained SVG panels (the radiograph is
base64-embedded) plus an index.html, so the whole folder can be zipped and sent
to a reviewer with no software to install.

The question being asked of the reviewer is printed on every panel: for the FDI
positions that carry NO annotation, is the tooth absent from the patient, or
present in the radiograph but simply not annotated?

That distinction decides whether "labeled teeth" is a valid recall denominator.

Usage:
    python3 paper_eval/export_review.py [--n 10]
"""
import argparse
import base64
import csv
import json
import os
import statistics
import subprocess
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(ROOT, "data", "final-di-stratified", "test")
SPLIT_JSON = os.path.join(SPLIT_DIR, "_annotations.coco.json")
MAP_CSV = os.path.join(ROOT, "paper_eval", "results", "class_to_position.csv")
OUT = os.path.join(ROOT, "paper_eval", "review")

FDI_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def git_state():
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
    try:
        d = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                    stderr=subprocess.DEVNULL).decode().strip()
        return f"{head}{'+dirty' if d else ''}"
    except subprocess.CalledProcessError:
        return f"{head}+dirty-unknown"


def load_mapping():
    """class_id -> (jaw, rank, fdi). Single source of truth: the Task B CSV."""
    m = {}
    with open(MAP_CSV) as fh:
        rows = csv.DictReader(l for l in fh if not l.startswith("#"))
        for r in rows:
            m[int(r["class_id"])] = (r["jaw"], int(r["position_rank"]), int(r["fdi"]))
    return m


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render(img_meta, anns, mapping, out_path):
    W, H = img_meta["width"], img_meta["height"]
    jpg = os.path.join(SPLIT_DIR, img_meta["file_name"])
    with open(jpg, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")

    present = {a["category_id"] for a in anns}
    missing_up = [(r, f) for c, (j, r, f) in sorted(mapping.items())
                  if j == "upper" and c not in present]
    missing_lo = [(r, f) for c, (j, r, f) in sorted(mapping.items())
                  if j == "lower" and c not in present]
    missing_up.sort(); missing_lo.sort()

    PANEL = 300
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W + PANEL}" height="{H}" viewBox="0 0 {W + PANEL} {H}">',
        '<style>'
        '.bx{fill:none;stroke:#00e05a;stroke-width:3}'
        '.lb{font:bold 15px monospace;fill:#00e05a}'
        '.lbg{fill:#000;fill-opacity:.65}'
        '.ttl{font:bold 17px sans-serif;fill:#fff}'
        '.tx{font:14px sans-serif;fill:#ddd}'
        '.q{font:bold 14px sans-serif;fill:#ffd24a}'
        '.miss{font:bold 14px monospace;fill:#ff6b6b}'
        '</style>',
        f'<image x="0" y="0" width="{W}" height="{H}" '
        f'xlink:href="data:image/jpeg;base64,{b64}"/>',
    ]
    for a in anns:
        x, y, w, h = a["bbox"]
        jaw, rank, fdi = mapping[a["category_id"]]
        parts.append(f'<rect class="bx" x="{x:.1f}" y="{y:.1f}" '
                     f'width="{w:.1f}" height="{h:.1f}"/>')
        ty = y - 4 if y > 18 else y + h + 15
        parts.append(f'<rect class="lbg" x="{x:.1f}" y="{ty-13:.1f}" width="86" height="17"/>')
        parts.append(f'<text class="lb" x="{x+3:.1f}" y="{ty:.1f}">'
                     f'c{a["category_id"]}&#8594;FDI{fdi}</text>')

    px = W + 12
    parts.append(f'<rect x="{W}" y="0" width="{PANEL}" height="{H}" fill="#111"/>')
    y = 28
    parts.append(f'<text class="ttl" x="{px}" y="{y}">{esc(img_meta["file_name"][:26])}</text>')
    y += 22
    parts.append(f'<text class="tx" x="{px}" y="{y}">image_id {img_meta["id"]} '
                 f'&#183; {len(anns)} of 32 labelled</text>')
    y += 30
    parts.append(f'<text class="q" x="{px}" y="{y}">QUESTION FOR REVIEWER:</text>')
    for line in ("For each FDI position listed below,",
                 "is the tooth ABSENT from the patient,",
                 "or PRESENT but not annotated?"):
        y += 18
        parts.append(f'<text class="tx" x="{px}" y="{y}">{line}</text>')
    y += 30
    parts.append(f'<text class="ttl" x="{px}" y="{y}">Unlabelled &#8212; upper</text>')
    for _, f in missing_up:
        y += 19
        parts.append(f'<text class="miss" x="{px}" y="{y}">FDI {f}    '
                     f'[ ] absent  [ ] present</text>')
    y += 28
    parts.append(f'<text class="ttl" x="{px}" y="{y}">Unlabelled &#8212; lower</text>')
    for _, f in missing_lo:
        y += 19
        parts.append(f'<text class="miss" x="{px}" y="{y}">FDI {f}    '
                     f'[ ] absent  [ ] present</text>')
    y += 30
    parts.append(f'<text class="tx" x="{px}" y="{y}">Green boxes = annotated ground truth.</text>')
    parts.append('</svg>')

    with open(out_path, "w") as fh:
        fh.write("\n".join(parts))
    return len(missing_up) + len(missing_lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    mapping = load_mapping()
    coco = json.load(open(SPLIT_JSON))
    images = {i["id"]: i for i in coco["images"]}
    per = defaultdict(list)
    for a in coco["annotations"]:
        if 1 <= a["category_id"] <= 32:
            per[a["image_id"]].append(a)

    quality = {}
    with open(os.path.join(ROOT, "paper_eval", "results", "label_quality.csv")) as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            quality[int(r["image_id"])] = r

    pool = [iid for iid, r in quality.items()
            if r["canonical_mapping"] == "1" and r["full_arch"] != "1"
            and int(r["labeled_teeth"]) > 0]
    # Spread the sample across "how many teeth are unlabelled", so the reviewer
    # sees both near-complete and heavily-sparse cases. Deterministic: sorted.
    pool.sort(key=lambda i: (-int(quality[i]["labeled_teeth"]), i))
    if len(pool) > args.n:
        step = len(pool) / float(args.n)
        chosen = [pool[int(k * step)] for k in range(args.n)]
    else:
        chosen = pool

    rows = []
    for iid in chosen:
        name = f"review_{iid:04d}.svg"
        n_missing = render(images[iid], per[iid], mapping, os.path.join(OUT, name))
        rows.append((iid, images[iid]["file_name"], len(per[iid]), n_missing, name))

    idx = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Tooth-AI label review</title>",
        "<style>body{font:15px/1.5 system-ui;margin:2rem;max-width:80rem}"
        "td,th{padding:.35rem .7rem;border-bottom:1px solid #ccc;text-align:left}"
        "code{background:#f4f4f4;padding:.1rem .3rem}</style>",
        "<h1>Tooth-AI &mdash; ground-truth label review</h1>",
        "<p><b>What we need to know:</b> in each radiograph below, some FDI "
        "positions carry no annotation. For each one, is the tooth <i>absent "
        "from the patient</i>, or <i>present in the radiograph but not "
        "annotated</i>?</p>",
        "<p>This decides whether the annotation count can be used as the "
        "denominator for detection recall. If unlabelled positions are mostly "
        "teeth that are present but unannotated, then a correct detection there "
        "would be scored as a false positive, and recall computed this way "
        "understates the model.</p>",
        f"<p>Split: <code>data/final-di-stratified/test</code> &middot; "
        f"{len(images)} images &middot; git <code>{git_state()}</code></p>",
        "<p>All images below have labels consistent with the standard "
        "position mapping, and an incomplete arch.</p>",
        "<table><tr><th>image_id</th><th>labelled</th><th>unlabelled</th>"
        "<th>panel</th><th>file</th></tr>",
    ]
    for iid, fn, nlab, nmiss, name in rows:
        idx.append(f"<tr><td>{iid}</td><td>{nlab}/32</td><td>{nmiss}</td>"
                   f"<td><a href='#p{iid}'>{name}</a></td><td><code>{esc(fn[:40])}</code></td></tr>")
    idx.append("</table>")
    # Embed each panel scaled to the page so a reviewer sees whole radiographs
    # without zooming; the standalone .svg keeps its full pixel size for print.
    for iid, fn, nlab, nmiss, name in rows:
        idx.append(f"<h2 id='p{iid}'>image_id {iid} &mdash; {nlab}/32 labelled, "
                   f"{nmiss} unlabelled</h2>")
        idx.append(f"<img src='{name}' style='max-width:100%;border:1px solid #999'>")
    with open(os.path.join(OUT, "index.html"), "w") as fh:
        fh.write("\n".join(idx))

    print(f"wrote {len(rows)} review panels to paper_eval/review/")
    print(f"{'image_id':>9} {'labelled':>9} {'unlabelled':>11}  file")
    for iid, fn, nlab, nmiss, name in rows:
        print(f"{iid:>9} {nlab:>7}/32 {nmiss:>11}  {fn[:44]}")
    print("\nopen paper_eval/review/index.html")


if __name__ == "__main__":
    main()
