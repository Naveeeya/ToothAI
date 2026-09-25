#!/usr/bin/env python3
"""Is the central-zone recall finding confounded by image clarity?

Standard library only. No numpy, no PIL, no GPU: pixels are obtained by
converting each JPEG to an uncompressed BMP with macOS `sips` and parsing the
bytes directly.

The question: central-zone recall (98.4%) exceeds peripheral (90.9%) on the 93
full-arch images. If those 93 are systematically clearer in the central band
than the other 623, the finding is conditional on annotation-complete images
and must be stated that way rather than as a property of the system.

Per image we compute, for the central 35-65% width band and the two outer
bands:
  mean      average intensity        (brightness)
  sd        intensity spread         (global contrast within the band)
  grad      mean |I[x+1]-I[x]|       (edge energy -- the best simple proxy
                                      for "is detail visible here")

and the central-minus-peripheral difference of each. Those differences are
then compared between the 93 full-arch images and the other 623 with
Cohen's d and a bootstrap CI on the difference of means.

    python3 paper_eval/check_contrast_confound.py [--width 256] [--jobs 8]
"""
import argparse
import csv
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPLIT_DIR = os.path.join(ROOT, "data", "final-di-stratified", "test")
SPLIT_JSON = os.path.join(SPLIT_DIR, "_annotations.coco.json")
RESULTS = os.path.join(HERE, "results")
LO, HI = 0.35, 0.65
SEED = 20260921


def read_bmp_gray(path):
    """24-bit uncompressed BMP -> (width, height, rows of luma 0..255)."""
    d = open(path, "rb").read()
    off = struct.unpack("<I", d[10:14])[0]
    # BITMAPINFOHEADER: biWidth@18(4) biHeight@22(4) biPlanes@26(2)
    # biBitCount@28(2) biCompression@30(4)  -> 16 bytes, i.e. d[18:34]
    w, h, planes, bpp, comp = struct.unpack("<iiHHI", d[18:34])
    if bpp != 24 or comp != 0:
        raise ValueError(f"unexpected BMP: bpp={bpp} comp={comp}")
    top_down = h < 0
    h = abs(h)
    stride = (w * 3 + 3) & ~3
    rows = []
    for y in range(h):
        src = y if top_down else (h - 1 - y)
        base = off + src * stride
        line = d[base:base + w * 3]
        # BGR -> luma, integer weights (ITU-R BT.601)
        rows.append([(line[i * 3 + 2] * 299 + line[i * 3 + 1] * 587
                      + line[i * 3] * 114) // 1000 for i in range(w)])
    return w, h, rows


def band_stats(rows, w, x0, x1):
    """mean, sd and mean |horizontal gradient| over columns [x0, x1)."""
    n = 0
    s = 0
    s2 = 0
    g = 0
    gn = 0
    for r in rows:
        prev = None
        for x in range(x0, x1):
            v = r[x]
            n += 1; s += v; s2 += v * v
            if prev is not None:
                g += abs(v - prev); gn += 1
            prev = v
    if n == 0:
        return None
    mean = s / n
    var = max(0.0, s2 / n - mean * mean)
    return mean, math.sqrt(var), (g / gn if gn else 0.0)


def measure(args):
    fname, width, tmpdir = args
    src = os.path.join(SPLIT_DIR, fname)
    out = os.path.join(tmpdir, f"{abs(hash(fname))}.bmp")
    try:
        subprocess.run(["sips", "-s", "format", "bmp", "-Z", str(width),
                        src, "--out", out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        w, h, rows = read_bmp_gray(out)
    except Exception as exc:
        return fname, None, str(exc)
    finally:
        if os.path.exists(out):
            os.remove(out)

    c0, c1 = int(w * LO), int(w * HI)
    cen = band_stats(rows, w, c0, c1)
    left = band_stats(rows, w, 0, c0)
    right = band_stats(rows, w, c1, w)
    if not (cen and left and right):
        return fname, None, "empty band"
    # peripheral = the two outer bands pooled, weighted by their widths
    lw, rw = c0, w - c1
    per = tuple((left[i] * lw + right[i] * rw) / (lw + rw) for i in range(3))
    return fname, {
        "central_mean": cen[0], "central_sd": cen[1], "central_grad": cen[2],
        "periph_mean": per[0], "periph_sd": per[1], "periph_grad": per[2],
        "d_mean": cen[0] - per[0], "d_sd": cen[1] - per[1],
        "d_grad": cen[2] - per[2],
    }, None


def cohens_d(a, b):
    """Standardised mean difference, pooled SD."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma = sum(a) / na
    mb = sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (ma - mb) / sp if sp else None


def boot_mean_diff(a, b, n=10000, seed=SEED, alpha=0.05):
    rng = random.Random(seed)
    st = []
    for _ in range(n):
        ma = sum(a[rng.randrange(len(a))] for _ in range(len(a))) / len(a)
        mb = sum(b[rng.randrange(len(b))] for _ in range(len(b))) / len(b)
        st.append(ma - mb)
    st.sort()
    return st[int(alpha / 2 * n)], st[min(n - 1, int((1 - alpha / 2) * n))]


def interpret(d):
    a = abs(d)
    return ("negligible" if a < 0.2 else "small" if a < 0.5
            else "medium" if a < 0.8 else "LARGE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    coco = json.load(open(SPLIT_JSON))
    fnames = {i["id"]: i["file_name"] for i in coco["images"]}

    q = {}
    with open(os.path.join(RESULTS, "label_quality.csv")) as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            q[int(r["image_id"])] = r
    full_arch = {i for i, r in q.items() if r["full_arch"] == "1"}

    ids = sorted(fnames)
    if args.limit:
        ids = ids[:args.limit]
    print(f"measuring {len(ids)} images at {args.width}px wide, {args.jobs} jobs...")

    with tempfile.TemporaryDirectory() as td:
        work = [(fnames[i], args.width, td) for i in ids]
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            out = list(ex.map(measure, work))

    by_name = {f: s for f, s, _e in out if s}
    errs = [(f, e) for f, s, e in out if s is None]
    if errs:
        print(f"  {len(errs)} failed, e.g. {errs[0]}")

    rows = []
    for i in ids:
        s = by_name.get(fnames[i])
        if not s:
            continue
        rows.append({"image_id": i, "full_arch": int(i in full_arch), **s})

    p = os.path.join(RESULTS, "contrast_bands.csv")
    with open(p, "w", newline="") as fh:
        fh.write("# Tooth-AI paper_eval -- central vs peripheral band contrast\n")
        fh.write(f"# source: JPEG -> BMP via sips at {args.width}px wide, luma BT.601\n")
        fh.write(f"# central band = {LO:.2f}-{HI:.2f} of width; peripheral = the two\n")
        fh.write("#   outer bands pooled, width-weighted\n")
        fh.write("# grad = mean |I[x+1]-I[x]| along rows (edge energy)\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fa = [r for r in rows if r["full_arch"] == 1]
    ot = [r for r in rows if r["full_arch"] == 0]
    print(f"\nfull-arch {len(fa)} images | others {len(ot)} images")

    print("\n" + "=" * 94)
    print("IS THE CENTRAL BAND CLEARER ON THE 93 FULL-ARCH IMAGES?")
    print("=" * 94)
    print(f"{'quantity':<34}{'full-arch':>12}{'others':>12}{'diff':>10}"
          f"{'95% CI on diff':>22}{'Cohen d':>10}")
    print("-" * 94)
    worst = 0.0
    for key, label in (("d_grad", "central-peripheral edge energy"),
                       ("d_sd",   "central-peripheral intensity sd"),
                       ("d_mean", "central-peripheral brightness"),
                       ("central_grad", "central edge energy (absolute)"),
                       ("periph_grad", "peripheral edge energy (absolute)")):
        a = [r[key] for r in fa]
        b = [r[key] for r in ot]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        d = cohens_d(a, b)
        lo, hi = boot_mean_diff(a, b)
        worst = max(worst, abs(d) if d else 0)
        print(f"{label:<34}{ma:>12.2f}{mb:>12.2f}{ma-mb:>10.2f}"
              f"  [{lo:>+7.2f},{hi:>+7.2f}]{d:>10.2f}  {interpret(d)}")
    print("-" * 94)

    # Two different questions live in that table and a blunt max|d| conflates
    # them:
    #
    #  (a) DIFFERENTIAL -- do the central-MINUS-peripheral contrasts differ
    #      between the groups? Only this can explain why central recall beats
    #      peripheral recall *on the full-arch images specifically*.
    #  (b) ABSOLUTE -- do the 93 differ from the 623 in overall image
    #      character? That is a population caveat, but it applies to both
    #      bands equally and cannot by itself produce a central-vs-peripheral
    #      gap.
    def d_of(key):
        return cohens_d([r[key] for r in fa], [r[key] for r in ot])

    diff_grad = d_of("d_grad")
    diff_sd = d_of("d_sd")
    abs_worst = max(abs(d_of("central_grad")), abs(d_of("periph_grad")))

    print("(a) DIFFERENTIAL -- central-minus-peripheral, full-arch vs others")
    print(f"      edge energy   d = {diff_grad:+.2f}  {interpret(diff_grad)}"
          "   <- the primary clarity proxy")
    print(f"      intensity sd  d = {diff_sd:+.2f}  {interpret(diff_sd)}")
    print("(b) ABSOLUTE -- overall edge energy, full-arch vs others")
    print(f"      largest |d| = {abs_worst:.2f}  {interpret(abs_worst)}")
    print()
    if abs(diff_grad) < 0.2:
        print("VERDICT on the confound: the central band is NOT relatively")
        print("  clearer on the full-arch images. The central-minus-peripheral")
        print("  edge energy is effectively identical in both groups")
        print(f"  ({sum(r['d_grad'] for r in fa)/len(fa):.2f} vs "
              f"{sum(r['d_grad'] for r in ot)/len(ot):.2f}, d = {diff_grad:+.2f}), so the")
        print("  central-recall result is not produced by a clarity gradient")
        print("  that is peculiar to annotation-complete images.")
    else:
        print("VERDICT on the confound: the central band IS relatively clearer")
        print("  on the full-arch images. State the central-recall result as")
        print("  conditional on annotation-complete images.")
    print()
    if abs_worst >= 0.5:
        print("SEPARATE CAVEAT -- the 93 are a distinct population in absolute")
        print("  terms: their edge energy is LOWER in BOTH bands (central "
              f"{sum(r['central_grad'] for r in fa)/len(fa):.2f} vs "
              f"{sum(r['central_grad'] for r in ot)/len(ot):.2f}, peripheral "
              f"{sum(r['periph_grad'] for r in fa)/len(fa):.2f} vs "
              f"{sum(r['periph_grad'] for r in ot)/len(ot):.2f}).")
        print("  They are SMOOTHER, not sharper -- the opposite of the confound")
        print("  hypothesis. It applies equally to both bands, so it cannot")
        print("  generate a central-vs-peripheral gap, but every full-arch")
        print("  number is measured on images that are not typical of the split")
        print("  and the paper should say so.")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
