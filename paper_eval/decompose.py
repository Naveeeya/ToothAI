#!/usr/bin/env python3
"""Decompose FDI accuracy into detection and numbering, on one denominator.

The headline FDI accuracy reported so far is a JOINT measure: its denominator
is every ground-truth tooth, so a tooth the detector never found counts as a
numbering error. Geometry runs on the 9-class detector and the network runs on
the Cascade, so a joint comparison confounds two different detectors with two
different numbering schemes.

This splits them, on identical images and the identical matching rule
(greedy, mask IoU >= 0.5):

    detection recall            matched / all ground-truth teeth
    numbering accuracy | det    correct FDI / matched
    jaw accuracy | det          correct jaw / matched
    joint                       correct FDI / all ground-truth teeth

and checks that recall x conditional reproduces the joint figure.

Reads results/per_image.csv only -- no GPU, no inference. Every configuration
in that file is decomposed, including network_numbering once its rows carry
match counts.

Usage:
    python3 paper_eval/decompose.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as K

RESULTS = os.path.join(K.ROOT, "paper_eval", "results")
PER_IMAGE = os.path.join(RESULTS, "per_image.csv")


def _int(v):
    return None if v in ("", None) else int(v)


def load_rows():
    with open(PER_IMAGE) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def subsets(manifests):
    """Denominators, each named for exactly what it contains.

    'canonical' (117) is the set the head-to-head numbering comparison ran on.
    'offset_subset' (112) is that set minus the 5 canonical images that sit in
    the sparse anomaly-annotation regime, where an unlabelled position is an
    unannotated tooth rather than an absent one. Both are reported because 112
    and 117 have been used interchangeably in discussion and they are not the
    same set.
    """
    return [
        ("canonical_117", manifests["canonical"]),
        # The complete-versus-partial split. This is the paper's main result:
        # the geometric engine numbers a complete arch well and a partial arch
        # badly, because sort_and_assign_fdi anchors the leftmost detection at
        # 18/48 unconditionally (app.py:126-135), so a missing tooth shifts the
        # whole jaw. These two subsets partition canonical_117 exactly.
        ("canonical_full_arch_67", manifests["canonical"] & manifests["full_arch"]),
        ("canonical_partial_50", manifests["canonical"] - manifests["full_arch"]),
        ("canonical_no_sparse_112", manifests["offset_subset"]),
        ("full_arch_93", manifests["full_arch"]),
    ]


# The complete/partial contrast, named once so the report and the CSV cannot
# drift apart.
CONTRAST = ("canonical_full_arch_67", "canonical_partial_50")


def two_group_delta(a_pairs, b_pairs, n=K.N_BOOT, seed=K.SEED, alpha=0.05):
    """Bootstrap CI on the difference between two DISJOINT groups of images.

    This is deliberately not the paired bootstrap used for config comparisons.
    The 67 complete arches and the 50 partial ones are different radiographs,
    so there is no image to pair on and nothing cancels; each group is
    resampled independently and the difference of the pooled ratios is taken.
    The interval is wider than a paired one would be, which is correct -- the
    between-image variance is real here, not a nuisance to be removed.
    """
    import random
    if not a_pairs or not b_pairs:
        return None

    def pooled(rows):
        num = sum(x for x, _ in rows)
        den = sum(y for _, y in rows)
        return (num / den) if den else float("nan")

    point = pooled(a_pairs) - pooled(b_pairs)
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        an = ad = bn = bd = 0
        for _ in range(len(a_pairs)):
            x, y = a_pairs[rng.randrange(len(a_pairs))]
            an += x; ad += y
        for _ in range(len(b_pairs)):
            x, y = b_pairs[rng.randrange(len(b_pairs))]
            bn += x; bd += y
        if ad and bd:
            stats.append(an / ad - bn / bd)
    if not stats:
        return None
    stats.sort()
    lo = stats[int((alpha / 2) * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return point, lo, hi, len(a_pairs), len(b_pairs)


def decompose(rows, ids):
    """Per-image (num, den) lists for each stage of the waterfall."""
    det, jaw, num_cond, joint = [], [], [], []
    n_img = 0
    for r in rows:
        iid = int(r["image_id"])
        if iid not in ids:
            continue
        n_img += 1
        rh, rt = _int(r["recall_hit"]), _int(r["recall_tot"])
        fh_, ft = _int(r["fdi_hit"]), _int(r["fdi_tot"])
        jh, jt = _int(r["jaw_hit"]), _int(r["jaw_tot"])
        if rh is not None and rt is not None:
            det.append((rh, rt))
        if jh is not None and jt is not None:
            jaw.append((jh, jt))
        if fh_ is not None and ft is not None:
            joint.append((fh_, ft))
            # Denominator for the conditional is the number MATCHED on that
            # same image, which is recall_hit: both loops walk the same pair
            # list, so a tooth counted as matched is exactly a tooth eligible
            # to be numbered.
            if rh is not None:
                num_cond.append((fh_, rh))
    return n_img, det, jaw, num_cond, joint


def main():
    manifests = K.load_manifests()
    rows = load_rows()
    configs = []
    for r in rows:
        if r["config"] not in configs:
            configs.append(r["config"])

    out = []
    stages = {}          # (config, subset) -> the per-image pair lists
    for cfg in configs:
        crows = [r for r in rows if r["config"] == cfg]
        for name, ids in subsets(manifests):
            n_img, det, jaw, cond, joint = decompose(crows, ids)
            if n_img == 0:
                continue
            stages[(cfg, name)] = {"detection_recall": det,
                                   "jaw_acc_given_detected": jaw,
                                   "fdi_acc_given_detected": cond,
                                   "fdi_acc_joint": joint}
            d = K.bootstrap_ci(det) if det else (float("nan"),) * 3
            j = K.bootstrap_ci(jaw) if jaw else (float("nan"),) * 3
            c = K.bootstrap_ci(cond) if cond else (float("nan"),) * 3
            t = K.bootstrap_ci(joint) if joint else (float("nan"),) * 3
            out.append({
                "config": cfg, "subset": name, "n_images": n_img,
                "gt_teeth": sum(y for _, y in joint) or sum(y for _, y in det),
                "matched": sum(x for x, _ in det),
                "detection_recall": d[0], "detection_recall_lo": d[1],
                "detection_recall_hi": d[2],
                "jaw_acc_given_detected": j[0], "jaw_acc_given_detected_lo": j[1],
                "jaw_acc_given_detected_hi": j[2],
                "fdi_acc_given_detected": c[0], "fdi_acc_given_detected_lo": c[1],
                "fdi_acc_given_detected_hi": c[2],
                "fdi_acc_joint": t[0], "fdi_acc_joint_lo": t[1],
                "fdi_acc_joint_hi": t[2],
                # recall x conditional should reproduce the joint figure when
                # both are computed over the same images; a mismatch means the
                # two denominators are not the same set.
                "product_check": (d[0] * c[0]) if det and cond else float("nan"),
            })

    p = os.path.join(RESULTS, "decomposition.csv")
    with open(p, "w", newline="") as fh:
        for line in K.provenance_header([
                "FDI accuracy split into detection and numbering.",
                "matching: greedy, highest score first, mask IoU >= 0.5",
                "detection recall     = matched / all ground-truth teeth",
                "jaw  | detected      = correct jaw / matched",
                "fdi  | detected      = correct FDI / matched",
                "joint                = correct FDI / all ground-truth teeth",
                "canonical_117            = every canonical-mapping image",
                "canonical_no_sparse_112  = 117 minus the 5 canonical images in",
                "                           the sparse anomaly-annotation regime",
                "full_arch_93             = complete arches (recall denominator)",
                "geometry configs detect with resnet50_9class_20k;",
                "network_numbering detects with resnext101_cascade_60k.",
                "Derived from results/per_image.csv -- no inference is re-run."]):
            fh.write(line + "\n")
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    # ---- complete versus partial ------------------------------------------
    a_name, b_name = CONTRAST
    contrast = []
    for cfg in configs:
        if (cfg, a_name) not in stages or (cfg, b_name) not in stages:
            continue
        row = {"config": cfg, "subset_a": a_name, "subset_b": b_name}
        for metric in ("detection_recall", "jaw_acc_given_detected",
                       "fdi_acc_given_detected", "fdi_acc_joint"):
            ap, bp = stages[(cfg, a_name)][metric], stages[(cfg, b_name)][metric]
            res = two_group_delta(ap, bp)
            if res is None:
                continue
            d, lo, hi, na, nb = res
            row[metric + "_a"] = sum(x for x, _ in ap) / sum(y for _, y in ap)
            row[metric + "_b"] = sum(x for x, _ in bp) / sum(y for _, y in bp)
            row[metric + "_delta"] = d
            row[metric + "_delta_lo"] = lo
            row[metric + "_delta_hi"] = hi
            row["n_images_a"], row["n_images_b"] = na, nb
        contrast.append(row)

    if contrast:
        cp = os.path.join(RESULTS, "complete_vs_partial.csv")
        with open(cp, "w", newline="") as fh:
            for line in K.provenance_header([
                    "FDI accuracy on COMPLETE arches versus PARTIAL ones.",
                    "Both subsets are canonical-mapping images and together they",
                    "partition canonical_117 exactly: 67 complete + 50 partial.",
                    "",
                    "The delta is an UNPAIRED two-group bootstrap: the 67 and the",
                    "50 are different radiographs, so there is no image to pair",
                    "on. Each group is resampled independently and the difference",
                    "of pooled ratios taken. This interval is wider than the",
                    "paired config-vs-config intervals in paired_deltas.txt and",
                    "is not comparable to them -- between-image variance is the",
                    "effect being measured here, not a nuisance removed by",
                    "pairing. Anyone quoting it should say 'unpaired'.",
                    "",
                    "Mechanism: sort_and_assign_fdi labels the leftmost detection",
                    "18/48 unconditionally and its gap check never runs before",
                    "the first tooth (app.py:126-135), so one missing tooth",
                    "shifts every FDI in that jaw. A complete arch has no gap to",
                    "mis-anchor on."]):
                fh.write(line + "\n")
            w = csv.DictWriter(fh, fieldnames=list(contrast[0].keys()))
            w.writeheader(); w.writerows(contrast)

        print(f"\n=== COMPLETE vs PARTIAL arches (canonical images only) ===")
        print(f"    {a_name} (n={contrast[0]['n_images_a']}) minus "
              f"{b_name} (n={contrast[0]['n_images_b']}), UNPAIRED bootstrap")
        print(f"{'config':<24}{'metric':<26}{'complete':>11}{'partial':>10}"
              f"{'delta':>11}{'95% interval':>22}")
        for r in contrast:
            for metric, label in (("fdi_acc_joint", "FDI accuracy (joint)"),
                                  ("fdi_acc_given_detected", "FDI | detected"),
                                  ("detection_recall", "detection recall"),
                                  ("jaw_acc_given_detected", "jaw | detected")):
                if metric + "_delta" not in r:
                    continue
                print(f"{r['config']:<24}{label:<26}"
                      f"{r[metric + '_a']*100:>10.1f}%{r[metric + '_b']*100:>9.1f}%"
                      f"{r[metric + '_delta']*100:>+10.2f}pp"
                      f"  [{r[metric + '_delta_lo']*100:>+6.2f}, "
                      f"{r[metric + '_delta_hi']*100:>+6.2f}]pp")
            print()

    for name, _ in subsets(manifests):
        sel = [r for r in out if r["subset"] == name]
        if not sel:
            continue
        print(f"\n=== {name}  (n={sel[0]['n_images']} images, "
              f"{sel[0]['gt_teeth']} ground-truth teeth) ===")
        print(f"{'config':<22}{'det recall':>22}{'jaw|det':>22}"
              f"{'FDI|det':>22}{'FDI joint':>22}{'r*c':>8}")
        for r in sel:
            def f(k):
                v, lo, hi = r[k], r[k + "_lo"], r[k + "_hi"]
                if v != v:
                    return f"{'--':>22}"
                return f"{v*100:>9.1f}% [{lo*100:>4.1f},{hi*100:>4.1f}]"
            pc = r["product_check"]
            pcs = f"{'--':>8}" if pc != pc else f"{pc*100:>7.1f}%"
            print(f"{r['config']:<22}{f('detection_recall')}"
                  f"{f('jaw_acc_given_detected')}{f('fdi_acc_given_detected')}"
                  f"{f('fdi_acc_joint')}{pcs}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
