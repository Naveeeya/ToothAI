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
        ("canonical_no_sparse_112", manifests["offset_subset"]),
        ("full_arch_93", manifests["full_arch"]),
    ]


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
    for cfg in configs:
        crows = [r for r in rows if r["config"] == cfg]
        for name, ids in subsets(manifests):
            n_img, det, jaw, cond, joint = decompose(crows, ids)
            if n_img == 0:
                continue
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
