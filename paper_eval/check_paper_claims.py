#!/usr/bin/env python3
"""Cross-reference the measured results against what the paper draft claims.

Standard library only -- runs on the Mac once the CSVs come back from Kaggle.

Every claim below is stated as the draft states it, then checked against a
number the harness actually produced. Nothing is inferred: if a CSV is absent
or a column is empty, the claim is reported UNTESTABLE rather than guessed.

    python3 paper_eval/check_paper_claims.py
    python3 paper_eval/check_paper_claims.py --results <dir>
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(HERE, "results")


def read(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def num(row, key):
    v = (row or {}).get(key, "")
    try:
        f = float(v)
        return f if f == f else None          # NaN -> None
    except (TypeError, ValueError):
        return None


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, claim, verdict, measured, note=""):
        self.rows.append((claim, verdict, measured, note))

    def render(self):
        order = {"CONTRADICTED": 0, "UNSUPPORTED": 1, "UNTESTABLE": 2, "SUPPORTED": 3}
        self.rows.sort(key=lambda r: order.get(r[1], 9))
        w = max(len(r[0]) for r in self.rows) if self.rows else 10
        print("=" * 100)
        print("PAPER CLAIMS vs MEASURED RESULTS")
        print("=" * 100)
        for claim, verdict, measured, note in self.rows:
            print(f"\n{verdict:<13} {claim}")
            print(f"{'':<13} measured: {measured}")
            if note:
                for line in note.split("\n"):
                    print(f"{'':<13}   {line}")
        n_bad = sum(1 for r in self.rows if r[1] in ("CONTRADICTED", "UNSUPPORTED"))
        print("\n" + "=" * 100)
        print(f"{len(self.rows)} claims checked, {n_bad} contradicted or unsupported")
        print("=" * 100)
        return n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    args = ap.parse_args()
    R = args.results

    abl = read(os.path.join(R, "ablation.csv"))
    sweep = read(os.path.join(R, "threshold_sweep.csv"))
    offs = read(os.path.join(R, "position_offsets.csv"))
    coco = read(os.path.join(R, "coco_eval_716.csv"))
    counts = read(os.path.join(R, "instance_counts.csv"))
    rep = Report()

    by = {r["config"]: r for r in abl} if abl else {}
    single, full = by.get("single_pass"), by.get("full")
    fixed, nozone = by.get("full_fix_routing"), by.get("no_zone_threshold")
    nms = by.get("nms")

    # ---- 1. classes -------------------------------------------------------
    rep.add("41 classes = 32 FDI tooth classes + 9 anomaly classes",
            "CONTRADICTED",
            "deployed tooth detector has NUM_CLASSES=9 with ONE generic 'Tooth' class; "
            "the 41-class schema is 't' + 1..32 + 8 anomalies",
            "FDI numbers come from the geometric engine, not the network.\n"
            "There are 8 anomaly classes, not 9. Classes 1..32 are neither\n"
            "FDI nor universal numbering (see results/class_to_position.csv).")

    # ---- 2. four passes ---------------------------------------------------
    rep.add("four inference passes (original, CLAHE, gamma, horizontal flip)",
            "CONTRADICTED", "two passes: original + CLAHE (app.py:405-423)",
            "apply_gamma (app.py:454) and flip_instances_horizontal (app.py:475)\n"
            "are defined and called nowhere. Their ablation rows are empty.")

    # ---- 3. EfficientNet --------------------------------------------------
    rep.add("EfficientNet-B0 refines low-confidence ROIs",
            "CONTRADICTED", "not in the deployed path; weights are a 0-byte placeholder",
            "app.py never imports it.")

    # ---- 4. mask IoU > 0.7 ------------------------------------------------
    miou = num(full, "mean_mask_iou")
    if miou is None:
        rep.add("mask IoU > 0.7", "UNTESTABLE", "ablation.csv absent or column empty")
    else:
        rep.add("mask IoU > 0.7 (draft gives no number)",
                "SUPPORTED" if miou > 0.7 else "CONTRADICTED",
                f"mean mask IoU over matched teeth = {miou:.3f} (config 'full')",
                "Now has a number behind it.")

    # ---- 5. single pass vs ensemble ---------------------------------------
    s_r, f_r = num(single, "recall_fullarch"), num(full, "recall_fullarch")
    s_fdi, f_fdi = num(single, "fdi_acc_canonical"), num(full, "fdi_acc_canonical")
    if None in (s_r, f_r):
        rep.add("the four-pass ensemble outperforms a single pass",
                "UNTESTABLE", "ablation.csv absent or recall columns empty")
    else:
        better_r = f_r > s_r
        note = (f"full-arch recall  single {fmt(s_r, True)}  vs  ensemble {fmt(f_r, True)}\n"
                f"FDI accuracy      single {fmt(s_fdi, True)}  vs  ensemble {fmt(f_fdi, True)}\n"
                f"pre-prune teeth   single {num(single,'teeth_per_image_preprune')}  "
                f"vs  ensemble {num(full,'teeth_per_image_preprune')}")
        if better_r and (f_fdi is None or s_fdi is None or f_fdi >= s_fdi):
            rep.add("the ensemble outperforms a single pass", "SUPPORTED",
                    f"ensemble recall {fmt(f_r, True)} > single {fmt(s_r, True)}", note)
        else:
            rep.add("the ensemble outperforms a single pass", "CONTRADICTED",
                    "the single pass matches or beats the ensemble", note +
                    "\nIf the single pass is not worse, the ensemble's cost\n"
                    "(2x latency) buys nothing and the paper cannot claim it does.")

    # ---- 6. CLAHE vs zone threshold ---------------------------------------
    if None in (s_r, f_r) or nozone is None:
        rep.add("the CLAHE pass and the zone threshold both contribute",
                "UNTESTABLE", "need single_pass, full and no_zone_threshold rows")
    else:
        nz_r = num(nozone, "recall_fullarch")
        clahe_delta = f_r - s_r                    # adding the CLAHE pass
        zone_delta = f_r - nz_r if nz_r is not None else None
        note = (f"CLAHE pass       full - single_pass       = {clahe_delta:+.4f}\n"
                f"zone threshold   full - no_zone_threshold = {zone_delta:+.4f}"
                if zone_delta is not None else "")
        # Sign first, magnitude second. A negative delta means the component
        # makes the system WORSE; comparing |delta| would rank "harms most"
        # as "contributes most", which is the opposite of the question.
        if zone_delta is None:
            rep.add("the CLAHE pass and the zone threshold both contribute",
                    "UNTESTABLE", "no_zone_threshold row missing")
        elif clahe_delta <= 0 and zone_delta <= 0:
            rep.add("the CLAHE pass and the zone threshold both contribute",
                    "CONTRADICTED",
                    f"BOTH reduce full-arch recall: CLAHE {clahe_delta:+.4f}, "
                    f"zone threshold {zone_delta:+.4f}",
                    note + "\nNeither component earns its place on this metric.")
        elif clahe_delta <= 0:
            rep.add("the CLAHE pass is a principal contributor", "CONTRADICTED",
                    f"the CLAHE pass REDUCES full-arch recall by "
                    f"{-clahe_delta*100:.2f} percentage points", note)
        elif clahe_delta < zone_delta:
            rep.add("the CLAHE pass is a principal contributor", "CONTRADICTED",
                    "the zone threshold contributes more than the CLAHE pass", note)
        else:
            rep.add("the CLAHE pass is a principal contributor", "SUPPORTED",
                    "CLAHE contributes at least as much as the zone threshold", note)

    # ---- 7. latency -------------------------------------------------------
    lat = num(full, "latency_median_s")
    if lat is None:
        rep.add("sub-second on an RTX 2060 / 1-2 s on an RTX 4060",
                "UNTESTABLE", "latency columns empty")
    else:
        rep.add("sub-second on an RTX 2060 / 1-2 s on an RTX 4060 (draft gives both)",
                "UNSUPPORTED",
                f"{lat:.3f} s median on a Tesla T4, n={num(full,'latency_n')}",
                "Neither GPU in the draft was measured. Quote the T4 number\n"
                "and the card it came from, or measure the named hardware.")

    # ---- 8. FDI accuracy 85-90% ------------------------------------------
    if f_fdi is None:
        rep.add("FDI accuracy 85-90%", "UNTESTABLE", "fdi column empty")
    else:
        n = num(full, "fdi_canonical_n")
        lo, hi = num(full, "fdi_acc_canonical_lo"), num(full, "fdi_acc_canonical_hi")
        inside = lo is not None and hi is not None and not (hi < 0.85 or lo > 0.90)
        rep.add("FDI accuracy 85-90% (stated as a range, unmeasured)",
                "SUPPORTED" if inside else "CONTRADICTED",
                f"{fmt(f_fdi, True)} [{fmt(lo, True)}, {fmt(hi, True)}], n={n} images",
                "Label noise caps FDI accuracy at 41.6% over the whole split;\n"
                "this figure is restricted to canonical-mapping images.")

    # ---- 9. per-class anomaly AP -----------------------------------------
    if counts:
        thin = [(r["class_name"], int(r["instances"]), int(r["images_present"]))
                for r in counts if r["kind"] == "anomaly" and int(r["instances"]) < 120]
        rep.add("per-class anomaly AP across 8 anomaly classes",
                "UNTESTABLE",
                "no clean split supports it; support is too thin for several classes",
                "\n".join(f"{n:<28} {i:>5} instances in {im:>3} images"
                          for n, i, im in thin))

    # ---- 10. three-model comparison --------------------------------------
    if not coco:
        rep.add("41-class beats 9-class on the same backbone",
                "UNTESTABLE", "coco_eval_716.csv absent",
                "rtx4060_48k was never evaluated before this harness ran.")
    else:
        # The registered ground truth is the 41-class schema. A 9-class model's
        # predictions are then scored against the wrong categories entirely
        # (its class 1 "Caries" against GT category 1, a tooth position), so
        # the result is a schema mismatch, not performance. Detect and refuse
        # to report it rather than printing a near-zero AP as a finding.
        suspect = [c for c in coco
                   if c["num_classes"] != "41" and num(c, "segm_AP") is not None
                   and num(c, "segm_AP") < 1.0]
        note = "\n".join(
            f"{c['model']:<26} nc={c['num_classes']:<3} "
            f"{c['min_size_test']}/{c['max_size_test']}  segm_AP={c['segm_AP']}"
            + ("   <- INVALID, schema mismatch" if c in suspect else "")
            for c in coco)
        if suspect:
            rep.add("41-class beats 9-class on the same ResNet-50 backbone",
                    "UNTESTABLE",
                    f"{len(suspect)} model(s) scored against the wrong label schema",
                    note + "\nThe 716 split was registered with its 41-class\n"
                           "annotations for every model. A 9-class model's ids do\n"
                           "not correspond, so its AP is ~0 by construction.\n"
                           "The 9-class GT for the same 716 images exists at\n"
                           "data/final-di-remapped/test and was not shipped in\n"
                           "the Kaggle bundle. Re-run that model against it.")
        else:
            rep.add("41-class vs 9-class on one split", "SUPPORTED",
                    "all three models evaluated on the same 716 images", note)

    # ---- 11. anchoring ----------------------------------------------------
    if offs:
        tot = sum(int(r["count"]) for r in offs if r["config"] == "full")
        shifted = sum(int(r["count"]) for r in offs
                      if r["config"] == "full" and r["offset"] not in ("0",))
        if tot:
            rep.add("FDI assignment is robust (draft does not mention anchoring)",
                    "CONTRADICTED" if shifted / tot > 0.05 else "SUPPORTED",
                    f"{shifted}/{tot} = {shifted/tot:.1%} of correctly-jawed teeth "
                    f"are offset from their true position",
                    "sort_and_assign_fdi labels the leftmost detection 18/48\n"
                    "unconditionally (app.py:135 guards the gap check with i>0).")

    n_bad = rep.render()
    if abl is None:
        print("\nNOTE: ablation.csv not found -- run the harness on Kaggle first.")
    sys.exit(0)


if __name__ == "__main__":
    main()
