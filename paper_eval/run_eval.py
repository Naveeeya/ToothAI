#!/usr/bin/env python3
"""Task D: ablation harness for the Tooth-AI paper.

Runs on the 716-image stratified test split. Requires torch + Detectron2, so
this is a Kaggle/GPU script -- it will not run on a CPU-only Mac.

Tooth metrics use ONLY the 9-class model (output/resnet50_9class_20k). That is
sound because nothing on the anomaly path alters the teeth list: verified in
round 2 by diffing every score mutation in app.py (:800 finding, :811 anomaly,
:2293 anomaly) -- none touches tooth['score'], and the two functions that
remove teeth (recursive_prune_teeth, resolve_fdi_conflicts) key only on
tooth score and FDI. The pathology model is loaded for its COCO eval only.

CONFIGURATIONS
    single_pass        original image only, deployed thresholds, no fusion
    full               deployed system: original + CLAHE, WBF(0.55, penalty .6)
    no_zone_threshold  two passes, uniform tau, no central relaxation
    nms                two passes, NMS instead of the custom fusion
    no_gamma           NOT RUN -- gamma pass is not in the deployed system
    no_flip            NOT RUN -- flip pass is not in the deployed system
    no_roi_refine      NOT RUN -- EfficientNet weights are a 0-byte placeholder

SUBSETS (each justified by a different ground-truth property)
    recall            reported separately for full-arch and sparse images,
                      because sparse labelling is concentrated on abnormal
                      teeth (43.1% of sparse labelled teeth overlap an anomaly
                      box, vs 0.0% of full-arch ones)
    fdi accuracy      117 canonical images (headline, includes missing teeth)
                      and 67 full-arch canonical images (clean-label bound)
    jaw accuracy      117 canonical images. An earlier version scored this on
                      all 93 full-arch images, on the claim that class->jaw was
                      consistent even where class->position was scrambled. That
                      claim was wrong: 25 of the 26 non-canonical full-arch
                      images use a different jaw partition, overlapping the
                      canonical upper set in 9 of 16 classes
                      (check_jaw_membership.py). The 93-image figure is kept
                      under jaw_acc_fullarch_confounded and should not be
                      quoted as a model result.
    offset histogram  117 canonical images, which include 50 partial arches
                      where a missing third molar can actually shift the jaw

Usage:
    python3 paper_eval/run_eval.py --config full
    python3 paper_eval/run_eval.py --all
    python3 paper_eval/run_eval.py --all --smoke-test
    python3 paper_eval/run_eval.py --coco-eval
    python3 paper_eval/run_eval.py --latency
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as K
import pipeline as P

RESULTS = os.path.join(K.ROOT, "paper_eval", "results")


def _load_leakage():
    """Exact train/test image overlap per model.

    Computed on the workstation (the bundle ships only test data) and shipped
    as JSON. A model that saw test images in training is optimistic on this
    split and every row it produces is indicative only.
    """
    p = os.path.join(RESULTS, "train_test_overlap.json")
    try:
        with open(p) as fh:
            return {k: v for k, v in json.load(fh).items()
                    if not k.startswith("_")}
    except Exception:
        return {}


LEAKAGE = _load_leakage()

RUNNABLE = ["single_pass", "full", "full_fix_routing", "no_zone_threshold", "nms",
            "single_pass_fix_routing"]

# full_fix_routing runs the deployed two-pass system with the app.py:953-958
# class-routing bug CORRECTED. It is a separate row rather than a flag on the
# whole run so the paper can show the deployed number and the fixed number
# side by side.
BASE_CONFIG = {"full_fix_routing": "full", "single_pass_fix_routing": "single_pass"}
FIXED_ROUTING = {"full_fix_routing", "single_pass_fix_routing"}
# single_pass_fix_routing: one pass (original image only) with the routing fix. The existing
# single_pass row carries the bug, so it cannot be compared like-for-like with full_fix_routing.
NOT_RUN = {
    "no_gamma": "not in deployed system (apply_gamma defined at app.py:454, called nowhere)",
    "no_flip": "not in deployed system (flip_instances_horizontal at app.py:475, called nowhere)",
    "no_roi_refine": "EfficientNet weights are a 0-byte placeholder "
                     "(workspace/tooth-ai/models/effnet_fdi_final.pth.placeholder)",
}

TEETH_TAU = 0.20        # app.py:2192 slider default
ANOM_TAU = 0.15         # app.py:2197 slider default
WBF_IOU = 0.55          # app.py:419
NMS_IOU = 0.4           # app.py:2314 (DIoU)


# --------------------------------------------------------------------------
LOADED_INPUT_SIZE = {}     # model_dir -> (min, max) as the cfg OBJECT holds it


def load_predictor(model_dir, score_thresh=0.05, announce=True):
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(model_dir, "config.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh   # app.py:375
    # Read the size off the live cfg object, not the YAML on disk: merge_from_file
    # plus any default can diverge from the file, and what the model actually
    # ran at is the only thing the paper may state.
    mn, mx = cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST
    LOADED_INPUT_SIZE[os.path.basename(model_dir.rstrip("/"))] = (mn, mx)
    if announce:
        print(f"  {os.path.basename(model_dir.rstrip('/')):<26} "
              f"cfg.INPUT.MIN_SIZE_TEST={mn}  cfg.INPUT.MAX_SIZE_TEST={mx}  "
              f"NUM_CLASSES={cfg.MODEL.ROI_HEADS.NUM_CLASSES}  device={cfg.MODEL.DEVICE}")
    return DefaultPredictor(cfg)


def nms_fuse(outputs_list, iou_threshold=WBF_IOU):
    """Plain per-class NMS across the concatenated passes.

    The `nms` ablation: same passes, same IoU, but the custom fusion is
    replaced by suppression. No box averaging and no consensus penalty, so it
    isolates what the fusion rule itself contributes.
    """
    import torch
    from detectron2.structures import Instances, Boxes
    boxes, scores, classes, masks = [], [], [], []
    for out in outputs_list:
        inst = out["instances"].to("cpu")
        if len(inst) == 0:
            continue
        boxes.append(inst.pred_boxes.tensor.numpy())
        scores.append(inst.scores.numpy())
        classes.append(inst.pred_classes.numpy())
        if inst.has("pred_masks"):
            masks.append(inst.pred_masks.numpy())
    size = outputs_list[0]["instances"].image_size
    if not boxes:
        e = Instances(size)
        e.pred_boxes = Boxes(torch.empty(0, 4)); e.scores = torch.empty(0)
        e.pred_classes = torch.empty(0, dtype=torch.int64)
        return {"instances": e}
    boxes = np.concatenate(boxes); scores = np.concatenate(scores)
    classes = np.concatenate(classes)
    masks = np.concatenate(masks) if masks else None

    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        idx = idx[np.argsort(-scores[idx])]
        while len(idx):
            i = idx[0]; keep.append(i)
            if len(idx) == 1:
                break
            rest = idx[1:]
            ious = np.array([P.compute_iou(boxes[i], boxes[j]) for j in rest])
            idx = rest[ious <= iou_threshold]

    keep = np.array(sorted(keep), dtype=int)
    inst = Instances(size)
    inst.pred_boxes = Boxes(torch.tensor(boxes[keep], dtype=torch.float32))
    inst.scores = torch.tensor(scores[keep], dtype=torch.float32)
    inst.pred_classes = torch.tensor(classes[keep], dtype=torch.int64)
    if masks is not None:
        inst.pred_masks = torch.tensor(masks[keep])
    return {"instances": inst}


def infer(predictor, image_bgr, config, threshold=0.05):
    """Produce fused model output for one configuration."""
    predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    out_orig = predictor(image_bgr)
    if config == "single_pass":
        return out_orig
    clahe = P.apply_clahe(image_bgr, clip_limit=2.5, tile_size=(8, 8))   # app.py:411
    out_clahe = predictor(clahe)
    if config == "nms":
        return nms_fuse([out_orig, out_clahe], iou_threshold=WBF_IOU)
    return P.weighted_box_fusion([out_orig, out_clahe],
                                 iou_threshold=WBF_IOU,
                                 skip_box_threshold=threshold)


def teeth_pipeline(outputs, width, config, tau=TEETH_TAU, fix_cls_routing=False):
    """Everything downstream of fusion, in app.py's order. Returns
    (final_teeth, pre_prune_count, pruning_fired)."""
    if config == "no_zone_threshold":
        # uniform tau: bypass the central relaxation by running the zone rule
        # with a width that puts nothing in the central band is fragile, so
        # filter explicitly instead, then hand the survivors on unchanged.
        raw, _ = P.process_predictions(outputs, image_width=width,
                                       teeth_threshold=tau, anomaly_threshold=ANOM_TAU,
                                       class_map=None, fix_cls_routing=fix_cls_routing)
        raw = [t for t in raw if t["score"] >= tau]
    else:
        raw, _ = P.process_predictions(outputs, image_width=width,
                                       teeth_threshold=tau, anomaly_threshold=ANOM_TAU,
                                       class_map=None, fix_cls_routing=fix_cls_routing)

    teeth_nms = P.DentalGeometricEngine.apply_nms(raw, iou_threshold=NMS_IOU)  # app.py:2314
    pre_prune = len(teeth_nms)

    upper, lower = P.DentalGeometricEngine.split_jaws(teeth_nms)
    all_teeth = (P.DentalGeometricEngine.sort_and_assign_fdi(upper, "upper")
                 + P.DentalGeometricEngine.sort_and_assign_fdi(lower, "lower"))
    all_teeth = P.DentalGeometricEngine.map_anomalies(all_teeth, [])
    all_teeth = P.clean_clinical_findings(all_teeth)
    fired = len(all_teeth) > 32
    all_teeth, _ = P.validate_anatomy(all_teeth, [], anomaly_threshold=ANOM_TAU)
    return all_teeth, pre_prune, fired


# --------------------------------------------------------------------------
def evaluate(config, predictor, images, gt_teeth, manifests, class_map,
             limit=None, fix_cls_routing=False, ids=None):
    """Run one configuration over the split and accumulate every metric."""
    infer_cfg = BASE_CONFIG.get(config, config)
    if config in FIXED_ROUTING:
        fix_cls_routing = True
    acc = {
        "recall_full": [], "recall_anom": [],
        "central_full": [], "periph_full": [],
        "per_image": [],
        "fdi_canonical": [], "fdi_fullarch": [],
        "jaw": [], "jaw_canonical": [], "offsets": Counter(),
        "mask_ious": [], "pre_prune": [], "final": [], "pruned": 0,
        "per_tooth": defaultdict(lambda: [0, 0]),
    }
    if ids is None:
        ids = sorted(gt_teeth) if limit is None else sorted(gt_teeth)[:limit]
    for n, iid in enumerate(ids, 1):
        meta = images[iid]
        path = os.path.join(K.SPLIT_DIR, meta["file_name"])
        import cv2
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        out = infer(predictor, img, infer_cfg)
        teeth, pre, fired = teeth_pipeline(out, w, infer_cfg,
                                           fix_cls_routing=fix_cls_routing)
        acc["pre_prune"].append(pre)
        acc["final"].append(len(teeth))
        acc["pruned"] += int(fired)

        gts = gt_teeth[iid]
        pairs = K.match_detections(gts, teeth, h, w)

        n_hit = n_tot = 0
        n_chit = n_ctot = 0
        n_phit = n_ptot = 0
        for gt, pred, iou in pairs:
            n_tot += 1
            central = K.is_central(gt["bbox"], meta["width"])
            n_ctot += int(central)
            n_ptot += int(not central)
            if pred is not None:
                n_hit += 1
                n_chit += int(central)
                n_phit += int(not central)
                acc["mask_ious"].append(iou)

        in_fa = iid in manifests["full_arch"]
        if in_fa:
            acc["recall_full"].append((n_hit, n_tot))
            # Peripheral recall alongside central, so the two can be compared
            # directly on identical images.
            acc["periph_full"].append((n_phit, n_ptot))
            # Central-zone recall comes ONLY from full-arch images: it is the
            # paper's core claim, and a complete arch is the only case where an
            # absent label means an absent tooth rather than an unannotated one.
            acc["central_full"].append((n_chit, n_ctot))
        elif iid in manifests["sparse"]:
            # anomaly-annotation regime -- reported separately, under its own
            # name, and excluded from central-zone and per-tooth recall
            acc["recall_anom"].append((n_hit, n_tot))

        # --- FDI / jaw / offset, only where the labels support it ----------
        in_canon = iid in manifests["canonical"]
        in_fdi = iid in manifests["fdi_evaluable"]
        in_full = iid in manifests["full_arch"]
        # Reset every iteration. Python scopes these to the function, so an
        # image that skips the block below would otherwise silently inherit
        # the previous image's counts into its per-image row.
        fdi_hit = fdi_tot = jaw_hit = jaw_tot = None
        if in_canon or in_full:
            fdi_hit = fdi_tot = jaw_hit = jaw_tot = 0
            for gt, pred, iou in pairs:
                jaw, rank, true_fdi = class_map[gt["category_id"]]
                if pred is None:
                    fdi_tot += 1
                    continue
                pf = pred.get("fdi")
                fdi_tot += 1
                if pf == true_fdi:
                    fdi_hit += 1
                pred_jaw = "upper" if pf in K.FDI_UPPER else "lower"
                jaw_tot += 1
                jaw_hit += int(pred_jaw == jaw)
                if pred_jaw == jaw and iid in manifests["offset_subset"]:
                    # Canonical images only (on the 26 scrambled ones the
                    # "true" rank is meaningless), and sparse ones dropped
                    # (their gaps are unannotated teeth, not absent ones), so
                    # an offset here really is the anchoring flaw.
                    seq = K.FDI_UPPER if jaw == "upper" else K.FDI_LOWER
                    acc["offsets"][seq.index(pf) - rank] += 1
            if in_canon:
                acc["fdi_canonical"].append((fdi_hit, fdi_tot))
            if in_fdi:
                acc["fdi_fullarch"].append((fdi_hit, fdi_tot))
            if in_canon:
                # The headline jaw figure. class->jaw is NOT reliable on the 26
                # non-canonical full-arch images: 25 of them use a different
                # jaw partition (check_jaw_membership.py), so jaw accuracy over
                # all 93 measures annotation convention as much as the model.
                acc["jaw_canonical"].append((jaw_hit, jaw_tot))
            if in_full:
                acc["jaw"].append((jaw_hit, jaw_tot))

        # --- per-position recall: FULL-ARCH IMAGES ONLY (n = 93) ------------
        # Same reasoning as central-zone recall: only on a complete arch does
        # an absent label mean an absent tooth, so only there is a per-position
        # denominator meaningful.
        if in_fa:
            for gt, pred, iou in pairs:
                _, _, true_fdi = class_map[gt["category_id"]]
                acc["per_tooth"][true_fdi][1] += 1
                if pred is not None:
                    acc["per_tooth"][true_fdi][0] += 1

        # One row per image, so paired (per-image) deltas between configs can
        # be bootstrapped later. Marginal CIs that overlap neither establish
        # nor refute a difference; only the paired delta does.
        acc["per_image"].append({
            "image_id": iid, "config": config,
            "full_arch": int(in_fa), "canonical": int(in_canon),
            "fdi_evaluable": int(in_fdi),
            "sparse": int(iid in manifests["sparse"]),
            "recall_hit": n_hit, "recall_tot": n_tot,
            "central_hit": n_chit, "central_tot": n_ctot,
            "periph_hit": n_phit, "periph_tot": n_ptot,
            "fdi_hit": "" if fdi_hit is None else fdi_hit,
            "fdi_tot": "" if fdi_tot is None else fdi_tot,
            "jaw_hit": "" if jaw_hit is None else jaw_hit,
            "jaw_tot": "" if jaw_tot is None else jaw_tot,
            "pred_preprune": pre, "pred_final": len(teeth),
        })
        if n % 25 == 0:
            print(f"    {config}: {n}/{len(ids)}", flush=True)
    return acc


# --------------------------------------------------------------------------
def network_numbering(predictor, images, gt_teeth, manifests, class_map,
                      subset="canonical", taus=(TEETH_TAU, 0.05), limit=None):
    """Score a 41-class model's OWN tooth numbering, end to end.

    The COCO comparison cannot settle the design question: one generic class
    is intrinsically easier to score than 32, so a higher AP-Tooth than
    mean-AP-over-positions is expected whatever the numbering is worth.

    This measures the same thing both ways instead -- same images, same
    matching rule, same denominator:

      network   the 41-class model's predicted class 1..32, mapped to an FDI
                through the empirical class_to_position table. No geometric
                engine at all.
      geometry  the deployed pipeline: one generic Tooth class, then y-median
                jaw split and left-to-right ordering.

    Duplicate FDIs are resolved winner-takes-all by score, mirroring
    resolve_fdi_conflicts so neither side gets a free pass on duplicates.

    Three things are recorded that the first version did not:

    1. DETECTION SEPARATELY FROM NUMBERING. Geometry detects with the 9-class
       model and the network detects with the Cascade, so a joint score
       (correct FDI / all ground-truth teeth) confounds two detectors with two
       numbering schemes. recall_hit / jaw_hit are written per image so the
       conditional "numbering accuracy given the tooth was found" can be
       computed on the same rows.

    2. A MAPPING-FREE CONTROL. class_to_position was derived from this split,
       so a sceptic can ask whether the result depends on it. The control runs
       a fully independent pipeline -- dedup keyed on the raw class id, its own
       matching pass, correct iff predicted class id == ground-truth category
       id -- and never consults class_map at all. The mapping is a bijection,
       so the two must agree on every image; the point is to demonstrate it
       rather than assert it.

    3. MORE THAN ONE THRESHOLD. Per-image rows are kept at the network's own
       optimum as well as at the deployed tau, so the paired delta against
       geometry can be taken where the baseline is strongest.
    """
    import cv2
    ids = sorted(i for i in gt_teeth if i in manifests[subset])
    if limit:
        ids = ids[:limit]
    taus = list(taus)
    print(f"  scoring network numbering on {len(ids)} {subset} images, "
          f"taus={taus}")

    # class_map is class_id -> (jaw, rank, fdi); invert to what a prediction means
    cls_to_fdi = {c: fdi for c, (_j, _r, fdi) in class_map.items()}
    per_img = {t: [] for t in taus}
    agg = {t: {"det": [], "jaw": [], "cond": [], "joint": [],
               "mapfree": [], "cand": [], "final": [],
               "disagree_images": 0} for t in taus}

    for n, iid in enumerate(ids, 1):
        meta = images[iid]
        img = cv2.imread(os.path.join(K.SPLIT_DIR, meta["file_name"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        inst = predictor(img)["instances"].to("cpu")
        boxes = inst.pred_boxes.tensor.numpy()
        scores = inst.scores.numpy()
        classes = inst.pred_classes.numpy()
        masks = inst.pred_masks.numpy() if inst.has("pred_masks") else None
        gts = gt_teeth[iid]
        gtm = K.gt_masks_for(gts, h, w)       # decode GT once, not per tau

        for tau in taus:
            cand = []
            for i in range(len(inst)):
                c = int(classes[i])
                if c not in cls_to_fdi or float(scores[i]) < tau:
                    continue                   # class 0 't' and 33-40 anomalies
                cand.append({"box": boxes[i].tolist(), "score": float(scores[i]),
                             "cls": c, "fdi": cls_to_fdi[c],
                             "mask": masks[i] if masks is not None else None})

            # --- mapped path: dedup per FDI, score against the mapped FDI ---
            best = {}
            for d in sorted(cand, key=lambda d: -d["score"]):
                best.setdefault(d["fdi"], d)
            preds = list(best.values())
            pairs = K.match_detections(gts, preds, h, w, gt_masks=gtm)

            det_hit = fdi_hit = jaw_hit = 0
            tot = len(pairs)
            for gt, pred, _iou in pairs:
                jaw, _rank, true_fdi = class_map[gt["category_id"]]
                if pred is None:
                    continue
                det_hit += 1
                if pred["fdi"] == true_fdi:
                    fdi_hit += 1
                pred_jaw = "upper" if pred["fdi"] in K.FDI_UPPER else "lower"
                jaw_hit += int(pred_jaw == jaw)

            # --- mapping-free control: never consults class_map -------------
            mbest = {}
            for d in sorted(cand, key=lambda d: -d["score"]):
                mbest.setdefault(d["cls"], d)
            mpreds = list(mbest.values())
            mpairs = K.match_detections(gts, mpreds, h, w, gt_masks=gtm)
            map_hit = sum(1 for gt, pred, _i in mpairs
                          if pred is not None and pred["cls"] == gt["category_id"])
            if map_hit != fdi_hit:
                agg[tau]["disagree_images"] += 1

            a = agg[tau]
            a["det"].append((det_hit, tot))
            a["joint"].append((fdi_hit, tot))
            a["cond"].append((fdi_hit, det_hit))
            a["jaw"].append((jaw_hit, det_hit))
            a["mapfree"].append((map_hit, tot))
            a["cand"].append(len(cand))
            a["final"].append(len(preds))

            per_img[tau].append({
                "image_id": iid, "config": _nn_name(tau),
                "full_arch": int(iid in manifests["full_arch"]),
                "canonical": int(iid in manifests["canonical"]),
                "fdi_evaluable": int(iid in manifests["fdi_evaluable"]),
                "sparse": int(iid in manifests["sparse"]),
                "recall_hit": det_hit, "recall_tot": tot,
                "central_hit": "", "central_tot": "",
                "periph_hit": "", "periph_tot": "",
                "fdi_hit": fdi_hit, "fdi_tot": tot,
                "jaw_hit": jaw_hit, "jaw_tot": det_hit,
                "pred_preprune": len(cand), "pred_final": len(preds)})
        if n % 25 == 0:
            print(f"    network_numbering: {n}/{len(ids)}", flush=True)

    out = []
    for tau in taus:
        a = agg[tau]
        r = {"config": _nn_name(tau), "tau": tau, "n_images": len(a["det"]),
             "gt_teeth": sum(y for _, y in a["det"]),
             "matched": sum(x for x, _ in a["det"])}
        for key, src in (("detection_recall", "det"),
                         ("jaw_acc_given_detected", "jaw"),
                         ("fdi_acc_given_detected", "cond"),
                         ("fdi_acc_joint", "joint"),
                         ("fdi_acc_joint_mapping_free", "mapfree")):
            v, lo, hi = K.bootstrap_ci(a[src])
            r[key], r[key + "_lo"], r[key + "_hi"] = v, lo, hi
        r["mapping_free_disagreements"] = a["disagree_images"]
        r["detections_per_image"] = (statistics.mean(a["cand"])
                                     if a["cand"] else float("nan"))
        r["after_dedup_per_image"] = (statistics.mean(a["final"])
                                      if a["final"] else float("nan"))
        out.append(r)

    p = os.path.join(RESULTS, "network_numbering_decomp.csv")
    with open(p, "w", newline="") as fh:
        for line in K.provenance_header([
                "network numbering, detection separated from numbering.",
                "matching: greedy, highest score first, mask IoU >= 0.5",
                "detection recall = matched / all ground-truth teeth",
                "fdi | detected   = correct FDI / matched",
                "joint            = correct FDI / all ground-truth teeth",
                "fdi_acc_joint_mapping_free scores predicted class id against",
                "  ground-truth category id with NO position mapping, through a",
                "  separate dedup and a separate matching pass. It must equal",
                "  fdi_acc_joint, because class_to_position is a bijection;",
                "  mapping_free_disagreements counts images where it did not."]):
            fh.write(line + "\n")
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    print(f"\n  {'config':<28}{'det recall':>20}{'jaw|det':>20}"
          f"{'FDI|det':>20}{'FDI joint':>20}{'mapping-free':>20}")
    for r in out:
        def f(k):
            v, lo, hi = r[k], r[k + "_lo"], r[k + "_hi"]
            return f"{v*100:>8.1f}% [{lo*100:>4.1f},{hi*100:>4.1f}]"
        print(f"  {r['config']:<28}{f('detection_recall')}"
              f"{f('jaw_acc_given_detected')}{f('fdi_acc_given_detected')}"
              f"{f('fdi_acc_joint')}{f('fdi_acc_joint_mapping_free')}")
    for r in out:
        nd = r["mapping_free_disagreements"]
        verdict = ("identical -- the result does not depend on the mapping"
                   if nd == 0 else "MISMATCH -- investigate before reporting")
        print(f"  mapping-free check @ tau={r['tau']}: {nd} of "
              f"{r['n_images']} images disagree ({verdict})")
    print(f"  wrote {p}")
    rows = [row for t in taus for row in per_img[t]]
    return rows, out


def _nn_name(tau):
    """Config label for a network-numbering run at one threshold."""
    return f"network_numbering_t{int(round(tau * 100)):03d}"


def numbering_sweep(predictor, images, gt_teeth, manifests, class_map,
                    subset="canonical", limit=None):
    """Sweep tau for network numbering and find the network's OWN optimum.

    Comparing geometry against the network at tau=0.20 is comparing it against
    a threshold chosen for a different model. If the gap only exists because
    the baseline is mistuned it is not a real finding, so the baseline gets
    its best shot: inference is run ONCE per image and only the threshold
    varies over the cached detections.
    """
    import cv2
    taus = [0.05, 0.08, 0.12, 0.18, 0.20, 0.25, 0.30, 0.35, 0.42, 0.50]
    cls_to_fdi = {c: fdi for c, (_j, _r, fdi) in class_map.items()}
    ids = sorted(i for i in gt_teeth if i in manifests[subset])
    if limit:
        ids = ids[:limit]
    print(f"  sweeping tau over {len(ids)} {subset} images "
          f"(inference cached, threshold varied)")

    per = {t: {"rows": [], "cand": [], "final": []} for t in taus}
    for n, iid in enumerate(ids, 1):
        meta = images[iid]
        img = cv2.imread(os.path.join(K.SPLIT_DIR, meta["file_name"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        inst = predictor(img)["instances"].to("cpu")
        boxes = inst.pred_boxes.tensor.numpy()
        scores = inst.scores.numpy()
        classes = inst.pred_classes.numpy()
        masks = inst.pred_masks.numpy() if inst.has("pred_masks") else None
        gts = gt_teeth[iid]
        gtm = K.gt_masks_for(gts, h, w)          # decode GT once, not per tau

        for t in taus:
            cand = []
            for i in range(len(inst)):
                c = int(classes[i])
                if c not in cls_to_fdi or float(scores[i]) < t:
                    continue
                cand.append({"box": boxes[i].tolist(), "score": float(scores[i]),
                             "fdi": cls_to_fdi[c],
                             "mask": masks[i] if masks is not None else None})
            best = {}
            for d in sorted(cand, key=lambda d: -d["score"]):
                best.setdefault(d["fdi"], d)
            preds = list(best.values())
            pairs = K.match_detections(gts, preds, h, w, gt_masks=gtm)
            hit = tot = 0
            for gt, pred, _iou in pairs:
                _j, _r, true_fdi = class_map[gt["category_id"]]
                tot += 1
                if pred is not None and pred.get("fdi") == true_fdi:
                    hit += 1
            per[t]["rows"].append((hit, tot))
            per[t]["cand"].append(len(cand))
            per[t]["final"].append(len(preds))
        if n % 25 == 0:
            print(f"    numbering sweep: {n}/{len(ids)}", flush=True)

    out = []
    for t in taus:
        v, lo, hi = K.bootstrap_ci(per[t]["rows"])
        c = statistics.mean(per[t]["cand"]) if per[t]["cand"] else float("nan")
        f = statistics.mean(per[t]["final"]) if per[t]["final"] else float("nan")
        out.append({"tau": t, "fdi_accuracy": v, "fdi_accuracy_lo": lo,
                    "fdi_accuracy_hi": hi,
                    "detections_per_image": c, "after_dedup_per_image": f,
                    "duplicates_discarded_per_image": c - f,
                    "n_images": len(per[t]["rows"])})

    p = os.path.join(RESULTS, "numbering_sweep.csv")
    with open(p, "w", newline="") as fh:
        for line in K.provenance_header([
                "network numbering (resnext101_cascade_60k classes 1-32 mapped",
                "  through the empirical class_to_position table), tau swept",
                "detections cached once per image; only the threshold varies",
                "duplicates_discarded = winner-takes-all per FDI, as",
                "  resolve_fdi_conflicts does for the geometric pipeline"]):
            fh.write(line + "\n")
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    print(f"\n{'tau':>6}{'FDI acc':>10}{'95% interval':>20}{'det/img':>10}"
          f"{'after dedup':>13}{'dups dropped':>14}")
    print("-" * 74)
    best = max(out, key=lambda r: (r["fdi_accuracy"] if r["fdi_accuracy"] ==
                                   r["fdi_accuracy"] else -1))
    for r in out:
        mark = "  <- best" if r is best else ""
        print(f"{r['tau']:>6.2f}{r['fdi_accuracy']*100:>9.1f}%"
              f"  [{r['fdi_accuracy_lo']*100:>5.1f},{r['fdi_accuracy_hi']*100:>5.1f}]"
              f"{r['detections_per_image']:>10.1f}{r['after_dedup_per_image']:>13.1f}"
              f"{r['duplicates_discarded_per_image']:>14.1f}{mark}")
    print("-" * 74)
    print(f"  network's OWN optimum: tau={best['tau']}, "
          f"FDI accuracy {best['fdi_accuracy']*100:.1f}% "
          f"[{best['fdi_accuracy_lo']*100:.1f}, {best['fdi_accuracy_hi']*100:.1f}]")
    print(f"  wrote {p}")
    return out, best


def routing_ab(predictor, images, gt_teeth, gt_anoms, n_images=5):
    """A/B the app.py:953-958 class-routing bug on ANOMALY-BEARING images.

    The bug routes an anomaly detection from the 9-class model into the TEETH
    list. It therefore cannot fire on an image where that model finds no
    anomaly -- and the 93 full-arch images carry zero anomaly annotations, so
    the ordinary smoke sample cannot settle the question at all.

    Sample from the 380 images that DO carry anomaly labels, run the teeth
    pipeline both ways, and diff the resulting teeth lists. If they are
    identical on every image, the bug does not affect tooth metrics and the
    audit must say so rather than implying otherwise.
    """
    import cv2
    pool = sorted(i for i in gt_teeth if gt_anoms.get(i) and gt_teeth.get(i))
    if not pool:
        print("  no anomaly-bearing images with tooth labels -- cannot test")
        return []
    step = max(1, len(pool) // n_images)
    chosen = [pool[k * step] for k in range(min(n_images, len(pool)))]

    print(f"\n{'='*78}")
    print("ROUTING A/B -- app.py:953-958, on ANOMALY-BEARING images")
    print(f"{'='*78}")
    print(f"pool: {len(pool)} images carrying both tooth and anomaly labels; "
          f"sampling {len(chosen)}")

    rows, any_diff = [], False
    for iid in chosen:
        meta = images[iid]
        img = cv2.imread(os.path.join(K.SPLIT_DIR, meta["file_name"]))
        if img is None:
            continue
        w = img.shape[1]
        out = infer(predictor, img, "full")
        both = {}
        for fix in (False, True):
            teeth, pre, _f = teeth_pipeline(out, w, "full", fix_cls_routing=fix)
            both[fix] = {
                "n_final": len(teeth),
                "n_preprune": pre,
                "fdis": sorted(t.get("fdi") for t in teeth),
                "sig": sorted((round(float(t["box"][0]), 1), round(float(t["box"][1]), 1),
                               round(float(t.get("score", 0)), 4)) for t in teeth),
            }
        same = both[False]["sig"] == both[True]["sig"]
        any_diff |= (not same)
        rows.append((iid, both, same))
        print(f"\n  image_id {iid}  ({meta['file_name'][:44]})")
        print(f"    GT: {len(gt_teeth[iid])} teeth, {len(gt_anoms[iid])} anomalies")
        for fix in (False, True):
            b = both[fix]
            tag = "FIXED " if fix else "buggy "
            print(f"    {tag} pre-prune {b['n_preprune']:>3}  final {b['n_final']:>3}  "
                  f"FDIs {b['fdis']}")
        print(f"    identical teeth lists: {same}")
        if not same:
            only_buggy = set(both[False]["sig"]) - set(both[True]["sig"])
            only_fixed = set(both[True]["sig"]) - set(both[False]["sig"])
            print(f"      only in buggy (x,y,score): {sorted(only_buggy)[:6]}")
            print(f"      only in fixed (x,y,score): {sorted(only_fixed)[:6]}")

    print(f"\n{'-'*78}")
    if any_diff:
        n = sum(1 for _i, _b, same in rows if not same)
        print(f"VERDICT: the teeth lists DIFFER on {n}/{len(rows)} images.")
        print("  The routing bug DOES affect tooth metrics. Report both rows.")
    else:
        print(f"VERDICT: teeth lists are IDENTICAL on all {len(rows)} images.")
        print("  On this evidence the bug does NOT affect tooth metrics, and")
        print("  the audit must be worded to say so -- it misroutes anomalies")
        print("  out of the anomaly list, but nothing reaches the teeth list.")
    print(f"{'-'*78}")
    return rows


def tau_sweep(predictor, images, gt_teeth, limit=None):
    """Sweep tau, caching post-fusion output once per image."""
    taus = [0.05, 0.08, 0.12, 0.18, 0.20, 0.25, 0.30, 0.35, 0.42, 0.50]
    ids = sorted(gt_teeth) if limit is None else sorted(gt_teeth)[:limit]
    per = {t: {"c": [], "p": [], "n": []} for t in taus}
    import cv2
    for n, iid in enumerate(ids, 1):
        meta = images[iid]
        img = cv2.imread(os.path.join(K.SPLIT_DIR, meta["file_name"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        out = infer(predictor, img, "full")          # cache: fuse ONCE
        gts = gt_teeth[iid]
        gtm = K.gt_masks_for(gts, h, w)              # decode GT ONCE, not per tau
        for t in taus:
            teeth, pre, _ = teeth_pipeline(out, w, "full", tau=t)
            pairs = K.match_detections(gts, teeth, h, w, gt_masks=gtm)
            ch = ct = ph = pt = 0
            for gt, pred, _iou in pairs:
                if K.is_central(gt["bbox"], meta["width"]):
                    ct += 1; ch += int(pred is not None)
                else:
                    pt += 1; ph += int(pred is not None)
            per[t]["c"].append((ch, ct)); per[t]["p"].append((ph, pt))
            per[t]["n"].append(pre)
        if n % 25 == 0:
            print(f"    sweep: {n}/{len(ids)}", flush=True)
    return taus, per


# --------------------------------------------------------------------------
def coco_eval(model_dirs, limit=None):
    """Re-evaluate each model on the 716 split at its OWN configured size."""
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    rows = []
    for md in model_dirs:
        name = os.path.basename(md)
        # Load first: the model's NUM_CLASSES decides which annotation schema
        # it may be scored against. Registering the 41-class GT for a 9-class
        # model produced segm_AP 0.0002 on the first full run -- a schema
        # mismatch reported as a result.
        pred = load_predictor(md, score_thresh=0.05)
        nc = pred.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        ann = K.annotations_for(nc)
        if not os.path.exists(ann):
            print(f"    SKIP {name}: needs {os.path.basename(os.path.dirname(ann))}"
                  f"/_annotations.coco.json for its {nc}-class schema, not shipped")
            rows.append({"model": name, "min_size_test": pred.cfg.INPUT.MIN_SIZE_TEST,
                         "max_size_test": pred.cfg.INPUT.MAX_SIZE_TEST,
                         "min_size_test_yaml": None, "max_size_test_yaml": None,
                         "num_classes": nc, "segm_AP": None, "segm_AP50": None,
                         "bbox_AP": None,
                         "sha256": K.sha256_file(os.path.join(md, "model_final.pth"))})
            continue
        print(f"    scoring {nc}-class model against {ann.split('/data/')[-1]}")
        ds = f"tooth716_{name}"
        if ds in DatasetCatalog.list():
            DatasetCatalog.remove(ds); MetadataCatalog.remove(ds)
        register_coco_instances(ds, {}, ann, K.SPLIT_DIR)
        # off the live cfg object, not the YAML: this is what actually ran
        mn = pred.cfg.INPUT.MIN_SIZE_TEST
        mx = pred.cfg.INPUT.MAX_SIZE_TEST
        mn_yaml, mx_yaml = K.config_input_size(md)
        if (mn, mx) != (mn_yaml, mx_yaml):
            print(f"    NOTE: cfg object says {mn}/{mx}, YAML on disk says "
                  f"{mn_yaml}/{mx_yaml} -- reporting the cfg object")
        ev = COCOEvaluator(ds, output_dir=os.path.join(RESULTS, f"coco_{name}"))
        loader = build_detection_test_loader(pred.cfg, ds)
        res = inference_on_dataset(pred.model, loader, ev)

        # Overall AP is dominated by Filling (2,504 instances vs ~150 per tooth
        # position), so it does not answer "can this model number teeth?".
        # Report the tooth signal on its own: mean over the 32 position classes
        # for a 41-class model, AP-Tooth for the 9-class one.
        segm = res.get("segm", {})
        pos = [v for k, v in segm.items()
               if k.startswith("AP-") and k[3:].isdigit() and v == v]
        generic = segm.get("AP-Tooth")
        leak = LEAKAGE.get(name, {})
        rows.append({
            "model": name,
            "annotations_used": os.path.relpath(ann, K.ROOT),
            "min_size_test": mn, "max_size_test": mx,
            "min_size_test_yaml": mn_yaml, "max_size_test_yaml": mx_yaml,
            "num_classes": nc,
            "segm_AP": segm.get("AP"),
            "segm_AP50": segm.get("AP50"),
            "bbox_AP": res.get("bbox", {}).get("AP"),
            "mean_AP_tooth_positions": (sum(pos) / len(pos)) if pos else None,
            "n_tooth_position_classes": len(pos),
            "generic_AP_Tooth": generic,
            "train_images": leak.get("train_images"),
            "test_overlap_images": leak.get("overlap_with_test"),
            "test_overlap_pct": leak.get("overlap_pct"),
            "contaminated": ("YES" if (leak.get("overlap_with_test") or 0) > 0
                             else "no"),
            "sha256": K.sha256_file(os.path.join(md, "model_final.pth")),
        })
    return rows


def measure_latency(predictor, images, gt_teeth, config, warmup=10, n=200):
    import cv2
    ids = sorted(gt_teeth)
    times = []
    loaded = []
    for iid in ids[:warmup + n]:
        img = cv2.imread(os.path.join(K.SPLIT_DIR, images[iid]["file_name"]))
        if img is not None:
            loaded.append((iid, img))
        if len(loaded) >= warmup + n:
            break
    for _, img in loaded[:warmup]:                      # warm-up, discarded
        infer(predictor, img, config)
    import torch
    for _, img in loaded[warmup:warmup + n]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = infer(predictor, img, config)
        teeth_pipeline(out, img.shape[1], config)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    if not times:
        return None
    times.sort()
    q1 = times[len(times) // 4]; q3 = times[(3 * len(times)) // 4]
    return {"n": len(times), "median_s": statistics.median(times),
            "iqr_lo_s": q1, "iqr_hi_s": q3}


# --------------------------------------------------------------------------
def write_ablation(rows, extra_header=()):
    p = os.path.join(RESULTS, "ablation.csv")
    cols = ["config", "status", "reason",
            "recall_fullarch", "recall_fullarch_lo", "recall_fullarch_hi", "recall_fullarch_n",
            "recall_anomaly_bearing_teeth", "recall_anomaly_bearing_teeth_lo",
            "recall_anomaly_bearing_teeth_hi", "recall_anomaly_bearing_teeth_n",
            "central_recall_fullarch", "central_recall_fullarch_lo",
            "central_recall_fullarch_hi", "central_recall_fullarch_n",
            "peripheral_recall_fullarch", "peripheral_recall_fullarch_lo",
            "peripheral_recall_fullarch_hi",
            "fdi_acc_canonical", "fdi_acc_canonical_lo", "fdi_acc_canonical_hi", "fdi_canonical_n",
            "fdi_acc_fullarch", "fdi_acc_fullarch_lo", "fdi_acc_fullarch_hi", "fdi_fullarch_n",
            "jaw_acc_canonical", "jaw_acc_canonical_lo",
            "jaw_acc_canonical_hi", "jaw_acc_canonical_n",
            "jaw_acc_fullarch_confounded", "jaw_acc_fullarch_confounded_lo",
            "jaw_acc_fullarch_confounded_hi", "jaw_acc_fullarch_confounded_n",
            "mean_mask_iou", "teeth_per_image_preprune", "teeth_per_image_final",
            "pruning_fired_images",
            "latency_median_s", "latency_iqr_lo_s", "latency_iqr_hi_s", "latency_n"]
    with open(p, "w", newline="") as fh:
        for line in K.provenance_header(list(extra_header) + [
                "jaw_acc_canonical is the jaw figure to quote: 117 canonical",
                "  images, where class->jaw is verifiably correct.",
                "jaw_acc_fullarch_confounded scores all 93 full-arch images.",
                "  25 of the 26 non-canonical ones use a DIFFERENT jaw",
                "  partition (9 of 16 upper classes shared -- see",
                "  check_jaw_membership.py), so that column measures",
                "  annotation convention as much as model behaviour."]):
            fh.write(line + "\n")
        fh.write("# empty row = configuration not run; see the reason column\n")
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def write_per_tooth(per_tooth_by_config):
    p = os.path.join(RESULTS, "per_tooth_recall.csv")
    with open(p, "w", newline="") as fh:
        for line in K.provenance_header([
                "SUBSET: the 93 FULL-ARCH images only (n = 93)",
                "  Only on a complete arch does an absent label mean an absent",
                "  tooth. Sparse images (1-10 labels) belong to the anomaly-",
                "  annotation regime and are EXCLUDED from this file entirely.",
                "config 'ensemble' is the 'full' two-pass system",
                "support = ground-truth instances of that FDI position across",
                "  those 93 images (max 93 per position)",
                "recall = detected at mask IoU >= 0.5 (detection, not numbering)"]):
            fh.write(line + "\n")
        w = csv.writer(fh)
        w.writerow(["fdi", "config", "recall", "support"])
        for cfg_out, table in per_tooth_by_config.items():
            for fdi in K.FDI_UPPER + K.FDI_LOWER:
                hit, sup = table.get(fdi, (0, 0))
                rec = (hit / sup) if sup else ""
                w.writerow([fdi, cfg_out, f"{rec:.6f}" if sup else "", sup])
    return p


def write_sweep(taus, per):
    p = os.path.join(RESULTS, "threshold_sweep.csv")
    with open(p, "w", newline="") as fh:
        for line in K.provenance_header([
                "central_threshold = max(tau*0.7, 0.08)  (app.py:963)",
                "peripheral threshold = tau               (app.py:965)",
                "post-fusion detections cached once per image; only tau varies"]):
            fh.write(line + "\n")
        w = csv.writer(fh)
        w.writerow(["tau", "central_threshold", "central_recall",
                    "peripheral_recall", "teeth_per_image"])
        for t in taus:
            c, _, _ = K.bootstrap_ci(per[t]["c"], n=1)
            pr, _, _ = K.bootstrap_ci(per[t]["p"], n=1)
            npi = statistics.mean(per[t]["n"]) if per[t]["n"] else float("nan")
            w.writerow([t, round(max(t * 0.7, 0.08), 4),
                        f"{c:.6f}", f"{pr:.6f}", f"{npi:.4f}"])
    return p


def write_per_image(rows_by_config, extra=()):
    """Per-image metrics for every config -- the input to paired deltas."""
    p = os.path.join(RESULTS, "per_image.csv")
    cols = ["image_id", "config", "full_arch", "canonical", "fdi_evaluable",
            "sparse", "recall_hit", "recall_tot", "central_hit", "central_tot",
            "periph_hit", "periph_tot", "fdi_hit", "fdi_tot", "jaw_hit",
            "jaw_tot", "pred_preprune", "pred_final"]
    with open(p, "w", newline="") as fh:
        for line in K.provenance_header(list(extra) + [
                "one row per (image, config). Every config ran on the SAME",
                "images in the same order, so these rows are PAIRED and a",
                "difference between two configs must be bootstrapped as a",
                "per-image delta, not inferred from overlapping marginal CIs."]):
            fh.write(line + "\n")
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for cfg in rows_by_config:
            w.writerows(rows_by_config[cfg])
    return p


def write_offsets(offsets_by_config):
    p = os.path.join(RESULTS, "position_offsets.csv")
    with open(p, "w", newline="") as fh:
        for line in K.provenance_header([
                "offset = predicted index in FDI_UPPER/FDI_LOWER minus true index",
                "restricted to teeth placed in the CORRECT jaw",
                "SUBSET: 112 images = 67 full-arch canonical + 45 mid canonical-partial.",
                "  The 5 sparse canonical-partial images are excluded: their gaps are",
                "  unannotated teeth, not absent ones, so an offset there would measure",
                "  annotation policy rather than the anchoring flaw.",
                "sort_and_assign_fdi always labels the leftmost detection 18/48 and",
                "its gap check never runs before the first tooth (app.py:126-135),",
                "so a missing third molar shifts the whole jaw by one."]):
            fh.write(line + "\n")
        w = csv.writer(fh)
        w.writerow(["config", "offset", "count"])
        for cfg, ctr in offsets_by_config.items():
            for off in sorted(ctr):
                w.writerow([cfg, off, ctr[off]])
    return p


def summarise(rows):
    print("\n" + "=" * 118)
    print("ABLATION SUMMARY")
    print("=" * 118)
    print("(bootstrap 95% CI in brackets; 10,000 resamples over images)")
    hdr = (f"{'config':<18} {'recall(full-arch)':>19} {'recall(anom-bear)':>19} "
           f"{'central(FA)':>19} {'FDI(canon)':>19} {'teeth/img':>10} {'lat':>8}")
    print(hdr); print("-" * 118)
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['config']:<18} {'-- not run --':>18}  {r.get('reason','')[:52]}")
            continue
        def ci(k):
            v, lo, hi = r.get(k), r.get(k + "_lo"), r.get(k + "_hi")
            return f"{v:.3f} [{lo:.3f},{hi:.3f}]" if isinstance(v, float) and v == v else "n/a"
        lat = r.get("latency_median_s")
        print(f"{r['config']:<18} {ci('recall_fullarch'):>19} "
              f"{ci('recall_anomaly_bearing_teeth'):>19} "
              f"{ci('central_recall_fullarch'):>19} {ci('fdi_acc_canonical'):>19} "
              f"{r.get('teeth_per_image_preprune', float('nan')):>10.1f} "
              f"{(f'{lat:.2f}s' if lat else 'n/a'):>8}")
    print("=" * 100)
    print("recall(full-arch)   n = 93 images, complete arches. THE paper number:")
    print("                    an absent label there means an absent tooth.")
    print("recall(anom-bear)   n = 491 sparse images from the ANOMALY-annotation")
    print("                    regime. 43.1% of their labelled teeth carry an")
    print("                    anomaly annotation; the full-arch set has 0.0%.")
    print("                    The two are NOT comparable and must never be")
    print("                    averaged or quoted as one recall figure.")
    print("central(FA)         central-zone recall, full-arch images only (n = 93).")
    print("teeth/img           PRE-PRUNE count (post DIoU-NMS, before the 32-cap),")
    print("                    the only over-detection measure valid on sparse GT.")


def gpu_identity():
    """Which physical card this ran on -- the box has 2x T4 and nothing here
    is multi-GPU, so the latency figure must name one card."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset -- all visible)")
    name = uuid = "unknown"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()
        if out:
            uuid = out[0].strip()
    except Exception:
        pass
    return {"CUDA_VISIBLE_DEVICES": vis, "torch_device_0": name, "nvidia_smi_row0": uuid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=RUNNABLE + list(NOT_RUN))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run the whole harness on 5 images to prove it works")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--coco-eval", action="store_true")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--numbering-sweep", action="store_true",
                    help="sweep tau for network numbering to find the "
                         "baseline's own optimum before comparing")
    ap.add_argument("--network-numbering", action="store_true",
                    help="score a 41-class model's own FDI numbering head to "
                         "head against detection+geometry on identical images")
    ap.add_argument("--numbering-model", default=None,
                    help="model dir for --network-numbering "
                         "(default: resnext101_cascade_60k)")
    ap.add_argument("--routing-ab", action="store_true",
                    help="A/B the app.py:953-958 routing bug on anomaly-bearing images")
    ap.add_argument("--fix-cls-routing", action="store_true",
                    help="correct the app.py:953-958 routing bug for EVERY config "
                         "(the full_fix_routing config does it for one row only)")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    limit = 5 if args.smoke_test else args.limit
    images, names, gt_teeth, gt_anoms = K.load_split()
    manifests = K.load_manifests()
    class_map = K.load_class_map()

    gid = gpu_identity()
    print("GPU pinning:")
    for k, v in gid.items():
        print(f"  {k:<22} {v}")
    extra = [f"fix_cls_routing={args.fix_cls_routing}"] + \
            [f"{k}={v}" for k, v in gid.items()]
    if limit:
        extra.append(f"*** LIMITED RUN: {limit} images -- NOT publishable ***")

    print(f"\nsplit: {len(images)} images | full-arch {len(manifests['full_arch'])} | "
          f"canonical {len(manifests['canonical'])} | fdi-evaluable "
          f"{len(manifests['fdi_evaluable'])} | sparse {len(manifests['sparse'])}")

    # --smoke-test only shrinks the workload; it must NOT select configs.
    # It used to imply --all, so `--routing-ab --smoke-test` silently ran the
    # entire ablation a second time.
    configs = RUNNABLE if args.all else ([args.config] if args.config else [])
    need_model = (bool(configs) or args.sweep or args.latency
                  or args.routing_ab)
    if not (need_model or args.coco_eval or args.network_numbering
            or args.numbering_sweep):
        ap.error("nothing to do: pass --config/--all/--coco-eval/--sweep/--latency/--routing-ab")

    predictor = None
    if need_model:
        print("\nloading teeth model (sizes read off the live cfg object):")
        predictor = load_predictor(K.TEETH_MODEL)
        extra.append(f"teeth cfg.INPUT MIN/MAX_SIZE_TEST="
                     f"{LOADED_INPUT_SIZE.get('resnet50_9class_20k')}")

    # ---- STAGE 0a: give the network baseline its best shot ----------------
    if args.numbering_sweep:
        md = args.numbering_model or K.CASCADE_MODEL
        print(f"\n--- numbering tau sweep: {os.path.basename(md)} ---")
        npred = load_predictor(md)
        numbering_sweep(npred, images, gt_teeth, manifests, class_map,
                        limit=limit)

    # ---- STAGE 0b: network numbering head-to-head -------------------------
    if args.network_numbering:
        md = args.numbering_model or K.CASCADE_MODEL
        print(f"\n--- network numbering: {os.path.basename(md)} ---")
        npred = load_predictor(md)
        nrows, _stat = network_numbering(npred, images, gt_teeth, manifests,
                                         class_map, limit=limit)
        # append to per_image.csv so paired_deltas can align it against the
        # geometric configs on identical image ids. Drop every previous
        # network row, whatever threshold it came from, so a rerun replaces
        # them rather than stacking a second copy alongside.
        pth = os.path.join(RESULTS, "per_image.csv")
        existing = []
        if os.path.exists(pth):
            with open(pth) as fh:
                existing = [r for r in csv.DictReader(
                    l for l in fh if not l.startswith("#"))
                    if not r["config"].startswith("network_numbering")]
        cols = (list(existing[0].keys()) if existing else list(nrows[0].keys()))
        with open(pth, "w", newline="") as fh:
            for line in K.provenance_header(
                    [f"network_numbering rows from {os.path.basename(md)}",
                     "one row per (image, config); configs are PAIRED on image_id"]):
                fh.write(line + "\n")
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(existing); w.writerows(nrows)
        print(f"  appended {len(nrows)} rows to {pth}")

    # ---- STAGE 0: routing A/B (cheap, and it can change the audit) --------
    if args.routing_ab:
        routing_ab(predictor, images, gt_teeth, gt_anoms,
                   n_images=5 if limit else 12)

    # ---- STAGE 1: the ablation. FIRST, and written as soon as it is done --
    rows, per_tooth, offsets, per_image = [], {}, {}, {}
    for cfg in configs:
        print(f"\n--- {cfg} ---", flush=True)
        acc = evaluate(cfg, predictor, images, gt_teeth, manifests, class_map,
                       limit=limit, fix_cls_routing=args.fix_cls_routing)
        r = {"config": cfg, "status": "ok", "reason": ""}
        for key, srcname in (("recall_fullarch", "recall_full"),
                             ("recall_anomaly_bearing_teeth", "recall_anom"),
                             ("central_recall_fullarch", "central_full"),
                             ("peripheral_recall_fullarch", "periph_full"),
                             ("fdi_acc_canonical", "fdi_canonical"),
                             ("fdi_acc_fullarch", "fdi_fullarch"),
                             ("jaw_acc_canonical", "jaw_canonical"),
                             ("jaw_acc_fullarch_confounded", "jaw")):
            v, lo, hi = K.bootstrap_ci(acc[srcname])
            r[key], r[key + "_lo"], r[key + "_hi"] = v, lo, hi
        r["recall_fullarch_n"] = len(acc["recall_full"])
        r["recall_anomaly_bearing_teeth_n"] = len(acc["recall_anom"])
        r["central_recall_fullarch_n"] = len(acc["central_full"])
        r["fdi_canonical_n"] = len(acc["fdi_canonical"])
        r["fdi_fullarch_n"] = len(acc["fdi_fullarch"])
        r["jaw_acc_canonical_n"] = len(acc["jaw_canonical"])
        r["jaw_acc_fullarch_confounded_n"] = len(acc["jaw"])
        r["mean_mask_iou"] = (statistics.mean(acc["mask_ious"])
                              if acc["mask_ious"] else float("nan"))
        r["teeth_per_image_preprune"] = (statistics.mean(acc["pre_prune"])
                                         if acc["pre_prune"] else float("nan"))
        r["teeth_per_image_final"] = (statistics.mean(acc["final"])
                                      if acc["final"] else float("nan"))
        r["pruning_fired_images"] = acc["pruned"]
        if args.latency:
            lat = measure_latency(predictor, images, gt_teeth,
                                  BASE_CONFIG.get(cfg, cfg),
                                  warmup=2 if limit else 10, n=3 if limit else 200)
            if lat:
                r["latency_median_s"] = lat["median_s"]
                r["latency_iqr_lo_s"] = lat["iqr_lo_s"]
                r["latency_iqr_hi_s"] = lat["iqr_hi_s"]
                r["latency_n"] = lat["n"]
        rows.append(r)
        offsets[cfg] = acc["offsets"]
        per_image[cfg] = acc["per_image"]
        if cfg in ("single_pass", "full"):
            per_tooth["ensemble" if cfg == "full" else cfg] = \
                {k: tuple(v) for k, v in acc["per_tooth"].items()}

        # Write after EVERY config, not at the end: a session that dies at
        # hour two must still leave the completed configs on disk.
        partial = rows + [{"config": c, "status": "not_run", "reason": why}
                          for c, why in NOT_RUN.items()]
        write_ablation(partial, extra)
        if per_tooth:
            write_per_tooth(per_tooth)
        if offsets:
            write_offsets(offsets)
        if per_image:
            write_per_image(per_image, extra)
        print(f"    [checkpointed {len(rows)}/{len(configs)} configs to results/]",
              flush=True)

    if rows:
        for c, why in NOT_RUN.items():
            rows.append({"config": c, "status": "not_run", "reason": why})
        print("\nwrote", write_ablation(rows, extra))
        if per_tooth:
            print("wrote", write_per_tooth(per_tooth))
        if offsets:
            print("wrote", write_offsets(offsets))
        if per_image:
            print("wrote", write_per_image(per_image, extra))
        summarise(rows)

    # ---- STAGE 2: tau sweep ----------------------------------------------
    if args.sweep:
        print("\n--- tau sweep ---", flush=True)
        taus, per = tau_sweep(predictor, images, gt_teeth, limit=limit)
        print("wrote", write_sweep(taus, per))

    # ---- STAGE 3: COCO re-evaluation (longest, last) ---------------------
    if args.coco_eval:
        print("\n--- COCO re-evaluation on the 716 split ---", flush=True)
        p = os.path.join(RESULTS, "coco_eval_716.csv")
        crows = []
        for md in (K.TEETH_MODEL, K.ANOM_MODEL, K.CASCADE_MODEL):
            crows += coco_eval([md], limit=limit)
            with open(p, "w", newline="") as fh:      # rewrite after each model
                for line in K.provenance_header(
                        ["each model evaluated at its OWN configured test size",
                         "sizes read off the live cfg object, not the YAML"] +
                        [f"{k}={v}" for k, v in gid.items()]):
                    fh.write(line + "\n")
                w = csv.DictWriter(fh, fieldnames=list(crows[0].keys()))
                w.writeheader(); w.writerows(crows)
            print(f"    [checkpointed {len(crows)}/3 models to {p}]", flush=True)
        print("wrote", p)
        for c in crows:
            print(f"  {c['model']:<24} nc={c['num_classes']:<3} "
                  f"{c['min_size_test']}/{c['max_size_test']}  segm_AP={c['segm_AP']}")


if __name__ == "__main__":
    main()
