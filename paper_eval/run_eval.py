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
    jaw accuracy      all 93 full-arch images -- jaw membership is consistent
                      in every one, including the 26 whose within-jaw ordering
                      is scrambled
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
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as K
import pipeline as P

RESULTS = os.path.join(K.ROOT, "paper_eval", "results")

RUNNABLE = ["single_pass", "full", "no_zone_threshold", "nms"]
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
def load_predictor(model_dir, score_thresh=0.05):
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(model_dir, "config.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh   # app.py:375
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
             limit=None, fix_cls_routing=False):
    """Run one configuration over the split and accumulate every metric."""
    acc = {
        "recall_full": [], "recall_sparse": [],
        "central_full": [], "central_sparse": [],
        "fdi_canonical": [], "fdi_fullarch": [],
        "jaw": [], "offsets": Counter(),
        "mask_ious": [], "pre_prune": [], "final": [], "pruned": 0,
        "per_tooth": defaultdict(lambda: [0, 0]),
    }
    ids = sorted(gt_teeth) if limit is None else sorted(gt_teeth)[:limit]
    for n, iid in enumerate(ids, 1):
        meta = images[iid]
        path = os.path.join(K.SPLIT_DIR, meta["file_name"])
        import cv2
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        out = infer(predictor, img, config)
        teeth, pre, fired = teeth_pipeline(out, w, config,
                                           fix_cls_routing=fix_cls_routing)
        acc["pre_prune"].append(pre)
        acc["final"].append(len(teeth))
        acc["pruned"] += int(fired)

        gts = gt_teeth[iid]
        pairs = K.match_detections(gts, teeth, h, w)

        n_hit = n_tot = 0
        n_chit = n_ctot = 0
        for gt, pred, iou in pairs:
            n_tot += 1
            central = K.is_central(gt["bbox"], meta["width"])
            n_ctot += int(central)
            if pred is not None:
                n_hit += 1
                n_chit += int(central)
                acc["mask_ious"].append(iou)

        bucket = "full" if iid in manifests["full_arch"] else (
            "sparse" if iid in manifests["sparse"] else None)
        if bucket:
            acc[f"recall_{bucket}"].append((n_hit, n_tot))
            acc[f"central_{bucket}"].append((n_chit, n_ctot))

        # --- FDI / jaw / offset, only where the labels support it ----------
        in_canon = iid in manifests["canonical"]
        in_fdi = iid in manifests["fdi_evaluable"]
        in_full = iid in manifests["full_arch"]
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
                if pred_jaw == jaw and in_canon:
                    # offsets only on canonical images: on the 26 scrambled
                    # ones the "true" rank is meaningless, so an offset there
                    # would measure label noise, not the anchoring flaw.
                    seq = K.FDI_UPPER if jaw == "upper" else K.FDI_LOWER
                    acc["offsets"][seq.index(pf) - rank] += 1
            if in_canon:
                acc["fdi_canonical"].append((fdi_hit, fdi_tot))
            if in_fdi:
                acc["fdi_fullarch"].append((fdi_hit, fdi_tot))
            if in_full:
                acc["jaw"].append((jaw_hit, jaw_tot))

        # --- per-position recall (detection only, all images) --------------
        for gt, pred, iou in pairs:
            _, _, true_fdi = class_map[gt["category_id"]]
            acc["per_tooth"][true_fdi][1] += 1
            if pred is not None:
                acc["per_tooth"][true_fdi][0] += 1

        if n % 25 == 0:
            print(f"    {config}: {n}/{len(ids)}", flush=True)
    return acc


# --------------------------------------------------------------------------
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
        for t in taus:
            teeth, pre, _ = teeth_pipeline(out, w, "full", tau=t)
            pairs = K.match_detections(gts, teeth, h, w)
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
        ds = f"tooth716_{name}"
        if ds in DatasetCatalog.list():
            DatasetCatalog.remove(ds); MetadataCatalog.remove(ds)
        register_coco_instances(ds, {}, K.SPLIT_JSON, K.SPLIT_DIR)
        mn, mx = K.config_input_size(md)
        print(f"  {name}: MIN_SIZE_TEST={mn} MAX_SIZE_TEST={mx}")
        pred = load_predictor(md, score_thresh=0.05)
        ev = COCOEvaluator(ds, output_dir=os.path.join(RESULTS, f"coco_{name}"))
        loader = build_detection_test_loader(pred.cfg, ds)
        res = inference_on_dataset(pred.model, loader, ev)
        rows.append({
            "model": name,
            "min_size_test": mn, "max_size_test": mx,
            "num_classes": pred.cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "segm_AP": res.get("segm", {}).get("AP"),
            "segm_AP50": res.get("segm", {}).get("AP50"),
            "bbox_AP": res.get("bbox", {}).get("AP"),
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
            "recall_fullarch", "recall_fullarch_lo", "recall_fullarch_hi",
            "recall_sparse", "recall_sparse_lo", "recall_sparse_hi",
            "central_recall_fullarch", "central_recall_fullarch_lo", "central_recall_fullarch_hi",
            "central_recall_sparse", "central_recall_sparse_lo", "central_recall_sparse_hi",
            "fdi_acc_canonical", "fdi_acc_canonical_lo", "fdi_acc_canonical_hi", "fdi_canonical_n",
            "fdi_acc_fullarch", "fdi_acc_fullarch_lo", "fdi_acc_fullarch_hi", "fdi_fullarch_n",
            "jaw_acc", "jaw_acc_lo", "jaw_acc_hi", "jaw_n",
            "mean_mask_iou", "teeth_per_image_preprune", "teeth_per_image_final",
            "pruning_fired_images",
            "latency_median_s", "latency_iqr_lo_s", "latency_iqr_hi_s", "latency_n"]
    with open(p, "w", newline="") as fh:
        for line in K.provenance_header(extra_header):
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
                "config 'ensemble' is the 'full' two-pass system",
                "support = ground-truth instances of that FDI position in the split",
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


def write_offsets(offsets_by_config):
    p = os.path.join(RESULTS, "position_offsets.csv")
    with open(p, "w", newline="") as fh:
        for line in K.provenance_header([
                "offset = predicted index in FDI_UPPER/FDI_LOWER minus true index",
                "restricted to teeth placed in the CORRECT jaw, canonical images only",
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
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY")
    print("=" * 100)
    hdr = f"{'config':<18} {'recall(FA)':>18} {'recall(sp)':>18} {'FDI(canon)':>18} {'teeth/img':>11} {'lat med':>9}"
    print(hdr); print("-" * 100)
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['config']:<18} {'-- not run --':>18}  {r.get('reason','')[:52]}")
            continue
        def ci(k):
            v, lo, hi = r.get(k), r.get(k + "_lo"), r.get(k + "_hi")
            return f"{v:.3f} [{lo:.3f},{hi:.3f}]" if isinstance(v, float) and v == v else "n/a"
        lat = r.get("latency_median_s")
        print(f"{r['config']:<18} {ci('recall_fullarch'):>18} {ci('recall_sparse'):>18} "
              f"{ci('fdi_acc_canonical'):>18} "
              f"{r.get('teeth_per_image_preprune', float('nan')):>11.1f} "
              f"{(f'{lat:.3f}s' if lat else 'n/a'):>9}")
    print("=" * 100)
    print("recall(FA) = full-arch images; recall(sp) = sparse images. They are NOT")
    print("comparable: 43.1% of sparse labelled teeth carry an anomaly, vs 0.0% of")
    print("full-arch ones. teeth/img is the PRE-PRUNE count (post DIoU-NMS).")


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
    ap.add_argument("--fix-cls-routing", action="store_true",
                    help="correct the app.py:953-958 class-routing bug (default OFF)")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    limit = 5 if args.smoke_test else args.limit
    images, names, gt_teeth, gt_anoms = K.load_split()
    manifests = K.load_manifests()
    class_map = K.load_class_map()

    extra = [f"fix_cls_routing={args.fix_cls_routing}"]
    if limit:
        extra.append(f"*** LIMITED RUN: {limit} images -- NOT publishable ***")

    print(f"split: {len(images)} images | full-arch {len(manifests['full_arch'])} | "
          f"canonical {len(manifests['canonical'])} | fdi-evaluable "
          f"{len(manifests['fdi_evaluable'])} | sparse {len(manifests['sparse'])}")
    print(f"gpu: {K.gpu_name()}")

    configs = RUNNABLE if (args.all or args.smoke_test) else (
        [args.config] if args.config else [])
    if not (configs or args.coco_eval or args.sweep or args.latency):
        ap.error("nothing to do: pass --config/--all/--coco-eval/--sweep/--latency")

    predictor = None
    if configs or args.sweep or args.latency:
        print(f"loading teeth model: {K.TEETH_MODEL}")
        predictor = load_predictor(K.TEETH_MODEL)

    rows, per_tooth, offsets = [], {}, {}
    for cfg in configs:
        print(f"\n--- {cfg} ---")
        acc = evaluate(cfg, predictor, images, gt_teeth, manifests, class_map,
                       limit=limit, fix_cls_routing=args.fix_cls_routing)
        r = {"config": cfg, "status": "ok", "reason": ""}
        for key, src in (("recall_fullarch", "recall_full"),
                         ("recall_sparse", "recall_sparse"),
                         ("central_recall_fullarch", "central_full"),
                         ("central_recall_sparse", "central_sparse"),
                         ("fdi_acc_canonical", "fdi_canonical"),
                         ("fdi_acc_fullarch", "fdi_fullarch"),
                         ("jaw_acc", "jaw")):
            v, lo, hi = K.bootstrap_ci(acc[src])
            r[key], r[key + "_lo"], r[key + "_hi"] = v, lo, hi
        r["fdi_canonical_n"] = len(acc["fdi_canonical"])
        r["fdi_fullarch_n"] = len(acc["fdi_fullarch"])
        r["jaw_n"] = len(acc["jaw"])
        r["mean_mask_iou"] = (statistics.mean(acc["mask_ious"])
                              if acc["mask_ious"] else float("nan"))
        r["teeth_per_image_preprune"] = (statistics.mean(acc["pre_prune"])
                                         if acc["pre_prune"] else float("nan"))
        r["teeth_per_image_final"] = (statistics.mean(acc["final"])
                                      if acc["final"] else float("nan"))
        r["pruning_fired_images"] = acc["pruned"]
        if args.latency:
            lat = measure_latency(predictor, images, gt_teeth, cfg,
                                  warmup=2 if limit else 10, n=3 if limit else 200)
            if lat:
                r["latency_median_s"] = lat["median_s"]
                r["latency_iqr_lo_s"] = lat["iqr_lo_s"]
                r["latency_iqr_hi_s"] = lat["iqr_hi_s"]
                r["latency_n"] = lat["n"]
        rows.append(r)
        offsets[cfg] = acc["offsets"]
        if cfg in ("single_pass", "full"):
            out_name = "ensemble" if cfg == "full" else cfg
            per_tooth[out_name] = {k: tuple(v) for k, v in acc["per_tooth"].items()}

    for cfg, reason in NOT_RUN.items():
        if args.all or args.smoke_test or args.config == cfg:
            rows.append({"config": cfg, "status": "not_run", "reason": reason})

    if rows:
        print("\nwrote", write_ablation(rows, extra))
    if per_tooth:
        print("wrote", write_per_tooth(per_tooth))
    if offsets:
        print("wrote", write_offsets(offsets))

    if args.sweep:
        print("\n--- tau sweep ---")
        taus, per = tau_sweep(predictor, images, gt_teeth, limit=limit)
        print("wrote", write_sweep(taus, per))

    if args.coco_eval:
        print("\n--- COCO re-evaluation on the 716 split ---")
        crows = coco_eval([K.TEETH_MODEL, K.ANOM_MODEL, K.CASCADE_MODEL], limit=limit)
        p = os.path.join(RESULTS, "coco_eval_716.csv")
        with open(p, "w", newline="") as fh:
            for line in K.provenance_header(
                    ["each model evaluated at its OWN configured test size"]):
                fh.write(line + "\n")
            w = csv.DictWriter(fh, fieldnames=list(crows[0].keys()))
            w.writeheader(); w.writerows(crows)
        print("wrote", p)
        for c in crows:
            print(f"  {c['model']:<24} nc={c['num_classes']:<3} "
                  f"{c['min_size_test']}/{c['max_size_test']}  segm_AP={c['segm_AP']}")

    if rows:
        summarise(rows)


if __name__ == "__main__":
    main()
