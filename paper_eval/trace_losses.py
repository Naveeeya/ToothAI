"""Where are teeth lost? (results/v1_tooth_losses.csv) Trace every labelled tooth on the full-arch test images through the deployed
pipeline (tooth_ai.pipeline) stage by stage. Prints numeric image ids and counts only (filenames hold
patient names). Run from the repo root with the Detectron2 env."""
import json, os, sys, time
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch

sys.path.insert(0, "paper_eval"); sys.path.insert(0, ".")
import evalkit as K
from tooth_ai import pipeline as P
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

torch.set_num_threads(int(os.environ.get("THREADS", "8")))
TAU = 0.20
OUT = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None

cfg = get_cfg(); cfg.merge_from_file("output/resnet50_9class_20k/config.yaml")
cfg.MODEL.WEIGHTS = "output/resnet50_9class_20k/model_final.pth"; cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
print("input size", cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "| train", cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, flush=True)
pred = DefaultPredictor(cfg)

images, names, gt_teeth, _ = K.load_split()
man = K.load_manifests()
ids = sorted(man["full_arch"])[:LIMIT]


def dets(instances, tooth_only=True):
    inst = instances.to("cpu")
    out = []
    for b, s, c, m in zip(inst.pred_boxes.tensor.numpy(), inst.scores.numpy(), inst.pred_classes.numpy(),
                          inst.pred_masks.numpy()):
        if tooth_only and int(c) != 0:
            continue
        out.append({"box": b.tolist(), "score": float(s), "mask": m})
    return out


def matched(gts, preds, h, w, gm):
    return [p is not None for _, p, _ in K.match_detections(gts, preds, h, w, gt_masks=gm)]


rows = []
t0 = time.time()
for n, iid in enumerate(ids):
    info = images[iid]
    img = cv2.imread(os.path.join(K.SPLIT_DIR, info["file_name"]))
    h, w = img.shape[:2]
    gts = gt_teeth[iid]
    gm = K.gt_masks_for(gts, h, w)
    o1 = pred(img); o2 = pred(P.apply_clahe(img, 2.5, (8, 8)))
    fused = P.weighted_box_fusion([o1, o2], iou_threshold=0.55, skip_box_threshold=0.05)
    s_seen = dets(o1["instances"]) + dets(o2["instances"])
    s_fused = dets(fused["instances"])
    raw, _ = P.process_predictions(fused, image_width=w, teeth_threshold=TAU, anomaly_threshold=0.25)
    nms = P.DentalGeometricEngine.apply_nms(raw, iou_threshold=0.4)
    up, lo = P.DentalGeometricEngine.split_jaws(nms)
    numbered = P.DentalGeometricEngine.sort_and_assign_fdi(up, "upper") + P.DentalGeometricEngine.sort_and_assign_fdi(lo, "lower")
    pruned = P.recursive_prune_teeth(list(numbered), max_teeth=32)
    final = P.resolve_fdi_conflicts([dict(t) for t in pruned])
    stages = {"seen": s_seen, "fused": s_fused, "threshold": raw, "nms": nms, "pruned": pruned, "final": final}
    hit = {k: matched(gts, v, h, w, gm) for k, v in stages.items()}
    # best fused score per GT (why it fell under the threshold), and who suppressed it at NMS
    best_fused = []
    for g, (_, p, _) in zip(gts, K.match_detections(gts, s_fused, h, w, gt_masks=gm)):
        best_fused.append(p["score"] if p else None)
    order = list(stages)
    for gi, g in enumerate(gts):
        lost = next((s for s in order if not hit[s][gi]), None)
        rows.append({"image": iid, "cat": g["category_id"], "lost_at": lost or "kept", "fused_score": best_fused[gi],
                     "central": K.is_central(g["bbox"], w)})
    fp_final = len(final) - sum(hit["final"])
    rows.append({"image": iid, "summary": True, "n_gt": len(gts), "n_nms": len(nms), "n_final": len(final),
                 "fp_final": fp_final, "w": w, "h": h})
    if (n + 1) % 5 == 0:
        print(f"{n + 1}/{len(ids)} images, {(time.time() - t0) / (n + 1):.1f} s/img", flush=True)
json.dump(rows, open(OUT, "w"))
print("done", OUT)
