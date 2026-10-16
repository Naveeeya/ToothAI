#!/usr/bin/env python3
"""Cache the 9-class teeth model's raw outputs, so engine variants can be scored without re-running it.

    python paper_eval/cache_outputs.py --split valid --min-teeth 14
    python paper_eval/cache_outputs.py --split test --canonical     # canonical + full-arch images

One JSON per image under paper_eval/.cache/<split>/ (git-ignored), keyed by numeric image id (file names
carry patient names and are never written). Passes: ``orig`` (what the app runs), ``clahe`` (the deployed
second pass), ``flip`` (horizontal flip, mapped back) and ``big`` (short edge 1000 px instead of 800).
Masks are COCO RLE. Runs on CPU; resumable.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evalkit as K  # noqa: E402
from tooth_ai import pipeline as P  # noqa: E402


def encode(instances, flip_w=None):
    inst = instances.to("cpu")
    boxes = inst.pred_boxes.tensor.numpy().copy()
    masks = inst.pred_masks.numpy()
    if flip_w is not None:
        boxes[:, [0, 2]] = flip_w - boxes[:, [2, 0]]
        masks = masks[:, :, ::-1]
    rles = mask_util.encode(np.asfortranarray(masks.transpose(1, 2, 0).astype(np.uint8)))
    return {"boxes": boxes.round(2).tolist(), "scores": inst.scores.numpy().round(5).tolist(),
            "classes": inst.pred_classes.numpy().tolist(),
            "masks": [{"size": r["size"], "counts": r["counts"].decode()} for r in rles]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "valid"], required=True)
    ap.add_argument("--min-teeth", type=int, default=0, help="only images with at least this many labelled teeth")
    ap.add_argument("--canonical", action="store_true",
                    help="test split: the 117 canonical images and the 93 full-arch images (recall)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--add41", action="store_true",
                    help="add the 41-class findings model's original-image pass (key 'net41') to cached files")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.split == "valid":
        K.SPLIT_DIR = K.SPLIT_DIR.replace(os.sep + "test", os.sep + "valid")
        K.SPLIT_JSON = os.path.join(K.SPLIT_DIR, "_annotations.coco.json")
    images, _, gt_teeth, _ = K.load_split()
    if args.canonical:
        man = K.load_manifests()
        ids = sorted(man["canonical"] | man["full_arch"])
    else:
        ids = sorted(i for i, g in gt_teeth.items() if len(g) >= args.min_teeth)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", args.split)
    os.makedirs(out_dir, exist_ok=True)

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(K.TEETH_MODEL, "config.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(K.TEETH_MODEL, "model_final.pth")
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    if args.add41:
        cfg.merge_from_file(os.path.join(K.ANOM_MODEL, "config.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(K.ANOM_MODEL, "model_final.pth")
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    pred = DefaultPredictor(cfg)
    base = (pred.aug.short_edge_length, pred.aug.max_size)
    if args.add41:
        t0, done = time.time(), 0
        for n, iid in enumerate(ids):
            path = os.path.join(out_dir, f"{iid}.json")
            if not os.path.exists(path):
                continue
            rec = json.load(open(path))
            if "net41" in rec:
                continue
            img = cv2.imread(os.path.join(K.SPLIT_DIR, images[iid]["file_name"]))
            rec["net41"] = encode(pred(img)["instances"])
            with open(path + ".tmp", "w") as fh:
                json.dump(rec, fh)
            os.replace(path + ".tmp", path)
            done += 1
            if done % 20 == 0:
                print(f"{n + 1}/{len(ids)} ({(time.time() - t0) / done:.1f} s/img)", flush=True)
        print("added net41 to", done, "files")
        return

    t0, done = time.time(), 0
    for n, iid in enumerate(ids):
        path = os.path.join(out_dir, f"{iid}.json")
        if os.path.exists(path):
            continue
        img = cv2.imread(os.path.join(K.SPLIT_DIR, images[iid]["file_name"]))
        h, w = img.shape[:2]
        rec = {"image_id": iid, "w": w, "h": h}
        rec["orig"] = encode(pred(img)["instances"])
        rec["clahe"] = encode(pred(P.apply_clahe(img, 2.5, (8, 8)))["instances"])
        rec["flip"] = encode(pred(np.ascontiguousarray(img[:, ::-1]))["instances"], flip_w=w)
        pred.aug.short_edge_length, pred.aug.max_size = (1000, 1000), 1777
        rec["big"] = encode(pred(img)["instances"])
        pred.aug.short_edge_length, pred.aug.max_size = base
        with open(path + ".tmp", "w") as fh:
            json.dump(rec, fh)
        os.replace(path + ".tmp", path)
        done += 1
        if done % 10 == 0:
            print(f"{n + 1}/{len(ids)} ({(time.time() - t0) / done:.1f} s/img)", flush=True)
    print("cached", len(ids), "images in", out_dir)


if __name__ == "__main__":
    main()
