#!/usr/bin/env python3
"""Task E: qualitative panel export for the paper figure.

    python3 paper_eval/export_panels.py --suggest-images
    python3 paper_eval/export_panels.py --image-id 293

--suggest-images ranks test images by how much the ensemble ('full') gains
over single_pass in CENTRAL-ZONE recall, and flags any candidate where the
ensemble introduces a central-zone false positive -- the Limitations section
claims that trade-off exists, so the figure should show one.

--image-id writes panel_gt.json, panel_single.json, panel_ensemble.json,
panel_full.json and the source PNG. Polygons and boxes are in ORIGINAL image
pixel coordinates. Masks are emitted as polygons, never RLE.

A note on 'central-zone false positive': with a median of 8 labelled teeth per
image, an unlabelled-but-real tooth also scores as a false positive. Candidates
are therefore drawn from the FDI-evaluable subset (complete, canonical arches),
where every tooth is labelled and an unmatched detection really is spurious.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as K
import run_eval as R
import pipeline as P

OUT = os.path.join(K.ROOT, "paper_eval", "results", "panels")


def mask_to_polygons(mask, epsilon_frac=0.004):
    """Boolean mask -> list of [[x,y], ...] rings, simplified.

    cv2.findContours + approxPolyDP; polygons, not RLE, as the figure script
    needs vertices it can draw.
    """
    import cv2
    m = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < 20:
            continue
        eps = epsilon_frac * cv2.arcLength(c, True)
        a = cv2.approxPolyDP(c, eps, True)
        if len(a) >= 3:
            polys.append([[int(p[0][0]), int(p[0][1])] for p in a])
    return polys


def central_crop(width, height, jaw_split_y):
    """Frame the central incisor region: the 35-65% band, padded."""
    x1 = int(width * 0.32)
    x2 = int(width * 0.68)
    y1 = max(0, int(jaw_split_y - height * 0.30))
    y2 = min(height, int(jaw_split_y + height * 0.30))
    return [x1, y1, x2, y2]


def jaw_split(teeth):
    if len(teeth) < 2:
        return None
    ys = [P.DentalGeometricEngine.get_box_center(t["box"])[1] for t in teeth]
    return int(np.median(ys))


def build_panel(image_name, teeth, jaw_y, crop, anomalies=()):
    dets = []
    for t in teeth:
        d = {"kind": "tooth", "fdi": t.get("fdi"),
             "score": round(float(t.get("score", 0.0)), 4)}
        m = t.get("mask")
        d["polygon"] = mask_to_polygons(m) if m is not None else []
        dets.append(d)
    for a in anomalies:
        x1, y1, x2, y2 = a["box"]
        dets.append({"kind": "anomaly", "label": a.get("class_name"),
                     "bbox": [int(x1), int(y1), int(x2), int(y2)],
                     "score": round(float(a.get("score", 0.0)), 4),
                     "fdi": a.get("associated_fdi")})
    return {"image": image_name, "jaw_split_y": jaw_y, "crop": crop,
            "detections": dets}


def gt_panel(image_name, gt_anns, class_map, jaw_y, crop, h, w):
    dets = []
    for a in gt_anns:
        _, _, fdi = class_map[a["category_id"]]
        segm = a["segmentation"]
        polys = []
        if isinstance(segm, list):
            for ring in segm:
                pts = [[int(ring[i]), int(ring[i + 1])] for i in range(0, len(ring) - 1, 2)]
                if len(pts) >= 3:
                    polys.append(pts)
        else:
            polys = mask_to_polygons(K.poly_to_mask(segm, h, w))
        dets.append({"kind": "tooth", "fdi": fdi, "polygon": polys, "score": 1.0})
    return {"image": image_name, "jaw_split_y": jaw_y, "crop": crop,
            "detections": dets}


def suggest(predictor, images, gt_teeth, manifests, top=10, limit=None):
    """Rank by central-zone recall gain of ensemble over single_pass."""
    import cv2
    ids = sorted(manifests["fdi_evaluable"])
    if limit:
        ids = ids[:limit]
    rows = []
    for n, iid in enumerate(ids, 1):
        meta = images[iid]
        img = cv2.imread(os.path.join(K.SPLIT_DIR, meta["file_name"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = gt_teeth[iid]
        central_gt = [g for g in gts if K.is_central(g["bbox"], meta["width"])]
        if not central_gt:
            continue
        res = {}
        for cfg in ("single_pass", "full"):
            out = R.infer(predictor, img, cfg)
            teeth, _pre, _f = R.teeth_pipeline(out, w, cfg)
            pairs = K.match_detections(gts, teeth, h, w)
            matched = {id(p) for _g, p, _i in pairs if p is not None}
            hit = sum(1 for g, p, _i in pairs
                      if p is not None and K.is_central(g["bbox"], meta["width"]))
            # a central detection matching no GT tooth, on a COMPLETE arch
            fp = sum(1 for t in teeth
                     if id(t) not in matched
                     and K.CENTRAL_LO < (P.DentalGeometricEngine.get_box_center(t["box"])[0]
                                         / float(meta["width"])) < K.CENTRAL_HI)
            res[cfg] = (hit, fp)
        gain = (res["full"][0] - res["single_pass"][0]) / float(len(central_gt))
        rows.append({
            "image_id": iid, "file": meta["file_name"],
            "central_gt": len(central_gt),
            "single_hit": res["single_pass"][0], "ensemble_hit": res["full"][0],
            "gain": gain,
            "single_fp": res["single_pass"][1], "ensemble_fp": res["full"][1],
            "new_fp": res["full"][1] - res["single_pass"][1],
        })
        if n % 10 == 0:
            print(f"  scanned {n}/{len(ids)}", flush=True)
    rows.sort(key=lambda r: (-r["gain"], -r["ensemble_hit"]))
    return rows[:top], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", type=int)
    ap.add_argument("--suggest-images", action="store_true")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    limit = 5 if args.smoke_test else args.limit

    images, names, gt_teeth, gt_anoms = K.load_split()
    manifests = K.load_manifests()
    class_map = K.load_class_map()
    predictor = R.load_predictor(K.TEETH_MODEL)

    if args.suggest_images:
        top, allrows = suggest(predictor, images, gt_teeth, manifests,
                               top=args.top, limit=limit)
        print("\n" + "=" * 92)
        print("CANDIDATE FIGURE IMAGES -- ranked by central-zone recall gain "
              "(ensemble over single pass)")
        print("=" * 92)
        print(f"{'image_id':>9} {'centralGT':>10} {'single':>7} {'ens':>5} "
              f"{'gain':>7} {'newFP':>6}  flag")
        for r in top:
            flag = "<-- ensemble ADDS a central FP (wanted for the figure)" if r["new_fp"] > 0 else ""
            print(f"{r['image_id']:>9} {r['central_gt']:>10} {r['single_hit']:>7} "
                  f"{r['ensemble_hit']:>5} {r['gain']:>7.2f} {r['new_fp']:>6}  {flag}")
        withfp = [r for r in allrows if r["new_fp"] > 0]
        print(f"\n{len(withfp)} of {len(allrows)} candidates show the ensemble introducing")
        print("a central-zone false positive. The Limitations section claims this")
        print("trade-off exists; pick from these to show it.")
        p = os.path.join(K.ROOT, "paper_eval", "results", "panel_candidates.csv")
        import csv as _csv
        with open(p, "w", newline="") as fh:
            for line in K.provenance_header([
                    "candidates drawn from the FDI-evaluable subset (complete,",
                    "canonical arches) so an unmatched central detection really is",
                    "spurious rather than merely unlabelled"]):
                fh.write(line + "\n")
            w = _csv.DictWriter(fh, fieldnames=list(allrows[0].keys()))
            w.writeheader(); w.writerows(allrows)
        print(f"\nwrote {p}")
        return

    if args.image_id is None:
        ap.error("pass --image-id or --suggest-images")

    import cv2
    iid = args.image_id
    if iid not in images:
        sys.exit(f"image_id {iid} is not in the split")
    meta = images[iid]
    img = cv2.imread(os.path.join(K.SPLIT_DIR, meta["file_name"]))
    h, w = img.shape[:2]
    os.makedirs(OUT, exist_ok=True)

    png_name = os.path.splitext(meta["file_name"])[0] + ".png"
    cv2.imwrite(os.path.join(OUT, png_name), img)

    panels = {}
    for cfg, out_name in (("single_pass", "panel_single"), ("full", "panel_ensemble")):
        out = R.infer(predictor, img, cfg)
        teeth, _pre, _f = R.teeth_pipeline(out, w, cfg)
        panels[out_name] = (teeth, jaw_split(teeth))

    # one jaw_split_y and one crop shared by ALL four files, as specified
    jaw_y = panels["panel_ensemble"][1] or h // 2
    crop = central_crop(w, h, jaw_y)

    for out_name, (teeth, _jy) in panels.items():
        d = build_panel(png_name, teeth, jaw_y, crop)
        json.dump(d, open(os.path.join(OUT, out_name + ".json"), "w"), indent=2)

    # panel_full = the complete deployed system, anomalies included
    anom_pred = R.load_predictor(K.ANOM_MODEL)
    aout = R.infer(anom_pred, img, "full")
    _t, anoms = P.process_predictions(aout, image_width=w, teeth_threshold=0.50,
                                      anomaly_threshold=R.ANOM_TAU,
                                      class_map=P.PATHOLOGY_CLASS_MAP)
    anoms = [a for a in anoms if a["class_name"] != "Tooth"]
    teeth_full = panels["panel_ensemble"][0]
    teeth_full = P.DentalGeometricEngine.map_anomalies(teeth_full, anoms)
    teeth_full = P.clean_clinical_findings(teeth_full)
    teeth_full, anoms = P.validate_anatomy(teeth_full, anoms,
                                           anomaly_threshold=R.ANOM_TAU)
    json.dump(build_panel(png_name, teeth_full, jaw_y, crop, anoms),
              open(os.path.join(OUT, "panel_full.json"), "w"), indent=2)

    json.dump(gt_panel(png_name, gt_teeth[iid], class_map, jaw_y, crop, h, w),
              open(os.path.join(OUT, "panel_gt.json"), "w"), indent=2)

    print(f"image_id {iid}  {meta['file_name']}  ({w}x{h})")
    print(f"jaw_split_y {jaw_y}   crop {crop}")
    print(f"{'file':<24} {'detections':>11}")
    for f in ("panel_gt", "panel_single", "panel_ensemble", "panel_full"):
        d = json.load(open(os.path.join(OUT, f + ".json")))
        print(f"  {f+'.json':<22} {len(d['detections']):>11}")
    print(f"  {png_name}")
    print(f"\nwrote to {OUT}")


if __name__ == "__main__":
    main()
