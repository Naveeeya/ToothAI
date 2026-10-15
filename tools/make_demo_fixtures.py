#!/usr/bin/env python3
"""Build the demo samples in assets/samples/ from cached model predictions.

Each sample is a validation-split OPG plus the teeth model's own saved predictions for it
(output/resnet50_9class_20k/inference/tooth_remapped_val/coco_instances_results.json:
single pass, score >= 0.30, COCO RLE masks). Demo mode feeds these through the real
geometric engine, so the chart is genuine pipeline output, not hand-made.

Source file names contain patient names; samples are written under neutral names and the
mapping is deliberately not recorded anywhere.

    .venv/bin/python tools/make_demo_fixtures.py
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "final-di-remapped" / "valid"
PREDS = ROOT / "output" / "resnet50_9class_20k" / "inference" / "tooth_remapped_val" / "coco_instances_results.json"
OUT = ROOT / "assets" / "samples"

# (validation image id, neutral name, description shown in the UI)
SAMPLES = [
    (20, "sample_01", "Complete dentition"),
    (45, "sample_02", "Complete arches with restorations"),
    (198, "sample_03", "Partial arches with restorations"),
]


def main():
    images = {im["id"]: im for im in json.loads((SPLIT / "_annotations.coco.json").read_text())["images"]}
    preds = json.loads(PREDS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for image_id, name, description in SAMPLES:
        im = images[image_id]
        shutil.copyfile(SPLIT / im["file_name"], OUT / f"{name}.jpg")
        dets = [
            {"category_id": p["category_id"], "bbox": [round(v, 2) for v in p["bbox"]],
             "score": round(p["score"], 4), "rle": p["segmentation"]["counts"]}
            for p in preds if p["image_id"] == image_id
        ]
        fixture = {"name": name, "description": description,
                   "width": im["width"], "height": im["height"],
                   "source": "Tooth-AI teeth model (resnet50_9class_20k), single pass, "
                             "cached predictions, score >= 0.30",
                   # COCO category ids are 1-based: 1 = Tooth ... 9 = Root canal obturation
                   "detections": dets}
        (OUT / f"{name}.json").write_text(json.dumps(fixture, separators=(",", ":")))
        index.append({"name": name, "description": description})
        print(name, len(dets), "detections")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
