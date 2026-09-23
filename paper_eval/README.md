# paper_eval

Evaluation exports for the IEEE paper. Nothing here modifies `app.py` or any
training code.

## What runs where

| script | needs | runs on |
|---|---|---|
| `compare_ap.py` | stdlib | Mac |
| `label_quality.py` | stdlib | Mac |
| `check_sparse_bias.py` | stdlib | Mac |
| `export_review.py` | stdlib | Mac |
| `kaggle/make_bundle.py` | stdlib | Mac |
| `pipeline.py` | numpy, torch, Detectron2 | Kaggle |
| `evalkit.py` | numpy, pycocotools | Kaggle |
| `run_eval.py` | torch, Detectron2, cv2 | Kaggle |
| `export_panels.py` | torch, Detectron2, cv2 | Kaggle |

This Mac is an Apple M1: no CUDA, no torch, no Detectron2. The GPU half runs on
Kaggle — see [`kaggle/README.md`](kaggle/README.md).

## Commands, in order

### On the Mac

```bash
python3 paper_eval/compare_ap.py output/resnet50_9class_20k output/rtx4060_48k output/resnext101_cascade_60k
python3 paper_eval/label_quality.py
python3 paper_eval/check_sparse_bias.py
python3 paper_eval/export_review.py --n 10
python3 paper_eval/kaggle/make_bundle.py
```

Runtime: under 30 s total, except `make_bundle.py` (~2 min, copies 1.7 GB).

`compare_ap.py` exits **2** when the comparison is invalid — it currently does,
because `rtx4060_48k` was never evaluated.

### On Kaggle

```bash
python3 paper_eval/run_eval.py --all --smoke-test --latency   # 5 images, < 1 min
python3 paper_eval/run_eval.py --all --latency                # ~60-90 min
python3 paper_eval/run_eval.py --sweep                        # ~25-40 min
python3 paper_eval/run_eval.py --coco-eval                    # ~20-30 min
python3 paper_eval/export_panels.py --suggest-images          # ~5 min
python3 paper_eval/export_panels.py --image-id <id>
```

Total: roughly **2–3 hours** on a T4. Estimated from image counts, not measured;
the first real run replaces these figures.

## Outputs

| file | contents |
|---|---|
| `results/ablation.csv` | one row per configuration; unrun rows are empty with a reason |
| `results/per_tooth_recall.csv` | `fdi,config,recall,support`, 32 positions, `single_pass` + `ensemble` |
| `results/threshold_sweep.csv` | `tau,central_threshold,central_recall,peripheral_recall,teeth_per_image` |
| `results/position_offsets.csv` | offset histogram, canonical images only |
| `results/coco_eval_716.csv` | three models re-evaluated on one split, each at its own test size |
| `results/panel_candidates.csv` | figure candidates ranked by central-zone recall gain |
| `results/panels/` | `panel_gt/single/ensemble/full.json` + source PNG |
| `results/class_to_position.csv` | the derived class→FDI mapping (Task B) |
| `results/label_quality.csv` | per-image label quality (Task B) |
| `results/instance_counts.csv` | ground-truth instances per class (Task B) |
| `manifests/fdi_evaluable.txt` | the 67 image ids FDI accuracy may use |
| `review/` | 10 SVG panels for clinical review of the labels |

Every CSV opens with a comment header carrying the checkpoint SHA-256, split,
git state, GPU name and seed.

## Which images each metric uses, and why

| metric | subset | reason |
|---|---|---|
| recall | full-arch (93) and sparse (≤10) **separately** | 43.1% of sparse labelled teeth overlap an anomaly box, vs 0.0% of full-arch ones — they are not the same measurement |
| FDI accuracy | 117 canonical (headline) and 67 full-arch canonical (clean-label bound) | the 117 include missing teeth, which is what real patients have; the 67 have a 100% label ceiling |
| jaw accuracy | all 93 full-arch | jaw membership is consistent in every full-arch image, including the 26 whose within-jaw ordering is scrambled |
| position offset | 117 canonical | includes 50 partial arches, the only images where a missing third molar can shift the jaw |

## Not run, and why

| configuration | reason |
|---|---|
| `no_gamma` | gamma pass is not in the deployed system (`app.py:454`, called nowhere) |
| `no_flip` | flip pass is not in the deployed system (`app.py:475`, called nowhere) |
| `no_roi_refine` | EfficientNet weights are a 0-byte placeholder |

These rows are written to `ablation.csv` empty, with the reason in the `reason`
column. No metric anywhere is estimated, interpolated or filled in by hand.

## Determinism

Fixed seed `20260921` for the 10,000-resample bootstrap. Image order is sorted
by id. The model is in eval mode with no test-time augmentation beyond the
CLAHE pass the deployed system performs.

See [`AUDIT.md`](AUDIT.md) for what the code does versus what the paper claims.
