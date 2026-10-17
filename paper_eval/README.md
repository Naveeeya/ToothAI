# paper_eval

Evaluation exports for the IEEE paper. Nothing here modifies `app.py` or any
training code.

> **Line references.** Every `app.py:NNN` citation in this folder resolves against
> the git tag `paper-eval-v1` (commit `c151d5b`): the single-file Streamlit app as it
> was before the Tooth-AI UI redesign. The redesign moved the pipeline into `tooth_ai/`
> and kept the old file as `app_legacy.py`; read cited lines with
> `git show paper-eval-v1:app.py`.


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
Kaggle: see [`kaggle/README.md`](kaggle/README.md).

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

`compare_ap.py` exits **2** when the comparison is invalid, and it currently does,
because `rtx4060_48k` was never evaluated.

### On Kaggle

```bash
python3 paper_eval/run_eval.py --routing-ab                   # bug A/B, ~2 min
python3 paper_eval/run_eval.py --all --smoke-test --latency   # 5 images, < 1 min
python3 paper_eval/run_eval.py --all --latency                # the ablation
python3 paper_eval/run_eval.py --sweep                        # tau sweep
python3 paper_eval/run_eval.py --coco-eval                    # 3 models
python3 paper_eval/export_panels.py --suggest-images
python3 paper_eval/export_panels.py --image-id <id>
```

`--smoke-test` only shrinks the workload to 5 images; it does **not** select
configurations. Pass `--all` or `--config <name>` for that.

Stages write their CSVs as they complete (`ablation.csv` after every config,
`coco_eval_716.csv` after every model), and the ablation runs first, so a
session that dies part-way still leaves finished work on disk.

Measured per-image latency from the smoke run on one pinned Tesla T4:
`single_pass` 0.23 s, the two-pass configs 0.57–0.61 s. A first projection of
the whole run at those rates came to ~9.5 h, close enough to Kaggle's cap to
matter; caching decoded ground-truth masks across taus and screening mask IoU
on bounding boxes first brought it down. Both are exact: a mask lies inside
its own box, so disjoint boxes imply zero mask IoU and no match changes.

### Engine v2 (on the Mac, CPU, from cached model outputs)

```bash
python3 paper_eval/cache_outputs.py --split valid --min-teeth 14   # ~25 min: model outputs, numeric ids only
python3 paper_eval/cache_outputs.py --split test --canonical       # ~25 min: canonical + full-arch images
python3 paper_eval/learn_template.py                               # arch template from the TRAIN split
python3 paper_eval/engine_lab.py --split valid                     # compare variants (design)
python3 paper_eval/engine_lab.py --split test --variants v1,v2 --out results/engine_v2.csv
```

`cache_outputs.py` stores the 9-class teeth model's raw outputs (boxes, scores, RLE masks) per image under
`paper_eval/.cache/` (git-ignored), so every engine variant is scored without re-running the model.
`engine_lab.py` uses this harness's matching (mask IoU >= 0.5), subsets and image bootstrap; with two
variants it also writes paired deltas. The `v1` row reproduces `full_fix_routing` exactly (recall 94.3 %,
FDI 61.6 / 76.6 / 35.4 %). On the validation split, which has no audited manifests, full arch means >= 28
labelled teeth, partial 14-27, and FDI uses only images passing `label_quality.py`'s canonical-mapping
test (the same test reproduces the 117 canonical test images exactly).

Every v2 design choice was made on the validation split; the test split was scored for the final
configuration (`results/engine_v2.csv`; validation in `results/engine_v2_valid.csv`).

## Outputs

| file | contents |
|---|---|
| `results/ablation.csv` | one row per configuration; unrun rows are empty with a reason |
| `results/engine_v2.csv` | engine v1 vs v2 on test: recall, FDI (all/complete/partial) with CIs and paired deltas |
| `results/engine_v2_valid.csv` | the same on the validation split (where v2 was designed) |
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

| metric | subset | n | reason |
|---|---|---|---|
| **recall** | full-arch | **93** | the paper number: only on a complete arch does an absent label mean an absent tooth |
| **central-zone recall** | full-arch | **93** | same; sparse images excluded entirely |
| **per-tooth recall** | full-arch | **93** | same; sparse images excluded entirely |
| `recall_anomaly_bearing_teeth` | sparse (1–10 labels) | 491 | the anomaly-annotation regime. 43.1% of its labelled teeth carry an anomaly box vs 0.0% of full-arch. Reported under its own name; never comparable to the above |
| FDI accuracy | canonical / full-arch canonical | 117 / 67 | the 117 include missing teeth, which is what real patients have; the 67 have a 100% label ceiling |
| **FDI, complete vs partial** | canonical, split by arch completeness | **67 / 50** | the headline result. 81.0% complete against 35.6% partial &mdash; the anchoring flaw, isolated. Detection is slightly *better* on the partial arches, so the collapse is numbering alone. See `complete_vs_partial.csv` and AUDIT.md &sect;6.1c |
| jaw accuracy | canonical | 117 | class&rarr;jaw is verifiably correct only here. 25 of the 26 non-canonical full-arch images use a different jaw partition (9 of 16 upper classes shared), so the 93-image figure measures annotation convention as much as the model &mdash; see `check_jaw_membership.py` and AUDIT.md &sect;6.1 |
| position offset | canonical, sparse dropped | **112** | 67 full-arch + 45 mid canonical-partial. The 5 sparse canonical-partial images are excluded: their gaps are unannotated teeth, not absent ones |

## Not run, and why

| configuration | reason |
|---|---|
| `no_gamma` | gamma pass is not in the deployed system (`app.py:454`, called nowhere) |
| `no_flip` | flip pass is not in the deployed system (`app.py:475`, called nowhere) |
| `no_roi_refine` | EfficientNet weights are a 0-byte placeholder |

These rows are written to `ablation.csv` empty, with the reason in the `reason`
column. No metric anywhere is estimated, interpolated or filled in by hand.

## The `full_fix_routing` row

`full` is the deployed system, carrying the `app.py:953-958` class-routing bug
verbatim. `full_fix_routing` is the same system with that bug corrected.

It is a row rather than a whole-run flag because both numbers belong in the
paper. The bug routes anomaly detections into the teeth list, so on the 491
sparse images, which are precisely the ones carrying anomaly labels, the
deployed rows' `recall_anomaly_bearing_teeth` and pre-prune tooth counts are
contaminated by phantom teeth. Measured on 5 anomaly-bearing images, the two
disagree on all 5, by up to 2.5x in reported tooth count (see AUDIT.md 5.3).

`full_fix_routing` is the only row where those two columns mean what their
names say.

## Determinism

Fixed seed `20260921` for the 10,000-resample bootstrap. Image order is sorted
by id. The model is in eval mode with no test-time augmentation beyond the
CLAHE pass the deployed system performs.

See [`AUDIT.md`](AUDIT.md) for what the code does versus what the paper claims.
