# Tooth-AI — code vs. paper audit

For co-authors and the supervisor. Every claim below is checked against the
source with a file and line number. Where the paper and the code disagree, the
code is reported as-is.

Repository state: commit `6b58c64`.

---

## 1. What the deployed system actually is

`app.py:2167` defaults the model selector to `index=0`, i.e. **"Complete Scan"**
(`app.py:318-335`). That loads **two** checkpoints:

| role | checkpoint | `cls_score` in weights | NUM_CLASSES |
|---|---|---|---|
| teeth | `output/resnet50_9class_20k/model_final.pth` | `(10, 1024)` | **9** |
| anomalies | `output/rtx4060_48k/model_final.pth` | `(42, 1024)` | **41** |

SHA-256:

```
d485b32de4fca7596af5bade38c02a108d76f79343de3e1f04b907558b3516d3  resnet50_9class_20k
51d9bb6018dd51c51afb5c5e86819969818f188543dc64fbdaf15ba7695b56ef  rtx4060_48k
79ce2132f11588039f6eb578ba48ed2f697971fa4a0c860923d2cc70199314fb  resnext101_cascade_60k
```

The 41-class model's tooth predictions are explicitly discarded at
`app.py:2281`.

**`MODEL_DIR`, `CONFIG_PATH`, `WEIGHTS_PATH` and `MODEL_VERSION` at
`app.py:23-26` are dead constants.** Nothing references them past line 25. The
banner string `"ResNeXt-101 Cascade 41-Class v2.0 (60k iter)"` is not what runs.

### The paper says 41 classes = 32 FDI + 9 anomalies. Both halves are wrong.

- The production tooth detector predicts **one generic `Tooth` class**. Every
  FDI number is assigned by the geometric engine: y-median jaw split
  (`app.py:82-104`) then left-to-right sort against the hardcoded `FDI_UPPER` /
  `FDI_LOWER` arrays (`app.py:57-58`). No network predicts an FDI number.
- The 41-class schema is `t` + `1`…`32` + **8** anomalies. There are eight
  anomaly classes, not nine; the "ninth" is `Tooth` itself in the 9-class schema.
- Classes `1`…`32` are **not FDI numbers and not universal numbering**. The
  mapping to arch position is documented nowhere in the repository. It was
  derived empirically from the 93 full-arch test images and is recorded in
  `results/class_to_position.csv`. The paper must state it.

Measured support for reporting the generic-plus-geometry design: on the same
692 validation images, the 9-class model scores `segm AP-Tooth` **71.12**, while
the 41-class cascade averages **13.99** over its 32 position classes.

---

## 2. Which checkpoints exist

| paper name | directory | NUM_CLASSES (from weights) | iterations reached | final checkpoint |
|---|---|---|---|---|
| ResNet-50 9-class 20k | `resnet50_9class_20k` | 9 | 20,000 | yes |
| ResNet-50 41-class 24k | `workspace/phase3/exp_41cls_24k` | — | — | **none**; `metrics.json` is 0 bytes |
| ResNeXt-101 9-class 40k | `resnext101_9class_40k` | config says 9 | **1,545** (22.9 s/it) | **none** |
| ResNeXt-101 Cascade "9-class" 60k | `resnext101_cascade_60k` | **41**, not 9 | 60,000 | yes |
| ResNet-50 41-class 48k | `rtx4060_48k` | 41 | 47,999 | yes |
| Cascade R101 41-class 70k | `cascade_r101_70k` | 41 | **30,918** | **none** |

Three of the six cannot be reported at all.

### `rtx4060_48k` was never evaluated

Its `metrics.json` has 2,401 lines and **none** carries a `segm/` or `bbox/`
key; its log has zero `Evaluation results for segm` blocks. The run ends with
Detectron2's `No evaluator found. Use DefaultTrainer.test(evaluators=)`.

The comparison the paper needs — 41-class vs 9-class on the same ResNet-50
backbone — has never been computed. The `16% mAP` in commit `c977f24` does not
come from this model's metrics. `paper_eval/run_eval.py --coco-eval` computes it.

---

## 3. The test split

| dataset | train / valid / test | note |
|---|---|---|
| `data/final-di` | 2,647 / 2,129 / 2,216 | original Roboflow split, 41-class labels |
| `data/final-di-stratified` | 5,584 / 692 / 716 | exactly 80/10/10, 41-class labels |
| `data/final-di-remapped` | 5,584 / 692 / 716 | same images, 9-class labels |
| `data/merged-9class` | 5,830 / 692 / 716 | remapped + 227 extra train images |

Both the paper's claims are true — of different datasets. The 80/10/10 comes
from `workspace/resplit_dataset_stratified.py` (seed 42). No separate split
manifest is written; the three `_annotations.coco.json` image lists *are* the
manifest.

### The 2,216-image split is disqualified

```
final-di/test (2,216)  ∩  final-di-remapped/train  =  1,772   (80%)
stratified/test (716)  ∩  final-di/train           =    292   (41%)
stratified/test (716)  ∩  final-di-remapped/train  =      0
```

**80% of the 2,216-image test set is training data for the deployed teeth
model.** All evaluation uses the 716-image stratified test set, which is clean
for the teeth model. It leaks 41% into the *anomaly* model, which is one reason
anomaly evaluation is out of scope.

`final-di-stratified/valid` and `final-di-remapped/valid` contain **exactly the
same 692 images** (verified: intersection 692, sets equal).

---

## 4. Where each stage lives

| the paper says | the code does |
|---|---|
| four passes (original, CLAHE, gamma, flip) | **two** — `app.py:405-423`, original + CLAHE only. `apply_gamma` (`:454`) and `flip_instances_horizontal` (`:475`) are defined and **called nowhere**. The docstring at `:392-395` says the 4-pass version was removed for producing "59 teeth instead of ~28-32". Not in git history either: `app.py` has one commit. |
| WBF at IoU 0.4 | **0.55** (`app.py:419`). Implementation at `:499`. It is not textbook WBF: single-pass detections take a 0.6 penalty (`:520`, `:620-627`) and the cluster score is `max`, not the weighted mean. The 0.4 is **DIoU-NMS on teeth** at `app.py:2314`, applied after thresholding. |
| tooth threshold 0.03 central / 0.35 outside | `max(τ·0.7, 0.08)` inside vs `τ` outside (`app.py:960-965`), τ default **0.20** (`:2192`) → effective **0.14 / 0.20**. The 35–65% band itself is correct (`:922-923`). |
| anomalies on a uniform 0.25 | slider default **0.15** (`:2197`); Crown/Implant/metal post get `max(τ·0.6, 0.10)` = 0.10 (`:973`). |
| anomalies mapped to teeth by mask IoU | **bounding-box** IoU ≥ 0.15 (`app.py:190-208`). |
| EfficientNet-B0 refines low-confidence ROIs | **not in the deployed path.** It exists as `workspace/tools/integrated_inference.py` (gate at `:233`) with a trainer at `workspace/classifier/train_effnet.py`. The weights are `workspace/tooth-ai/models/effnet_fdi_final.pth.placeholder` — **0 bytes**. `app.py` never imports it. |
| CLAHE clip 2.5, 8×8, L channel in LAB | correct (`app.py:411`, `:426-450`). |
| `SCORE_THRESH_TEST` 0.05 at inference | correct (`app.py:375`, overriding the config's 0.3). |
| WBF skip-box 0.05 | correct. |
| jaw split at y-median | correct (`app.py:82-104`). |
| FDI by spatial sort within quadrant | correct (`app.py:106+`). |

---

## 5. Things the paper does not mention

### 5.1 The jaw split is an unconditional median

`split_jaws` (`app.py:82-104`) takes the median of the detections' y-centres,
then assigns `y < median_y` to the upper jaw and `y >= median_y` to the lower
(`app.py:100-103`). The split is therefore forced to be even to within one
tooth, regardless of the patient: upper receives the detections strictly above
the median, lower receives the rest. Whenever the true number of upper and
lower teeth differs — or the detector finds different numbers — teeth are
placed in the wrong jaw by construction, and no evidence about the actual jaw
boundary is consulted. `run_eval.py` reports jaw-assignment accuracy separately
on all 93 full-arch images so the size of this effect is visible.

### 5.2 FDI assignment is anchored on the leftmost detection

`sort_and_assign_fdi` (`app.py:108-170`) labels the leftmost detection in each
jaw `18` (or `48`) unconditionally, then walks right. Its gap-detection check
is guarded by `if i > 0` (`app.py:135`), so it runs only *between* consecutive
teeth — never before the first one. A missing third molar, common in adults, shifts the entire jaw by
one position. `results/position_offsets.csv` is the histogram of exactly this,
computed on the 117 canonical images because those include 50 partial arches
where the shift can actually occur.

### 5.3 The class-routing bug at `app.py:953-958`

```python
is_tooth = False
if class_name == "Tooth":
    is_tooth = True  # 9-class model
elif cls_id <= 32:  # 41-class model: IDs 0-32 are individual teeth
    is_tooth = True
```

For the 9-class model the class ids are `0=Tooth, 1=Caries, 2=Crown, 3=Filling,
4=Implant, 5=Prefabricated metal post, 6=Retained root, 7=Root canal filling,
8=Root canal obturation` — **every one of them ≤ 32**. The `elif` fires for all
eight anomaly classes.

**This affects teeth, not only anomalies.** Every anomaly the primary model
detects is routed into the *teeth* list, where it enters DIoU-NMS, the median
jaw split, FDI assignment, recursive pruning and FDI conflict resolution. The
`elif class_name in ANOMALY_CLASSES` branch below it is unreachable for the
9-class model, so in practice every reported anomaly comes from the 41-class
hybrid model alone.

The harness carries the bug verbatim by default and corrects it under
`--fix-cls-routing`, so both numbers exist. The paper reports the fixed numbers
and discloses the bug.

### 5.4 There is no 1024×512 resize

`app.py` never resizes. It relies on `DefaultPredictor`'s `ResizeShortestEdge`,
driven by each config — and **the two deployed models differ**:

| model | `MIN_SIZE_TEST` | `MAX_SIZE_TEST` |
|---|---|---|
| `resnet50_9class_20k` (teeth) | 800 | 1333 |
| `rtx4060_48k` (anomalies) | 512 | 1024 |

The two halves of one pipeline see the same radiograph at different
resolutions. `workspace/configs/mask_rcnn_1024x512.yaml` exists but is a
training config used by neither deployed checkpoint.

### 5.5 `FDIValidator` is not imported by `app.py`

`fdi_validator.py` defines `FDIValidator`, and the project documents an "FDI
Confidence Penalty (20% per critical warning)". `grep` over `app.py` finds no
import of it. That penalty is **not** in the deployed path.

### 5.6 Undescribed post-processing subsystems

All run on every inference, none appear in the paper:

| subsystem | location | what it does |
|---|---|---|
| gap detection | `app.py:126-135` | skips an FDI slot when the x-gap to the previous tooth is large |
| Global Anatomical Constraint | `resolve_fdi_conflicts`, `app.py:674` | winner-takes-all on duplicate FDI numbers; losers move to the nearest free slot in the quadrant **or are discarded** |
| metal artifact suppression | `suppress_metal_artifacts`, `app.py:742` | halves the score of `Caries` within 5 px of radiopaque work and renames it |
| anatomical count check | `recursive_prune_teeth`, `app.py:818` | raises the confidence floor in 0.02 steps until ≤ 32 teeth remain |
| finding hierarchy | `clean_clinical_findings`, `app.py:1028` | drops `Filling` when a root canal or crown is present on the same tooth |
| master orchestrator | `validate_anatomy`, `app.py:857` | runs prune → conflicts → metal → re-threshold |

`recursive_prune_teeth` caps the reported count at 32, which hides
over-detection. `run_eval.py` therefore reports the **pre-prune** count (after
DIoU-NMS, before pruning) as well as the final count, and how often pruning
fired.

### 5.7 Region labels in the reports divide by a hardcoded 1000

`app.py:1486`, `:1528`, `:1692` and `:1761`:

```python
location = "Center" if 0.35 < x_center / 1000 < 0.65 else (...)
```

The divisor is the literal `1000`, not the image width. For any radiograph
whose width is not 1000 px the Left/Center/Right label in the generated
clinical report and PDF is wrong. The detection zone logic itself
(`app.py:922-923`) correctly uses `image_width`; only the report labels are
affected.

---

## 6. Ground-truth quality on the 716 split

Measured in `results/label_quality.csv` and `results/instance_counts.csv`.

8,107 tooth instances over 716 images — mean 11.32, **median 8** per image,
against ~28–32 teeth actually present. Annotation is not exhaustive.

| quantity | value |
|---|---|
| full-arch images (all 32 classes once) | 93 |
| canonical-mapping images | 117 |
| images with duplicate tooth classes | 9 |
| FDI-evaluable (full arch ∧ canonical) | **67** (9.4%) |

### 6.1 Label noise caps FDI accuracy at 41.6%

Of 93 full-arch images, only 67 use the canonical class→position mapping. The
other 26 are **wholesale relabellings**, not an orientation convention:

| test | result |
|---|---|
| mirrored left–right | 0 of 26 |
| upper/lower swapped | 0 of 26 |
| mirrored + jaw swapped | 0 of 26 |
| single jaw reversed | 0 of 26 |
| **distinct orderings among the 26** | **18** (the most common shared by only 6) |

25 of the 26 differ from canonical in 15–16 of 16 upper positions; only one is
a near-miss explainable by adjacent swaps. There is no single alternative
convention to remap them to.

**Jaw membership, however, is preserved in all 26** — the same 16 classes are
upper in every full-arch image. So class→jaw is reliable across all 93 even
though class→position is not, which is why jaw-assignment accuracy is scored on
93 images and FDI accuracy on 67.

Over the whole split:

```
ceiling = tooth instances in canonical images / all tooth instances
        = 3,370 / 8,107 = 41.6%
```

A tooth in a non-canonical image cannot be scored correct whatever the model
predicts. On the 67-image subset the ceiling is 100% by construction, which is
why `n` is printed beside every FDI figure.

### 6.2 The split contains two disjoint annotation regimes

| | has anomaly labels | no anomaly labels |
|---|---|---|
| **has tooth labels** | 380 | 268 |
| **no tooth labels** | 67 | 1 |

- The 380 anomaly-bearing images average **6.9** labelled teeth and **none** is
  a full arch.
- The 268 images without anomalies average **20.5** labelled teeth and contain
  all 93 full arches.
- **0 of the 67 FDI-evaluable images carry any anomaly label.**

Fraction of labelled teeth overlapping an anomaly box (IoU > 0):

| group | images | teeth | overlapping | fraction |
|---|---|---|---|---|
| full-arch (32 labelled) | 93 | 2,976 | 0 | **0.0%** |
| sparse (1–10 labelled) | 491 | 3,425 | 1,477 | **43.1%** |
| mid (11–31 labelled) | 64 | 1,706 | 0 | 0.0% |

Recall on sparse images is measured on a population where 43% of the labelled
teeth are restored or diseased; recall on full-arch images on one where none
are. They are not the same measurement and `ablation.csv` reports them in
separate columns.

### 6.3 Anomaly evaluation is out of scope

Support on the 716 split: Filling 2,504 / Caries 385 / Root canal filling 323 /
Root canal obturation 286 / Crown 101 / Prefabricated metal post 65 / Retained
root 44 / **Implant 10 (in 4 images)**. Combined with the 41% train leak into
the anomaly model and the disjoint annotation regimes above, no defensible
per-class anomaly AP can be computed. `results/instance_counts.csv` reports the
counts so the paper can state why.

### 6.4 Open question for clinical review

`paper_eval/review/` holds 10 canonical-partial images rendered with their
ground-truth boxes and FDI labels, and a list of the unlabelled positions. The
question for a reviewer at the dental college: are those positions **absent
teeth** or **present but unannotated**? If mostly the latter, "labelled teeth"
is not a valid recall denominator and every recall figure understates the model.

---

## 7. Repository hygiene found during the audit

### 7.1 `.gitignore` contained a UTF-16 line that git read as `*`

The last line of `.gitignore` was `**/model_final.pth` encoded as **UTF-16LE**
(a PowerShell `>>` redirect). Git reads `.gitignore` as bytes and stops at the
first NUL, so the pattern parsed as a bare **`*`**.

**Every untracked file in the repository was being ignored.** `git status`
reported zero untracked files while `paper_eval/`, `.graphifyignore` and
`ap_summary.json` all existed.

It arrived in **`6b58c64`**, "Update gitignore and gitattributes for model
weight exclusion" — the most recent commit. Repaired to UTF-8 during this audit.

Note the resulting order: the repaired `**/model_final.pth` at line 66 sits
*after* `!output/*/model_final.pth` at line 36, so it overrides that negation
and **new** checkpoints will be ignored. Already-tracked checkpoints are
unaffected. Left as-is by decision.

### 7.2 `git status` was failing entirely

`.gitattributes` routes `*.pth` through the LFS filter. Without `git-lfs`
installed, `git status` exits 128 (`git-lfs filter-process: git-lfs: command not
found`), which is what hid 7.1. Install `git-lfs` before trusting `git status`
on this repository.

### 7.3 The working tree is CRLF, HEAD is LF

`git status` reports 459 files as modified with **zero content change** — the
working tree has CRLF line endings, the committed blobs have LF, and
`core.autocrlf` is unset. Not renormalised: teammates work on Windows and that
is a team decision. Stage by explicit path until it is settled.
