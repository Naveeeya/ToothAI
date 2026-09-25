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
computed on 112 images (67 full-arch canonical + 45 mid canonical-partial),
which include partial arches where the shift can actually occur.

A second, sharper failure mode sits in the same function. When more than 16
detections land in one jaw, `fdi_index` runs past the end of the 16-element
sequence and every surplus tooth is assigned the **same** final FDI number:

```python
if fdi_index < len(fdi_sequence):
    fdi_number = fdi_sequence[fdi_index]
else:
    fdi_number = fdi_sequence[-1]  # Fallback to last   (app.py:151-153)
```

So over-detection in one jaw collapses the excess onto FDI 28 (upper) or 38
(lower). `resolve_fdi_conflicts` then sees a large duplicate group and either
reassigns the losers to arbitrary free slots in that quadrant or discards them
outright (`app.py:721-732`). Because the median jaw split forces an even
division, a surplus in one jaw is not a rare event. This is one reason the
harness reports the **pre-prune** detection count: by the time the final teeth
list is produced, the evidence of over-detection has been rewritten into
plausible-looking FDI numbers.

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

**Measured, not inferred.** `run_eval.py --routing-ab` runs the teeth pipeline
both ways on images sampled from the 380 that carry anomaly labels — the bug
cannot fire where the 9-class model finds no anomaly, and all 93 full-arch
images have zero anomaly annotations, so the question is unanswerable on them.
On a Tesla T4, kernel v4:

| image_id | GT teeth / anomalies | teeth reported, deployed | teeth reported, corrected |
|---|---|---|---|
| 2 | 6 / 2 | **15** | 6 |
| 150 | 8 / 4 | **25** | 16 |
| 288 | 5 / 5 | **12** | 6 |
| 436 | 6 / 4 | **12** | 6 |
| 582 | 7 / 10 | **28** | 23 |

The teeth lists differ on **5 of 5** images. The surplus detections are the
anomalies themselves: on image 288 the six phantom teeth carry scores 0.98,
0.94, 0.99, 0.98, 0.74 and 0.34 — high-confidence fillings and crowns entering
the teeth list, being assigned FDI numbers, and being reported to the clinician
as teeth. In the worst case the deployed system reports **2.5x** the teeth the
corrected one does.

This is a clinical defect, not only a metrics artifact: the tooth count is
inflated roughly in proportion to how much restorative work the patient has
had. A mouth with many fillings is reported as having many more teeth than it
does.

Two cautions on reading that table. The corrected count is **not** validated by
matching the GT count — these images label only 5–8 of the ~28–32 teeth
present, so a corrected count of 6 may equally reflect severe under-detection.
The A/B establishes the bug's effect, not either version's correctness. And the
sample is 5 images; the full-run `full_fix_routing` row is the number to quote.

The harness carries the bug verbatim by default and corrects it in the
`full_fix_routing` configuration, which appears as its own row in
`ablation.csv` so the deployed and corrected numbers sit side by side. The
paper reports the corrected numbers and discloses the bug.

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

## 5.7b Network numbering versus detection plus geometry

The design question behind the whole paper is whether a network should predict
FDI numbers directly or whether a generic tooth detector plus a geometric
engine should assign them. The COCO comparison in section 2 cannot settle it:
one generic class is intrinsically easier to score than 32, so a higher
`AP-Tooth` than mean-AP-over-positions is expected regardless of how good the
numbering is.

Measured the same way on both sides instead -- FDI accuracy, the same 117
canonical images, the same mask-IoU >= 0.5 matching, the same denominator, the
same score threshold (tau = 0.20), and winner-takes-all deduplication per FDI
on both sides so neither gets a free pass on duplicates:

| method | FDI accuracy | paired delta vs network |
|---|---|---|
| detection + geometry (`single_pass`) | **64.5%** | **+20.03 pp** [+13.59, +26.21] |
| detection + geometry (`full`) | 60.9% | +16.38 pp [+10.04, +22.70] |
| network numbering (`resnext101_cascade_60k`, classes 1-32) | **44.5%** [41.1, 47.8] | — |

`single_pass` beats the network on **77 of 117** images, loses on 32, ties on
8.

### The baseline at its own optimum

tau = 0.20 is the threshold the *deployed* pipeline uses, not one chosen for
this model, so the network was given its best shot: inference cached once per
image, tau swept (`results/numbering_sweep.csv`).

| tau | FDI accuracy | detections/image | after dedup | duplicates dropped |
|---|---|---|---|---|
| **0.05** | **46.3%** [42.9, 49.6] | 146.8 | 31.1 | 115.7 |
| 0.08 | 46.3% [42.9, 49.6] | 120.2 | 30.8 | 89.4 |
| 0.12 | 46.1% [42.8, 49.5] | 96.1 | 30.2 | 65.9 |
| 0.18 | 45.1% [41.8, 48.5] | 65.6 | 28.0 | 37.6 |
| 0.20 | 44.5% [41.1, 47.8] | 57.3 | 26.9 | 30.3 |
| 0.30 | 36.3% [33.1, 39.5] | 29.8 | 18.8 | 10.9 |
| 0.50 | 13.2% [11.8, 14.7] | 6.8 | 5.8 | 1.0 |

The network's optimum is **46.3%** at tau = 0.05-0.08, where it was already
near-optimal at the deployed threshold. Tuning recovers **1.8 pp** of a
roughly 20-point gap:

| comparison | gap |
|---|---|
| geometry (`single_pass` 64.5%) vs network at tau = 0.20 (44.5%) | +20.03 pp [+13.59, +26.21] **paired** |
| geometry (`single_pass` 64.5%) vs network at its optimum (46.3%) | **+18.2 pp**, unpaired |
| geometry (`full` 60.9%) vs network at its optimum | +14.6 pp, unpaired |

**Two honest qualifications.** The +18.2 pp figure is a difference of point
estimates: per-image rows were only retained at tau = 0.20, so the *paired*
delta exists only there. And the optimum sits at the low edge of the swept
range, so 46.3% is marginally a lower bound -- though accuracy has plateaued
(0.05 and 0.08 are identical) and `SCORE_THRESH_TEST` is itself 0.05, so
little is left below it.

The conclusion does not depend on either. Geometry's 64.5% exceeds the **upper
end of the network's interval at its own optimum** (49.6%) by 14.9 points. Even
granting the network the most favourable reading of its best configuration, it
loses by a wide margin.

### Why the network struggles

Its own output shows it. At its optimum it emits **146.8** tooth-class
detections per image against a maximum of 32 real teeth, and winner-takes-all
deduplication discards **115.7** of them -- roughly five competing position
claims per tooth. Raising tau suppresses the duplicates but costs accuracy
monotonically (30.3 duplicates and 44.5% at tau = 0.20; 1.0 duplicate and
13.2% at tau = 0.50). The model is not confidently assigning positions; it is
proposing many and relying on the tie-break, and it needs that redundancy to
reach even 46%.

This is the result that justifies the deployed architecture. The paper should
report it, because the architecture is currently justified in the draft by a
class count that does not match the code.

## 5.7c Why the network loses: it trains through annotation-convention noise

Section 6.1 records that 26 of 93 full-arch **test** images use a
class-to-position mapping that disagrees with every other image. The same
check applied to the data the 41-class model was *trained* on:

| split | images | full-arch (checkable) | canonical | **non-canonical** | distinct orderings |
|---|---|---|---|---|---|
| **train** | 5,584 | 737 | 524 | **213 (28.9%)** | **116** |
| valid | 692 | 84 | 56 | 28 (33.3%) | 28 |
| test | 716 | 93 | 67 | 26 (28.0%) | 19 |

The cascade model saw 64,161 tooth instances. Of the 737 training images where
the convention can be checked at all, **213 use a non-canonical mapping, spread
across 116 mutually inconsistent orderings**.

That is the mechanism behind the 20-point gap in 5.7b. Each of those images
teaches the network a *different* answer for the same anatomical position. A
model that predicts position directly has to learn through that noise, and
116 conflicting conventions over 737 examples is not noise it can average
away.

The geometric engine never learns positions at all. It derives them at
inference from left-to-right order within a jaw, so it is **structurally
immune** to this particular corruption -- it cannot be taught a wrong
convention because it is not taught any convention.

The 28.9% is measurable only on the 737 full-arch training images, which are
13% of the training set. The remaining 4,847 cannot be checked: their
conventions may be equally inconsistent and simply invisible. The true
exposure is therefore a lower bound.

## 5.8 The central zone is the EASY zone

The system's central-relaxation rule exists because the spinal shadow is
believed to suppress detection near the midline (`app.py:960-965`, the
comment reads "model scores lower here"). Measured on the 93 full-arch
images, the opposite is true.

| configuration | central recall (35-65%) | peripheral recall | gap |
|---|---|---|---|
| `single_pass` | **98.4** [97.6, 99.1] | 90.9 [89.1, 92.6] | **+7.6** |
| `full` | 98.4 [97.5, 99.1] | 89.2 [87.3, 91.1] | +9.1 |
| `full_fix_routing` | 98.3 [97.4, 99.0] | 90.1 [88.3, 91.8] | +8.2 |
| `no_zone_threshold` | 98.4 [97.5, 99.1] | 89.8 [87.9, 91.6] | +8.6 |
| `nms` | 98.4 [97.5, 99.1] | 89.0 [86.9, 91.0] | +9.4 |

Paired on the same 93 images, `single_pass` central minus peripheral is
**+7.55 pp [+5.76, +9.33]**, and central exceeds peripheral on **63 of 93**
images individually. Detection is *better* in the central zone by a wide and
statistically clear margin, under every configuration.

Four checks support the number rather than an artefact:

- **Coordinate space.** COCO `width`/`height` match the actual JPEG on
  **716/716** images. Ground truth uses `meta["width"]` and predictions use
  `img.shape[1]`; these are the same value, so both sides use one boundary.
- **Zone occupancy.** The 35-65% band holds **1,529 of 2,976** full-arch tooth
  instances (**51.4%**), not the ~25-30% a uniform-width assumption suggests.
  Incisors and canines are narrow and crowded at the midline; molars are wide
  and reach the edges.
- **Threshold sweep.** Central recall is flat at 98.2-98.7% across tau from
  0.05 to 0.50. There is no operating point at which the central relaxation
  changes the outcome.
- **Image clarity.** Measured directly (`check_contrast_confound.py`: JPEG to
  uncompressed BMP via `sips`, parsed with `struct`, no numpy or PIL). At
  matched native resolution the full-arch images have a **smaller**
  central-minus-peripheral edge-energy advantage than other images of the same
  size (0.96 vs 1.40, d = -0.69). The clarity gradient runs slightly *against*
  the recall result rather than producing it.

### A resolution caveat, and a batch caveat

An earlier pass reported that the 93 full-arch images are "smoother" in both
bands (d = -1.12 central, -1.36 peripheral). **That was an artefact of the
measurement**, not a property of the images. All 93 are natively **1991 px**
wide while most of the split is **1056 px** (d = +2.73 on native width), and
downsampling everything to a fixed 256 px compresses a larger image more,
softening its edges. Absolute edge energy correlates with native width at
r = -0.36 (central) and -0.50 (peripheral); restricted to 1991 px images the
peripheral difference collapses to **d = +0.02**. The claim is withdrawn.

The `central-minus-peripheral intensity sd` proxy correlates with native width
at **r = +0.570**, so it was tracking resolution rather than contrast. Where
the two proxies disagreed, edge energy is the one to trust.

What survives is a **batch** caveat, which the paper should state: all 93
full-arch images come from one acquisition family (`cate*` filenames at
1991 px), while the 1056 px images are a separately-named family. The
annotation-complete subset is not a random sample of the split, so every
full-arch number in this study -- recall, central recall, per-tooth recall,
jaw accuracy -- is measured on one batch.

The zone holds half the dentition and it is the half made of simple,
well-separated, high-contrast anterior teeth. The weakness is in the
**periphery** -- the molar regions, where teeth are wide, overlapping, and
frequently cut off at the image edge.

**Consequence for the paper.** The zone-adaptive threshold addresses a
weakness that this test set does not exhibit, in the region that is in fact
easiest. Removing it (`no_zone_threshold`) *improves* full-arch recall by
0.27 pp [0.10, 0.44] paired. If a spinal-shadow effect exists in clinical
data, it is not present here and cannot be claimed from these results.

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

**CORRECTION (this round). Jaw membership is *not* preserved.** An earlier
version of this section claimed the same 16 classes were upper in every
full-arch image, and therefore scored jaw accuracy on all 93. That claim was
asserted, never checked by code, and it is wrong.

`check_jaw_membership.py` splits each full-arch image's ground-truth boxes at
the y-median of their own centres — the same rule `split_jaws` uses
(`app.py:82-104`) — and asks which class ids land upper. No model is involved.

| subset | ground-truth label jaw matches geometry |
|---|---|
| canonical full-arch (n=67) | 2,144 / 2,144 = **100.0%** |
| non-canonical full-arch (n=26) | 482 / 832 = **57.9%** |
| all full-arch (n=93) | 2,626 / 2,976 = 88.2% |

**25 of the 26** share a single alternative jaw partition whose upper set is
`{1,2,3,4,5,6,7,8,12,23,27,28,29,30,31,32}` — overlapping the canonical upper
set in only **9 of 16** classes. The 26th is canonical at the jaw level and
scrambled only within jaws.

So there *is* a second convention in this dataset, consistent at the jaw level
across 25 images, even though within-jaw ordering varies over 18 distinct
orderings. Class→jaw is reliable on canonical images and on those alone.

**What this invalidates.** The jaw accuracy of **87.2%** previously reported on
93 images is not a model result — it is 99% on the images whose labels are
right and 57% on the images whose labels put the tooth in the other jaw.
Measured on the same detections:

| subset | jaw accuracy given detection (`single_pass`) |
|---|---|
| canonical (n=117) | **97.9%** [97.3, 98.5] |
| canonical full-arch (n=67) | **98.96%** |
| non-canonical full-arch (n=26) | 57.2% — label noise, not model error |
| all full-arch (n=93) | 87.2% — **do not quote** |

The jaw figure to quote is **97.9% on 117 canonical images**. `ablation.csv`
now carries both columns, the confounded one named as such.

This surfaced only because the FDI decomposition put detection, jaw and
numbering on one denominator (§6.1b). It is the second claim this round that
did not survive being recomputed rather than restated; the first was the
smoothness claim in §5.8.

Over the whole split:

```
ceiling = tooth instances in canonical images / all tooth instances
        = 3,370 / 8,107 = 41.6%
```

A tooth in a non-canonical image cannot be scored correct whatever the model
predicts. On the 67-image subset the ceiling is 100% by construction, which is
why `n` is printed beside every FDI figure.

### 6.2 The dataset contains two disjoint annotation regimes

This is a property of the **published dataset**, not of our split, and it
belongs in the paper. Images were annotated either for *teeth* or for
*anomalies*, essentially never for both.

Contingency table over the 716-image test split:

|  | has anomaly labels | no anomaly labels | total |
|---|---|---|---|
| **has tooth labels** | 380 | 268 | 648 |
| **no tooth labels** | 67 | 1 | 68 |
| **total** | 447 | 269 | 716 |

The two populations behave completely differently:

- the 380 anomaly-bearing images average **6.9** labelled teeth, every one has
  ≤ 10, and **not one is a complete arch**;
- the 268 images without anomalies average **20.5** labelled teeth and contain
  **all 93** full arches;
- **0 of the 67 FDI-evaluable images carry any anomaly label**.

Fraction of labelled teeth whose bounding box overlaps an anomaly box (IoU > 0):

| group | images | labelled teeth | overlapping an anomaly | fraction |
|---|---|---|---|---|
| full-arch (32 labelled) | 93 | 2,976 | 0 | **0.0%** |
| sparse (1–10 labelled) | 491 | 3,425 | 1,477 | **43.1%** |
| mid (11–31 labelled) | 64 | 1,706 | 0 | 0.0% |

The separation is total: every image with 11 or more labelled teeth has zero
anomaly annotations.

**Consequences for evaluation.** Recall on sparse images is measured on a
population where 43% of the labelled teeth are restored or diseased; recall on
full-arch images on one where none are. These are different measurements and
must never be averaged or quoted as a single figure. Therefore:

- recall, **central-zone recall** and **per-tooth recall** are computed on the
  **93 full-arch images only** (`n = 93`, printed in every header). Those are
  the paper's numbers.
- sparse-image recall is reported under the explicit name
  `recall_anomaly_bearing_teeth`, with its own `n`, and is excluded from
  central-zone and per-tooth recall entirely.
- the position-offset histogram uses **112** images — 67 full-arch canonical
  plus 45 mid canonical-partial. Of the 50 canonical-partial images, 45 are mid
  (1,206 tooth instances, **zero** anomaly annotations) and 5 are sparse (20
  tooth instances, 3 of them carrying anomaly labels). The 5 are dropped: their
  gaps are unannotated teeth, not absent ones, so an offset there would measure
  annotation policy rather than the FDI anchoring flaw.

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
