# Model card: ToothAI Photo Screening

ToothAI Photo Screening turns 1 to 5 smartphone photos of the teeth into a screening report: possible
decay, tooth wear and restorations, a triage level and a screening index. When something concerning is
seen it recommends a panoramic X-ray (OPG). It is a **screening aid, not a diagnosis**.

> Research prototype. Not a medical device and not for diagnostic use. All findings require review by a
> qualified dentist.

Version 0.1.0 · October 2026 · Measured results: [EVAL.md](EVAL.md)

## Intended use

- Self-screening at home or in outreach camps, to decide whether to see a dentist and get an OPG.
- Teaching and research at dental colleges, as a demonstration of photo screening and triage.

## Not intended use

- Diagnosis or treatment planning, or ruling disease out. A routine result does not mean healthy teeth.
- Children's primary teeth outside the AKU data's range, X-rays, or photos of anything but the mouth.
- Any use without a dentist reviewing the findings.

## Models

| Model | Architecture | Finds | Trained on | Licence | Role |
| --- | --- | --- | --- | --- | --- |
| `alphadent` | YOLOv8x-seg, 960 px | Wear, filling, crown, caries | AlphaDent (DSLR, mirror and macro photos) | Apache-2.0 | Wear and restorations; caries only as agreement |
| `caries_aku` | YOLOv8s, 640 px | Caries (`d` primary, `D` permanent teeth) | AKU intraoral dataset, 5,011 smartphone photos | Weights ours; data CC BY 4.0 | The caries source |
| `teeth_dentalai` | YOLOv8s-seg, 640 px | Tooth, cavity, crack | DentalAI, 1,991 photos | Weights ours; data CC BY 4.0 | Mouth check and tooth positions; cavity/crack off |

Every report lists each model's SHA-256 prefix (`model_versions`); `scripts/fetch_weights.py` records the
hashes. Training: `notebooks/train_caries_aku.ipynb` and `notebooks/train_teeth_dentalai.ipynb` (Kaggle, T4).

## Data and attribution

- **AKU intraoral caries dataset**, Aga Khan University: *Annotated intraoral image dataset for dental
  caries detection*, Zenodo, [doi:10.5281/zenodo.14827784](https://doi.org/10.5281/zenodo.14827784),
  CC BY 4.0. Official Benchmarking splits; the bundled sample photos come from its test split.
- **DentalAI** (tooth, caries, cavity, crack masks), CC BY 4.0, via Kaggle `pawanvalluri/dental-segmentation`
  (a Roboflow semantic-mask export; tooth instances split by watershed, so they are approximate).
- **AlphaDent** pretrained weights, [github.com/ZFTurbo/AlphaDent](https://github.com/ZFTurbo/AlphaDent),
  Apache-2.0.
- **MouthCare** intraoral sample: evaluation only, never trained on, never shipped (not yet used).

## How a report is decided

Fixed rules decide every finding, band, triage level and the index (`app/rules.py`, unit-tested). An
optional AI model may only add advisory gum and hygiene notes and rephrase the summary.

1. **Quality gate.** Hard checks: the photo's shorter side is at least 400 px, blur (Laplacian variance at
   least 20 at 640 px) and exposure (mean 45 to 225, at most 25 % clipped). A failed photo is skipped with
   a retake tip. The mouth check is a soft warning, or hard ("at least one tooth") when the tooth model is
   loaded.
2. **Detection and fusion.** Marks of the same type are merged when they overlap (IoU above 0.55) or one
   lies mostly (60 %) inside the other. The leading model supplies score and shape; agreeing models are
   kept as `sources`.
3. **What is reported.**
   - Caries seen only by AlphaDent is dropped when the phone model is loaded: AlphaDent caries measured
     AP@50 0.006 on phone photos.
   - Cavity and crack are off by default: precision 0.34 and recall 0.25 even on DentalAI's own test split.
   - Wear is kept from score 0.55. It is not validated on phone photos and over-fires on incisal edges.
4. **Confidence bands.** Caries thresholds come from the AKU validation curve: keep from 0.48 (recall at
   least 0.90), Medium from 0.55 (best F1), High from 0.74 (precision at least 0.90). High also needs a
   phone-trained model or two agreeing models. AlphaDent-only caries would always be Low.
5. **Triage.**

   | Level | When |
   | --- | --- |
   | PROMPT, "See a dentist promptly" | High caries, or cavity/crack at Medium or above |
   | SOON, "Book a check-up soon" | Medium caries, or 2 or more Low decay/cavity/crack findings |
   | ROUTINE | Anything else |

   An OPG is recommended for PROMPT and SOON. Every report also says what photos cannot show.
6. **Screening index** (0 to 100; never called a health score):

   ```
   index = clamp(100 − 20·(cavities + cracks) − min(40, caries) − min(3, 3·wear) − min(5, fillings + crowns), 0, 100)
   caries per finding: High 15, Medium 8, Low 3
   ```

   The caps are deliberate: caries can cost at most 40 points, wear 3 and restorations 5, so no single type
   dominates. Bands: 85 to 100 low concern, 60 to 84 some concern, below 60 higher concern.

## Performance (summary of [EVAL.md](EVAL.md))

On the AKU official test split (628 smartphone photos; 622 passed the quality gate, 226 of them with
caries), with the service defaults:

| Measure | Value (95 % CI) |
| --- | --- |
| Photos with caries referred for a check-up or dentist | 88.1 % [83.2, 91.7] |
| Caries-free photos referred anyway | 4.0 % [2.5, 6.5] |
| Image-level caries sensitivity / specificity | 89.4 % / 95.5 % |
| Caries model alone, mAP@50 (published YOLOv8s baseline 0.841) | 0.764; classes merged 0.895 |
| Tooth detection (DentalAI test) box / mask AP@50 | 0.739 / 0.775 |

Speed: about 0.2 s per photo for all three models on a T4, about 1 s on an Apple M1 (MPS).

## Limitations

- **What photos can't see:** decay between teeth, below the gum line or inside the tooth, bone loss and
  periapical disease. That is why the OPG is recommended.
- **Domain gap:** AlphaDent was trained on DSLR and macro photos, by a single annotator, with one camera
  setup; its caries class does not transfer to phones. Its wear, filling and crown classes are not
  validated on phone photos.
- **Data range:** all measured photos come from two public datasets (AKU: one centre and its patients'
  ages; DentalAI). Real users' phones, lighting, ages and skin tones will differ.
- **Labels:** AKU marks caries only, so false wear and restoration marks are not measured. DentalAI tooth
  instances are approximate.
- **Regions, not tooth numbers:** positions are given as sides and "about the Nth tooth from the middle".
  No FDI numbers are derived from photos.

## Privacy and ethics

- Photos are processed in memory and never stored. There are no accounts and no database, and logs
  contain no image data or filenames.
- Consent is required before upload.
- AI observations are off by default and run locally (Ollama) when on. A remote provider shows an explicit
  notice naming where photos go.
- Findings never use diagnostic wording. The disclaimer is on every page and report, and the triage
  always points to a dentist and an OPG, not to self-treatment.
