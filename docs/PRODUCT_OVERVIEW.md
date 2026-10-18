# Tooth-AI: product overview

**Tooth-AI: Integrated Deep Learning and Geometric Reasoning System for Detection, Segmentation,
Numbering and Anatomical Mapping of Teeth in Panoramic Dental Radiographs.**

*Every tooth, found and explained.*

Tooth-AI reads a panoramic radiograph (OPG) and returns a mask for every tooth, its FDI number,
the findings on it and a dental chart, with a PDF report. Detection is learned. Numbering is
geometric, so every number can be explained, and every case carries a numbering-reliability
flag. It is a research prototype for teaching and research, and a second read for a dentist.
It is not a medical device.

A collaboration between Bharati Vidyapeeth (Deemed to be University) College of Engineering
and Bharati Vidyapeeth Dental College and Hospital, Pune, under the C-CAMP Inter-Institutional
Biomedical Innovations Programme (AICTE–C-CAMP IBIP).

Brand and UX rules are in [`docs/BRAND_AND_UX_SPEC.md`](BRAND_AND_UX_SPEC.md).

## What the app does

| Page | Purpose |
|---|---|
| **Analyze** | Upload an OPG or open a sample case. You get a dark radiograph viewer with layer toggles (masks, numbers, findings, geometry), colouring by finding or by quadrant, zoom and click-to-inspect. Alongside it are a reliability banner, a 32-tooth dental chart, a tooth detail card with **Why this number?**, tables, and PDF, JSON, PNG and Markdown exports. |
| **How it works** | The pipeline, and why numbering is geometric. |
| **Performance** | Measured results with confidence intervals, read live from `paper_eval/results/`. |
| **About** | Institutions, data licence, citation, disclaimer. |

There are two modes. **Full chart** numbers the teeth and maps findings to them. **Findings
only** skips numbering. Notation can be FDI (the default) or Universal.

## Pipeline

1. **Detection.** Mask R-CNN (ResNet-50, 9 classes: tooth plus 8 finding types) runs on the
   original image and on a CLAHE contrast-enhanced copy. The two passes are fused. In Full chart
   mode a second model (41 classes) adds findings.
2. **Geometric numbering.** Distance-IoU NMS removes duplicates, the jaws are split at the median
   tooth height, and each jaw is ordered from the patient's right to left. FDI numbers are then
   assigned in sequence; a gap wider than 1.6× a tooth's width skips numbers for missing teeth.
3. **Anatomical rules.** No FDI number may be used twice, a mouth has at most 32 teeth, and caries
   next to metal restorations is down-weighted.
4. **Mapping.** Each finding is attached to the tooth it overlaps. Findings that overlap no tooth
   are listed separately.

## Detected findings

Caries, retained root, filling, crown, root canal filling, root canal obturation, implant,
and prefabricated metal post.

## Measured performance

Measured on 716 held-out OPGs with no training overlap, in the configuration the app runs.
Numbering is scored on 117 images with consistent labels.

| Measure | Result |
|---|---|
| Tooth detection recall | 94.9 % [94.0, 95.8] |
| Correct jaw, given detected | 97.9 % |
| FDI correct, complete arches (n = 67) | 76.6 % [70.7, 82.2] |
| FDI correct, partial arches (n = 50) | 35.4 % [27.7, 42.8] |
| Geometry vs a network that numbers teeth directly (single pass, paired) | +18.2 pp [+11.8, +24.3] |
| AP, generic tooth class | 72.5 |

The partial-arch drop is the main limitation: one missing tooth shifts every later number in that
jaw. The app flags it on every case. Finding accuracy has not been validated. Full details are in
`paper_eval/AUDIT.md`.

## Run

```bash
pip install -r requirements-app.txt
TOOTH_AI_DEMO=1 streamlit run app.py   # sample cases, no Detectron2 needed
streamlit run app.py                   # full mode: torch + detectron2 + weights in output/
```

The previous single-file app (branded "Dent-AI") is kept as `workspace/app_legacy.py` for comparison.
