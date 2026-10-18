# Tooth-AI: context for Claude Code

**Tooth-AI: Integrated Deep Learning and Geometric Reasoning System for Detection,
Segmentation, Numbering and Anatomical Mapping of Teeth in Panoramic Dental Radiographs.**

A research prototype that takes a panoramic radiograph (OPG) and returns per-tooth
instance masks, FDI numbers, mapped findings (caries, restorations, endodontic,
prosthetic), a dental chart and a PDF report.

- Collaboration: Bharati Vidyapeeth (Deemed to be University) College of Engineering
  and Bharati Vidyapeeth Dental College & Hospital, Pune.
- Programme: C-CAMP Inter-Institutional Biomedical Innovations Programme (AICTE–C-CAMP IBIP).
  Partner logos live in `assets/logo_{aicte,bvu,ccamp,cibip}.png`.
- Target venue: an IEEE paper. Evaluation harness: `paper_eval/`.
- Brand and UI rules: `docs/BRAND_AND_UX_SPEC.md` and `assets/brand/BRAND.md`.
  The product name is **ToothAI** (Brand Guidelines v1.0, Oct 2026: wordmark "Tooth" + "AI" in Spark
  Blue; Plus Jakarta Sans + JetBrains Mono; palette in `tooth_ai/ui/theme.py`). "Tooth-AI" and
  "Dent-AI" are retired; assets come from `tools/build_brand.py`.

## Layout

| path | what |
|---|---|
| `app.py` | Streamlit entry point: page config, theme, `st.navigation` |
| `workspace/app_legacy.py` | the old single-file "Dent-AI" app, kept for comparison; do not extend |
| `tooth_ai/pipeline.py` | pure inference + geometric engine (no Streamlit import) |
| `tooth_ai/analysis.py` | `analyze()` orchestration → `AnalysisResult`, reliability assessment |
| `tooth_ai/render.py` | overlay drawing (layers, colour-by, geometry, selection) |
| `tooth_ai/notation.py` | FDI ↔ Universal, tooth names |
| `tooth_ai/report_pdf.py`, `report_md.py` | PDF / Markdown / JSON exports |
| `tooth_ai/demo.py` | demo mode: cached model detections through the real engine, no Detectron2 |
| `tooth_ai/evidence.py` | measured numbers, parsed from `paper_eval/results/` |
| `assets/brand/`, `static/fonts/` | logo/marks/tokens (generated), self-hosted Plus Jakarta Sans + JetBrains Mono (OFL) |
| `assets/samples/` | de-identified demo OPGs + cached detections |
| `tools/` | `build_brand.py`, `make_demo_fixtures.py`, `report_media/` (report figures), `dataset/` (one-off dataset scripts) |
| `tests/` | pytest: routing fix, regions, notation, engine parity vs paper_eval, reliability |
| `tooth_ai/ui/` | theme tokens, CSS, reusable components |
| `tooth_ai/views/` | pages: analyze, method, performance, about |
| `paper_eval/` | evaluation harness and audit; source of every number we show |
| `workspace/` | older phase work, a legacy proof-of-concept UI and `legacy_root/` (old notes and scripts); not on the deployed path |
| `docs/` | brand and UX spec, product overview, requirements, `setup/` (Mac and Windows set-up) |
| `output/<run>/` | Detectron2 configs + `model_final.pth` (weights are git-ignored) |
| `photo_screening/` | smartphone photo screening: FastAPI service, rules, PDF, UI page, notebooks, eval |
| `docker/`, `deploy/` | app image (Detectron2), production compose + Caddy, Oracle VM setup and push scripts |

## Published pipeline (v1, Numbering → Published)

The app's default is engine v2 (`tooth_ai/engine_v2.py`): one pass, then its own clean-up, bite-line jaw
split, symmetry midline and template numbering in place of steps 1 and 4 below.

1. Two passes: original + CLAHE (clip 2.5, 8×8), fused by weighted box fusion at IoU 0.55.
   Gamma and flip passes exist in code but are **not** called.
2. `process_predictions`: split detections into teeth and findings (centre-zone threshold relaxation).
3. "Full chart" mode also runs the 41-class model (`output/rtx4060_48k`) for findings and
   merges them (IoU > 0.3 keeps the higher score).
4. `DentalGeometricEngine`: DIoU-NMS (0.4) → `split_jaws` (median Y) → `sort_and_assign_fdi`
   (sort by X, walk `FDI_UPPER`/`FDI_LOWER`, skip numbers when the gap > 1.6× tooth width).
5. `map_anomalies`: each finding goes to the best-IoU tooth (> 0.15), else "unassigned".
6. `validate_anatomy`: cap at 32 teeth, resolve FDI conflicts, penalise caries near metal.

Models: teeth `output/resnet50_9class_20k` (9 classes: Tooth + 8 findings);
findings `output/rtx4060_48k` (41 classes: ids 0–32 tooth, 33–40 findings).

Data shapes: tooth `{box, fdi, score, quadrant, mask, findings[], trace}`,
finding `{type, score, box, mask}`.

## Numbers we are allowed to show

Only measured `paper_eval` figures (716 held-out OPGs, no train overlap, Tesla T4,
bootstrap 95 % CIs; see `paper_eval/AUDIT.md`, `paper_eval/results/*.csv`). **Always say which
configuration a number comes from.** In code, read them through `tooth_ai/evidence.py` (it parses
the CSVs) instead of hard-coding.

Published engine v1 (`full_fix_routing`, two passes + routing fix; `evidence.DEPLOYED`, the paper's configuration):
- Tooth detection recall **94.9 % [94.0, 95.8]** (n = 117); jaw given detected **97.9 %**.
- FDI end-to-end **61.6 %**: complete arches **76.6 % [70.7, 82.2]** (n = 67), partial arches
  **35.4 % [27.7, 42.8]** (n = 50). The partial-arch collapse is the headline limitation: one
  missing tooth shifts every number in that jaw.

Best measured (`single_pass`): recall **95.3 %**, FDI **64.5 % [58.7, 70.1]**, complete 81.0 %,
partial 35.6 %. Single pass beats two passes (+3.65 pp FDI, paired): a candidate change.

- Geometry (single pass) vs a direct-numbering 41-class network: **+18.2 pp [+11.8, +24.3]**
  paired (64.5 % vs 46.3 %); most of the gap is detection, not numbering.
- AP-Tooth (segm) **72.5**. Latency: 0.23 s/image single pass, 0.57–0.61 s two-pass (T4).

Engine v2 (`tooth_ai/engine_v2.py`, the app default since 18 Oct 2026; `paper_eval/results/engine_v2.csv`, designed on the validation
split, `paper_eval/engine_lab.py` on cached outputs): FDI **85.8 %** (complete 90.6 %, partial **77.4 %**),
recall **95.5 %**; paired vs v1 FDI **+24.2 pp [+18.6, +29.9]**. v1 stays selectable and is still
`evidence.DEPLOYED` (the paper reports it); `TOOTH_AI_ENGINE=v1` makes it a deployment's default again.

**Never show or write:** 87 % / 85–90 % FDI, 0.78 mask IoU, 92 % quadrant accuracy,
"< 500 ms", "4-pass Smart Ensemble", "EfficientNet refinement", "6 production models",
any anomaly/caries accuracy, "clinically validated", "95 % time reduction". These were
never measured on this system or were contradicted by `paper_eval`.

Every user-facing surface carries the disclaimer: *Research prototype. Not a medical
device and not for diagnostic use. All findings require review by a qualified dentist.*

## Rules

- **paper_eval line references** (`app.py:NNN`) resolve against git tag `paper-eval-v1`.
  `paper_eval/pipeline.py` is a verbatim copy of the old pure functions: don't edit it to
  match new code. `tests/test_engine_parity.py` keeps `tooth_ai/pipeline.py` honest.
- **Bugs fixed in the redesign** (behaviour differs from `paper-eval-v1` on purpose):
  9-class finding ids 1–8 were routed into teeth (`cls_id <= 32` branch now applies only to
  the 41-class model); region labels divided by a hard-coded 1000 instead of image width.
- **Privacy:** filenames under `data/test data/` and `data/final-di*/` contain patient names,
  dates, ages, sex. Never surface them in UI, fixtures, logs or commits. Demo samples live in
  `assets/samples/` under neutral names. The dataset is CC BY-NC 4.0
  (Adnan et al., *Data in Brief*, 2024): attribute it and treat it as non-commercial.
- **Copy:** sentence case, no emoji (use `:material/…:` icons), "findings" not "anomalies",
  modes are "Full chart" and "Findings only".
- Repo files mostly use CRLF line endings (Windows origin); new files use LF.

## Run

```bash
pip install -r requirements-app.txt          # UI deps (no Detectron2)
TOOTH_AI_DEMO=1 streamlit run app.py         # demo mode, any machine
streamlit run app.py                         # full mode: needs torch + detectron2 + weights
python3 -m pytest tests/                     # unit tests (no GPU)
```

The Windows venv in `venv/` (Python 3.10, Streamlit 1.52.2, detectron2 0.6,
torch 2.5.1+cu121) is the full-mode environment. On the Mac (M1, no CUDA) use demo mode with
`.venv/` (Python 3.10, `requirements-app.txt`). Run from the repo root: model paths are relative.

Notes for working on the UI:
- The repo lives on an external volume the desktop preview runner can't read: start Streamlit
  from the shell (as above) and attach the browser pane to `http://localhost:8501`
  (`.claude/launch.json`). Restart the server after editing `tooth_ai/` modules or CSS: imported
  modules and the CSS are cached per process.
- `st.html` strips inline `<svg>`: use `components.svg_image()` (data-URI `<img>`).
- The viewer uses Plotly `dragmode="zoom"`: in pan mode plotly.js never emits `plotly_selected`,
  so click-to-select silently stops working.
- Demo samples: `tools/make_demo_fixtures.py` (cached model detections, neutral names).
  Brand assets: `tools/build_brand.py` (regenerates every SVG/PNG in `assets/brand/`).
