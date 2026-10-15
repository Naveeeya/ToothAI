# ToothAI — brand and UX specification

Source of truth for the product's look, voice and interaction design. Implementation
lives in `tooth_ai/ui/` (tokens, CSS, components) and `assets/brand/` (marks, usage guide).
Researched October 2026.

> **Identity superseded.** The name, logo, colour and type sections below describe the first
> "Tooth-AI" identity. They are replaced by **ToothAI Brand Guidelines v1.0** (Oct 2026): see
> `assets/brand/BRAND.md` and `tooth_ai/ui/theme.py`. The UX, voice and report sections still apply.

---

## 1. Competitive landscape

| Product | Company | Panoramic capability | Regulatory | Brand colour |
|---|---|---|---|---|
| Overjet | Overjet Inc. (Boston) | caries, calculus, bone level, PARL — bitewing/PA focus | 11 FDA 510(k) | deep violet `#3D2276` |
| Second Opinion | Pearl (LA) | up to 18 findings; panoramic clearance Dec 2025 | FDA, EU MDR | cyan `#23BFEE` on pastels |
| Videa AI | VideaHealth | 30+ conditions incl. panoramic | FDA K251002 | chartreuse `#DFDA02` |
| Diagnocat | DGNCT | OPG/IOPA/CBCT, 65+ conditions, colour dental chart | EU MDR; FDA (CBCT PAL) | violet `#5241CC` + red |
| Denti.AI | Denti.AI (Toronto) | Detect + Auto-Chart (numbering) on panoramic | FDA (2022, 2023) | mint `#00CD90` |
| Align X-ray Insights | Align (ex-dentalXrai Pro) | caries, PAL, bone loss, restorations | CE/UK/CA; US pending | cyan `#009ACE` (legacy) |
| Allisone | Allisone (France) | structures + pathology on pano/PA | CE IIa | navy + purple→sky gradient |
| Velmeni | Velmeni (Sunnyvale) | cavities, restorations, implants | FDA 510(k) | orange `#F57326` |
| scanO | scanO (**Pune**) | kiosk + intraoral/pano/bitewing, 40+ conditions | CDSCO Class B, CE MDR | violet `#6A3DF0` |

Others: CranioCatch (academic, purple), Promaton (B2B), Planmeca Romexis 7 and
Carestream (via Pearl), Dexis DTX Studio, Dürr VistaSoft AI; Indian: AI Dent, epikdoc.

Benchmarks to position against: DENTEX (MICCAI 2023, quadrant–tooth–diagnosis on OPG),
STS 2024 (semi-supervised per-tooth FDI segmentation), OdontoAI/O²PR (4,000 OPGs, numbering
leaderboard), Tufts Dental Database (1,000 OPGs with eye-tracking), ToothFairy (CBCT).

### What dentists now expect

1. The radiograph is the main surface, on a **dark** viewer, with zoom/pan.
2. Overlays colour-coded **by finding type**, with per-layer toggles and opacity.
3. A status-coloured **odontogram** linked to the image (de-facto convention: healthy
   white, treated, pathology red, missing ×, uncertain yellow).
4. Notation setting (FDI for India/EU, Universal for the US).
5. Uncertainty shown, not hidden; the AI framed as a second read.
6. A one-page PDF: annotated image, chart, per-tooth table.

### Where Tooth-AI differs

No competitor explains **why** a tooth got its number. Tooth-AI's numbering is geometric,
so it can show its reasoning:

1. **"Why this number?"** per tooth — jaw (above/below the split line), rank along the arch,
   gap skips, conflict reassignments.
2. **Geometry layer** — jaw split line, midline, fitted arch curves, quadrant colouring.
3. **Numbering reliability flag** per case, driven by arch completeness (the measured
   failure mode: 77 % FDI on complete arches vs 35 % on partial ones, as deployed).
4. **Honest performance page** — measured numbers with n and 95 % CIs.
5. **FDI by default**, built for Indian dental colleges and clinics.

### Territory to avoid

- Colours: violet/purple (Overjet, Diagnocat, CranioCatch, scanO, Allisone), cyan (Pearl,
  dentalXrai), mint (Denti.AI), chartreuse (Videa), orange (Velmeni).
- Type: Inter, DM Sans, Space Grotesk, Plus Jakarta Sans.
- Mascots and playful patient copy.
- Red, yellow and purple are reserved for chart status meanings — never brand colours.
- Name risk: "Tooth-AI" sits close to "Toothsi" (Indian aligner brand). Fine for academic
  use; run a trademark search before any commercial launch.

---

## 2. Brand

| | |
|---|---|
| Name | **Tooth-AI** |
| Full title | Tooth-AI: Integrated Deep Learning and Geometric Reasoning System for Detection, Segmentation, Numbering and Anatomical Mapping of Teeth in Panoramic Dental Radiographs |
| Tagline | **Every tooth, found and explained.** |
| Descriptor | Detection · Segmentation · Numbering · Anatomical mapping — for panoramic radiographs |
| Voice | evidence-led, calm, clinical; sentence case; no emoji; never overclaim |

The full title appears on About, How it works, the PDF methodology section and the paper.

### Logo — "Arch Nodes" (primary)

Deep learning finds the teeth (nodes); geometry connects and orders them (the curve).

- 64 × 64 viewBox, rounded-square tile, radius 14, fill ink-900.
- Smile-curve arch `y = 44 − 0.08·(x − 32)²`, x ∈ [14, 50], stroke 1.5 slate-300 @ 60 %.
- Seven nodes evenly spaced by arc length on the curve, r = 3.2, fill bone-50.
- The centre node is a saffron tooth glyph (crown + two roots, ~9 × 10).
- Favicon (≤ 32 px): no curve, five nodes + the saffron tooth.

Alternate "Quadrant Tooth": a molar silhouette crossed by a thin FDI crosshair, one
quadrant saffron. Both are rendered in `assets/brand/logo-explorations.html`.

Wordmark: IBM Plex Sans SemiBold, tracking −1 %, hyphen in saffron.
UI motif: detection-style corner brackets mark selection and focus — never the logo.

### Colour tokens

| Token | Hex | Use |
|---|---|---|
| ink-900 | `#0B1B2B` | headings, header band, logo tile |
| ink-700 | `#1E3A56` | primary buttons, links (`primaryColor`) |
| ink-500 | `#3F5D7A` | secondary text, icons |
| slate-300 | `#C6D0DA` | borders, dividers |
| slate-100 | `#EEF2F5` | secondary background, sidebar |
| bone-50 | `#F7F5F0` | app background (enamel ivory) |
| white | `#FFFFFF` | cards |
| viewer-950 | `#0A0E13` | radiograph canvas (always dark) |
| saffron-500 | `#E8A317` | brand accent: logo, hyphen, focus ring, active nav. **Never inside the viewer or chart.** |

Semantic colours (viewer and chart only; always paired with an icon, label or pattern):

| Meaning | Hex |
|---|---|
| Caries | `#E5484D` |
| Retained root | `#C2410C` |
| Filling | `#3B82F6` |
| Crown | `#1D4ED8` |
| Root canal filling / obturation | `#0F9F8F` |
| Implant | `#64748B` |
| Prefabricated metal post | `#94A3B8` |
| Uncertain (dashed + "?") | `#CA8A04` |
| Not detected (dashed grey + ×) | `#9AA5B1` |
| Quadrant mode Q1 / Q2 / Q3 / Q4 | `#7FB3D5` / `#9BC79B` / `#E3B679` / `#C9A0C9` |

### Typography

Plus Jakarta Sans (headlines 800, UI 400/500), JetBrains Mono (FDI numbers, metrics). All SIL OFL.
TTFs in `static/fonts/` for the app and the PDF (Brand Guidelines v1.0).

### Partner logos

AICTE, Bharati Vidyapeeth, C-CAMP, CIBIP — once per surface (About page, Analyze footer,
PDF page-1 footer), equal height on white chips, original colours, never recoloured or
cropped. Caption: *Developed under the C-CAMP Inter-Institutional Biomedical Innovations
Programme (AICTE–C-CAMP IBIP).*

### Copy rules

- "Findings", not "anomalies". Modes: **Full chart** (teeth + findings), **Findings only**.
- "Caries (near metal, penalized)" → "Caries" + flag "Near metal — lower confidence".
- "Confidence" → "Mean detection score", in detail views only.
- Disclaimer on every surface: *Research prototype. Not a medical device and not for
  diagnostic use. All findings require review by a qualified dentist.*

---

## 3. Information architecture

Top navigation: **Analyze** (default) · **How it works** · **Performance** · **About**.
Header badge: "Research prototype".

### Analyze

```
┌ Sidebar: CASE ─────────┐┌ Viewer (dark) ──────────────────────────┐┌ Insight column ─────────────────┐
│ Upload OPG             ││ View [Overlay|Original|Compare]         ││ Reliability banner              │
│ or try a sample        ││ Layers (Masks)(Numbers)(Findings)(Geom.)││ KPIs: teeth · findings · U/L ·  │
│ Case ID (no names)     ││ Colour by [Finding|Quadrant]            ││       time                      │
│ Mode [Full chart|      ││ ┌─────────────────────────────────────┐ ││ Odontogram (FDI 18→28 / 48→38)  │
│       Findings only]   ││ │ zoom/pan, hover per tooth,          │ ││ Selected tooth: number, name,   │
│ Advanced detection ▸   ││ │ click = select                      │ ││  score, findings,               │
│ Notation [FDI|Univ.]   ││ └─────────────────────────────────────┘ ││  "Why this number?"             │
│ Model version          ││ Legend                                  ││ Tabs: Teeth | Findings |        │
│ Disclaimer             │└─────────────────────────────────────────┘│       Unassigned                │
└────────────────────────┘ Report: PDF · JSON · PNG · Markdown · partner strip · disclaimer         ┘
```

States:

- **Empty** — hero with mark, "Upload a panoramic radiograph (OPG)", formats, samples,
  three-step strip (Detect → Number → Map), privacy note.
- **Loading** — `st.status` lists each stage, collapses to "Done in 0.6 s".
- **Model missing** — setup card listing expected weight paths; never a half page.
- **Error** — friendly message, traceback in an expander.
- **Not an OPG** — warning when aspect ratio < 1.6; analysis still allowed.

Reliability banner:

| State | Rule | Copy |
|---|---|---|
| ok | both jaws ≥ 14 teeth, no gap skip, no conflict reassignment | Complete arches — numbering reliable |
| caution | a jaw ≤ 13 teeth, or a gap skip fired | Partial arch (lower jaw) — numbers may be shifted; review |
| alert | < 10 teeth in total | Low tooth count — check image quality |

Interactions: display controls never re-run inference (results cached in session state by
image hash + mode + thresholds). Selecting a tooth (viewer click or table row) syncs the
viewer brackets, the odontogram ring and the detail card. Downloads are named
`ToothAI_<case>_<YYYYMMDD-HHMM>.<ext>`.

### How it works

Pipeline diagram: OPG → contrast pass (CLAHE) + fusion → tooth instances (Mask R-CNN) and
findings model → geometric engine → anatomical validation → mapping and report. Plus
"Why geometry": ~29 % of checkable training images follow one of 116 inconsistent label
conventions; geometry learns none of them.

### Performance

Read live from `paper_eval/results/`. KPI tiles with n and CIs, complete-vs-partial chart,
test-set description, known limitations. No anomaly accuracy.

### About

Full title, tagline, institutions, partner strip, team, dataset attribution
(CC BY-NC 4.0), citation, disclaimer.

---

## 4. PDF report

A4, Plus Jakarta Sans and JetBrains Mono embedded.

- **Page 1** — ink-900 header band (white lockup, "Panoramic radiograph report"); case meta
  grid; reliability statement; annotated OPG at full width, aspect preserved; legend;
  odontogram; partner strip in the footer.
- **Page 2** — per-tooth table; unassigned findings; methodology (full title + summary);
  limitations; disclaimer.
- **Every page** — "Tooth-AI · Research prototype — not for diagnostic use · Page n of m".
