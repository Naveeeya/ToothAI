# Running the evaluation on Kaggle

Everything here needs a GPU. The Mac this repo lives on is an Apple M1 with no
CUDA and no torch/Detectron2, so nothing in Task D or E can run locally.

## 0. Check before you push

```bash
python3 paper_eval/check_imports.py
```

The GPU half cannot be imported on this Mac (no torch, Detectron2, cv2 or
numpy), so a missing module-level import is invisible locally and only shows
up minutes into a paid GPU session. This walks the AST instead. It is how the
`cv2` import missing from `pipeline.py` was found — after it had already cost
one Kaggle run.

## 1. Build the bundles (on the Mac)

There are **two** datasets on purpose:

| dataset | size | contents | changes |
|---|---|---|---|
| `tooth-ai-paper-eval` | ~2 GB | `data/test/`, `output/<model>/` | never |
| `tooth-ai-paper-eval-code` | ~200 KB | `paper_eval/` | every fix |

Splitting them makes a harness fix a **2-second** upload instead of 14 minutes.

```bash
python3 paper_eval/kaggle/make_bundle.py --dry-run     # size check
python3 paper_eval/kaggle/make_bundle.py               # the big one, once
python3 paper_eval/kaggle/make_bundle.py --code-only   # the small one
```

Produces `paper_eval/kaggle/bundle/` — about **1.7 GB**, 735 files:

| group | contents |
|---|---|
| `data/test/` | the 716 test images + `_annotations.coco.json` |
| `output/` | three checkpoints + configs (R50 9-class, R50 41-class, X101 Cascade) |
| `paper_eval/` | the harness, plus the Task B CSVs and the evaluable manifest |

No training data is included.

## 2. Create the Kaggle Dataset — **private**

```bash
# big dataset, once
kaggle datasets create -p paper_eval/kaggle/bundle --dir-mode zip -t

# code dataset, once
kaggle datasets create -p paper_eval/kaggle/bundle_code --dir-mode zip -t

# code dataset, every time the harness changes
python3 paper_eval/kaggle/make_bundle.py --code-only
kaggle datasets version -p paper_eval/kaggle/bundle_code --dir-mode zip -t -m "what changed"
```

`--dir-mode zip` is **required**. Without it the CLI prints
`Skipping folder: paper_eval` and uploads nothing but the metadata.

`-t` / `--keep-tabular` matters: without it Kaggle converts tabular files to
CSV, which would rewrite the COCO annotation JSON and the result CSVs. They
must stay byte-exact.

Privacy is the **default** for `datasets create` in CLI 2.2.x — `--public` is
opt-in. Do not pass it. `dataset-metadata.json` also sets `"isPrivate": true`
and `make_bundle.py` asserts it.

Processing is asynchronous: after the upload returns, Kaggle unpacks and
indexes the dataset, and `kaggle datasets files <slug>` returns 403 until that
finishes. For a ~2 GB bundle allow 10-30 minutes before the kernel can attach
it.

The source data is CC BY-NC and these are **medical radiographs**. The dataset
must be private, permanently. Do not make it public and do not share the slug
outside the project.

`dataset-metadata.json` must exist in the bundle directory before this runs;
`kaggle datasets init -p paper_eval/kaggle/bundle` creates a template. Set
`"isPrivate": true`.

## 3. Push the notebook

`kernel-metadata.json` next to `run_eval.ipynb` must carry:

```json
{
  "id": "<username>/tooth-ai-paper-eval",
  "title": "Tooth-AI paper eval",
  "code_file": "run_eval.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["<username>/tooth-ai-paper-eval",
                      "<username>/tooth-ai-paper-eval-code"],
  "competition_sources": [],
  "kernel_sources": []
}
```

`enable_internet` is required — Detectron2 is built from source at run time.

The kernel `id` must **not** equal a dataset slug in the same namespace, or
`SaveKernel` returns `409 Conflict`. Hence `tooth-ai-paper-eval-run`.

`machine_shape` does not exist in CLI 2.2.x. Accelerator is a push flag:
`--accelerator nvidiaTeslaT4`.

```bash
kaggle kernels push -p paper_eval/kaggle
kaggle kernels status <username>/tooth-ai-paper-eval
kaggle kernels output <username>/tooth-ai-paper-eval -p paper_eval/results
```

## 4. Order of operations inside the notebook

1. **GPU gate.** Cell 1 fails loudly if `nvidia-smi` shows nothing. The push API
   has been reported to allocate a different machine than requested without
   raising an error, so this is checked, not trusted.
2. **Pinned install.** torch `2.5.1+cu121` from the PyTorch cu121 index, then
   Detectron2 at commit `fd27788985af0f4ca800bca563acdb700bb890e2`. Each falls
   back only on failure, and every fallback is collected into `DEVIATIONS` and
   printed in a banner. If that banner appears, record it in the paper's
   reproducibility note — numbers may differ slightly.
3. **Smoke test** (`--smoke-test`, 5 images). Confirm the summary table is
   populated before committing to the full run.
4. Full run, sweep, COCO eval, panel candidates, zip.

## 5. Expected runtime (T4)

| step | images | rough time |
|---|---|---|
| smoke test | 5 | < 1 min |
| `--all --latency` | 716 × 4 configs | ~60–90 min |
| `--sweep` | 716 × 10 taus (fused once) | ~25–40 min |
| `--coco-eval` | 716 × 3 models | ~20–30 min |
| `--suggest-images` | 67 × 2 configs | ~5 min |

These are estimates from image count and model size, not measurements — the
first real run replaces them.

## 6. Bringing results back

```bash
kaggle kernels output <username>/tooth-ai-paper-eval -p paper_eval/results
```

Every CSV carries a comment header with the checkpoint SHA-256, the split, the
git state and the GPU name. Check the GPU line matches what you expect before
quoting any latency figure.
