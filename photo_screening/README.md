# ToothAI Photo Screening

Smartphone photos of the teeth go in; a screening report comes out: quality-checked, annotated
photos, possible findings, a triage level, a screening index and a PDF. When something concerning
is seen, it recommends a panoramic X-ray (OPG), which the main ToothAI app analyses.

> Research prototype. Not a medical device and not for diagnostic use. This is a screening aid, not
> a diagnosis: photos cannot show between teeth, below the gum line or the bone. All findings
> require review by a qualified dentist.

How it decides, what each model is, and its measured accuracy are in [MODEL_CARD.md](MODEL_CARD.md)
and [EVAL.md](EVAL.md).

## Run it

From the repository root, with Python 3.11:

```bash
uv venv -p 3.11 ~/.venvs/photo && VIRTUAL_ENV=~/.venvs/photo uv pip install -r photo_screening/requirements.txt
~/.venvs/photo/bin/python photo_screening/scripts/fetch_weights.py
~/.venvs/photo/bin/uvicorn photo_screening.app.main:app --port 8001
```

Then start the ToothAI app (`streamlit run app.py`) and open **Photo screening** in the top bar.
The page talks to the API at `PHOTO_API_URL` (default `http://localhost:8001`). API docs are at
`http://localhost:8001/docs`.

Settings are environment variables with the `PHOTO_` prefix, or a `photo_screening/.env` file. See
[.env.example](.env.example).

### Docker (CPU)

```bash
docker compose up --build photo-api          # from the repository root; image about 2.4 GB
```

The weights are mounted read-only from `photo_screening/weights/`, never baked into the image. Measured on
an M1 with Colima (2 CPUs, 4 GB): about 10 s per photo, against about 1 s natively on MPS, so the native
path is the one for demos. Give the Docker VM at least 4 GB: with 2 GB the kernel kills the service
during the first analysis. Colima only shares your home folder by default; if the repository is on an
external drive, add the weights folder to `mounts` in `~/.colima/default/colima.yaml`.

## Models and weights

| Model | Finds | How to get it |
|---|---|---|
| `alphadent` (required) | Wear, filling, crown, caries | `scripts/fetch_weights.py` (GitHub release, Apache-2.0) |
| `caries_aku` (optional, preferred for caries) | Caries in phone photos | Notebook A, then `fetch_weights.py --import-zip caries_aku_artifacts.zip` |
| `teeth_dentalai` (optional) | Teeth, cavity, crack | Notebook B, then `fetch_weights.py --import-zip teeth_dentalai_artifacts.zip` |

The service starts with whatever is present: missing optional models are reported at
`GET /v1/photo-screening/models`, never a crash. Weight versions (SHA-256) appear in every report
and in the PDF footer. After importing new weights, run `scripts/calibrate.py` and restart.

## Train (Kaggle, free T4)

`notebooks/train_caries_aku.ipynb` and `notebooks/train_teeth_dentalai.ipynb` run unattended
("Run all"). Kernel metadata for `kaggle kernels push` is in `notebooks/kaggle/`.

## Evaluate

```bash
python photo_screening/scripts/evaluate.py --aku "<path>/Benchmarking Dataset" --dentalai "<path>/dentalai-2"
```

The AKU zip downloads slowly from Zenodo outside Kaggle, so `scripts/build_eval_kernel.py` builds a
Kaggle kernel that runs the same evaluation there. Results go to `eval_results.json`, summarised in
`EVAL.md`.

## Test

```bash
cd photo_screening && ~/.venvs/photo/bin/python -m pytest && ~/.venvs/photo/bin/ruff check .
```

The tests need no weights and no network.

## Privacy

- Photos are processed in memory and never stored.
- The service has no accounts and no database.
- Logs carry no image data or filenames.
- AI observations run locally unless you point them at a remote provider, in which case the page
  says where photos are sent.
