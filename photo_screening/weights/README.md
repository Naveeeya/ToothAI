# Photo Screening weights

Nothing in this folder is committed except this file. The service starts with whatever is
present and reports anything missing at `GET /v1/photo-screening/models`.

| Folder | Model | Required | How to get it | Licence |
|---|---|---|---|---|
| `alphadent/` | AlphaDent YOLOv8x-seg, 4 classes (abrasion, filling, crown, caries), 960 px | Yes | `python photo_screening/scripts/fetch_weights.py` (`--variant 9c640` or `9c960` for the alternatives) | Apache-2.0, [ZFTurbo/AlphaDent](https://github.com/ZFTurbo/AlphaDent) |
| `caries_aku/` | YOLOv8s caries detector trained on phone photos (Notebook A) | Optional; preferred for caries when present | Run `notebooks/train_caries_aku.ipynb`, then `fetch_weights.py --import-zip caries_aku_artifacts.zip` | Weights trained on CC BY 4.0 data (AKU, doi:10.5281/zenodo.14827784) |
| `teeth_dentalai/` | YOLOv8s-seg tooth, caries, cavity and crack (Notebook B) | Optional; enables tooth positions and the tooth-based mouth check | Run `notebooks/train_teeth_dentalai.ipynb`, then `fetch_weights.py --import-zip teeth_dentalai_artifacts.zip` | Weights trained on CC BY 4.0 data (DentalAI) |

`manifest.json` is written by `fetch_weights.py`. It records each file's size, source and the
SHA-256 computed at download or import time. The service reports those hashes as
`model_versions` in every report and in the PDF footer.

- **AlphaDent downloads:** the byte count is checked against the GitHub release API.
- **Notebook zips:** `best.pt` must match the SHA-256 the notebook recorded, or the import is
  rejected and nothing is changed.

Restart the service after adding weights. Models are loaded once at start-up.
