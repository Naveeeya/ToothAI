# Sample photos

The "Try sample photos" button in the app uses the images in this folder. It's hidden when the
folder has no images.

Notebook A copies five images from the AKU **Benchmarking Dataset test split**, plus an
`ATTRIBUTION.md`, into its artifact zip. `fetch_weights.py --import-zip caries_aku_artifacts.zip`
puts them here. They are licensed CC BY 4.0 (doi:10.5281/zenodo.14827784), and the attribution
file must stay with them.

The images are git-ignored, so a fresh clone has none until the zip is imported.
