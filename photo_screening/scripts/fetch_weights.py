#!/usr/bin/env python3
"""Fetch and register model weights for ToothAI Photo Screening.

    python photo_screening/scripts/fetch_weights.py                       # AlphaDent 4-class 960 px
    python photo_screening/scripts/fetch_weights.py --variant 9c640       # smaller-input fallback
    python photo_screening/scripts/fetch_weights.py --import-zip caries_aku_artifacts.zip
    python photo_screening/scripts/fetch_weights.py --list

AlphaDent weights come from the ZFTurbo/AlphaDent GitHub release (Apache-2.0). The file size is
checked against the release API; the SHA-256 is computed on download (never compared with a
hard-coded value) and recorded in ``weights/manifest.json``, which the service reports as
``model_versions``. Notebook artifact zips are unpacked into ``weights/<model_id>/`` and their
``best.pt`` is checked against the SHA-256 the notebook recorded. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = Path(os.environ.get("PHOTO_WEIGHTS_DIR", PKG_DIR / "weights"))
SAMPLES_DIR = Path(os.environ.get("PHOTO_SAMPLES_DIR", PKG_DIR / "samples"))
MANIFEST = WEIGHTS_DIR / "manifest.json"

ALPHADENT_RELEASE_API = "https://api.github.com/repos/ZFTurbo/AlphaDent/releases/tags/v1.0"
ALPHADENT_VARIANTS = {
    "4c960": "yolov8x_AlphaDent_4_classes_960px.pt",
    "9c640": "yolov8x_AlphaDent_9_classes_640px.pt",
    "9c960": "yolov8x_AlphaDent_9_classes_960px.pt",
}
ARTIFACT_MODELS = {"caries_aku", "teeth_dentalai"}
ARTIFACT_FILES = ("best.pt", "metrics.json", "thresholds.json", "test_files.txt", "args.yaml", "results.csv")


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    """Read weights/manifest.json ({} if absent)."""
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def record(model_id: str, entry: dict) -> None:
    """Upsert one model's entry in the manifest."""
    m = load_manifest()
    m[model_id] = entry
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "tooth-ai"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download(url: str, dest: Path, expected_size: int, tries: int = 5) -> None:
    """Resumable download to ``dest`` that only succeeds when the byte count equals ``expected_size``."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > expected_size:                       # stale partial from a different file
            part.unlink()
            have = 0
        if have < expected_size:
            headers = {"User-Agent": "tooth-ai"}
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                    mode = "ab" if have and r.status == 206 else "wb"
                    done = have if mode == "ab" else 0
                    last = time.time()
                    with open(part, mode) as f:
                        while chunk := r.read(1 << 20):
                            f.write(chunk)
                            done += len(chunk)
                            if time.time() - last > 2:
                                print(f"\r  {done / 1e6:7.1f} / {expected_size / 1e6:.1f} MB", end="", flush=True)
                                last = time.time()
                print()
            except OSError as e:
                print(f"\n  attempt {attempt} failed: {e}", file=sys.stderr)
                time.sleep(3 * attempt)
                continue
        if part.stat().st_size == expected_size:
            part.replace(dest)
            return
        print(f"  size {part.stat().st_size} != expected {expected_size}, retrying", file=sys.stderr)
    raise SystemExit(f"download failed after {tries} attempts: {url}")


def fetch_alphadent(variant: str) -> Path:
    """Download one AlphaDent variant (size-checked), record its SHA-256, return its path."""
    name = ALPHADENT_VARIANTS[variant]
    release = _get_json(ALPHADENT_RELEASE_API)
    asset = next((a for a in release.get("assets", []) if a["name"] == name), None)
    if asset is None:
        raise SystemExit(f"{name} not found in AlphaDent release {release.get('tag_name')}")
    dest = WEIGHTS_DIR / "alphadent" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == asset["size"]:
        print(f"already present: {dest} ({asset['size']} bytes, size matches release)")
    else:
        print(f"downloading {name} ({asset['size'] / 1e6:.1f} MB) from {asset['browser_download_url']}")
        download(asset["browser_download_url"], dest, asset["size"])
    digest = sha256(dest)
    record("alphadent", {
        "file": str(dest.relative_to(WEIGHTS_DIR)), "variant": variant, "size": asset["size"], "sha256": digest,
        "source": asset["browser_download_url"], "release": release.get("tag_name"), "licence": "Apache-2.0",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    print(f"ok: {dest}\n    sha256 {digest} (recorded in {MANIFEST.relative_to(PKG_DIR)})")
    return dest


def _visible(name: str) -> bool:
    parts = Path(name).parts
    return not any(p.startswith("._") or p == "__MACOSX" or p == ".DS_Store" for p in parts)


def import_zip(zip_path: Path) -> str:
    """Unpack a notebook artifact zip into weights/<model_id>/ (and samples/), verify best.pt."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if _visible(n) and not n.endswith("/")]
        prefix = os.path.commonpath(names) if len(names) > 1 else ""
        prefix = prefix if prefix and not any(n == prefix for n in names) else ""

        def rel(n: str) -> str:
            return n[len(prefix):].lstrip("/") if prefix else n

        metrics_name = next((n for n in names if rel(n) == "metrics.json"), None)
        if metrics_name is None:
            raise SystemExit("not a notebook artifact zip (no metrics.json at its top level)")
        metrics = json.loads(z.read(metrics_name))
        model_id = metrics.get("model_id")
        if model_id not in ARTIFACT_MODELS:
            raise SystemExit(f"unknown model_id {model_id!r} in metrics.json")
        # Stage everything, verify best.pt, then swap in: a rejected zip leaves no trace.
        out = WEIGHTS_DIR / model_id
        stage = WEIGHTS_DIR / f".{model_id}.incoming"
        shutil.rmtree(stage, ignore_errors=True)
        stage_root = stage.resolve()
        copied = 0
        for n in names:
            r = rel(n)
            if r in ARTIFACT_FILES or r.startswith("plots/") or (r.startswith("samples/") and model_id == "caries_aku"):
                # Zip slip: "plots/../../x" passes the prefix check, so confine the resolved target.
                target = (stage / r).resolve()
                if Path(r).is_absolute() or ".." in Path(r).parts or not target.is_relative_to(stage_root):
                    shutil.rmtree(stage, ignore_errors=True)
                    raise SystemExit(f"unsafe path in artifact zip: {n!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                copied += 1
    staged_best = stage / "best.pt"
    expected = metrics.get("best_pt_sha256")
    digest = sha256(staged_best) if staged_best.exists() else None
    if digest is None or (expected and expected != digest):
        shutil.rmtree(stage, ignore_errors=True)
        raise SystemExit("artifact zip has no best.pt" if digest is None else
                         f"best.pt SHA-256 {digest} does not match the notebook's record {expected}")
    if (stage / "samples").exists():
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        for p in [q for q in (stage / "samples").iterdir() if _visible(q.name)]:
            shutil.move(str(p), SAMPLES_DIR / p.name)
        shutil.rmtree(stage / "samples")
    shutil.rmtree(out, ignore_errors=True)
    stage.replace(out)
    best = out / "best.pt"
    record(model_id, {
        "file": str(best.relative_to(WEIGHTS_DIR)), "size": best.stat().st_size, "sha256": digest,
        "source": zip_path.name, "arch": metrics.get("arch"), "licence": metrics.get("dataset", {}).get("licence"),
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    print(f"ok: {model_id} -> {out} ({copied} files); best.pt sha256 {digest}"
          + (" (matches notebook)" if expected else ""))
    return model_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(ALPHADENT_VARIANTS), default="4c960",
                    help="AlphaDent weights to download (default 4c960: 4 classes, 960 px)")
    ap.add_argument("--import-zip", type=Path, metavar="ZIP", help="register a notebook artifact zip instead")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    args = ap.parse_args()
    if args.list:
        print(json.dumps(load_manifest(), indent=2))
    elif args.import_zip:
        import_zip(args.import_zip)
    else:
        fetch_alphadent(args.variant)


if __name__ == "__main__":
    main()
