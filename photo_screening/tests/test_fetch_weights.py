"""fetch_weights --import-zip: a hostile artifact zip must not write outside weights/."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_weights.py"


def load(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOTO_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("PHOTO_SAMPLES_DIR", str(tmp_path / "samples"))
    spec = importlib.util.spec_from_file_location("fetch_weights_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_zip(path: Path, extra: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("caries_aku_artifacts/metrics.json", json.dumps({"model_id": "caries_aku"}))
        z.writestr("caries_aku_artifacts/best.pt", b"weights")
        for name, data in extra.items():
            z.writestr(f"caries_aku_artifacts/{name}", data)
    return path


@pytest.mark.parametrize("name", ["plots/../../escaped.txt", "samples/../../../escaped.txt"])
def test_zip_slip_is_rejected(monkeypatch, tmp_path, name):
    mod = load(monkeypatch, tmp_path)
    (tmp_path / "weights").mkdir()
    z = make_zip(tmp_path / "bad.zip", {name: b"x"})
    with pytest.raises(SystemExit, match="unsafe path"):
        mod.import_zip(z)
    assert not list(tmp_path.rglob("escaped.txt"))
    assert not (tmp_path / "weights" / ".caries_aku.incoming").exists()


def test_a_clean_zip_imports(monkeypatch, tmp_path):
    mod = load(monkeypatch, tmp_path)
    (tmp_path / "weights").mkdir()
    z = make_zip(tmp_path / "ok.zip", {"plots/pr.png": b"png"})
    mod.import_zip(z)
    assert (tmp_path / "weights" / "caries_aku" / "best.pt").read_bytes() == b"weights"
    assert (tmp_path / "weights" / "caries_aku" / "plots" / "pr.png").exists()
