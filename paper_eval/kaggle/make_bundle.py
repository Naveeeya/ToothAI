#!/usr/bin/env python3
"""Task F: build the Kaggle upload bundle.

Standard library only -- runs on the Mac.

Contents: the 716 test images, their annotation JSON, the three checkpoints
needed (two deployed + the cascade, for the COCO comparison), their configs,
and paper_eval/ itself. Nothing else -- in particular no training data.

    python3 paper_eval/kaggle/make_bundle.py            # build
    python3 paper_eval/kaggle/make_bundle.py --dry-run  # size only
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUNDLE = os.path.join(ROOT, "paper_eval", "kaggle", "bundle")
CODE_BUNDLE = os.path.join(ROOT, "paper_eval", "kaggle", "bundle_code")

SPLIT_DIR = os.path.join(ROOT, "data", "final-di-stratified", "test")
MODELS = ["output/resnet50_9class_20k",
          "output/rtx4060_48k",
          "output/resnext101_cascade_60k"]
CODE = ["compare_ap.py", "label_quality.py", "check_sparse_bias.py",
        "evalkit.py", "pipeline.py", "run_eval.py", "export_panels.py",
        "export_review.py", "README.md"]


def git_state():
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
    try:
        d = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                    stderr=subprocess.DEVNULL).decode().strip()
        return f"{head}{'+dirty' if d else ''}"
    except subprocess.CalledProcessError:
        return f"{head}+dirty-unknown"


def write_provenance(dest_dir):
    """Stamp the commit into the bundle: Kaggle has no git repo, so without
    this every results CSV would read '# git: unknown'."""
    os.makedirs(dest_dir, exist_ok=True)
    rec = {"git": git_state(),
           "built_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .isoformat(timespec="seconds")}
    with open(os.path.join(dest_dir, "BUILD_PROVENANCE.json"), "w") as fh:
        json.dump(rec, fh, indent=2)
    return rec


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024.0
    return f"{n:.1f} TB"


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build_code_bundle():
    """Code-only dataset: paper_eval/ and its Task B outputs, nothing else."""
    if os.path.exists(CODE_BUNDLE):
        shutil.rmtree(CODE_BUNDLE)
    total = n = 0
    for f in CODE + ["check_imports.py", "AUDIT.md"]:
        src = os.path.join(ROOT, "paper_eval", f)
        if not os.path.exists(src):
            continue
        dst = os.path.join(CODE_BUNDLE, "paper_eval", f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst); total += os.path.getsize(src); n += 1
    for sub in ("results/class_to_position.csv", "results/label_quality.csv",
                "results/instance_counts.csv", "manifests/fdi_evaluable.txt"):
        src = os.path.join(ROOT, "paper_eval", sub)
        if not os.path.exists(src):
            continue
        dst = os.path.join(CODE_BUNDLE, "paper_eval", sub)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst); total += os.path.getsize(src); n += 1

    rec = write_provenance(os.path.join(CODE_BUNDLE, "paper_eval"))
    print(f"  stamped git: {rec['git']}")

    meta = {"title": "Tooth-AI paper eval code",
            "id": "alphaleporus/tooth-ai-paper-eval-code",
            "licenses": [{"name": "other"}], "isPrivate": True}
    json.dump(meta, open(os.path.join(CODE_BUNDLE, "dataset-metadata.json"), "w"),
              indent=2)

    # Sweep AFTER every write: the exFAT volume creates a "._<name>" sidecar
    # for each file as it lands, including dataset-metadata.json itself.
    swept = 0
    for root, dirs, files in os.walk(CODE_BUNDLE):
        for f in files:
            if f.startswith("._"):
                os.remove(os.path.join(root, f)); swept += 1
    print(f"code bundle at {CODE_BUNDLE}")
    print(f"  {n} files, {human(total)}" + (f", swept {swept} sidecars" if swept else ""))
    print("\n  --dir-mode zip is REQUIRED or the paper_eval/ folder is skipped.")
    print("  first time :  kaggle datasets create  -p paper_eval/kaggle/bundle_code --dir-mode zip -t")
    print("  thereafter :  kaggle datasets version -p paper_eval/kaggle/bundle_code --dir-mode zip -t -m '<msg>'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--code-only", action="store_true",
                    help="build only the ~200 KB paper_eval/ bundle. The data "
                         "and checkpoints never change, so iterating on the "
                         "harness should not re-upload 1.7 GB each time.")
    args = ap.parse_args()

    if args.code_only:
        build_code_bundle()
        return

    coco = os.path.join(SPLIT_DIR, "_annotations.coco.json")
    if not os.path.exists(coco):
        sys.exit(f"missing {coco}")
    imgs = [i["file_name"] for i in json.load(open(coco))["images"]]

    plan, total = [], 0
    plan.append((coco, "data/test/_annotations.coco.json"))
    for fn in imgs:
        # Skip macOS AppleDouble sidecars. This repo lives on an exFAT volume,
        # which materialises a "._<name>" file next to anything carrying
        # extended attributes. They are pure noise in the bundle, and "._x.jpg"
        # matches a *.jpg glob, so they can inflate an image count downstream.
        if fn.startswith("._"):
            continue
        src = os.path.join(SPLIT_DIR, fn)
        if os.path.exists(src):
            plan.append((src, f"data/test/{fn}"))
    for md in MODELS:
        for f in ("model_final.pth", "config.yaml"):
            src = os.path.join(ROOT, md, f)
            if os.path.exists(src):
                plan.append((src, f"{md}/{f}"))
            elif f == "model_final.pth":
                print(f"  WARNING: {md}/{f} missing -- its COCO eval will be skipped")
    for f in CODE:
        src = os.path.join(ROOT, "paper_eval", f)
        if os.path.exists(src):
            plan.append((src, f"paper_eval/{f}"))
    for sub in ("results/class_to_position.csv", "results/label_quality.csv",
                "results/instance_counts.csv", "manifests/fdi_evaluable.txt"):
        src = os.path.join(ROOT, "paper_eval", sub)
        if os.path.exists(src):
            plan.append((src, f"paper_eval/{sub}"))

    by_group = {}
    for src, dst in plan:
        sz = os.path.getsize(src)
        total += sz
        by_group[dst.split("/")[0]] = by_group.get(dst.split("/")[0], 0) + sz

    print(f"{'group':<16} {'size':>12}")
    for g, s in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print(f"{g:<16} {human(s):>12}")
    print(f"{'TOTAL':<16} {human(total):>12}   ({len(plan)} files)")
    if total > 100 * 1024 ** 3:
        print("  !! over Kaggle's private-dataset limit")

    if args.dry_run:
        return

    if os.path.exists(BUNDLE):
        shutil.rmtree(BUNDLE)
    for src, dst in plan:
        d = os.path.join(BUNDLE, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(src, d)

    # copy2 preserves xattrs, which makes the exFAT volume create a fresh
    # "._<name>" sidecar beside every file it just wrote. Sweep them.
    swept = 0
    for root, dirs, files in os.walk(BUNDLE):
        for f in files:
            if f.startswith("._"):
                os.remove(os.path.join(root, f))
                swept += 1
    if swept:
        print(f"  swept {swept} AppleDouble sidecars from the bundle")

    write_provenance(os.path.join(BUNDLE, "paper_eval"))
    manifest = {"files": len(plan), "bytes": total, "checkpoints": {}}
    for md in MODELS:
        p = os.path.join(BUNDLE, md, "model_final.pth")
        if os.path.exists(p):
            manifest["checkpoints"][md] = sha256(p)
    json.dump(manifest, open(os.path.join(BUNDLE, "BUNDLE_MANIFEST.json"), "w"),
              indent=2)
    print(f"\nbundle at {BUNDLE}")
    for md, h in manifest["checkpoints"].items():
        print(f"  {md:<34} {h[:16]}...")
    print("\nNOTE: these are medical images under a CC BY-NC dataset licence.")
    print("The Kaggle dataset MUST be created private. Never public.")


if __name__ == "__main__":
    main()
