#!/usr/bin/env python3
"""Extract and compare per-class AP from Detectron2 metrics.json files.

No dependencies beyond the standard library. Runs anywhere python3 does.

Usage:
    python3 compare_ap.py output/resnet50_9class_20k output/rtx4060_48k

Pass the OUTPUT DIRECTORIES (not the json files). It reads metrics.json and,
if present, config.yaml, so it can tell you which dataset each model was
evaluated on -- which decides whether the comparison is valid at all.
"""
import json
import os
import re
import sys

TOOTH_POSITION = re.compile(r"^\d+$")          # 41-class schema: "1".."32"
GENERIC_TOOTH = {"tooth"}


def read_eval_records(path):
    """metrics.json is JSONL. Return every line that carries COCO eval results."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(k.startswith(("segm/", "bbox/")) for k in rec):
                records.append(rec)
    return records


def split_classes(rec, task="segm"):
    """Bucket per-class AP keys into tooth positions, generic tooth, and other."""
    prefix = f"{task}/AP-"
    positions, generic, other = {}, {}, {}
    for key, value in rec.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if TOOTH_POSITION.match(name):
            positions[name] = value
        elif name.lower() in GENERIC_TOOTH:
            generic[name] = value
        else:
            other[name] = value
    return positions, generic, other


def finite(values):
    """COCO writes NaN for classes with no test instances. Those must not be
    averaged in -- a NaN class means no evidence, not zero performance."""
    return [v for v in values if isinstance(v, (int, float)) and v == v]


class DatasetReadError(Exception):
    """config.yaml exists but DATASETS.TEST could not be read from it."""


def find_eval_dataset(outdir):
    """Return DATASETS.TEST as a list of dataset names.

    Returns None when there is no config.yaml at all. Raises DatasetReadError
    when a config exists but its DATASETS.TEST cannot be parsed -- an
    unreadable provenance must stop the comparison, not silently pass it.

    Handles both YAML spellings Detectron2 emits:

        DATASETS:              DATASETS:
          TEST: []               TEST:
                                 - tooth_val

    A plain substring search for "TEST:" is wrong here: the same block holds
    PROPOSAL_FILES_TEST and PRECOMPUTED_PROPOSAL_TOPK_TEST, and matching those
    reports an empty list for every model, which reads as agreement.
    """
    cfg = os.path.join(outdir, "config.yaml")
    if not os.path.exists(cfg):
        return None
    lines = open(cfg, errors="replace").read().splitlines()

    # 1. locate the top-level DATASETS: block
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^DATASETS:\s*$", ln):
            start = i + 1
            break
    if start is None:
        raise DatasetReadError(f"{cfg}: no top-level 'DATASETS:' block")

    body = []
    for ln in lines[start:]:
        if ln.strip() and not ln[:1].isspace():      # next top-level key
            break
        body.append(ln)

    # 2. find the TEST key inside it -- exact key, not a suffix match
    for j, ln in enumerate(body):
        m = re.match(r"^(\s+)TEST:\s*(.*)$", ln)
        if not m:
            continue
        indent, inline = m.group(1), m.group(2).strip()
        if inline:                                    # flow list or scalar
            if inline.startswith("[") and inline.endswith("]"):
                inner = inline[1:-1].strip()
                return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
            return [inline.strip("'\"")]
        # block sequence: '- name' lines at >= the key's indent
        items = []
        for nxt in body[j + 1:]:
            if not nxt.strip():
                continue
            lead = len(nxt) - len(nxt.lstrip())
            s = nxt.strip()
            if s.startswith("- ") and lead >= len(indent):
                items.append(s[2:].strip().strip("'\""))
                continue
            break
        return items
    raise DatasetReadError(f"{cfg}: 'DATASETS:' block has no 'TEST:' key")


def report(outdir, task="segm"):
    mpath = os.path.join(outdir, "metrics.json")
    if not os.path.exists(mpath):
        print(f"\n=== {outdir}  [{task}] ===")
        print(f"  !! NO metrics.json in {outdir}")
        return {"dir": outdir, "task": task, "status": "no_metrics"}

    records = read_eval_records(mpath)
    if not records:
        n_lines = sum(1 for _ in open(mpath))
        print(f"\n=== {outdir}  [{task}] ===")
        print(f"  !! NO EVALUATION RECORDS in {mpath}")
        print(f"     ({n_lines} metrics lines, none carrying a segm/ or bbox/ key)")
        print("     This model was never evaluated. It cannot enter a comparison.")
        return {"dir": outdir, "task": task, "status": "no_eval_records",
                "metrics_lines": n_lines}

    rec = records[-1]
    positions, generic, other = split_classes(rec, task)
    pos_vals = finite(positions.values())

    try:
        eval_dataset = find_eval_dataset(outdir)
        dataset_error = None
    except DatasetReadError as exc:
        eval_dataset, dataset_error = None, str(exc)

    summary = {
        "status": "ok",
        "task": task,
        "dataset_error": dataset_error,
        "dir": outdir,
        "eval_records": len(records),
        "iteration": rec.get("iteration"),
        "eval_dataset": eval_dataset,
        f"{task}/AP": rec.get(f"{task}/AP"),
        f"{task}/AP50": rec.get(f"{task}/AP50"),
        f"{task}/AP75": rec.get(f"{task}/AP75"),
        "n_tooth_position_classes": len(positions),
        "n_tooth_position_with_data": len(pos_vals),
        "mean_AP_tooth_positions": (sum(pos_vals) / len(pos_vals)) if pos_vals else None,
        "generic_tooth_AP": next(iter(generic.values()), None),
        "per_position_AP": positions,
        "other_class_AP": other,
    }

    print(f"\n=== {outdir}  [{task}] ===")
    print(f"  iteration at last eval : {summary['iteration']}")
    if dataset_error:
        print(f"  eval dataset (config)  : UNKNOWN -- {dataset_error}")
    elif eval_dataset is None:
        print("  eval dataset (config)  : UNKNOWN -- no config.yaml")
    else:
        print(f"  eval dataset (config)  : {', '.join(eval_dataset) or '(empty list)'}")
    print(f"  overall {task}/AP        : {summary[f'{task}/AP']}")
    print(f"  AP50 / AP75            : {summary[f'{task}/AP50']} / {summary[f'{task}/AP75']}")
    if generic:
        for k, v in generic.items():
            print(f"  generic tooth AP-{k}   : {v}")
    if positions:
        print(f"  tooth-position classes : {len(positions)} "
              f"({len(pos_vals)} with test instances, "
              f"{len(positions) - len(pos_vals)} NaN)")
        print(f"  mean AP over positions : {summary['mean_AP_tooth_positions']}")
        ranked = sorted(((v, k) for k, v in positions.items() if v == v), reverse=True)
        if ranked:
            top = ", ".join(f"{k}={v:.2f}" for v, k in ranked[:5])
            bot = ", ".join(f"{k}={v:.2f}" for v, k in ranked[-5:])
            print(f"    best  : {top}")
            print(f"    worst : {bot}")
    if other:
        print("  other classes:")
        for k, v in sorted(other.items()):
            flag = "  (no test instances)" if v != v else ""
            print(f"    {k:<28} {v}{flag}")
    return summary


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        sys.exit(1)

    out = {}
    for task in ("segm", "bbox"):
        for d in dirs:
            s = report(d.rstrip("/"), task)
            if s:
                out[f"{d.rstrip('/')}::{task}"] = s

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    failures = []

    # 1. a model that was never evaluated cannot be compared, at all.
    never_evaluated = sorted({v["dir"] for v in out.values()
                              if v.get("status") == "no_eval_records"})
    no_metrics = sorted({v["dir"] for v in out.values()
                         if v.get("status") == "no_metrics"})
    for d in never_evaluated:
        line = f"NO EVAL RECORDS: {d} was never evaluated -- no AP exists for it."
        print(line)
        failures.append(line)
    for d in no_metrics:
        line = f"NO METRICS FILE: {d} has no metrics.json."
        print(line)
        failures.append(line)

    ok = [v for v in out.values() if v.get("status") == "ok"]

    # 2. unreadable provenance is a failure, never an all-clear.
    unreadable = sorted({v["dir"] for v in ok
                         if v.get("dataset_error") or v.get("eval_dataset") is None})
    for d in unreadable:
        err = next((v.get("dataset_error") for v in ok if v["dir"] == d
                    and v.get("dataset_error")), "no config.yaml")
        line = f"UNKNOWN eval dataset for {d}: {err}"
        print(line)
        failures.append(line)

    # 3. differing datasets invalidate the comparison.
    datasets = {tuple(v["eval_dataset"]) for v in ok if v.get("eval_dataset")}
    if len(datasets) > 1:
        print("DIFFERENT eval datasets -- the comparison is not valid:")
        seen = set()
        for v in sorted(ok, key=lambda x: x["dir"]):
            if v.get("eval_dataset") is not None and v["dir"] not in seen:
                seen.add(v["dir"])
                print(f"    {v['dir']:<34} {', '.join(v['eval_dataset'])}")
        print("  NOTE: a shared dataset NAME is not proof of shared data --")
        print("  the same name may be registered to different directories in")
        print("  different runs. Confirm against the training log.")
        failures.append("models evaluated on different datasets")
    elif len(datasets) == 1:
        only = ", ".join(next(iter(datasets))) or "(empty list)"
        print(f"All evaluated models used dataset: {only}")
        print("  NOTE: a shared dataset NAME is not proof of shared data --")
        print("  confirm the directory against the training log.")

    with open("ap_summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote ap_summary.json  <- send me this file")

    if failures:
        print("\nCOMPARISON INVALID -- " + str(len(failures)) + " blocking issue(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(2)
    print("\nComparison is valid.")
    sys.exit(0)


if __name__ == "__main__":
    main()
