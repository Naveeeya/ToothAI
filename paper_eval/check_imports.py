#!/usr/bin/env python3
"""Find names a module reads but never binds, without importing it.

The GPU half of paper_eval cannot be imported on this Mac -- no torch, no
Detectron2, no cv2, no numpy. That means a missing module-level import is
invisible locally and only surfaces on Kaggle, minutes into a paid GPU
session. This walks the AST instead and reports undefined names.

It caught exactly that: pipeline.py used cv2 in apply_clahe without importing
it, because app.py imports cv2 at module scope (app.py:15) and the verbatim
extraction did not carry module-level imports across.

    python3 paper_eval/check_imports.py            # all harness modules
    python3 paper_eval/check_imports.py a.py b.py  # specific files
"""
import ast
import builtins
import os
import sys

DEFAULT = ["pipeline.py", "evalkit.py", "run_eval.py", "export_panels.py",
           "export_review.py", "label_quality.py", "check_sparse_bias.py",
           "compare_ap.py", "decompose.py", "paired_deltas.py",
           "check_contrast_confound.py", "check_label_exposure.py",
           "check_paper_claims.py", "kaggle/make_bundle.py"]


def _bind_targets(node, into):
    for x in ast.walk(node):
        if isinstance(x, ast.Name):
            into.add(x.id)


def _collect_bindings(node, into):
    """Every name `node`'s subtree binds. Nested defs contribute their own
    name and their parameters, so their bodies do not report false positives."""
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                into.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                _bind_targets(t, into)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            _bind_targets(n.target, into)
        elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
            _bind_targets(n.target, into)
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            _bind_targets(n.optional_vars, into)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            into.add(n.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            into.add(n.name)
            args = getattr(n, "args", None)
            if args is not None:
                for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
                    into.add(a.arg)
                if args.vararg:
                    into.add(args.vararg.arg)
                if args.kwarg:
                    into.add(args.kwarg.arg)
        elif isinstance(n, ast.Lambda):
            for a in list(n.args.posonlyargs) + list(n.args.args) + list(n.args.kwonlyargs):
                into.add(a.arg)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            into.update(n.names)


def check(path):
    tree = ast.parse(open(path).read())
    top = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    _collect_bindings(tree, top)

    problems, seen = [], set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = set(top)
        _collect_bindings(node, local)
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id not in local and (node.name, n.id) not in seen:
                    seen.add((node.name, n.id))
                    problems.append((n.lineno, node.name, n.id))
    return sorted(problems)


def module_exports(path):
    """Every top-level name a module defines."""
    t = ast.parse(open(path).read())
    names = set()
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for tgt in targets:                 # walk: tuple targets count too
                for y in ast.walk(tgt):
                    if isinstance(y, ast.Name):
                        names.add(y.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
    return names


def check_cross_module(here):
    """Verify every K.<name> / P.<name> the harness calls actually exists.

    The undefined-name pass cannot see these: `K.gt_masks_for` is an attribute
    access on an imported module, not a bare name. A patch that renamed a
    helper in evalkit.py while leaving the caller intact shipped exactly that
    bug, and it would have surfaced as an AttributeError on Kaggle.
    """
    alias = {"K": "evalkit", "P": "pipeline"}
    exports = {}
    for mod in set(alias.values()):
        path = os.path.join(here, mod + ".py")
        if os.path.exists(path):
            exports[mod] = module_exports(path)
    bad = []
    for caller in ("run_eval.py", "export_panels.py"):
        path = os.path.join(here, caller)
        if not os.path.exists(path):
            continue
        tree = ast.parse(open(path).read())
        for n in ast.walk(tree):
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id in alias):
                mod = alias[n.value.id]
                if mod in exports and n.attr not in exports[mod]:
                    bad.append((caller, n.lineno, f"{n.value.id}.{n.attr}", mod))
    return bad


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    targets = sys.argv[1:] or [os.path.join(here, f) for f in DEFAULT]
    bad = 0
    for p in targets:
        if not os.path.exists(p):
            print(f"  SKIP   {p} (missing)")
            continue
        probs = check(p)
        rel = os.path.relpath(p, os.path.dirname(here))
        if probs:
            bad += 1
            print(f"  FAIL   {rel}")
            for line, fn, name in probs:
                print(f"           line {line:>5}  {fn}()  ->  {name}")
        else:
            print(f"  ok     {rel}")
    cross = check_cross_module(here)
    if cross:
        print("\n  FAIL   cross-module references that do not resolve:")
        for caller, line, ref, mod in cross:
            print(f"           {caller}:{line}  {ref}  not defined in {mod}.py")
    else:
        print("  ok     cross-module references (K.* / P.*)")

    if bad or cross:
        if bad:
            print(f"\n{bad} module(s) reference names they never bind.")
        if cross:
            print(f"{len(cross)} unresolved cross-module reference(s).")
        print("Fix before pushing to Kaggle.")
        sys.exit(1)
    print("\nAll modules bind every name they read, and every K.*/P.* resolves.")


if __name__ == "__main__":
    main()
