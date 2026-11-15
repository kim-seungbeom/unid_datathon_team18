"""Compare Temperature scaling and L2 logistic stacking using existing Train OOF.

The base models, OOF proxy heads, caches and previous submissions are immutable.
Validation is comparison only; Test applies saved fusion parameters, without fitting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom import inner_best_prior_neighbors as previous  # noqa: E402
from experiments.seungbeom import probability_fusion_core as core  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    read_json,
    sha256,
    write_json,
)
from schema import read_frame  # noqa: E402

prior30 = previous.prior30
cache = prior30.cache
original = prior30.original
KIND = "seungbeom_probability_fusion"
SELF = "experiments/seungbeom/probability_fusion.py"
CORE = "experiments/seungbeom/probability_fusion_core.py"


def sources():
    return {**previous.sources(), SELF: sha256(ROOT / SELF), CORE: sha256(ROOT / CORE)}


def versions():
    return {"scipy": scipy.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__}


def unchanged(base, manifest, files, saved_sources, device):
    previous.unchanged(base, manifest, files, previous.sources(), device)
    if sources() != saved_sources:
        raise ValueError("Fusion source changed during execution.")


def load_oof(args, old, manifest, files):
    """Authenticate the existing prior30 OOF, including Train rows/group isolation."""
    c = old["correction"]
    info_path, npz_path = args.oof_dir / "oof.json", args.oof_dir / "oof.npz"
    info = read_json(info_path)
    status = read_json(args.oof_dir / "status.json")
    if (
        status.get("status") != "complete"
        or status.get("oof_json_sha256") != sha256(info_path)
        or info != c["oof_info"]
        or info.get("source_sha256") != prior30.sources()
        or sha256(npz_path) != info["oof_sha256"]
        or sha256(args.train_csv) != manifest["inputs"]["train"]["csv_sha256"]
        or sha256(args.train_csv) != info["train_csv_sha256"]
    ):
        raise ValueError("Train/OOF provenance mismatch.")
    files.update(
        original.file_hashes([info_path, npz_path, args.oof_dir / "status.json", args.train_csv])
    )
    frame = read_frame(args.train_csv, labeled=True)
    with np.load(npz_path, allow_pickle=False) as z:
        prior30.check_oof_rows(z, frame)
        values = core.members(z["neural"], z["extra_trees"])
        y, folds = z["labels"].copy(), z["fold"].copy()
    baseline = core.score(y, prior30.mixture(values[:, 0], values[:, 1]))["classification"]
    if baseline != c["baseline"]:
        raise ValueError("Original Train OOF baseline does not replay.")
    return values[:, 0], values[:, 1], y, folds


def prediction_labels(neural, tree, state, old_correction):
    probs = core.apply(neural, tree, state)
    labels = {name: value.argmax(1) + 1 for name, value in probs.items()}
    labels[prior30.METHOD] = prior30.corrected(
        probs[core.BASELINE], old_correction["prior"], old_correction["tau"]
    )
    return probs, labels


def duplicate_decisions(labels):
    # Deduplication only: no Test labels, distribution targets or performance ranking.
    seen = {core.BASELINE: labels[core.BASELINE], prior30.METHOD: labels[prior30.METHOD]}
    result = {}
    for method in core.METHODS:
        duplicate = next(
            (name for name, pred in seen.items() if np.array_equal(pred, labels[method])), None
        )
        result[method] = "export" if duplicate is None else f"skip: identical to {duplicate}"
        if duplicate is None:
            seen[method] = labels[method]
    return result


def display(recipe, folds=False):
    state, diagnosis = recipe["fusion"], recipe["diagnostics"]
    print("Temperature member order = neural, ExtraTrees; final ratio = 30:70, no prior tau.")
    for name, t in zip(("neural", "extra_trees"), state["temperatures"], strict=True):
        print(
            f"{name}: T={t['T']:.6f} fit_NLL={t['before_nll']:.6f}->{t['fit_nll']:.6f} "
            f"at_bound={t['at_bound']}"
        )
    print(f"Stacking: raw probabilities 12 -> six classes; fixed L2 C={core.STACK_C}.")
    print("method\tfold_excluded_macro_f1\tfold_excluded_NLL\tfit_macro_f1\tfit_NLL")
    for name in (core.BASELINE, *core.METHODS):
        held = diagnosis["fold_excluded_diagnostic"][name]
        fitted = diagnosis["full_fit_diagnostic"][name]
        print(
            f"{name}\t{held['classification']['macro_f1']:.6f}\t{held['log_loss']:.6f}\t"
            f"{fitted['classification']['macro_f1']:.6f}\t{fitted['log_loss']:.6f}"
        )
    if folds:
        print("fold\tmethod\tmacro_f1\tNLL")
        for fold in diagnosis["per_fold"]:
            for name, m in fold["metrics"].items():
                print(
                    f"{fold['fold']}\t{name}\t{m['classification']['macro_f1']:.6f}\t{m['log_loss']:.6f}"
                )
    prior30.display(recipe["validation_metrics"])
    print("Validation NLL (lower is better):", recipe["validation_nll"])
    print("Changed Validation rows vs prior30:", recipe["changed_rows"])
    print("OOF diagnostic improvement vs raw 30:70:", recipe["diagnostic_improved"])
    print(diagnosis["limitations"])
    print("Validation reused for comparison; lower NLL does not guarantee higher Macro F1.")


def compare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    saved_sources, environment = sources(), versions()
    old, reference, base, manifest, files = previous.load_reference(
        args.baseline_recipe, args.device
    )
    if sha256(args.input) != base["validation"]["csv_sha256"]:
        raise ValueError("Comparison input must be the original Validation CSV.")
    # All fitting happens before reading any Validation data/cache.
    neural, tree, y, folds = load_oof(args, old, manifest, files)
    state, diagnosis = core.crossfit(neural, tree, y, folds)
    files.update(original.file_hashes([args.input]))
    frame = read_frame(args.input, labeled=True)
    members, images = cache.cached_members(
        args, frame, base, manifest, files, cache.source_hashes()
    )
    if images != base["validation"]["images_sha256"]:
        raise ValueError("Validation images changed.")
    existing = cache.previous.candidates(*members)
    if any(
        prior30.classification_metrics(frame.label, existing[name].argmax(1) + 1)
        != reference["metrics"][name]
        for name in cache.REPLAY
    ):
        raise ValueError("Original member/ratio Validation replay failed.")
    probs, labels = prediction_labels(*members, state, old["correction"])
    metrics = {name: prior30.classification_metrics(frame.label, v) for name, v in labels.items()}
    if any(metrics[name] != old["metrics"][name] for name in (core.BASELINE, prior30.METHOD)):
        raise ValueError("Original 30:70 / prior30 Validation replay failed.")
    unchanged(base, manifest, files, saved_sources, args.device)
    if versions() != environment:
        raise ValueError("Fusion libraries changed during execution.")
    diagnostic = diagnosis["fold_excluded_diagnostic"]
    recipe = {
        "kind": KIND,
        "source_sha256": saved_sources,
        "versions": environment,
        "contract": core.CONTRACT,
        "baseline_path": str(args.baseline_recipe.resolve()),
        "baseline": old,
        "fusion": state,
        "diagnostics": diagnosis,
        "validation_metrics": metrics,
        "validation_nll": {n: core.log_loss(frame.label, p) for n, p in probs.items()},
        "changed_rows": {
            n: int(np.count_nonzero(v != labels[prior30.METHOD])) for n, v in labels.items()
        },
        "diagnostic_improved": {
            n: bool(
                diagnostic[n]["classification"]["macro_f1"]
                > diagnostic[core.BASELINE]["classification"]["macro_f1"]
            )
            for n in core.METHODS
        },
        "provenance": files,
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "recipe.json"
    write_json(path, recipe)
    write_json(args.output_dir / "status.json", {"status": "complete", "sha256": sha256(path)})
    display(recipe)
    return recipe


def read_recipe(path):
    status = read_json(path.parent / "status.json")
    if status.get("status") != "complete" or status.get("sha256") != sha256(path):
        raise ValueError("Incomplete/modified fusion recipe.")
    r = read_json(path)
    if (
        r.get("kind") != KIND
        or r.get("source_sha256") != sources()
        or r.get("versions") != versions()
        or r.get("contract") != core.CONTRACT
    ):
        raise ValueError("Fusion source/version/contract mismatch.")
    core.apply(np.full((1, 6), 1 / 6), np.full((1, 6), 1 / 6), r["fusion"])
    return r


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    r = read_recipe(args.recipe)
    old, _, base, manifest, files = previous.load_reference(Path(r["baseline_path"]), args.device)
    if old != r["baseline"]:
        raise ValueError("Saved baseline changed.")
    files.update(
        original.file_hashes(
            [args.recipe, args.recipe.parent / "status.json", args.input, args.sample_submission]
        )
    )
    frame = read_frame(args.input)
    ids = frame.sample_id.tolist()
    order = prior30.template_ids(args.sample_submission, ids)
    members, images = cache.cached_members(
        args, frame, base, manifest, files, cache.source_hashes()
    )
    _, labels = prediction_labels(*members, r["fusion"], old["correction"])
    decisions = duplicate_decisions(labels)
    unchanged(base, manifest, files, r["source_sha256"], args.device)
    args.output_dir.mkdir(parents=True)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    outputs = {}
    try:
        for method in core.METHODS:
            print(
                f"{method}: {decisions[method]}; "
                f"OOF diagnostic improved={r['diagnostic_improved'][method]}"
            )
            if decisions[method] != "export":
                continue
            values = pd.Series(labels[method], index=ids).loc[order].to_numpy(dtype=np.int64)
            output = args.output_dir / f"submission_{method}.csv"
            with output.open("x", encoding="utf-8", newline="") as handle:
                pd.DataFrame({"sample_id": order, "label": values}).to_csv(handle, index=False)
            outputs[method] = check_csv(output, order)
            write_json(
                output.with_suffix(".manifest.json"),
                {
                    "recipe": r,
                    "method": method,
                    "input_files": files,
                    "images_sha256": images,
                    "output": outputs[method],
                },
            )
            print(f"Saved {output}")
        unchanged(base, manifest, files, r["source_sha256"], args.device)
        write_json(
            status,
            {
                "status": "complete",
                "outputs": outputs,
                "decisions": decisions,
                "changed_rows_vs_prior30": {
                    n: int(np.count_nonzero(labels[n] != labels[prior30.METHOD]))
                    for n in core.METHODS
                },
                "candidate_disagreement_rows": int(
                    np.count_nonzero(labels[core.METHODS[0]] != labels[core.METHODS[1]])
                ),
            },
        )
        print("Fixed fusion inference only; no fitting or Train/Validation/OOF data access.")
        print(
            "CSV generation does not mean the candidate improved; "
            "keep the existing best submission."
        )
        return outputs
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    comp = commands.add_parser("compare")
    for name in ("baseline_recipe", "oof_dir", "train_csv"):
        comp.add_argument("--" + name, type=Path, required=True)
    comp.set_defaults(handler=compare)
    export = commands.add_parser("predict")
    for name in ("recipe", "sample_submission"):
        export.add_argument("--" + name, type=Path, required=True)
    export.set_defaults(handler=predict)
    for command in (comp, export):
        for name in ("input", "image_root", "cache_dir", "output_dir"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    report = commands.add_parser("report")
    report.add_argument("--recipe", type=Path, required=True)
    report.add_argument("--folds", action="store_true")
    report.set_defaults(handler=lambda args: display(read_recipe(args.recipe), args.folds))
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
