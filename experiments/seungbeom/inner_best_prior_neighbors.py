"""Reuse Train OOF to select priors for two predefined ratios: 35:65 and 25:75.

No model training or image inference. Validation compares fixed corrections;
Test only applies them and detects duplicate labels, never fits or ranks them.
Leave the fingerprinted prior30 code and all previous artifacts unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom import inner_best_prior30 as prior30  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    read_json,
    sha256,
    write_json,
)
from schema import LABELS, read_frame, validate_labels  # noqa: E402

KIND = "seungbeom_inner_best_prior_neighbors"
SELF = "experiments/seungbeom/inner_best_prior_neighbors.py"
RATIOS = {"inner_best_35_65": 0.35, "inner_best_25_75": 0.25}
cache = prior30.cache
original = prior30.original
metrics_for = prior30.classification_metrics


def sources():
    return {**prior30.sources(), SELF: sha256(ROOT / SELF)}


def mixture(neural, tree, weight):
    if weight not in (*RATIOS.values(), 0.3):
        raise ValueError("Only predefined ratios are supported.")
    p = prior30.validate_probabilities(np.stack([neural, tree], axis=1))
    return weight * p[:, 0] + (1 - weight) * p[:, 1]


def select_ratio(neural, tree, labels, folds, weight):
    """Same diagnostics as prior30, without changing its fixed-ratio arithmetic."""
    y = validate_labels(labels)
    p = mixture(neural, tree, weight)
    folds = np.asarray(folds)
    if folds.shape != y.shape or len(p) != len(y) or len(np.unique(folds)) < 3:
        raise ValueError("At least three aligned folds required.")
    baseline = metrics_for(y, p.argmax(1) + 1)
    held_predictions = np.zeros(len(y), dtype=int)
    per_fold = []
    for f in np.unique(folds):
        fit, held = folds != f, folds == f
        prior = prior30.prior_for(y[fit])
        tau, _ = prior30.choose(p[fit], y[fit], prior)
        held_predictions[held] = prior30.corrected(p[held], prior, tau)
        per_fold.append(
            {
                "fold": int(f),
                "tau": tau,
                "baseline": metrics_for(y[held], p[held].argmax(1) + 1),
                "corrected": metrics_for(y[held], held_predictions[held]),
            }
        )
    prior = prior30.prior_for(y)
    tau, grid = prior30.choose(p, y, prior)
    diagnostic = metrics_for(y, held_predictions)
    return {
        "weight": weight,
        "prior": prior.tolist(),
        "tau": tau,
        "baseline": baseline,
        "fold_excluded_diagnostic": diagnostic,
        "fit_diagnostic": metrics_for(y, prior30.corrected(p, prior, tau)),
        "per_fold": per_fold,
        "grid": dict(zip(map(str, prior30.TAUS), grid, strict=True)),
        "adopt": bool(tau > 0 and diagnostic["macro_f1"] > baseline["macro_f1"]),
        "at_grid_edge": tau == prior30.TAUS[-1],
    }


def load_reference(path, device):
    old = prior30.read_saved(path)
    c = old["correction"]
    if c["weight"] != 0.3 or c["labels"] != list(LABELS):
        raise ValueError("Expected the existing 30:70 correction recipe.")
    reference, base, manifest, files = cache.load_reference(Path(c["reference_path"]), device)
    if reference != c["reference"]:
        raise ValueError("Existing baseline reference changed.")
    files.update(original.file_hashes([path, path.parent / "status.json"]))
    return old, reference, base, manifest, files


def unchanged(base, manifest, files, saved_sources, device):
    prior30.finish_checks(base, manifest, files, prior30.sources(), device)
    if sources() != saved_sources:
        raise ValueError("Neighbor correction source changed.")


def fit_from_oof(args, old, manifest, files):
    """Require exactly the OOF/Train inputs used by the prior30 selection."""
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
        raise ValueError("Original OOF/Train provenance mismatch.")
    files.update(
        original.file_hashes([info_path, npz_path, args.oof_dir / "status.json", args.train_csv])
    )
    frame = read_frame(args.train_csv, labeled=True)
    with np.load(npz_path, allow_pickle=False) as z:
        prior30.check_oof_rows(z, frame)
        replay = prior30.select_arrays(z["neural"], z["extra_trees"], z["labels"], z["fold"])
        if any(c[key] != value for key, value in replay.items()):
            raise ValueError("Original 30:70 tau selection does not replay.")
        return {
            name: select_ratio(z["neural"], z["extra_trees"], z["labels"], z["fold"], w)
            for name, w in RATIOS.items()
        }


def predictions(neural, tree, old_correction, selections):
    p = prior30.mixture(neural, tree)
    result = {
        prior30.BASELINE: p.argmax(1) + 1,
        prior30.METHOD: prior30.corrected(p, old_correction["prior"], old_correction["tau"]),
    }
    for name, weight in RATIOS.items():
        p = mixture(neural, tree, weight)
        c = selections[name]
        result[name] = p.argmax(1) + 1
        result[name + "_prior_tau"] = prior30.corrected(p, c["prior"], c["tau"])
    return result


def export_decisions(preds, selections):
    """Deduplication is not Test-based optimization; tau and weights stay fixed."""
    seen = {prior30.BASELINE: preds[prior30.BASELINE], prior30.METHOD: preds[prior30.METHOD]}
    decisions = {}
    for name in RATIOS:  # fixed order, never sorted by Validation or Test predictions
        method = name + "_prior_tau"
        if not selections[name]["adopt"]:
            decisions[method] = "skip: OOF diagnostic did not support correction"
            continue
        duplicate = next(
            (key for key, values in seen.items() if np.array_equal(values, preds[method])), None
        )
        if duplicate is not None:
            decisions[method] = f"skip: identical labels to {duplicate}"
        else:
            decisions[method] = "export"
            seen[method] = preds[method]
    return decisions


def display(recipe):
    print("Train OOF selection; neural:tree weights are fixed before comparison.")
    print("method\ttau\tadopt\tbaseline_oof\tfold_excluded_diagnostic\tfit_diagnostic")
    for name, c in recipe["selections"].items():
        print(
            f"{name}\t{c['tau']:.2f}\t{c['adopt']}\t{c['baseline']['macro_f1']:.6f}\t"
            f"{c['fold_excluded_diagnostic']['macro_f1']:.6f}\t{c['fit_diagnostic']['macro_f1']:.6f}"
        )
        print(f"  at_grid_edge={c['at_grid_edge']}")
    prior30.display(recipe["metrics"])
    print("Validation changed rows vs existing 30:70+tau:", recipe["changed_rows"])
    print("Proxy OOF selection diagnostics are not independent nested CV; Validation is reused.")


def compare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    saved_sources = sources()
    old, reference, base, manifest, files = load_reference(args.baseline_recipe, args.device)
    # Select from Train before opening Validation; no validation-dependent tau search.
    selections = fit_from_oof(args, old, manifest, files)
    if sha256(args.input) != base["validation"]["csv_sha256"]:
        raise ValueError("Comparison requires the original Validation input.")
    files.update(original.file_hashes([args.input]))
    frame = read_frame(args.input, labeled=True)
    members, images = cache.cached_members(
        args, frame, base, manifest, files, cache.source_hashes()
    )
    if images != base["validation"]["images_sha256"]:
        raise ValueError("Validation images changed.")
    existing = cache.previous.candidates(*members)
    if any(
        metrics_for(frame.label, existing[name].argmax(1) + 1) != reference["metrics"][name]
        for name in cache.REPLAY
    ):
        raise ValueError("Existing member/ratio Validation replay failed.")
    preds = predictions(*members, old["correction"], selections)
    metrics = {name: metrics_for(frame.label, p) for name, p in preds.items()}
    if any(metrics[name] != old["metrics"][name] for name in (prior30.BASELINE, prior30.METHOD)):
        raise ValueError("Original 30:70+tau Validation replay failed.")
    unchanged(base, manifest, files, saved_sources, args.device)
    recipe = {
        "kind": KIND,
        "ratios": RATIOS,
        "source_sha256": saved_sources,
        "baseline_path": str(args.baseline_recipe.resolve()),
        "baseline": old,
        "selections": selections,
        "metrics": metrics,
        "provenance": files,
        "changed_rows": {
            name: int(np.count_nonzero(p != preds[prior30.METHOD])) for name, p in preds.items()
        },
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
        raise ValueError("Incomplete/changed neighbor recipe.")
    r = read_json(path)
    if (
        r.get("kind") != KIND
        or r.get("source_sha256") != sources()
        or r.get("ratios") != RATIOS
        or set(r.get("selections", {})) != set(RATIOS)
    ):
        raise ValueError("Neighbor recipe/source identity mismatch.")
    for name, weight in RATIOS.items():
        c = r["selections"][name]
        if c["weight"] != weight or type(c["adopt"]) is not bool:
            raise ValueError("Invalid correction weight/adoption.")
        prior30.corrected(np.full((1, 6), 1 / 6), c["prior"], c["tau"])
    return r


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    r = read_recipe(args.recipe)
    old, _, base, manifest, files = load_reference(Path(r["baseline_path"]), args.device)
    if old != r["baseline"]:
        raise ValueError("Original prior30 recipe changed.")
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
    preds = predictions(*members, old["correction"], r["selections"])
    decisions = export_decisions(preds, r["selections"])
    unchanged(base, manifest, files, r["source_sha256"], args.device)
    args.output_dir.mkdir(parents=True)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    outputs = {}
    try:
        for method, decision in decisions.items():
            print(f"{method}: {decision}")
            if decision != "export":
                continue
            output = args.output_dir / f"submission_{method}.csv"
            values = pd.Series(preds[method], index=ids).loc[order].to_numpy(dtype=np.int64)
            with output.open("x", encoding="utf-8", newline="") as handle:
                pd.DataFrame({"sample_id": order, "label": values}).to_csv(handle, index=False)
            outputs[method] = check_csv(output, order)
            write_json(
                output.with_suffix(".manifest.json"),
                {
                    "recipe": r,
                    "input_files": files,
                    "images_sha256": images,
                    "output": outputs[method],
                },
            )
            print(f"Saved {output}")
        unchanged(base, manifest, files, r["source_sha256"], args.device)
        names = [name + "_prior_tau" for name in RATIOS]
        write_json(
            status,
            {
                "status": "complete",
                "outputs": outputs,
                "decisions": decisions,
                "changed_rows_vs_prior30": {
                    n: int(np.count_nonzero(preds[n] != preds[prior30.METHOD])) for n in names
                },
                "candidate_disagreement_rows": int(
                    np.count_nonzero(preds[names[0]] != preds[names[1]])
                ),
            },
        )
        print(
            f"Exported {len(outputs)} CSV(s). "
            "No training/inference or Train/Validation/OOF data read."
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
    report.set_defaults(handler=lambda args: display(read_recipe(args.recipe)))
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
