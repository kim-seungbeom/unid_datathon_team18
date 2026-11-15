"""Train-OOF prior correction for fixed inner_best:ExtraTrees = 40:60.

OOF heads are proxies, and fold-excluded correction diagnostics are NOT fully
nested CV. Validation is comparison only; Test never selects priors or tau.
Reuses prior30 Train OOF members, but recomputes the fixed 40:60 mixture and tau.
Grid: 0..2 in steps of 0.05, ties keep smaller tau. No model training.
Compare/export require existing verified probability caches, without inference.
This is a new experiment, not a reconstruction of the earlier highest submission.
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

from evaluate import classification_metrics  # noqa: E402
from experiments.seungbeom import inner_best_cached_ratios as cache  # noqa: E402
from experiments.seungbeom import inner_best_prior30 as origin  # noqa: E402
from experiments.seungbeom.ensemble_core import validate_probabilities  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from schema import LABELS, read_frame, validate_labels  # noqa: E402

KIND = "seungbeom_inner_best_prior40"
SELF = "experiments/seungbeom/inner_best_prior40.py"
BASELINE = "inner_best_40_60"
METHOD = BASELINE + "_prior_tau"
TAUS = tuple(round(i * 0.05, 2) for i in range(41))
original = cache.previous.original


def sources():
    return {**origin.sources(), SELF: sha256(ROOT / SELF)}


def mixture(neural, tree):
    values = validate_probabilities(np.stack([neural, tree], axis=1))
    return 0.4 * values[:, 0] + 0.6 * values[:, 1]


prior_for = origin.prior_for
corrected = origin.corrected
choose = origin.choose


def select_arrays(neural, tree, labels, folds):
    labels = validate_labels(labels)
    p = mixture(neural, tree)
    folds = np.asarray(folds)
    if folds.shape != labels.shape or len(p) != len(labels) or len(np.unique(folds)) < 3:
        raise ValueError("At least three aligned folds required.")
    baseline = classification_metrics(labels, p.argmax(1) + 1)
    held_predictions = np.zeros(len(labels), dtype=int)
    per_fold = []
    for fold in np.unique(folds):
        fit, held = folds != fold, folds == fold
        prior = prior_for(labels[fit])
        tau, _ = choose(p[fit], labels[fit], prior)
        held_predictions[held] = corrected(p[held], prior, tau)
        per_fold.append(
            {
                "fold": int(fold),
                "tau": tau,
                "baseline": classification_metrics(labels[held], p[held].argmax(1) + 1),
                "corrected": classification_metrics(labels[held], held_predictions[held]),
            }
        )
    prior = prior_for(labels)
    tau, scores = choose(p, labels, prior)
    diagnostic = classification_metrics(labels, held_predictions)
    return {
        "tau": tau,
        "prior": prior.tolist(),
        "baseline": baseline,
        "fold_excluded_diagnostic": diagnostic,
        "per_fold": per_fold,
        "fit_diagnostic": classification_metrics(labels, corrected(p, prior, tau)),
        "grid": dict(zip(map(str, TAUS), scores, strict=True)),
        "at_grid_edge": tau == TAUS[-1],
        "adopt": bool(tau > 0 and diagnostic["macro_f1"] > baseline["macro_f1"]),
        "limitations": "Proxy heads; epoch reused from inner selection; fold-excluded "
        "tau diagnostics reuse base OOF and are not independent nested CV.",
    }


check_oof_rows = origin.check_oof_rows


def finish_checks(base, manifest, files, saved_sources, device):
    cache.unchanged(base, manifest, files, cache.source_hashes(), device)
    if sources() != saved_sources:
        raise ValueError("Prior-correction source changed during execution.")


def select(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    reference, base, manifest, files = cache.load_reference(args.baseline_recipe, args.device)
    saved_sources = sources()
    info = read_json(args.oof_dir / "oof.json")
    status = read_json(args.oof_dir / "status.json")
    if (
        status.get("status") != "complete"
        or status.get("oof_json_sha256") != sha256(args.oof_dir / "oof.json")
        or info.get("kind") != origin.KIND + "_oof"
        or info.get("source_sha256") != origin.sources()
        or info.get("reference") != reference
        or info["train_csv_sha256"] != manifest["inputs"]["train"]["csv_sha256"]
        or sha256(args.train_csv) != info["train_csv_sha256"]
        or sha256(args.oof_dir / "oof.npz") != info["oof_sha256"]
    ):
        raise ValueError(
            "OOF provenance/Train input changed; use the existing prior30 completed OOF."
        )
    files.update(
        original.file_hashes(
            [
                args.train_csv,
                args.oof_dir / "oof.json",
                args.oof_dir / "status.json",
                args.oof_dir / "oof.npz",
            ]
        )
    )
    frame = read_frame(args.train_csv, labeled=True)
    with np.load(args.oof_dir / "oof.npz", allow_pickle=False) as z:
        check_oof_rows(z, frame)
        result = select_arrays(z["neural"], z["extra_trees"], z["labels"], z["fold"])
    finish_checks(base, manifest, files, saved_sources, args.device)
    result.update(
        {
            "kind": KIND,
            "weight": 0.4,
            "labels": list(LABELS),
            "reference": reference,
            "reference_path": str(args.baseline_recipe.resolve()),
            "source_sha256": saved_sources,
            "selection_provenance": files,
            "oof_info": info,
        }
    )
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "correction.json"
    write_json(path, result)
    write_json(args.output_dir / "status.json", {"status": "complete", "sha256": sha256(path)})
    for name in ("baseline", "fold_excluded_diagnostic", "fit_diagnostic"):
        print(f"{name}\t{result[name]['macro_f1']:.6f}")
    print("fold\ttau\tbaseline_macro_f1\tcorrected_macro_f1")
    for row in result["per_fold"]:
        print(
            f"{row['fold']}\t{row['tau']:.2f}\t{row['baseline']['macro_f1']:.6f}\t"
            f"{row['corrected']['macro_f1']:.6f}"
        )
    print(f"Selected tau={result['tau']}; adopt={result['adopt']}; edge={result['at_grid_edge']}")
    print(result["limitations"])
    return result


def read_saved(path):
    status = read_json(path.parent / "status.json")
    if status.get("status") != "complete" or status.get("sha256") != sha256(path):
        raise ValueError("Incomplete or modified saved result.")
    value = read_json(path)
    if value.get("kind") != KIND or value.get("source_sha256") != sources():
        raise ValueError("Correction source/kind differs.")
    return value


def read_correction(path):
    c = read_saved(path)
    if c.get("weight") != 0.4 or c.get("labels") != list(LABELS):
        raise ValueError("Invalid correction identity.")
    corrected(np.full((1, 6), 1 / 6), c["prior"], c["tau"])
    return c


def display(metrics):
    print("method\tvalidation_macro_f1\t" + "\t".join(f"f1_{i}" for i in LABELS))
    for name, m in metrics.items():
        print(
            f"{name}\t{m['macro_f1']:.6f}\t"
            + "\t".join(f"{m['class_f1'][str(i)]:.6f}" for i in LABELS)
        )
        cm = np.asarray(m["confusion_matrix"])
        print(f"  errors 3->2={cm[2, 1]} 5->4={cm[4, 3]} 2->3={cm[1, 2]} 4->5={cm[3, 4]}")


def compare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    correction = read_correction(args.correction)
    reference, base, manifest, files = cache.load_reference(
        Path(correction["reference_path"]), args.device
    )
    if (
        reference != correction["reference"]
        or sha256(args.input) != base["validation"]["csv_sha256"]
    ):
        raise ValueError("Reference changed or input is not original Validation.")
    files.update(
        original.file_hashes([args.input, args.correction, args.correction.parent / "status.json"])
    )
    frame = read_frame(args.input, labeled=True)
    members, images = cache.cached_members(
        args, frame, base, manifest, files, cache.source_hashes()
    )
    if images != base["validation"]["images_sha256"]:
        raise ValueError("Validation image mismatch.")
    old = cache.previous.candidates(*members)
    if any(
        classification_metrics(frame.label, old[n].argmax(1) + 1) != reference["metrics"][n]
        for n in cache.REPLAY
    ):
        raise ValueError("Saved member/40:60 Validation replay differs.")
    p = mixture(*members)
    metrics = {
        BASELINE: classification_metrics(frame.label, p.argmax(1) + 1),
        METHOD: classification_metrics(
            frame.label, corrected(p, correction["prior"], correction["tau"])
        ),
    }
    finish_checks(base, manifest, files, correction["source_sha256"], args.device)
    recipe = {
        "kind": KIND,
        "source_sha256": sources(),
        "correction": correction,
        "metrics": metrics,
        "validation_provenance": files,
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "recipe.json"
    write_json(path, recipe)
    write_json(args.output_dir / "status.json", {"status": "complete", "sha256": sha256(path)})
    display(metrics)
    print("Tau fixed from Train OOF. Reused Validation comparison, not independent evaluation.")
    return recipe


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    recipe = read_saved(args.recipe)
    c = recipe["correction"]
    if not c["adopt"]:
        print(
            "Exploratory export: held-fold diagnostics did not support correction; "
            "not a verified upgrade."
        )
    reference, base, manifest, files = cache.load_reference(Path(c["reference_path"]), args.device)
    if reference != c["reference"]:
        raise ValueError("Original reference changed.")
    files.update(
        original.file_hashes(
            [args.input, args.sample_submission, args.recipe, args.recipe.parent / "status.json"]
        )
    )
    frame = read_frame(args.input)
    ids = frame.sample_id.tolist()
    order = template_ids(args.sample_submission, ids)
    members, images = cache.cached_members(
        args, frame, base, manifest, files, cache.source_hashes()
    )
    predictions = corrected(mixture(*members), c["prior"], c["tau"])
    values = pd.Series(predictions, index=ids).loc[order].to_numpy(dtype=np.int64)
    finish_checks(base, manifest, files, recipe["source_sha256"], args.device)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / f"submission_{METHOD}.csv"
    with output.open("x", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"sample_id": order, "label": values}).to_csv(handle, index=False)
    record = check_csv(output, order)
    write_json(
        output.with_suffix(".manifest.json"),
        {
            "recipe": recipe,
            "output": record,
            "input_files": files,
            "images_sha256": images,
            "original_submission_match_verified": False,
            "public_score": None,
            "new_40_60_tau_experiment": True,
        },
    )
    write_json(args.output_dir / "status.json", {"status": "complete", "output": record})
    print(f"Saved {output}. No training/inference; no Train/Validation/OOF data read.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("select")
    for name in ("baseline_recipe", "oof_dir", "train_csv", "output_dir"):
        selection.add_argument("--" + name, type=Path, required=True)
    selection.set_defaults(handler=select)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--correction", type=Path, required=True)
    comparison.set_defaults(handler=compare)
    export = commands.add_parser("predict")
    export.add_argument("--recipe", type=Path, required=True)
    export.add_argument("--sample_submission", type=Path, required=True)
    export.set_defaults(handler=predict)
    for command in (comparison, export):
        for name in ("input", "image_root", "cache_dir", "output_dir"):
            command.add_argument("--" + name, type=Path, required=True)
    for command in (selection, comparison, export):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
