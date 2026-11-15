"""Train-OOF prior correction for fixed inner_best:ExtraTrees = 30:70.

OOF heads are proxies, and fold-excluded correction diagnostics are NOT fully
nested CV. Validation is comparison only; Test never selects priors or tau.
Compare/export require existing verified probability caches, without inference.
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
from experiments.seungbeom import prior_oof_reference as oof_code  # noqa: E402
from experiments.seungbeom.ensemble_core import validate_probabilities  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from schema import LABELS, read_frame, validate_labels  # noqa: E402

KIND = "seungbeom_inner_best_prior30"
SELF = "experiments/seungbeom/inner_best_prior30.py"
BASELINE = "inner_best_30_70"
METHOD = BASELINE + "_prior_tau"
TAUS = tuple(round(i * 0.05, 2) for i in range(41))
original = cache.previous.original


def sources():
    return {
        **cache.source_hashes(),
        SELF: sha256(ROOT / SELF),
        oof_code.THIS_FILE: sha256(ROOT / oof_code.THIS_FILE),
    }


def mixture(neural, tree):
    values = validate_probabilities(np.stack([neural, tree], axis=1))
    return 0.3 * values[:, 0] + 0.7 * values[:, 1]


def prior_for(labels):
    labels = validate_labels(labels)
    counts = np.array([(labels == c).sum() for c in LABELS], dtype=float)
    if (counts <= 0).any():
        raise ValueError("All six classes required in each prior-fitting split.")
    return counts / counts.sum()


def corrected(probabilities, prior, tau):
    p = validate_probabilities(np.asarray(probabilities)[:, None, :])[:, 0]
    prior = np.asarray(prior, dtype=float)
    if (
        prior.shape != (6,)
        or not np.isfinite(prior).all()
        or (prior <= 0).any()
        or not np.isclose(prior.sum(), 1)
        or tau not in TAUS
    ):
        raise ValueError("Invalid fixed prior/tau.")
    return (p / prior**tau).argmax(1) + 1


def choose(p, labels, prior):
    scores = [classification_metrics(labels, corrected(p, prior, t))["macro_f1"] for t in TAUS]
    return TAUS[int(np.argmax(scores))], scores


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


def check_oof_rows(z, frame):
    ids = np.asarray(z["sample_ids"]).astype(str)
    if len(ids) != len(frame) or len(set(ids)) != len(ids) or set(ids) != set(frame.sample_id):
        raise ValueError("OOF must contain each Train row exactly once.")
    aligned = frame.set_index("sample_id").loc[ids]
    if not np.array_equal(validate_labels(aligned.label), z["labels"]):
        raise ValueError("OOF labels differ from Train.")
    folds = np.asarray(z["fold"])
    if folds.shape != (len(ids),) or folds.dtype.kind not in "iu" or len(np.unique(folds)) < 3:
        raise ValueError("Invalid OOF folds.")
    if aligned.experiment_id.isna().any():
        raise ValueError("Missing experiment ID.")
    if (
        pd.DataFrame({"group": aligned.experiment_id.to_numpy(), "fold": folds})
        .groupby("group")
        .fold.nunique()
        .max()
        != 1
    ):
        raise ValueError("OOF folds split an experiment.")


def finish_checks(base, manifest, files, saved_sources, device):
    cache.unchanged(base, manifest, files, cache.source_hashes(), device)
    if sources() != saved_sources:
        raise ValueError("Prior-correction source changed during execution.")


def generate_oof(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    saved_sources = sources()
    reference, base, manifest, files = cache.load_reference(args.baseline_recipe, args.device)
    tree_run = Path(manifest["reference"]["base_recipe"]["ensemble_run"])
    inputs = [
        args.data_dir / "train/train.csv",
        tree_run / "train_oof.npz",
        tree_run / "embeddings/train/features.npy",
        tree_run / "embeddings/train/metadata.json",
        tree_run / "embeddings/train/sample_ids.csv",
    ]
    files.update(original.file_hashes(inputs))
    frame = read_frame(inputs[0], labeled=True)
    with np.load(inputs[1], allow_pickle=False) as z:
        check_oof_rows(z, frame)
    try:
        oof_code.oof(
            argparse.Namespace(
                output_dir=args.output_dir,
                device=args.device,
                data_dir=args.data_dir,
                tree_run=tree_run,
                checkpoint_run=Path(base["run_dir"]),
            )
        )
        finish_checks(base, manifest, files, saved_sources, args.device)
        info = read_json(args.output_dir / "oof.json")
        info.update(
            {
                "source_sha256": saved_sources,
                "reference": reference,
                "provenance": files,
                "kind": KIND + "_oof",
            }
        )
        write_json(args.output_dir / "oof.json", info)
        write_json(
            args.output_dir / "status.json",
            {"status": "complete", "oof_json_sha256": sha256(args.output_dir / "oof.json")},
        )
    except BaseException as error:
        if args.output_dir.exists():
            write_json(
                args.output_dir / "status.json",
                {"status": "failed", "error_type": type(error).__name__},
            )
        raise


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
        or info.get("kind") != KIND + "_oof"
        or info.get("source_sha256") != saved_sources
        or info.get("reference") != reference
        or info["train_csv_sha256"] != manifest["inputs"]["train"]["csv_sha256"]
        or sha256(args.train_csv) != info["train_csv_sha256"]
        or sha256(args.oof_dir / "oof.npz") != info["oof_sha256"]
    ):
        raise ValueError("OOF provenance/Train input changed; use this command's completed OOF.")
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
            "weight": 0.3,
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
    if c.get("weight") != 0.3 or c.get("labels") != list(LABELS):
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
        raise ValueError("Saved member/30:70 Validation replay differs.")
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
        raise ValueError("OOF diagnostic did not support correction; keep original 30:70.")
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
        {"recipe": recipe, "output": record, "input_files": files, "images_sha256": images},
    )
    write_json(args.output_dir / "status.json", {"status": "complete", "output": record})
    print(f"Saved {output}. No training/inference; no Train/Validation/OOF data read.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    oof = commands.add_parser("oof")
    for name in ("baseline_recipe", "data_dir", "output_dir"):
        oof.add_argument("--" + name, type=Path, required=True)
    oof.set_defaults(handler=generate_oof)
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
    for command in (oof, selection, comparison, export):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
