"""User-run 3-fold experiment-group CV; preserves the original 50:50 submission.

Reads only the original Train CSV and frozen Train embedding cache, plus saved
model metadata. Official Validation and Test CSVs/images are never opened.
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
from experiments.seungbeom import blend_ratios  # noqa: E402
from experiments.seungbeom import group_cv_training as training  # noqa: E402
from experiments.seungbeom import midpoint_blend as midpoint  # noqa: E402
from experiments.seungbeom import multimodal as mm  # noqa: E402
from experiments.seungbeom.ensemble_core import validate_probabilities  # noqa: E402
from experiments.seungbeom.export_submissions import read_json, sha256, write_json  # noqa: E402
from experiments.seungbeom.neural_data import embedding_contract  # noqa: E402
from schema import LABELS, RAW_SENSOR_COLUMNS, read_frame  # noqa: E402
from sensor_features import build_sensor_features  # noqa: E402

KIND = "seungbeom_train_group_cv"
HELPERS = (
    "experiments/seungbeom/group_cv.py",
    "experiments/seungbeom/group_cv_training.py",
    "experiments/seungbeom/blend_ratios.py",
)
ARTIFACTS = (
    "result.json",
    "oof.npz",
    "neural.pt",
    "extra_trees.joblib",
    "inner_history.json",
    "refit_history.json",
)


def source_hashes():
    return {**midpoint.source_hashes(), **{name: sha256(ROOT / name) for name in HELPERS}}


def load_inputs(args):
    base = read_json(args.recipe)
    # Verify Train identity before reading any CSV, even if a labeled Validation
    # path was accidentally supplied. Never infer paths to other data splits.
    if sha256(args.train_csv) != base["provenance"]["train_csv_sha256"]:
        raise ValueError("--train_csv must be the original Train CSV, not Validation/Test.")
    loaded = blend_ratios.load_base(base, args.device)
    config = loaded["config"]
    if config["seed"] != 42 or config["loss"] != "cross_entropy":
        raise ValueError("Use the original seed42 cross-entropy 50:50 recipe.")
    cache_hashes = {
        name: sha256(args.train_embeddings / name)
        for name in ("features.npy", "sample_ids.csv", "metadata.json")
    }
    if cache_hashes != loaded["identity"]["input_provenance"]["train"]:
        raise ValueError("Use the exact frozen Train embedding cache of the original neural run.")
    frame = read_frame(args.train_csv, labeled=True)
    plan = training.split_plan(frame, config["seed"])
    vectors, meta = mm.load_embeddings(args.train_embeddings, frame["sample_id"])
    contract = embedding_contract(meta)
    if contract != embedding_contract(loaded["checkpoint"]["visual_contract"]):
        raise ValueError("Image feature contract differs from the reference.")
    sensors = build_sensor_features(frame, RAW_SENSOR_COLUMNS).reset_index(drop=True)
    sensors.columns = [f"sensor_{name}" for name in sensors.columns]
    features = pd.concat(
        [
            sensors,
            pd.DataFrame(
                vectors, columns=[f"{mm.EMBED_PREFIX}{i:04d}" for i in range(vectors.shape[1])]
            ),
        ],
        axis=1,
    )
    bundle = loaded["bundle"]
    if list(features.columns) != bundle["columns"]:
        raise ValueError("Reference tree feature order differs.")
    files = {
        **base["provenance"]["files"],
        str(args.recipe.resolve()): sha256(args.recipe),
        str(args.train_csv.resolve()): sha256(args.train_csv),
        **{
            str((args.train_embeddings / name).resolve()): value
            for name, value in cache_hashes.items()
        },
    }
    manifest = {
        "kind": KIND,
        "format_version": 1,
        "scope": "original_train_only_no_official_validation_or_test_access",
        "outer_split": "GroupKFold(n_splits=3)",
        "inner_split": "first StratifiedGroupKFold(min(5,groups), shuffle=True, seed=42) split",
        "epoch_selection": "inner_holdout_then_fresh_outer_train_refit_same_epoch_count",
        "rule": midpoint.RULE,
        "config": config,
        "source_sha256": source_hashes(),
        "input_sha256": files,
        "runtime": midpoint.runtime_signature(args.device),
        "plan": plan,
        "reference_validation": base["metrics"],
        "reference_note": "Different evaluation data; do not rank directly against CV scores.",
    }
    tree = bundle["pipelines"][bundle["models"].index("extra_trees")]
    return manifest, frame, vectors, features, tree, contract


def check_unchanged(manifest, device):
    midpoint.check_files(manifest["input_sha256"])
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Code changed during CV; use the original code or a new output directory.")
    if midpoint.runtime_signature(device) != manifest["runtime"]:
        raise ValueError("Runtime changed during CV.")


def read_fold(directory, part, labels):
    saved = read_json(directory / f"fold_{part['fold']}.json")
    attempt = directory / saved["attempt"]
    if attempt.parent != directory or not attempt.name.startswith(f"fold_{part['fold']}_attempt_"):
        raise ValueError("Invalid saved fold location.")
    if set(saved["sha256"]) != set(ARTIFACTS):
        raise ValueError("Incomplete saved fold artifacts.")
    if any(sha256(attempt / name) != digest for name, digest in saved["sha256"].items()):
        raise ValueError("Completed fold artifacts changed; do not reuse them.")
    result = read_json(attempt / "result.json")
    with np.load(attempt / "oof.npz", allow_pickle=False) as source:
        positions = source["positions"]
        probabilities = {name: source[name].copy() for name in training.METHODS}
    if positions.tolist() != part["holdout"] or result["fold"] != part["fold"]:
        raise ValueError("OOF rows differ from the planned fold.")
    values = np.stack(list(probabilities.values()), axis=1)
    validate_probabilities(values)
    if values.shape != (len(positions), 3, 6):
        raise ValueError("Invalid OOF shape.")
    if not np.array_equal(
        probabilities["blend_50_50"],
        midpoint.blend_probabilities(probabilities["neural"], probabilities["extra_trees"]),
    ):
        raise ValueError("Saved fold is not the fixed 50:50 blend.")
    for name, prob in probabilities.items():
        if (
            classification_metrics(labels[positions], prob.argmax(axis=1) + 1)
            != result["metrics"][name]
        ):
            raise ValueError("Saved fold metrics do not replay from OOF predictions.")
    return result, probabilities


def display(summary):
    print("fold\tmethod\tselected_epoch\tmacro_f1\t" + "\t".join(f"f1_{c}" for c in LABELS))
    for fold in summary["folds"]:
        for name in training.METHODS:
            metrics = fold["metrics"][name]
            print(
                f"{fold['fold']}\t{name}\t{fold['selected_epoch']}\t{metrics['macro_f1']:.6f}\t"
                + "\t".join(f"{metrics['class_f1'][str(c)]:.6f}" for c in LABELS)
            )
        for partition, counts in fold["counts"].items():
            missing = [c for c, count in counts["support"].items() if count == 0]
            if missing:
                print(
                    f"fold={fold['fold']} {partition}: absent classes={','.join(missing)}; "
                    "Macro F1 still averages all six classes."
                )
    print(
        "\nmethod\tfold_mean\tfold_sample_std\tpooled_oof_f1\t"
        + "\t".join(f"f1_{c}" for c in LABELS)
    )
    for name, item in summary["methods"].items():
        metrics = item["pooled_oof"]
        print(
            f"{name}\t{item['fold_macro_f1_mean']:.6f}\t{item['fold_macro_f1_std']:.6f}\t"
            f"{metrics['macro_f1']:.6f}\t"
            + "\t".join(f"{metrics['class_f1'][str(c)]:.6f}" for c in LABELS)
        )
    if "reference_validation" in summary:
        print("\nHistorical official Validation (different rows; no direct CV delta):")
        for name in training.METHODS:
            print(f"{name}\t{summary['reference_validation'][name]['macro_f1']:.6f}")
    capped = [str(f["fold"]) for f in summary["folds"] if f["inner_epoch_limit_reached"]]
    if capped:
        print("Inner best epoch reached configured maximum in fold(s): " + ", ".join(capped))
    print("Fold std describes partition variability, not a confidence interval.")
    print("Architecture previously selected on Validation/Public; this is not a new blind test.")
    print("Original 50:50 artifacts are unchanged. No submission CSV was generated.")


def run(args):
    from runtime import select_device

    directory = args.output_dir.resolve()
    if directory.exists() and not args.resume:
        raise ValueError("Output directory exists; use --resume or a new directory.")
    if args.resume and not (directory / "experiment.json").is_file():
        raise ValueError("--resume requires an existing CV experiment.json.")
    manifest, frame, vectors, features, tree, contract = load_inputs(args)
    check_unchanged(manifest, args.device)
    if args.resume:
        if read_json(directory / "experiment.json") != manifest:
            raise ValueError("Inputs, splits, code or settings changed; cannot resume this CV.")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "experiment.json", manifest)
    device = select_device(args.device)
    status = directory / "status.json"
    write_json(status, {"status": "running"})
    labels = frame["label"].to_numpy()
    oof = {name: np.full((len(frame), 6), np.nan) for name in training.METHODS}
    results = []
    try:
        for part in manifest["plan"]:
            marker = directory / f"fold_{part['fold']}.json"
            if marker.exists():
                print(f"Reusing verified complete fold {part['fold']}/3.", flush=True)
            else:
                attempt_number = 1
                while (directory / f"fold_{part['fold']}_attempt_{attempt_number:03d}").exists():
                    attempt_number += 1
                attempt = directory / f"fold_{part['fold']}_attempt_{attempt_number:03d}"
                attempt.mkdir()
                counts = part["counts"]
                print(
                    f"Starting fold {part['fold']}/3: "
                    f"fit groups={counts['fit']['experiments']}, "
                    f"holdout groups={counts['holdout']['experiments']}",
                    flush=True,
                )
                training.fit_fold(
                    frame,
                    vectors,
                    features,
                    tree,
                    manifest["config"],
                    part,
                    attempt,
                    device,
                    contract,
                )
                check_unchanged(manifest, args.device)
                write_json(
                    marker,
                    {
                        "attempt": attempt.name,
                        "sha256": {name: sha256(attempt / name) for name in ARTIFACTS},
                    },
                )
            result, probabilities = read_fold(directory, part, labels)
            results.append(result)
            for name in oof:
                oof[name][part["holdout"]] = probabilities[name]
        if any(not np.isfinite(values).all() for values in oof.values()):
            raise ValueError("Some rows have no held-out prediction.")
        check_unchanged(manifest, args.device)
        summary = training.summarize(labels, results, oof)
        summary["reference_validation"] = manifest["reference_validation"]
        write_json(directory / "summary.json", summary)
        write_json(
            status, {"status": "complete", "summary_sha256": sha256(directory / "summary.json")}
        )
        display(summary)
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def report(args):
    status = read_json(args.output_dir / "status.json")
    if status.get("status") != "complete":
        completed = sum((args.output_dir / f"fold_{i}.json").exists() for i in range(1, 4))
        raise ValueError(
            f"CV not complete ({completed}/3 saved folds); continue run with --resume."
        )
    if sha256(args.output_dir / "summary.json") != status.get("summary_sha256"):
        raise ValueError("Saved summary changed.")
    display(read_json(args.output_dir / "summary.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--recipe", type=Path, required=True)
    execute.add_argument("--train_csv", type=Path, required=True)
    execute.add_argument("--train_embeddings", type=Path, required=True)
    execute.add_argument("--output_dir", type=Path, required=True)
    execute.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    execute.add_argument("--resume", action="store_true")
    execute.set_defaults(handler=run)
    summary = commands.add_parser("report")
    summary.add_argument("--output_dir", type=Path, required=True)
    summary.set_defaults(handler=report)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
