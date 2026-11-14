"""Train-only OOF classwise fusion; run real experiments manually on the organizer server."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom.ensemble_core import (  # noqa: E402
    blend_scores,
    classification_metrics,
    fit_classwise_weights,
    validate_probabilities,
    validate_weights,
)
from schema import LABELS, read_frame, validate_split  # noqa: E402
from sensor_features import uses_history  # noqa: E402

MODEL_NAMES = ("xgboost", "lightgbm", "extra_trees", "logistic", "mlp")
ENCODERS = ("resnet18", "dinov2", "convnext_tiny")
DEFAULT_MODELS = ("xgboost", "lightgbm", "extra_trees")
FORMAT_VERSION = 1
SOURCE_FILES = (
    "schema.py",
    "sensor_features.py",
    "temporal.py",
    "image_features/encoders.py",
    "image_features/extract_features.py",
    "experiments/seungbeom/classwise_ensemble.py",
    "experiments/seungbeom/ensemble_core.py",
    "experiments/seungbeom/multimodal.py",
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata() -> dict:
    versions = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "xgboost",
        "lightgbm",
        "torch",
        "torchvision",
        "transformers",
        "Pillow",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return {
        "python": sys.version,
        "os": platform.platform(),
        "packages": versions,
        "git_commit": result.stdout.strip() if result.returncode == 0 else None,
        "source_sha256": {name: digest_file(ROOT / name) for name in SOURCE_FILES},
    }


def group_splits(frame: pd.DataFrame, folds: int):
    from sklearn.model_selection import GroupKFold

    if folds < 2:
        raise ValueError("folds must be at least 2.")
    groups = frame["experiment_id"].to_numpy()
    count = min(folds, len(np.unique(groups)))
    if count < 2:
        raise ValueError("At least two independent training experiments are required.")
    splits = list(GroupKFold(n_splits=count).split(frame, groups=groups))
    coverage = np.zeros(len(frame), dtype=np.int64)
    for tr, held in splits:
        if set(groups[tr]) & set(groups[held]):
            raise ValueError("OOF train/held-out experiments overlap.")
        coverage[held] += 1
    if not np.all(coverage == 1):
        raise ValueError("Every Train row must be held out exactly once.")
    return splits


def prediction_cube(models, features: pd.DataFrame, *, stable_rows: bool = False):
    from experiments.seungbeom.multimodal import predict_proba_six, predict_proba_six_stable

    method = predict_proba_six_stable if stable_rows else predict_proba_six
    members = [method(model, features) for model in models]
    return validate_probabilities(np.stack(members, axis=1))


def comparison_metrics(probabilities, labels, weights, names):
    equal = np.full_like(weights, 1 / len(names))
    comparisons = {
        name: classification_metrics(labels, probabilities[:, i].argmax(axis=1) + 1)
        for i, name in enumerate(names)
    }
    comparisons["equal_average"] = classification_metrics(
        labels, blend_scores(probabilities, equal).argmax(axis=1) + 1
    )
    comparisons["classwise_ensemble"] = classification_metrics(
        labels, blend_scores(probabilities, weights).argmax(axis=1) + 1
    )
    return comparisons


def save_comparison(path: Path, train_metrics: dict, val_metrics: dict) -> None:
    rows = []
    for name, metrics in val_metrics.items():
        rows.append(
            {
                "method": name,
                "train_oof_f1_fit_diagnostic": train_metrics[name]["macro_f1"],
                "validation_macro_f1": metrics["macro_f1"],
                **{f"f1_{label}": metrics["class_f1"][str(label)] for label in LABELS},
                "used_for_prediction": name == "classwise_ensemble",
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _validate_train_args(args) -> None:
    if len(args.models) < 2 or len(set(args.models)) != len(args.models):
        raise ValueError("Specify at least two distinct models.")
    if args.folds < 2 or args.pca_dim < 0:
        raise ValueError("folds must be >=2 and pca_dim must be >=0.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32).")
    if not np.isfinite(args.regularization) or args.regularization < 0:
        raise ValueError("regularization must be finite and nonnegative.")
    if not 1 <= args.max_rounds <= 20 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid search rounds, batch size, or worker count.")
    if (args.train_embeddings is None) != (args.val_embeddings is None):
        raise ValueError("Supply both train_embeddings and val_embeddings, or neither.")


def train(args) -> None:
    import joblib

    from experiments.seungbeom import multimodal as mm

    _validate_train_args(args)
    train_csv = args.data_dir / "train" / "train.csv"
    val_csv = args.data_dir / "validation" / "validation.csv"
    train_frame = read_frame(train_csv, labeled=True)
    val_frame = read_frame(val_csv, labeled=True)
    validate_split(train_frame, val_frame)
    splits = group_splits(train_frame, args.folds)
    spec = mm.sensor_spec(args.sensor_config)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    run = args.run_dir
    write_json(run / "status.json", {"status": "training"})
    write_json(run / "environment.json", environment_metadata())
    write_json(
        run / "config.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in ("handler",)
            },
            "sensor_spec": spec,
            "effective_folds": len(splits),
            "weight_fit_split": "train_group_oof_only",
            "base_fit_split": "train_only",
            "validation_used_for_fit": False,
        },
    )
    if uses_history(spec["sensor_features"], spec["sensor_imputation"]):
        print("History enabled: removing required past rows can change predictions.", flush=True)
    try:
        features, embedding_meta = mm.make_features(
            train_frame,
            train_csv,
            train_csv.parent,
            spec,
            embeddings_dir=args.train_embeddings,
            encoder=args.encoder,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            work_dir=run / "embeddings" / "train",
        )
        embedding_spec = mm.embedding_info(embedding_meta)
        columns = list(features.columns)
        n_emb = sum(name.startswith(mm.EMBED_PREFIX) for name in columns)
        if n_emb == 0:
            raise ValueError("This experiment requires image embeddings and sensor features.")
        labels = train_frame["label"].to_numpy()
        oof = np.full((len(features), len(args.models), len(LABELS)), np.nan)
        fold_ids = np.full(len(features), -1, dtype=np.int64)
        for fold, (_, held) in enumerate(splits):
            fold_ids[held] = fold
        fold_scores = []
        for index, name in enumerate(args.models):
            for fold, (tr, held) in enumerate(splits):
                print(f"OOF {name}: fold {fold + 1}/{len(splits)}", flush=True)
                pipeline = mm.build_pipeline(
                    name, columns, mm.effective_pca_dim(args.pca_dim, len(tr), n_emb), args.seed
                )
                pipeline.fit(features.iloc[tr], labels[tr])
                oof[held, index] = mm.predict_proba_six(pipeline, features.iloc[held])
                score = classification_metrics(labels[held], oof[held, index].argmax(axis=1) + 1)
                fold_scores.append({"model": name, "fold": fold, "macro_f1": score["macro_f1"]})
                print(f"  macro_f1={score['macro_f1']:.6f}", flush=True)
        validate_probabilities(oof)
        weights, search_report = fit_classwise_weights(
            oof,
            labels,
            seed=args.seed,
            regularization=args.regularization,
            max_rounds=args.max_rounds,
        )
        train_metrics = comparison_metrics(oof, labels, weights, args.models)
        write_json(run / "weight_search.json", search_report)
        write_json(
            run / "class_weights.json",
            {
                "models": args.models,
                "classes": list(LABELS),
                "weights": weights.tolist(),
                "fit_split": "train_group_oof_only",
                "selected_strategy": search_report["selected_strategy"],
            },
        )
        np.savez_compressed(
            run / "train_oof.npz",
            probabilities=oof,
            labels=labels,
            fold=fold_ids,
            sample_ids=train_frame["sample_id"].to_numpy(dtype=str),
            experiment_ids=train_frame["experiment_id"].to_numpy(dtype=str),
            model_names=np.asarray(args.models),
        )
        # Neither labels nor predictions from Validation enter weight selection.
        # The final classifiers use full Train with fresh Train-fitted preprocessing.
        pipelines = []
        for name in args.models:
            print(f"Refit {name} on full Train", flush=True)
            pipeline = mm.build_pipeline(
                name, columns, mm.effective_pca_dim(args.pca_dim, len(features), n_emb), args.seed
            )
            pipeline.fit(features, labels)
            pipelines.append(pipeline)
        bundle = {
            "kind": "seungbeom_classwise_multimodal",
            "format_version": FORMAT_VERSION,
            "label_values": list(LABELS),
            "models": args.models,
            "pipelines": pipelines,
            "weights": weights,
            "sensor_spec": spec,
            "columns": columns,
            "encoder": args.encoder,
            "embedding": embedding_spec,
            "seed": args.seed,
            "refit_train_val": False,
            "weight_fit_split": "train_group_oof_only",
            "selected_strategy": search_report["selected_strategy"],
        }
        with (run / "ensemble.joblib").open("xb") as handle:
            joblib.dump(bundle, handle)
        # Validation is evaluated only after the inference bundle has been fixed.
        val_features, _ = mm.make_features(
            val_frame,
            val_csv,
            val_csv.parent,
            spec,
            embeddings_dir=args.val_embeddings,
            encoder=args.encoder,
            embedding_spec=embedding_spec,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            work_dir=run / "embeddings" / "validation",
        )
        if list(val_features.columns) != columns:
            raise ValueError("Validation feature columns differ from Train.")
        val_prob = prediction_cube(pipelines, val_features, stable_rows=True)
        val_labels = val_frame["label"].to_numpy()
        val_metrics = comparison_metrics(val_prob, val_labels, weights, args.models)
        write_json(
            run / "metrics.json",
            {
                "train_oof": train_metrics,
                "validation": val_metrics,
                "fold_scores": fold_scores,
                "oof_note": "Weights were fit on Train OOF; optimized OOF is not independent.",
                "selection_note": "Validation scores do not change weights or the saved predictor.",
            },
        )
        save_comparison(run / "comparison.tsv", train_metrics, val_metrics)
        np.savez_compressed(
            run / "validation_predictions.npz",
            probabilities=val_prob,
            labels=val_labels,
            sample_ids=val_frame["sample_id"].to_numpy(dtype=str),
            model_names=np.asarray(args.models),
        )
        write_json(
            run / "split.json",
            {
                "train_csv_sha256": digest_file(train_csv),
                "validation_csv_sha256": digest_file(val_csv),
                "train_rows": len(train_frame),
                "validation_rows": len(val_frame),
                "train_experiments": int(train_frame["experiment_id"].nunique()),
                "validation_experiments": int(val_frame["experiment_id"].nunique()),
                "group_column": "experiment_id",
                "folds": len(splits),
            },
        )
        write_json(run / "status.json", {"status": "complete"})
        report(argparse.Namespace(run_dir=run))
    except BaseException as error:
        write_json(run / "status.json", {"status": "failed", "error_type": type(error).__name__})
        raise


def load_bundle(path: Path) -> dict:
    import joblib

    from experiments.seungbeom import multimodal as mm

    bundle = joblib.load(path)
    if (
        bundle.get("kind") != "seungbeom_classwise_multimodal"
        or bundle.get("format_version") != FORMAT_VERSION
        or bundle.get("label_values") != list(LABELS)
    ):
        raise ValueError("Unsupported Seungbeom ensemble bundle.")
    if len(bundle["models"]) != len(bundle["pipelines"]) or len(set(bundle["models"])) != len(
        bundle["models"]
    ):
        raise ValueError("Invalid ensemble model ordering.")
    validate_weights(bundle["weights"], len(bundle["models"]))
    mm.check_versions(bundle["sensor_spec"])
    if bundle.get("refit_train_val") or bundle.get("weight_fit_split") != "train_group_oof_only":
        raise ValueError("This bundle does not have the expected Train-only fitting contract.")
    return bundle


def predict(args) -> None:
    from experiments.seungbeom import multimodal as mm

    if args.output.exists():
        raise ValueError("Output exists; choose a new path.")
    bundle = load_bundle(args.checkpoint)
    frame = read_frame(args.input)
    # No cached embeddings in this command: blank/modified images must affect
    # features and only the manifest's rows are supplied to the frozen encoder.
    features, _ = mm.make_features(
        frame,
        args.input,
        args.image_root,
        bundle["sensor_spec"],
        encoder=bundle["encoder"],
        embedding_spec=bundle["embedding"],
        device=args.device,
        batch_size=1,
        workers=0,
    )
    if list(features.columns) != bundle["columns"]:
        raise ValueError("Input feature columns differ from training.")
    probabilities = prediction_cube(bundle["pipelines"], features, stable_rows=True)
    labels = blend_scores(probabilities, bundle["weights"]).argmax(axis=1) + 1
    ids = frame["sample_id"].tolist()
    if args.sample_submission is not None:
        template = pd.read_csv(args.sample_submission, dtype=str, keep_default_na=False)
        template_ids = template["sample_id"].tolist()
        if (
            not template["sample_id"].is_unique
            or len(template_ids) != len(ids)
            or set(template_ids) != set(ids)
        ):
            raise ValueError("sample_submission IDs must match the supplied input exactly.")
        labels = pd.Series(labels, index=ids).loc[template_ids].to_numpy()
        ids = template_ids
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"sample_id": ids, "label": labels}).to_csv(handle, index=False)
    print("Prediction complete: sample_id,label CSV written in the requested order.")


def report(args) -> None:
    status = json.loads((args.run_dir / "status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("Run is not complete; check status.json before comparing scores.")
    table = pd.read_csv(args.run_dir / "comparison.tsv", sep="\t")
    print("\nValidation comparison (Train OOF column is a weight-fitting diagnostic):")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    info = json.loads((args.run_dir / "class_weights.json").read_text())
    print("\nFixed class weights learned from Train OOF; rows=models, columns=labels:")
    print(
        pd.DataFrame(info["weights"], index=info["models"], columns=info["classes"]).to_string(
            float_format=lambda value: f"{value:.3f}"
        )
    )
    print("Selected by Train OOF:", info["selected_strategy"])
    if getattr(args, "class_details", False):
        metrics = json.loads((args.run_dir / "metrics.json").read_text())["validation"]
        rows = []
        for name, values in metrics.items():
            for label in LABELS:
                key = str(label)
                rows.append(
                    {
                        "method": name,
                        "class": label,
                        "precision": values["class_precision"][key],
                        "recall": values["class_recall"][key],
                        "f1": values["class_f1"][key],
                        "support": values["support"][key],
                    }
                )
        print("\nValidation class details:")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("Classwise scores are not calibrated probabilities. Test was not used for fitting.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train grouped OOF models and a fixed ensemble")
    training.add_argument("--data_dir", type=Path, required=True)
    training.add_argument("--run_dir", type=Path, required=True)
    training.add_argument("--sensor_config", type=Path)
    training.add_argument("--encoder", choices=ENCODERS, default="resnet18")
    training.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(DEFAULT_MODELS))
    training.add_argument("--train_embeddings", type=Path)
    training.add_argument("--val_embeddings", type=Path)
    training.add_argument("--folds", type=int, default=5)
    training.add_argument("--pca_dim", type=int, default=32)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--regularization", type=float, default=0.001)
    training.add_argument("--max_rounds", type=int, default=3)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    training.add_argument("--batch_size", type=int, default=1)
    training.add_argument("--workers", type=int, default=0)
    training.set_defaults(handler=train)
    inference = commands.add_parser("predict", help="Infer from saved models and supplied images")
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--image_root", type=Path, required=True)
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--sample_submission", type=Path)
    inference.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    inference.set_defaults(handler=predict)
    summary = commands.add_parser(
        "report", help="Print existing Validation scores and fixed weights"
    )
    summary.add_argument("--run_dir", type=Path, required=True)
    summary.add_argument("--class_details", action="store_true")
    summary.set_defaults(handler=report)
    return result


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
