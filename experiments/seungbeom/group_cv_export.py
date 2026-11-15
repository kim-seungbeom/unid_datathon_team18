"""Compare and export the fixed equal average of three completed CV ExtraTrees.

No fitting or weight search. Predict reads only supplied input images/rows and
saved models/metadata; it does not open Train, Validation or embedding caches.
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
from experiments.seungbeom import blend_ratios, group_cv  # noqa: E402
from experiments.seungbeom import group_cv_training as training  # noqa: E402
from experiments.seungbeom import midpoint_blend as midpoint  # noqa: E402
from experiments.seungbeom import multimodal as mm  # noqa: E402
from experiments.seungbeom.ensemble_core import validate_probabilities  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    checked_score,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.neural_data import embedding_contract  # noqa: E402
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from schema import LABELS, read_frame  # noqa: E402

KIND = "seungbeom_extra_trees_cv3"
RULE = {"method": "probability_mean", "folds": [1, 2, 3], "weights": [1 / 3] * 3}
SELF = "experiments/seungbeom/group_cv_export.py"
CANDIDATE = "extra_trees_cv3"


def source_hashes():
    return {**group_cv.source_hashes(), SELF: sha256(ROOT / SELF)}


def probability_mean(values):
    if len(values) != 3:
        raise ValueError("Exactly three fold probability arrays are required.")
    return validate_probabilities(np.stack(values, axis=1)).mean(axis=1)


def load_cv(directory, base_path, base, baseline):
    """Verify CV metadata and model hashes, without following training-data paths."""
    import joblib
    from sklearn.base import clone

    directory = directory.resolve()
    manifest = read_json(directory / "experiment.json")
    status = read_json(directory / "status.json")
    if status.get("status") != "complete":
        raise ValueError("All three CV folds must be complete before export.")
    if sha256(directory / "summary.json") != status.get("summary_sha256"):
        raise ValueError("CV summary hash differs from completed status.")
    if (
        manifest.get("kind") != group_cv.KIND
        or manifest.get("format_version") != 1
        or manifest.get("scope") != "original_train_only_no_official_validation_or_test_access"
        or manifest.get("outer_split") != "GroupKFold(n_splits=3)"
        or manifest.get("config") != baseline["config"]
        or manifest.get("rule") != midpoint.RULE
        or manifest.get("source_sha256") != group_cv.source_hashes()
        or manifest.get("runtime") != base["runtime_signature"]
        or manifest.get("reference_validation") != base["metrics"]
    ):
        raise ValueError("CV source, configuration or runtime differs from the reference.")
    expected = {**base["provenance"]["files"], str(base_path.resolve()): sha256(base_path)}
    if any(manifest["input_sha256"].get(path) != value for path, value in expected.items()):
        raise ValueError("CV was not trained from this exact original 50:50 recipe.")
    summary = read_json(directory / "summary.json")
    if [part["fold"] for part in manifest["plan"]] != [1, 2, 3]:
        raise ValueError("Expected exactly folds 1, 2 and 3.")
    if [part["fold"] for part in summary["folds"]] != [1, 2, 3]:
        raise ValueError("CV summary has different folds.")
    files = {
        str(directory / name): sha256(directory / name)
        for name in ("experiment.json", "status.json", "summary.json")
    }
    bundle = baseline["bundle"]
    reference = bundle["pipelines"][bundle["models"].index("extra_trees")]
    signature = joblib.hash(clone(reference))
    paths, pipelines = [], []
    for part, result in zip(manifest["plan"], summary["folds"], strict=True):
        number = part["fold"]
        marker = directory / f"fold_{number}.json"
        saved = read_json(marker)
        attempt = (directory / saved["attempt"]).resolve()
        if attempt.parent != directory or not attempt.name.startswith(f"fold_{number}_attempt_"):
            raise ValueError("CV fold path escapes its experiment directory.")
        files[str(marker)] = sha256(marker)
        for name in ("result.json", "extra_trees.joblib"):
            path = attempt / name
            if sha256(path) != saved["sha256"][name]:
                raise ValueError("CV fold model or result changed after training.")
            files[str(path)] = saved["sha256"][name]
        if read_json(attempt / "result.json") != result:
            raise ValueError("Fold result differs from the completed CV summary.")
        path = attempt / "extra_trees.joblib"
        pipeline = joblib.load(path)
        if (
            joblib.hash(clone(pipeline)) != signature
            or list(pipeline.feature_names_in_) != bundle["columns"]
        ):
            raise ValueError("Fold tree configuration or feature order differs from the reference.")
        paths.append(str(path))
        pipelines.append(pipeline)
    midpoint.check_files(files)
    return pipelines, paths, files


def extract_features(frame, image_root, spec, device):
    from runtime import seed_everything

    seed_everything(42)
    with midpoint.precision_context(spec["image_precision"]):
        features, meta = mm.make_features(
            frame.drop(columns="label", errors="ignore"),
            None,
            image_root,
            spec["sensor_spec"],
            encoder="resnet18",
            embedding_spec=spec["embedding"],
            device=device,
            batch_size=1,
            workers=0,
        )
    if list(features.columns) != spec["columns"] or embedding_contract(meta) != embedding_contract(
        spec["embedding"]
    ):
        raise ValueError("Live input feature contract differs from the fitted models.")
    return features


def check_unchanged(files, sources, runtime, device):
    midpoint.check_files(files)
    if source_hashes() != sources or midpoint.runtime_signature(device) != runtime:
        raise ValueError("Source code or runtime changed; repeat comparison in a new directory.")


def show_metrics(metrics):
    print("method\tvalidation_macro_f1\t" + "\t".join(f"f1_{i}" for i in LABELS))
    for name, item in metrics.items():
        print(
            f"{name}\t{item['macro_f1']:.6f}\t"
            + "\t".join(f"{item['class_f1'][str(i)]:.6f}" for i in LABELS)
        )
    print("extra_trees_cv3 is an equal probability mean, distinct from pooled OOF F1.")


def compare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    base = read_json(args.recipe)
    digest = sha256(args.input)
    if (
        digest != base["provenance"]["validation_csv_sha256"]
        or digest == base["provenance"]["train_csv_sha256"]
    ):
        raise ValueError("Compare requires the original official Validation CSV.")
    baseline = blend_ratios.load_base(base, args.device)
    pipelines, paths, artifact_files = load_cv(args.cv_dir, args.recipe, base, baseline)
    sources = source_hashes()
    # Only these artifacts are needed by predict. Dataset paths in CV metadata
    # and the comparison input hash below are recorded but never opened there.
    files = {
        **artifact_files,
        **base["provenance"]["files"],
        str(args.recipe.resolve()): sha256(args.recipe),
        str(args.input.resolve()): digest,
    }
    spec = {
        "columns": baseline["bundle"]["columns"],
        "sensor_spec": baseline["bundle"]["sensor_spec"],
        "embedding": baseline["bundle"]["embedding"],
        "image_precision": base["image_precision"],
    }
    frame = read_frame(args.input, labeled=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        features = extract_features(frame, args.image_root, spec, args.device)
        print(
            "Image features ready; predicting original members and three fold trees...", flush=True
        )
        image_columns = [
            f"{mm.EMBED_PREFIX}{i:04d}" for i in range(baseline["config"]["embedding_dim"])
        ]
        from runtime import select_device

        target = select_device(args.device)
        neural_prob = training.neural_probabilities(
            baseline["model"].to(target),
            baseline["preprocessor"],
            frame,
            features[image_columns].to_numpy(dtype=np.float32),
            target,
        )
        bundle = baseline["bundle"]
        original_tree = bundle["pipelines"][bundle["models"].index("extra_trees")]
        tree_prob = mm.predict_proba_six_stable(original_tree, features)
        probabilities = {
            "neural": neural_prob,
            "extra_trees": tree_prob,
            "blend_50_50": midpoint.blend_probabilities(neural_prob, tree_prob),
        }
        folds = [mm.predict_proba_six_stable(model, features) for model in pipelines]
        probabilities.update({f"extra_trees_fold{i}": prob for i, prob in enumerate(folds, 1)})
        probabilities[CANDIDATE] = probability_mean(folds)
        metrics = {
            name: classification_metrics(frame["label"], prob.argmax(axis=1) + 1)
            for name, prob in probabilities.items()
        }
        replay = {name: metrics[name] == base["metrics"][name] for name in midpoint.METHODS}
        write_json(
            args.output_dir / "validation_metrics.json", {"metrics": metrics, "replay": replay}
        )
        if not all(replay.values()):
            raise ValueError(
                "Original Validation scores do not replay; inspect before Test export."
            )
        check_unchanged(files, sources, base["runtime_signature"], args.device)
        recipe = {
            "kind": KIND,
            "format_version": 1,
            "status": "complete",
            "rule": RULE,
            "label_values": list(LABELS),
            "models": paths,
            "artifact_sha256": artifact_files,
            "source_sha256": sources,
            "runtime": base["runtime_signature"],
            "features": spec,
            "metrics": metrics,
            "replay_matches": replay,
            "validation_csv_sha256": digest,
        }
        write_json(args.output_dir / "recipe.json", recipe)
        write_json(status, {"status": "complete"})
        show_metrics(metrics)
        print("Original scores replayed. Fixed CV3 recipe saved; no fitting or Test access.")
        return recipe
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def read_recipe(path):
    recipe = read_json(path)
    if (
        recipe.get("kind") != KIND
        or recipe.get("format_version") != 1
        or recipe.get("status") != "complete"
        or recipe.get("rule") != RULE
        or recipe.get("label_values") != list(LABELS)
        or recipe.get("replay_matches") != dict.fromkeys(midpoint.METHODS, True)
        or len(recipe.get("models", [])) != 3
        or len(set(recipe["models"])) != 3
        or any(path not in recipe["artifact_sha256"] for path in recipe["models"])
    ):
        raise ValueError("Use a completed fixed CV3 Validation comparison recipe.")
    for metrics in recipe["metrics"].values():
        checked_score(metrics)
    return recipe


def predict(args):
    import joblib

    from runtime import seed_everything

    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    recipe = read_recipe(args.recipe)
    seed_everything(42)
    files = {
        **recipe["artifact_sha256"],
        str(args.recipe.resolve()): sha256(args.recipe),
        str(args.input.resolve()): sha256(args.input),
    }
    if args.sample_submission is not None:
        files[str(args.sample_submission.resolve())] = sha256(args.sample_submission)
    # Check hashes BEFORE deserialization. No original baseline or training cache
    # is loaded: these are the three saved fold pipelines only.
    check_unchanged(files, recipe["source_sha256"], recipe["runtime"], args.device)
    models = [joblib.load(path) for path in recipe["models"]]
    if any(list(model.feature_names_in_) != recipe["features"]["columns"] for model in models):
        raise ValueError("Fold feature order differs from the compared recipe.")
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    ordered = template_ids(args.sample_submission, ids) if args.sample_submission else ids
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        features = extract_features(frame, args.image_root, recipe["features"], args.device)
        print("Image features ready; averaging three ExtraTrees probabilities...", flush=True)
        values = probability_mean(
            [mm.predict_proba_six_stable(model, features) for model in models]
        )
        labels = (
            pd.Series(values.argmax(axis=1) + 1, index=ids).loc[ordered].to_numpy(dtype=np.int64)
        )
        check_unchanged(files, recipe["source_sha256"], recipe["runtime"], args.device)
        output = args.output_dir / "submission_extratrees_cv3.csv"
        with output.open("x", encoding="utf-8", newline="") as handle:
            pd.DataFrame({"sample_id": ordered, "label": labels}).to_csv(handle, index=False)
        result = check_csv(output, ordered)
        write_json(
            output.with_suffix(".manifest.json"),
            {
                "kind": KIND,
                "rule": RULE,
                "recipe": recipe,
                "input_sha256": {str(args.input.resolve()): files[str(args.input.resolve())]},
                "template_sha256": files.get(str(args.sample_submission.resolve()))
                if args.sample_submission
                else None,
                "output": result,
            },
        )
        write_json(status, {"status": "complete", "output": result})
        print("Submission CSV saved:", output)
        return result
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("compare")
    export = commands.add_parser("predict")
    for command in (check, export):
        for name in ("recipe", "input", "image_root", "output_dir"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    check.add_argument("--cv_dir", type=Path, required=True)
    check.set_defaults(handler=compare)
    export.add_argument("--sample_submission", type=Path)
    export.set_defaults(handler=predict)
    report = commands.add_parser("report")
    report.add_argument("--recipe", type=Path, required=True)
    report.set_defaults(handler=lambda args: show_metrics(read_recipe(args.recipe)["metrics"]))
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
