"""Train-only ExtraTrees leaf 2/5/10 ablation with the saved frozen 50:50 blend.

User-run organizer-server experiment. No neural training, Test fitting, ratio
search, or modification of previous recipes. Optional CV evaluates trees only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import classification_metrics  # noqa: E402
from experiments.seungbeom import frozen_tree_blend as frozen_blend  # noqa: E402
from experiments.seungbeom import midpoint_blend as midpoint  # noqa: E402
from experiments.seungbeom import multimodal as mm  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    checked_score,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.neural_data import embedding_contract  # noqa: E402
from experiments.seungbeom.replay_precision import precision_context, runtime_signature  # noqa: E402
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from schema import LABELS, RAW_SENSOR_COLUMNS, read_frame, validate_split  # noqa: E402

KIND = "seungbeom_extratrees_leaf_sweep"
LEAVES = (2, 5, 10)
PARAMETER = "classifier__estimator__min_samples_leaf"
CACHE_FILES = ("features.npy", "sample_ids.csv", "metadata.json")
SELF = "experiments/seungbeom/extratrees_leaf_sweep.py"
METHODS = (
    "frozen",
    *(name for leaf in LEAVES for name in (f"extra_trees_leaf{leaf}", f"blend_leaf{leaf}_50_50")),
)
REPLAY_NAMES = ("frozen", "extra_trees", "frozen_tree_50_50", "refitted_leaf2_probabilities")


def source_hashes():
    return {**frozen_blend.source_hashes(), SELF: sha256(ROOT / SELF)}


def tree_template(legacy):
    from sklearn.ensemble import ExtraTreesClassifier

    bundle = legacy["bundle"]
    model = bundle["pipelines"][bundle["models"].index("extra_trees")]
    estimator = model.named_steps["classifier"].estimator
    if (
        not isinstance(estimator, ExtraTreesClassifier)
        or estimator.min_samples_leaf != 2
        or estimator.random_state != 42
        or bundle.get("refit_train_val") is not False
        or bundle["sensor_spec"] != mm.sensor_spec()
        or legacy["config"]["sensor_features"] != list(RAW_SENSOR_COLUMNS)
    ):
        raise ValueError("Use the original Train-only seed42/raw5 ExtraTrees with leaf=2.")
    return model


def load_reference(reference, device):
    if reference["source_sha256"] != frozen_blend.source_hashes():
        raise ValueError("Reference source changed; restore the recorded source version.")
    midpoint.check_files(reference["frozen"]["files"])
    legacy, frozen, record = frozen_blend.load_members(
        reference["base_recipe"], Path(reference["frozen"]["run_dir"]), device
    )
    if record != reference["frozen"]:
        raise ValueError("Frozen model differs from the reference recipe.")
    tree_template(legacy)
    return legacy, frozen


def check_unchanged(reference, sources, files, device):
    midpoint.check_files(reference["base_recipe"]["provenance"]["files"])
    midpoint.check_files(reference["frozen"]["files"])
    midpoint.check_files(files)
    if (
        source_hashes() != sources
        or runtime_signature(device) != reference["base_recipe"]["runtime_signature"]
    ):
        raise ValueError("Source/runtime changed; use the recorded environment and code.")


def feature_matrix(frame, bundle, *, cache=None, image_root=None, device="cuda", policy="fp32"):
    with precision_context(policy):
        features, meta = mm.make_features(
            frame.drop(columns="label", errors="ignore"),
            None,
            image_root,
            bundle["sensor_spec"],
            embeddings_dir=cache,
            encoder=bundle["encoder"],
            embedding_spec=bundle["embedding"],
            device=device,
            batch_size=1,
            workers=0,
        )
    if list(features.columns) != bundle["columns"] or embedding_contract(
        meta
    ) != embedding_contract(bundle["embedding"]):
        raise ValueError("Image/sensor feature contract differs from the reference.")
    return features


def fit_candidate(template, features, labels, leaf):
    """Clone discards all learned state; the sole parameter change is leaf size."""
    from sklearn.base import clone

    if leaf not in LEAVES:
        raise ValueError("Use leaf 2, 5 or 10.")
    model = clone(template).set_params(**{PARAMETER: leaf})
    model.fit(features, labels)
    return model


def parameter_fingerprint(model, *, normalize_leaf=False):
    import joblib
    from sklearn.base import clone

    copied = clone(model)
    if normalize_leaf:
        copied.set_params(**{PARAMETER: 2})
    return joblib.hash(copied)


def load_inputs(args, reference, legacy):
    files = {
        str(args.recipe.resolve()): sha256(args.recipe),
        str((args.recipe.parent / "status.json").resolve()): sha256(
            args.recipe.parent / "status.json"
        ),
    }
    for split in ("train", "validation"):
        path = getattr(args, f"{split}_csv")
        digest = sha256(path)
        if digest != reference["base_recipe"]["provenance"][f"{split}_csv_sha256"]:
            raise ValueError(f"Use the exact recorded {split} CSV, not a different split or Test.")
        files[str(path.resolve())] = digest
    cache_hashes = {name: sha256(args.train_embeddings / name) for name in CACHE_FILES}
    if cache_hashes != legacy["identity"]["input_provenance"]["train"]:
        raise ValueError("Use the original Train embedding cache.")
    files.update(
        {
            str((args.train_embeddings / name).resolve()): value
            for name, value in cache_hashes.items()
        }
    )
    train_frame = read_frame(args.train_csv, labeled=True)
    val_frame = read_frame(args.validation_csv, labeled=True)
    validate_split(train_frame, val_frame)
    if set(train_frame["label"]) != set(LABELS):
        raise ValueError("The original Train must contain all six classes.")
    features = feature_matrix(
        train_frame, legacy["bundle"], cache=args.train_embeddings, device=args.device
    )
    return train_frame, val_frame, features, files


def tree_group_cv(frame, features, template):
    """Same GroupKFold(3) outer split as group_cv.py; no full-Train neural in OOF."""
    from sklearn.model_selection import GroupKFold

    groups = frame["experiment_id"]
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any() or groups.nunique() < 3:
        raise ValueError("Three-fold CV needs at least three nonempty experiment IDs.")
    labels = frame["label"].to_numpy()
    oof = {leaf: np.zeros((len(frame), len(LABELS))) for leaf in LEAVES}
    coverage = np.zeros(len(frame), dtype=int)
    folds = []
    for number, (fit, holdout) in enumerate(GroupKFold(3).split(frame, groups=groups), 1):
        validate_split(frame.iloc[fit], frame.iloc[holdout])
        coverage[holdout] += 1
        metrics = {}
        for leaf in LEAVES:
            print(f"Tree-only group CV: fold={number}/3 leaf={leaf}", flush=True)
            model = fit_candidate(template, features.iloc[fit], labels[fit], leaf)
            prob = mm.predict_proba_six_stable(model, features.iloc[holdout])
            oof[leaf][holdout] = prob
            metrics[str(leaf)] = classification_metrics(labels[holdout], prob.argmax(1) + 1)
        folds.append(
            {"fold": number, "fit": fit.tolist(), "holdout": holdout.tolist(), "metrics": metrics}
        )
    if not np.all(coverage == 1):
        raise ValueError("Every Train row must be held out exactly once.")
    summary = {}
    for leaf in LEAVES:
        scores = [part["metrics"][str(leaf)]["macro_f1"] for part in folds]
        summary[str(leaf)] = {
            "fold_mean": float(np.mean(scores)),
            "fold_sample_std": float(np.std(scores, ddof=1)),
            "pooled_oof": classification_metrics(labels, oof[leaf].argmax(1) + 1),
        }
    return {
        "split": "GroupKFold(n_splits=3)",
        "scope": "train_only_trees_only",
        "folds": folds,
        "summary": summary,
    }


def infer(frame, image_root, reference, legacy, frozen, models, device):
    current = frame.drop(columns="label", errors="ignore").copy()
    print("Predicting saved frozen control once (FP32)...", flush=True)
    neural = frozen_blend.frozen_probabilities(current, image_root, frozen, device)
    print("Extracting current image vectors once for all tree candidates...", flush=True)
    features = feature_matrix(
        current,
        legacy["bundle"],
        image_root=image_root,
        device=device,
        policy=reference["base_recipe"]["image_precision"],
    )
    probabilities = {"frozen": neural}
    for leaf, model in models.items():
        print(f"Predicting ExtraTrees leaf={leaf} and fixed 50:50...", flush=True)
        prob = mm.predict_proba_six_stable(model, features)
        probabilities[f"extra_trees_leaf{leaf}"] = prob
        probabilities[f"blend_leaf{leaf}_50_50"] = midpoint.blend_probabilities(neural, prob)
    return probabilities, features


def display(metrics, cv=None, *, class_details=False):
    baseline = checked_score(metrics["blend_leaf2_50_50"])
    print(
        "method\tvalidation_macro_f1\tdelta_vs_leaf2_blend\t" + "\t".join(f"f1_{c}" for c in LABELS)
    )
    for name in METHODS:
        m = metrics[name]
        print(
            f"{name}\t{m['macro_f1']:.6f}\t{m['macro_f1'] - baseline:+.6f}\t"
            + "\t".join(f"{m['class_f1'][str(c)]:.6f}" for c in LABELS)
        )
    if class_details:
        print("\nmethod\tclass\tprecision\trecall\tf1\tsupport")
        for name in METHODS:
            m = metrics[name]
            for c in LABELS:
                k = str(c)
                print(
                    f"{name}\t{c}\t{m['class_precision'][k]:.6f}\t{m['class_recall'][k]:.6f}\t"
                    f"{m['class_f1'][k]:.6f}\t{m['support'][k]}"
                )
    if cv:
        print("\nTrain-only tree GroupKFold(3); not a frozen+tree ensemble score.")
        print("fold\tleaf\tmacro_f1")
        for part in cv["folds"]:
            for leaf in LEAVES:
                print(f"{part['fold']}\t{leaf}\t{part['metrics'][str(leaf)]['macro_f1']:.6f}")
        print("leaf\tfold_mean\tfold_sample_std\tpooled_oof_f1")
        for leaf in LEAVES:
            m = cv["summary"][str(leaf)]
            print(
                f"{leaf}\t{m['fold_mean']:.6f}\t{m['fold_sample_std']:.6f}\t"
                f"{m['pooled_oof']['macro_f1']:.6f}"
            )
        print("Fold variability is not a confidence interval; CV and official Validation differ.")


def train(args):
    import joblib

    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    reference = frozen_blend.read_comparison(args.recipe)
    legacy, frozen = load_reference(reference, args.device)
    train_frame, val_frame, features, files = load_inputs(args, reference, legacy)
    sources = source_hashes()
    check_unchanged(reference, sources, files, args.device)
    template = tree_template(legacy)
    parameter_hash = parameter_fingerprint(template)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        models, model_records = {}, {}
        for leaf in LEAVES:
            started = time.perf_counter()
            print(f"Fitting leaf={leaf} on Train only; all other parameters fixed...", flush=True)
            model = fit_candidate(template, features, train_frame["label"].to_numpy(), leaf)
            if parameter_fingerprint(model, normalize_leaf=True) != parameter_hash:
                raise ValueError("A parameter other than min_samples_leaf changed.")
            path = args.output_dir / f"leaf{leaf}.joblib"
            joblib.dump(model, path)
            models[leaf] = joblib.load(path)
            model_records[str(leaf)] = {
                "file": path.name,
                "sha256": sha256(path),
                "parameters": parameter_fingerprint(model),
                "fit_seconds": time.perf_counter() - started,
            }
            print(f"leaf={leaf} fitted in {time.perf_counter() - started:.1f}s", flush=True)
        probabilities, val_features = infer(
            val_frame, args.image_root, reference, legacy, frozen, models, args.device
        )
        metrics = {
            name: classification_metrics(val_frame["label"], prob.argmax(1) + 1)
            for name, prob in probabilities.items()
        }
        original_prob = mm.predict_proba_six_stable(template, val_features)
        replay = {
            "frozen": metrics["frozen"] == reference["metrics"]["frozen"],
            "extra_trees": metrics["extra_trees_leaf2"] == reference["metrics"]["extra_trees"],
            "frozen_tree_50_50": metrics["blend_leaf2_50_50"]
            == reference["metrics"]["frozen_tree_50_50"],
            "refitted_leaf2_probabilities": bool(
                np.array_equal(original_prob, probabilities["extra_trees_leaf2"])
            ),
        }
        write_json(
            args.output_dir / "validation_metrics.json",
            {"metrics": metrics, "replay_matches": replay},
        )
        display(metrics)
        if not all(replay.values()):
            raise ValueError(
                "Baseline/leaf2 replay differs. Inspect Train/cache/runtime "
                "before interpreting leaf changes."
            )
        cv = tree_group_cv(train_frame, features, template) if args.group_cv else None
        files.update(
            {
                str((args.output_dir / r["file"]).resolve()): r["sha256"]
                for r in model_records.values()
            }
        )
        check_unchanged(reference, sources, files, args.device)
        recipe = {
            "kind": KIND,
            "format_version": 1,
            "status": "complete",
            "leaves": list(LEAVES),
            "label_values": list(LABELS),
            "rule": frozen_blend.RULE,
            "fit_splits": ["train"],
            "reference": reference,
            "models": model_records,
            "template_parameters": parameter_hash,
            "source_sha256": sources,
            "training_input_sha256": files,
            "replay_matches": replay,
            "metrics": metrics,
            "group_cv": cv,
            "selection": "three_fixed_leaf_sizes_no_automatic_winner_no_test",
        }
        write_json(args.output_dir / "recipe.json", recipe)
        write_json(
            status, {"status": "complete", "recipe_sha256": sha256(args.output_dir / "recipe.json")}
        )
        display(metrics, cv)
        print(
            "Complete: Train-only leaf ablation. Existing models/CSVs unchanged; no Test accessed."
        )
        print("Validation has informed earlier choices; it is not an independent final score.")
        return recipe
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def read_sweep(path):
    recipe = read_json(path)
    status = read_json(path.parent / "status.json")
    if (
        recipe.get("kind") != KIND
        or recipe.get("format_version") != 1
        or recipe.get("status") != "complete"
        or status.get("status") != "complete"
        or status.get("recipe_sha256") != sha256(path)
        or recipe.get("leaves") != list(LEAVES)
        or recipe.get("label_values") != list(LABELS)
        or recipe.get("rule") != frozen_blend.RULE
        or recipe.get("fit_splits") != ["train"]
        or recipe.get("replay_matches") != dict.fromkeys(REPLAY_NAMES, True)
        or set(recipe.get("models", {})) != {str(leaf) for leaf in LEAVES}
        or set(recipe.get("metrics", {})) != set(METHODS)
    ):
        raise ValueError("Use a complete, unchanged Train-only leaf sweep recipe.")
    for item in recipe["metrics"].values():
        checked_score(item)
    return recipe


def predict(args):
    import joblib

    leaves = tuple(args.leaves)
    if not leaves or len(set(leaves)) != len(leaves) or any(leaf not in LEAVES for leaf in leaves):
        raise ValueError("Choose distinct leaves from 2, 5, 10.")
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    recipe = read_sweep(args.recipe)
    if recipe["source_sha256"] != source_hashes():
        raise ValueError("Inference source differs from the saved experiment.")
    reference = recipe["reference"]
    legacy, frozen = load_reference(reference, args.device)
    paths = [args.recipe, args.recipe.parent / "status.json", args.input]
    if args.sample_submission:
        paths.append(args.sample_submission)
    files = {str(path.resolve()): sha256(path) for path in paths}
    models = {}
    for leaf in leaves:
        record = recipe["models"][str(leaf)]
        if record["file"] != f"leaf{leaf}.joblib":
            raise ValueError("Invalid leaf model filename.")
        path = args.recipe.parent / record["file"]
        files[str(path.resolve())] = record["sha256"]
        midpoint.check_files({str(path): record["sha256"]})
        model = joblib.load(path)
        if (
            parameter_fingerprint(model) != record["parameters"]
            or parameter_fingerprint(model, normalize_leaf=True) != recipe["template_parameters"]
            or model.get_params()[PARAMETER] != leaf
            or list(model.feature_names_in_) != legacy["bundle"]["columns"]
        ):
            raise ValueError("Saved leaf model parameters/features differ.")
        models[leaf] = model
    # Training CSV/cache paths are provenance only; never open them in predict.
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    ordered = template_ids(args.sample_submission, ids) if args.sample_submission else ids
    check_unchanged(reference, recipe["source_sha256"], files, args.device)
    outputs, _ = infer(frame, args.image_root, reference, legacy, frozen, models, args.device)
    check_unchanged(reference, recipe["source_sha256"], files, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        result = {}
        for leaf in leaves:
            name = f"blend_leaf{leaf}_50_50"
            labels = outputs[name].argmax(1) + 1
            ordered_labels = pd.Series(labels, index=ids).loc[ordered].to_numpy(dtype=np.int64)
            path = args.output_dir / f"submission_frozen_extratrees_leaf{leaf}_50_50.csv"
            with path.open("x", encoding="utf-8", newline="") as handle:
                pd.DataFrame({"sample_id": ordered, "label": ordered_labels}).to_csv(
                    handle, index=False
                )
            result[name] = check_csv(path, ordered)
            print("CSV saved:", path)
        write_json(
            args.output_dir / "manifest.json",
            {
                "kind": KIND,
                "recipe": recipe,
                "input_sha256": files,
                "outputs": result,
                "leaves": list(leaves),
            },
        )
        write_json(
            status,
            {"status": "complete", "manifest_sha256": sha256(args.output_dir / "manifest.json")},
        )
        return result
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def report(args):
    recipe = read_sweep(args.recipe)
    display(recipe["metrics"], recipe["group_cv"], class_details=args.class_details)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser(
        "train", help="Train trees on Train, replay baseline, compare Validation"
    )
    for name in (
        "recipe",
        "train_csv",
        "validation_csv",
        "train_embeddings",
        "image_root",
        "output_dir",
    ):
        fitting.add_argument("--" + name, type=Path, required=True)
    fitting.add_argument(
        "--group_cv", action="store_true", help="Also run tree-only Train GroupKFold(3)"
    )
    fitting.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    fitting.set_defaults(handler=train)
    reporting = commands.add_parser("report")
    reporting.add_argument("--recipe", type=Path, required=True)
    reporting.add_argument("--class_details", action="store_true")
    reporting.set_defaults(handler=report)
    prediction = commands.add_parser("predict", help="Explicit leaf candidates; no fitting")
    for name in ("recipe", "input", "image_root", "output_dir"):
        prediction.add_argument("--" + name, type=Path, required=True)
    prediction.add_argument("--leaves", nargs="+", type=int, choices=LEAVES, default=[5, 10])
    prediction.add_argument("--sample_submission", type=Path)
    prediction.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prediction.set_defaults(handler=predict)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
