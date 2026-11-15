"""Recover an audited leaf comparison while keeping original and refit trees distinct.

Imports trusted local failed-run artifacts, replays live Validation, and optionally
runs Train-only tree CV. Does not refit full-Train models, change the old guard,
or call the refitted leaf2 an exact copy of the original submission model.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom import extratrees_leaf_audit as audit  # noqa: E402
from experiments.seungbeom import extratrees_leaf_sweep as sweep  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    checked_score,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from schema import LABELS, read_frame  # noqa: E402

KIND = "seungbeom_audited_leaf_recovery"
SELF = "experiments/seungbeom/extratrees_leaf_recovery.py"
AUDIT_SELF = "experiments/seungbeom/extratrees_leaf_audit.py"
ORIGINAL_NAMES = {
    "frozen": "frozen",
    "extra_trees": "original_extra_trees",
    "frozen_tree_50_50": "original_frozen_tree_50_50",
}
METHODS = (
    *ORIGINAL_NAMES.values(),
    *(f"refit_{name}" for name in sweep.METHODS if name != "frozen"),
)
LEGACY_FAILURE = {**dict.fromkeys(sweep.REPLAY_NAMES, True), "refitted_leaf2_probabilities": False}
POLICY = "original_saved_tree_preserved_refitted_leaf2_is_a_distinct_candidate"


def audit_sources():
    return {**sweep.source_hashes(), AUDIT_SELF: sha256(ROOT / AUDIT_SELF)}


def source_hashes():
    return {**audit_sources(), SELF: sha256(ROOT / SELF)}


def check_unchanged(reference, sources, files, device):
    sweep.check_unchanged(reference, sweep.source_hashes(), files, device)
    if source_hashes() != sources:
        raise ValueError("Recovery source changed during execution.")


def validate_model(model, template, leaf, columns):
    if (
        model.get_params()[sweep.PARAMETER] != leaf
        or sweep.parameter_fingerprint(model, normalize_leaf=True)
        != sweep.parameter_fingerprint(template)
        or list(model.feature_names_in_) != columns
        or not np.array_equal(model.classes_, template.classes_)
    ):
        raise ValueError(
            "Imported model parameters, classes or columns differ from the leaf contract."
        )


def load_imports(args, legacy, rows):
    import joblib

    report = read_json(args.audit)
    if (
        report.get("kind") != "seungbeom_leaf_replay_audit"
        or report.get("scope") != "validation_cache_diagnostic_only"
        or report.get("source_sha256") != audit_sources()
        or report.get("prior_replay_matches") != LEGACY_FAILURE
        or report.get("cache_matches_saved_live_metrics") != {"original": True, "refitted": True}
    ):
        raise ValueError(
            "Use the unchanged successful audit of this specific leaf2 replay failure."
        )
    audited_files = report["input_sha256"]
    required = [
        args.recipe,
        args.recipe.parent / "status.json",
        args.validation_csv,
        args.failed_run / "status.json",
        args.failed_run / "validation_metrics.json",
        args.failed_run / "leaf2.joblib",
    ]
    if any(audited_files.get(str(path.resolve())) != sha256(path) for path in required):
        raise ValueError("Audit references a different recipe, failed run or Validation input.")
    sweep.midpoint.check_files(audited_files)
    if read_json(args.failed_run / "status.json") != {
        "status": "failed",
        "error_type": "ValueError",
    }:
        raise ValueError("Use the unchanged failed run; never import an active run.")
    prior = read_json(args.failed_run / "validation_metrics.json")
    if prior.get("replay_matches") != LEGACY_FAILURE or set(prior["metrics"]) != set(sweep.METHODS):
        raise ValueError("Failed run is not the audited leaf probability replay failure.")
    template = sweep.tree_template(legacy)
    models, records = {}, {}
    files = {**audited_files, str(args.audit.resolve()): sha256(args.audit)}
    prep_hash = None
    for leaf in sweep.LEAVES:
        path = args.failed_run / f"leaf{leaf}.joblib"
        digest = sha256(path)
        model = joblib.load(path)  # trusted local models; all original artifacts remain read-only
        validate_model(model, template, leaf, legacy["bundle"]["columns"])
        current = joblib.hash(model[:-1])
        if prep_hash is None:
            prep_hash = current
        elif current != prep_hash:
            raise ValueError(
                "Refit candidates have different fitted preprocessing; do not pool this ablation."
            )
        image = model.named_steps["features"].named_transformers_["image"]
        if not np.all(np.asarray(image.named_steps["scale"].n_samples_seen_) == rows):
            raise ValueError(
                "Imported preprocessing was not fitted on the recorded Train row count."
            )
        if "pca" in image.named_steps and image.named_steps["pca"].n_samples_ != rows:
            raise ValueError("Imported PCA training row count differs.")
        estimator = model.named_steps["classifier"].estimator_
        if not estimator.bootstrap and any(
            t.tree_.n_node_samples[0] != rows for t in estimator.estimators_
        ):
            raise ValueError("Imported trees were not fitted on the recorded Train row count.")
        files[str(path.resolve())] = digest
        records[str(leaf)] = {
            "file": path.name,
            "sha256": digest,
            "parameters": sweep.parameter_fingerprint(model),
        }
        models[leaf] = model
    return models, records, prior, report, files, prep_hash


def infer(frame, image_root, reference, legacy, frozen, models, device):
    values, features = sweep.infer(frame, image_root, reference, legacy, frozen, models, device)
    outputs = {"frozen": values["frozen"]}
    outputs.update({f"refit_{name}": value for name, value in values.items() if name != "frozen"})
    original = sweep.mm.predict_proba_six_stable(sweep.tree_template(legacy), features)
    outputs["original_extra_trees"] = original
    outputs["original_frozen_tree_50_50"] = sweep.midpoint.blend_probabilities(
        values["frozen"], original
    )
    return outputs


def display(metrics, cv=None, *, class_details=False):
    original = checked_score(metrics["original_frozen_tree_50_50"])
    refit = checked_score(metrics["refit_blend_leaf2_50_50"])
    print(
        "method\tvalidation_macro_f1\tdelta_vs_original_blend\tdelta_vs_refit_leaf2_blend\t"
        + "\t".join(f"f1_{c}" for c in LABELS)
    )
    for name in METHODS:
        m = metrics[name]
        print(
            f"{name}\t{m['macro_f1']:.6f}\t{m['macro_f1'] - original:+.6f}\t"
            f"{m['macro_f1'] - refit:+.6f}\t"
            + "\t".join(f"{m['class_f1'][str(c)]:.6f}" for c in LABELS)
        )
    if class_details:
        print("\nmethod\tclass\tprecision\trecall\tf1\tsupport")
        for name in METHODS:
            m = metrics[name]
            for label in LABELS:
                key = str(label)
                print(
                    f"{name}\t{label}\t{m['class_precision'][key]:.6f}\t"
                    f"{m['class_recall'][key]:.6f}\t{m['class_f1'][key]:.6f}\t{m['support'][key]}"
                )
    if cv:
        print("\nTree-only Train GroupKFold(3); no full-Train neural in OOF.")
        print("fold\tleaf\tmacro_f1")
        for fold in cv["folds"]:
            for leaf in sweep.LEAVES:
                print(f"{fold['fold']}\t{leaf}\t{fold['metrics'][str(leaf)]['macro_f1']:.6f}")
        print("leaf\tfold_mean\tfold_sample_std\tpooled_oof_f1")
        for leaf in sweep.LEAVES:
            m = cv["summary"][str(leaf)]
            print(
                f"{leaf}\t{m['fold_mean']:.6f}\t{m['fold_sample_std']:.6f}\t{m['pooled_oof']['macro_f1']:.6f}"
            )
    print("Original and refitted leaf2 are distinct fitted models, even if their F1 scores match.")


def recover(args):
    import joblib

    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    reference = sweep.frozen_blend.read_comparison(args.recipe)
    legacy, frozen = sweep.load_reference(reference, args.device)
    train_frame, val_frame, features, files = sweep.load_inputs(args, reference, legacy)
    models, records, prior, audit_report, imported_files, prep_hash = load_imports(
        args, legacy, len(train_frame)
    )
    files.update(imported_files)
    sources = source_hashes()
    check_unchanged(reference, sources, files, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir / "status.json"
    write_json(status_path, {"status": "running"})
    try:
        for leaf in sweep.LEAVES:
            path = args.output_dir / records[str(leaf)]["file"]
            shutil.copyfile(args.failed_run / path.name, path)
            if sha256(path) != records[str(leaf)]["sha256"]:
                raise ValueError("Candidate copy differs from the imported file.")
            models[leaf] = joblib.load(path)
            files[str(path.resolve())] = records[str(leaf)]["sha256"]
        print(
            "Reusing audited full-Train trees. Replaying current Validation images...", flush=True
        )
        outputs = infer(val_frame, args.image_root, reference, legacy, frozen, models, args.device)
        metrics = {
            name: sweep.classification_metrics(val_frame["label"], value.argmax(1) + 1)
            for name, value in outputs.items()
        }
        original_replay = {
            old: metrics[new] == reference["metrics"][old] for old, new in ORIGINAL_NAMES.items()
        }
        candidate_replay = {
            name: metrics[name if name == "frozen" else f"refit_{name}"] == prior["metrics"][name]
            for name in sweep.METHODS
        }
        difference = audit.probability_difference(
            outputs["original_extra_trees"], outputs["refit_extra_trees_leaf2"]
        )
        difference["changed_blend_predictions"] = int(
            np.count_nonzero(
                outputs["original_frozen_tree_50_50"].argmax(1)
                != outputs["refit_blend_leaf2_50_50"].argmax(1)
            )
        )
        write_json(
            args.output_dir / "validation_metrics.json",
            {
                "metrics": metrics,
                "original_replay": original_replay,
                "candidate_replay": candidate_replay,
                "live_leaf2_difference": difference,
            },
        )
        display(metrics)
        if not all(original_replay.values()) or not all(candidate_replay.values()):
            raise ValueError(
                "Original or imported candidate Validation metrics did not replay; stop."
            )
        cv = (
            sweep.tree_group_cv(train_frame, features, sweep.tree_template(legacy))
            if args.group_cv
            else None
        )
        check_unchanged(reference, sources, files, args.device)
        recipe = {
            "kind": KIND,
            "format_version": 1,
            "status": "complete",
            "policy": POLICY,
            "rule": sweep.frozen_blend.RULE,
            "leaves": list(sweep.LEAVES),
            "label_values": list(LABELS),
            "fit_splits": ["train"],
            "reference": reference,
            "source_sha256": sources,
            "models": records,
            "refit_preprocessing_hash": prep_hash,
            "metrics": metrics,
            "original_replay": original_replay,
            "candidate_replay": candidate_replay,
            "live_leaf2_difference": difference,
            "audit": audit_report,
            "group_cv": cv,
            "import_input_sha256": files,
            "import_note": (
                "Legacy failed run had no pre-fit manifest. Import-time hashes, "
                "shared fitted preprocessing, parameters, Train row counts and live "
                "metrics checked; not retroactive source attestation."
            ),
        }
        write_json(args.output_dir / "recipe.json", recipe)
        write_json(
            status_path,
            {"status": "complete", "recipe_sha256": sha256(args.output_dir / "recipe.json")},
        )
        display(metrics, cv)
        print("Recovery complete. Original models/submissions unchanged. No Test accessed.")
        print("Validation is reused for comparison, not an independent final score.")
        return recipe
    except BaseException as error:
        write_json(status_path, {"status": "failed", "error_type": type(error).__name__})
        raise


def read_recipe(path):
    recipe = read_json(path)
    status = read_json(path.parent / "status.json")
    if (
        recipe.get("kind") != KIND
        or recipe.get("format_version") != 1
        or recipe.get("status") != "complete"
        or status.get("status") != "complete"
        or status.get("recipe_sha256") != sha256(path)
        or recipe.get("policy") != POLICY
        or recipe.get("rule") != sweep.frozen_blend.RULE
        or recipe.get("fit_splits") != ["train"]
        or recipe.get("leaves") != list(sweep.LEAVES)
        or recipe.get("label_values") != list(LABELS)
        or recipe.get("original_replay") != dict.fromkeys(ORIGINAL_NAMES, True)
        or recipe.get("candidate_replay") != dict.fromkeys(sweep.METHODS, True)
        or set(recipe.get("models", {})) != {str(leaf) for leaf in sweep.LEAVES}
        or set(recipe.get("metrics", {})) != set(METHODS)
    ):
        raise ValueError("Use a complete, unchanged audited recovery recipe.")
    for item in recipe["metrics"].values():
        checked_score(item)
    return recipe


def predict(args):
    import joblib

    leaves = tuple(args.leaves)
    if (
        not leaves
        or len(set(leaves)) != len(leaves)
        or any(leaf not in sweep.LEAVES for leaf in leaves)
    ):
        raise ValueError("Choose distinct leaves from 2, 5, 10.")
    if args.output_dir.exists():
        raise ValueError("Choose a new output_dir; existing submissions are preserved.")
    recipe = read_recipe(args.recipe)
    if recipe["source_sha256"] != source_hashes():
        raise ValueError("Recovery source changed; use the recorded code.")
    reference = recipe["reference"]
    legacy, frozen = sweep.load_reference(reference, args.device)
    paths = [args.recipe, args.recipe.parent / "status.json", args.input]
    if args.sample_submission:
        paths.append(args.sample_submission)
    files = {str(path.resolve()): sha256(path) for path in paths}
    models = {}
    for leaf in leaves:
        record = recipe["models"][str(leaf)]
        if record["file"] != f"leaf{leaf}.joblib":
            raise ValueError("Invalid imported candidate filename.")
        path = args.recipe.parent / record["file"]
        files[str(path.resolve())] = record["sha256"]
        sweep.midpoint.check_files({str(path.resolve()): record["sha256"]})
        model = joblib.load(path)
        validate_model(model, sweep.tree_template(legacy), leaf, legacy["bundle"]["columns"])
        if (
            sweep.parameter_fingerprint(model) != record["parameters"]
            or joblib.hash(model[:-1]) != recipe["refit_preprocessing_hash"]
        ):
            raise ValueError("Candidate parameter/preprocessing fingerprint differs.")
        models[leaf] = model
    # Old training/validation/cache/audit/import paths are provenance only in prediction.
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    ordered = template_ids(args.sample_submission, ids) if args.sample_submission else ids
    check_unchanged(reference, recipe["source_sha256"], files, args.device)
    outputs = infer(frame, args.image_root, reference, legacy, frozen, models, args.device)
    check_unchanged(reference, recipe["source_sha256"], files, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir / "status.json"
    write_json(status_path, {"status": "running"})
    try:
        records = {}
        for leaf in leaves:
            name = f"refit_blend_leaf{leaf}_50_50"
            labels = (
                pd.Series(outputs[name].argmax(1) + 1, index=ids)
                .loc[ordered]
                .to_numpy(dtype=np.int64)
            )
            path = args.output_dir / f"submission_frozen_refit_extratrees_leaf{leaf}_50_50.csv"
            with path.open("x", encoding="utf-8", newline="") as handle:
                pd.DataFrame({"sample_id": ordered, "label": labels}).to_csv(handle, index=False)
            records[name] = check_csv(path, ordered)
            print("Candidate CSV saved:", path)
        manifest = {
            "kind": KIND,
            "recipe": recipe,
            "leaves": list(leaves),
            "input_sha256": files,
            "outputs": records,
        }
        write_json(args.output_dir / "manifest.json", manifest)
        write_json(
            status_path,
            {"status": "complete", "manifest_sha256": sha256(args.output_dir / "manifest.json")},
        )
        return records
    except BaseException as error:
        write_json(status_path, {"status": "failed", "error_type": type(error).__name__})
        raise


def report(args):
    recipe = read_recipe(args.recipe)
    display(recipe["metrics"], recipe["group_cv"], class_details=args.class_details)
    print("Live original vs refitted leaf2 diagnostics:", recipe["live_leaf2_difference"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recovery = commands.add_parser(
        "recover", help="Import audited trees, replay live Validation, optionally run tree CV"
    )
    for name in (
        "recipe",
        "failed_run",
        "audit",
        "train_csv",
        "validation_csv",
        "train_embeddings",
        "image_root",
        "output_dir",
    ):
        recovery.add_argument("--" + name, type=Path, required=True)
    recovery.add_argument("--group_cv", action="store_true")
    recovery.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    recovery.set_defaults(handler=recover)
    reporting = commands.add_parser("report")
    reporting.add_argument("--recipe", type=Path, required=True)
    reporting.add_argument("--class_details", action="store_true")
    reporting.set_defaults(handler=report)
    prediction = commands.add_parser("predict")
    for name in ("recipe", "input", "image_root", "output_dir"):
        prediction.add_argument("--" + name, type=Path, required=True)
    prediction.add_argument("--leaves", nargs="+", type=int, choices=sweep.LEAVES, default=[5, 10])
    prediction.add_argument("--sample_submission", type=Path)
    prediction.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prediction.set_defaults(handler=predict)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
