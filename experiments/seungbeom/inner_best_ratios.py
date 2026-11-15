"""Replay saved inner_best + original ExtraTrees, compare ratios, export without fitting.

Human-run organizer server only; local tests use independent synthetic inputs.
Weights below always mean inner_best neural : ExtraTrees, never label averaging.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import classification_metrics  # noqa: E402
from experiments.seungbeom import frozen_checkpoint as checkpoint  # noqa: E402
from experiments.seungbeom.ensemble_core import validate_probabilities  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    checked_score,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from schema import LABELS, read_frame  # noqa: E402

KIND = "seungbeom_inner_best_ratios"
SELF = "experiments/seungbeom/inner_best_ratios.py"
DRIVER = "experiments/seungbeom/frozen_checkpoint.py"
# Reviewed f639869 -> d0df9bf diff only adds training-budget CLI/metadata.
# restore(), probability(), checkpoint_identity(), read_result() are unchanged.
# All dependency hashes remain exact; no general ignore-source-change option.
REVIEWED_DRIVERS = {
    "60c51f4e9768a93e14f8df7c7cf7b6d4ee3fdc0ae69e8b7771b9723ed4bbc3ef",
    "6451deb86b2cf1a253f92ed62afa2f299272d5a49df50a0a87afbac94a281144",
}
RATIOS = {"inner_best_50_50": 0.5, "inner_best_40_60": 0.4, "inner_best_60_40": 0.6}
EXPORTS = ("inner_best_40_60", "inner_best_60_40")
REPLAY = ("inner_best_neural", "extra_trees", "inner_best_50_50")
METHODS = ("inner_best_neural", "extra_trees", *RATIOS)
blend = checkpoint.blend


def source_hashes():
    return {**checkpoint.source_hashes(), SELF: sha256(ROOT / SELF)}


def check_training_sources(recorded, current):
    if not isinstance(recorded, dict) or set(recorded) != set(current):
        raise ValueError("Checkpoint source file set changed.")
    if recorded == current:
        return "exact"
    if (
        recorded.get(DRIVER) in REVIEWED_DRIVERS
        and current.get(DRIVER) in REVIEWED_DRIVERS
        and all(recorded[name] == current[name] for name in current if name != DRIVER)
    ):
        return "reviewed_training_budget_only_change"
    raise ValueError("Checkpoint inference source changed; use the recorded revision.")


def file_hashes(paths):
    return {str(path.resolve()): sha256(path) for path in paths}


def run_files(directory):
    return file_hashes([directory / "status.json", *(directory / n for n in checkpoint.FILES)])


def image_digest(frame, image_root):
    """Same content digest as the checkpoint manifest, using the actual inference root."""
    root = image_root.resolve()
    digest = hashlib.sha256()
    for relative in frame["image_path"]:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image escapes its explicit split directory.")
        digest.update(relative.encode() + b"\0" + sha256(path).encode())
    return digest.hexdigest()


def weighted_probabilities(neural, tree, name):
    if name not in RATIOS:
        raise ValueError("Use a predefined neural/tree ratio.")
    if name == "inner_best_50_50":
        return blend.midpoint.blend_probabilities(neural, tree)
    left, right = np.asarray(neural), np.asarray(tree)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Members must have matching [rows, 6] probability shapes.")
    values = validate_probabilities(np.stack([left, right], axis=1))
    weight = RATIOS[name]
    return weight * values[:, 0] + (1 - weight) * values[:, 1]


def load_members(directory, device):
    files = run_files(directory)
    manifest, result = checkpoint.read_result(directory)
    compatibility = check_training_sources(manifest["source_sha256"], checkpoint.source_hashes())
    target = checkpoint.ft.select_device(device)
    checkpoint.ft.seed_everything(manifest["config"]["seed"])
    if checkpoint.runtime_signature(target) != manifest["runtime"]:
        raise ValueError("Use the checkpoint's recorded runtime environment.")
    model, prep = checkpoint.restore(
        directory / "inner_best.pt", result["identities"]["inner_best"], target
    )
    base = manifest["reference"]["base_recipe"]
    legacy = blend.ratios.load_base(base, device)
    blend.midpoint.check_files(files)
    return manifest, result, files, compatibility, (model, prep, legacy, target)


def infer_members(frame, image_root, manifest, loaded):
    model, prep, legacy, device = loaded
    current = frame.drop(columns="label", errors="ignore")
    print("Predicting saved inner_best neural (no training)...", flush=True)
    neural = checkpoint.probability(model, prep, current, image_root, manifest["config"], device)
    print("Predicting original ExtraTrees; reusing both members for all ratios...", flush=True)
    base = manifest["reference"]["base_recipe"]
    old = blend.midpoint.method_probabilities(
        current, image_root, legacy, str(device), image_precision=base["image_precision"]
    )
    # The legacy loader also knows the historical neural; it is NOT the inner_best member.
    return neural, old["extra_trees"]


def check_unchanged(manifest, sources, files, device):
    blend.midpoint.check_files(files)
    blend.midpoint.check_files(manifest["reference"]["base_recipe"]["provenance"]["files"])
    if source_hashes() != sources or checkpoint.runtime_signature(device) != manifest["runtime"]:
        raise ValueError("Sources/runtime changed during inference.")


def display(metrics, class_details=False):
    reference = checked_score(metrics["inner_best_50_50"])
    print("Ratios = inner_best neural : original ExtraTrees (probabilities, then argmax).")
    print("method\tvalidation_macro_f1\tdelta_vs_50_50\t" + "\t".join(f"f1_{i}" for i in LABELS))
    for name in METHODS:
        item = metrics[name]
        print(
            f"{name}\t{item['macro_f1']:.6f}\t{item['macro_f1'] - reference:+.6f}\t"
            + "\t".join(f"{item['class_f1'][str(i)]:.6f}" for i in LABELS)
        )
    if class_details:
        print("\nmethod\tclass\tprecision\trecall\tf1\tsupport")
        for name in METHODS:
            item = metrics[name]
            for label in LABELS:
                key = str(label)
                print(
                    f"{name}\t{label}\t{item['class_precision'][key]:.6f}\t"
                    f"{item['class_recall'][key]:.6f}\t{item['class_f1'][key]:.6f}\t"
                    f"{item['support'][key]}"
                )


def compare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    sources = source_hashes()
    manifest, result, files, compatibility, loaded = load_members(args.run_dir, args.device)
    files.update(file_hashes([args.input]))
    digest = files[str(args.input.resolve())]
    if (
        digest != manifest["inputs"]["validation"]["csv_sha256"]
        or digest == manifest["inputs"]["train"]["csv_sha256"]
    ):
        raise ValueError("Compare requires the original official Validation CSV, not Train/Test.")
    frame = read_frame(args.input, labeled=True)
    images = image_digest(frame, args.image_root)
    if images != manifest["inputs"]["validation"]["images_sha256"]:
        raise ValueError("Validation images differ from the saved checkpoint run.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        neural, tree = infer_members(frame, args.image_root, manifest, loaded)
        probabilities = {"inner_best_neural": neural, "extra_trees": tree}
        probabilities.update({name: weighted_probabilities(neural, tree, name) for name in RATIOS})
        metrics = {
            name: classification_metrics(frame["label"], values.argmax(1) + 1)
            for name, values in probabilities.items()
        }
        replay = {name: metrics[name] == result["metrics"][name] for name in REPLAY}
        write_json(
            args.output_dir / "validation_metrics.json",
            {"metrics": metrics, "replay_matches": replay},
        )
        if not all(replay.values()):
            raise ValueError(
                "Member/50:50 Validation replay differs; inspect before Test inference."
            )
        check_unchanged(manifest, sources, files, loaded[-1])
        if image_digest(frame, args.image_root) != images:
            raise ValueError("Validation images changed during inference.")
        recipe = {
            "kind": KIND,
            "format_version": 1,
            "status": "complete",
            "label_values": list(LABELS),
            "ratios_neural_weight": RATIOS,
            "run_dir": str(args.run_dir.resolve()),
            "run_files": run_files(args.run_dir),
            "source_sha256": sources,
            "source_compatibility": compatibility,
            "runtime": manifest["runtime"],
            "validation": {"csv_sha256": digest, "images_sha256": images},
            "metrics": metrics,
            "replay_matches": replay,
            "selection": "predefined_ratios_no_fitting_no_test_access",
        }
        path = args.output_dir / "recipe.json"
        write_json(path, recipe)
        write_json(status, {"status": "complete", "recipe_sha256": sha256(path)})
        display(metrics)
        print("Saved comparison; all three baseline metrics replayed. No training or Test access.")
        print("Validation is reused for comparison, not an independent final score.")
        return recipe
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def read_comparison(path):
    status = read_json(path.parent / "status.json")
    if status.get("status") != "complete" or status.get("recipe_sha256") != sha256(path):
        raise ValueError("Comparison incomplete or recipe changed.")
    recipe = read_json(path)
    if (
        recipe.get("kind") != KIND
        or recipe.get("format_version") != 1
        or recipe.get("status") != "complete"
        or recipe.get("label_values") != list(LABELS)
        or recipe.get("ratios_neural_weight") != RATIOS
        or recipe.get("replay_matches") != dict.fromkeys(REPLAY, True)
        or set(recipe.get("metrics", {})) != set(METHODS)
    ):
        raise ValueError("Invalid inner_best ratio comparison.")
    for name in METHODS:
        checked_score(recipe["metrics"][name])
    return recipe


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    recipe = read_comparison(args.recipe)
    sources = source_hashes()
    if recipe["source_sha256"] != sources:
        raise ValueError("Comparison source changed; compare again in a new directory.")
    blend.midpoint.check_files(recipe["run_files"])
    manifest, result, files, compatibility, loaded = load_members(
        Path(recipe["run_dir"]), args.device
    )
    if (
        files != recipe["run_files"]
        or compatibility != recipe["source_compatibility"]
        or manifest["runtime"] != recipe["runtime"]
        or recipe["validation"] != manifest["inputs"]["validation"]
        or any(recipe["metrics"][name] != result["metrics"][name] for name in REPLAY)
    ):
        raise ValueError("Models or replay provenance differ from the comparison.")
    files.update(
        file_hashes(
            [args.recipe, args.recipe.parent / "status.json", args.input, args.sample_submission]
        )
    )
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    ordered = template_ids(args.sample_submission, ids)
    images = image_digest(frame, args.image_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        neural, tree = infer_members(frame, args.image_root, manifest, loaded)
        check_unchanged(manifest, sources, files, loaded[-1])
        if image_digest(frame, args.image_root) != images:
            raise ValueError("Input images changed during inference.")
        outputs = {}
        for name in EXPORTS:
            labels = weighted_probabilities(neural, tree, name).argmax(1) + 1
            values = pd.Series(labels, index=ids).loc[ordered].to_numpy(dtype=np.int64)
            output = args.output_dir / f"submission_{name}.csv"
            with output.open("x", encoding="utf-8", newline="") as handle:
                pd.DataFrame({"sample_id": ordered, "label": values}).to_csv(handle, index=False)
            outputs[name] = check_csv(output, ordered)
            write_json(
                output.with_suffix(".manifest.json"),
                {
                    "kind": KIND,
                    "method": name,
                    "weights": {"inner_best_neural": RATIOS[name], "extra_trees": 1 - RATIOS[name]},
                    "recipe": recipe,
                    "input_sha256": files,
                    "input_images_sha256": images,
                    "output": outputs[name],
                },
            )
            print("Submission CSV saved:", output)
        check_unchanged(manifest, sources, files, loaded[-1])
        write_json(status, {"status": "complete", "outputs": outputs})
        print("Two CSVs exported with one inference pass per member. Existing 50:50 is unchanged.")
        return outputs
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def report(args):
    display(read_comparison(args.recipe)["metrics"], args.class_details)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    comparison = commands.add_parser(
        "compare", help="Replay baseline and compare three fixed ratios"
    )
    export = commands.add_parser("predict", help="Export both new ratios with no training")
    for command in (comparison, export):
        for name in ("input", "image_root", "output_dir"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    comparison.add_argument("--run_dir", type=Path, required=True)
    comparison.set_defaults(handler=compare)
    export.add_argument("--recipe", type=Path, required=True)
    export.add_argument("--sample_submission", type=Path, required=True)
    export.set_defaults(handler=predict)
    saved = commands.add_parser("report", help="Read saved Validation results without inference")
    saved.add_argument("--recipe", type=Path, required=True)
    saved.add_argument("--class_details", action="store_true")
    saved.set_defaults(handler=report)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
