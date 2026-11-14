"""Saved frozen-control ResNet18 + original ExtraTrees, with a fixed 50:50 soft vote.

Run by the user on the organizer server. No fitting, weight search, or Test
adaptation. Existing neural/tree sources and artifacts remain read-only.
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
from experiments.seungbeom import blend_ratios as ratios  # noqa: E402
from experiments.seungbeom import midpoint_blend as midpoint  # noqa: E402
from experiments.seungbeom import resnet_finetune as finetune  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    checked_score,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.replay_precision import precision_context, runtime_signature  # noqa: E402
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from runtime import select_device  # noqa: E402
from schema import LABELS, read_frame  # noqa: E402

KIND = "seungbeom_frozen_control_extratrees_blend"
RULE = {"method": "probability_mean", "frozen": 0.5, "extra_trees": 0.5}
OLD_NAMES = {
    "neural": "historical_neural",
    "extra_trees": "extra_trees",
    "blend_50_50": "baseline_50_50",
}
METHODS = ("frozen", "extra_trees", "frozen_tree_50_50", "baseline_50_50", "historical_neural")
REPLAY_NAMES = ("frozen", *OLD_NAMES.values())
SELF = "experiments/seungbeom/frozen_tree_blend.py"


def source_hashes():
    return {**ratios.source_hashes(), **finetune.source_hashes(), SELF: sha256(ROOT / SELF)}


def load_members(base, frozen_run, device):
    """Load trusted saved artifacts, checking split, source, mode and runtime identity."""
    legacy = ratios.load_base(base, device)
    frozen_run = Path(frozen_run).resolve()
    result = finetune.check_trial(frozen_run)
    if result.get("mode") != "frozen" or not result.get("checkpoint_replay_matches"):
        raise ValueError("Use the completed frozen control, not the layer4 fine-tuned candidate.")
    status = read_json(frozen_run / "status.json")
    files = {
        str(frozen_run / name): sha256(frozen_run / name)
        for name in ("status.json", *status["files"])
    }
    model, prep, checkpoint = finetune.load_checkpoint(frozen_run, device)
    if checkpoint["mode"] != "frozen":
        raise ValueError("Frozen-control checkpoint mode differs.")
    for key in ("train_csv_sha256", "validation_csv_sha256"):
        if checkpoint.get(key) != base["provenance"][key] or result.get(key) != checkpoint.get(key):
            raise ValueError("Frozen and original tree must use identical Train/Validation splits.")
    if checkpoint["runtime"] != base["runtime_signature"]:
        raise ValueError("Original tree and frozen control require the same runtime environment.")
    checked_score(result["metrics"])
    record = {
        "run_dir": str(frozen_run),
        "files": files,
        "mode": "frozen",
        "config": checkpoint["config"],
        "model_config": checkpoint["model_config"],
        "selected_epoch": checkpoint["selected_epoch"],
        "metrics": result["metrics"],
        "train_csv_sha256": checkpoint["train_csv_sha256"],
        "validation_csv_sha256": checkpoint["validation_csv_sha256"],
    }
    midpoint.check_files(files)
    return legacy, {"model": model, "preprocessor": prep, "checkpoint": checkpoint}, record


def frozen_probabilities(frame, image_root, saved, device):
    """Keep the new raw-image FP32 path; never substitute legacy cached vectors."""
    loader = finetune.evaluation_loader(
        frame, image_root, saved["preprocessor"], saved["checkpoint"]["config"], labeled=False
    )
    with precision_context("fp32"):
        return finetune.probabilities(
            saved["model"], loader, select_device(device), progress_every=2048
        )


def infer_members(frame, image_root, legacy, frozen, base, device):
    """The comparison and CSV export deliberately use exactly the same path."""
    frame = frame.drop(columns="label", errors="ignore").copy()
    print("Predicting saved frozen control (FP32, current input images)...", flush=True)
    fresh = frozen_probabilities(frame, image_root, frozen, device)
    print("Predicting original tree recipe with its saved image precision...", flush=True)
    # The historical neural head is cheap to replay alongside the tree's one
    # image extraction. Reuse the established path instead of extracting the
    # image features a third time solely to test a separate export path.
    old = midpoint.method_probabilities(
        frame, image_root, legacy, device, image_precision=base["image_precision"]
    )
    outputs = {new: old[old_name] for old_name, new in OLD_NAMES.items()}
    outputs["frozen"] = fresh
    outputs["frozen_tree_50_50"] = midpoint.blend_probabilities(fresh, outputs["extra_trees"])
    return outputs


def check_unchanged(base, frozen, sources, files, device):
    midpoint.check_files(base["provenance"]["files"])
    midpoint.check_files(frozen["files"])
    midpoint.check_files(files)
    if source_hashes() != sources or runtime_signature(device) != base["runtime_signature"]:
        raise ValueError("Source or runtime changed during inference.")


def show_metrics(metrics, *, class_details=False):
    reference = checked_score(metrics["frozen"])
    print(
        "method\tvalidation_macro_f1\tdelta_vs_frozen\t"
        + "\t".join(f"f1_{label}" for label in LABELS)
    )
    for name in METHODS:
        item = metrics[name]
        print(
            f"{name}\t{item['macro_f1']:.6f}\t{item['macro_f1'] - reference:+.6f}\t"
            + "\t".join(f"{item['class_f1'][str(label)]:.6f}" for label in LABELS)
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
    files = {str(path.resolve()): sha256(path) for path in (args.baseline_recipe, args.input)}
    base = read_json(args.baseline_recipe)
    legacy, frozen, record = load_members(base, args.frozen_run, args.device)
    digest = files[str(args.input.resolve())]
    if (
        digest != base["provenance"]["validation_csv_sha256"]
        or digest == base["provenance"]["train_csv_sha256"]
    ):
        raise ValueError("Compare requires the recorded official Validation CSV, not Train/Test.")
    frame = read_frame(args.input, labeled=True)
    sources = source_hashes()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        outputs = infer_members(frame, args.image_root, legacy, frozen, base, args.device)
        metrics = {
            name: classification_metrics(frame["label"], outputs[name].argmax(1) + 1)
            for name in METHODS
        }
        replay = {"frozen": metrics["frozen"] == record["metrics"]}
        replay.update({new: metrics[new] == base["metrics"][old] for old, new in OLD_NAMES.items()})
        write_json(
            args.output_dir / "validation_metrics.json",
            {"metrics": metrics, "replay_matches": replay},
        )
        show_metrics(metrics)
        if not all(replay.values()):
            raise ValueError("Saved member/baseline Validation metrics did not replay; stop here.")
        check_unchanged(base, record, sources, files, args.device)
        recipe = {
            "kind": KIND,
            "format_version": 1,
            "status": "complete",
            "rule": RULE,
            "label_values": list(LABELS),
            "base_recipe": base,
            "frozen": record,
            "source_sha256": sources,
            "metrics": metrics,
            "replay_matches": replay,
            "comparison_input_sha256": files,
            "weight_selection": "fixed_equal_before_validation_no_search_no_test",
            "image_execution": {"frozen": "fp32", "extra_trees": base["image_precision"]},
        }
        write_json(args.output_dir / "recipe.json", recipe)
        write_json(
            status, {"status": "complete", "recipe_sha256": sha256(args.output_dir / "recipe.json")}
        )
        print("All saved members and the original 50:50 replayed. No training or Test access.")
        print("Validation is a candidate comparison, not an independent final score.")
        return recipe
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def read_comparison(path):
    path = Path(path)
    recipe = read_json(path)
    status = read_json(path.parent / "status.json")
    if (
        recipe.get("kind") != KIND
        or recipe.get("format_version") != 1
        or recipe.get("status") != "complete"
        or recipe.get("rule") != RULE
        or recipe.get("label_values") != list(LABELS)
        or recipe.get("replay_matches") != dict.fromkeys(REPLAY_NAMES, True)
        or status.get("status") != "complete"
        or status.get("recipe_sha256") != sha256(path)
    ):
        raise ValueError("Use a completed, unchanged frozen/ExtraTrees Validation recipe.")
    if set(recipe["metrics"]) != set(METHODS):
        raise ValueError("Unexpected comparison methods.")
    for name in METHODS:
        checked_score(recipe["metrics"][name])
    if (
        recipe["metrics"]["frozen"] != recipe["frozen"]["metrics"]
        or any(
            recipe["metrics"][new] != recipe["base_recipe"]["metrics"][old]
            for old, new in OLD_NAMES.items()
        )
        or recipe.get("image_execution")
        != {"frozen": "fp32", "extra_trees": recipe["base_recipe"]["image_precision"]}
    ):
        raise ValueError("Comparison replay records are inconsistent.")
    return recipe


def predict(args):
    sidecar = args.output.with_suffix(".manifest.json")
    if args.output.exists() or sidecar.exists():
        raise ValueError("Output or manifest exists; choose a new output filename.")
    recipe = read_comparison(args.recipe)
    if recipe["source_sha256"] != source_hashes():
        raise ValueError("Inference source changed; compare again with the saved member sources.")
    base = recipe["base_recipe"]
    midpoint.check_files(recipe["frozen"]["files"])
    legacy, frozen, record = load_members(base, Path(recipe["frozen"]["run_dir"]), args.device)
    if record != recipe["frozen"]:
        raise ValueError("Frozen model differs from the validated candidate.")
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    ordered = template_ids(args.sample_submission, ids) if args.sample_submission else ids
    paths = [args.recipe, args.recipe.parent / "status.json", args.input]
    if args.sample_submission:
        paths.append(args.sample_submission)
    files = {str(path.resolve()): sha256(path) for path in paths}
    # The comparison CSV/cache paths recorded in the recipe are provenance only.
    # They are deliberately not opened or checked during prediction.
    outputs = infer_members(frame, args.image_root, legacy, frozen, base, args.device)
    labels = outputs["frozen_tree_50_50"].argmax(1) + 1
    values = pd.Series(labels, index=ids).loc[ordered].to_numpy(dtype=np.int64)
    check_unchanged(base, record, recipe["source_sha256"], files, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"sample_id": ordered, "label": values}).to_csv(handle, index=False)
    metadata = check_csv(args.output, ordered)
    write_json(
        sidecar,
        {"kind": KIND, "rule": RULE, "recipe": recipe, "input_sha256": files, "output": metadata},
    )
    print("Frozen + original ExtraTrees 50:50 CSV saved:", args.output)
    return metadata


def report(args):
    recipe = read_comparison(args.recipe)
    show_metrics(recipe["metrics"], class_details=args.class_details)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    comparison = commands.add_parser("compare", help="Replay members and compare fixed 50:50")
    for name in ("baseline_recipe", "frozen_run", "input", "image_root", "output_dir"):
        comparison.add_argument("--" + name, type=Path, required=True)
    comparison.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    comparison.set_defaults(handler=compare)
    reporting = commands.add_parser("report", help="Read the saved aggregate comparison")
    reporting.add_argument("--recipe", type=Path, required=True)
    reporting.add_argument("--class_details", action="store_true")
    reporting.set_defaults(handler=report)
    prediction = commands.add_parser("predict", help="Saved fixed blend; no fitting")
    for name in ("recipe", "input", "image_root", "output"):
        prediction.add_argument("--" + name, type=Path, required=True)
    prediction.add_argument("--sample_submission", type=Path)
    prediction.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prediction.set_defaults(handler=predict)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
