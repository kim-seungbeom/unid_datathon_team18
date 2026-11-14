"""Compare/export fixed 25:75 and 20:80 blends from verified server-local caches.

No model fitting, loading or image inference. Missing/mismatched caches fail;
this command never silently falls back to inference. Preserve the 30:70 baseline.
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
from experiments.seungbeom import inner_best_refine as previous  # noqa: E402
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

KIND = "seungbeom_inner_best_cached_ratios"
SELF = "experiments/seungbeom/inner_best_cached_ratios.py"
BASELINE = "inner_best_30_70"
RATIOS = {"inner_best_25_75": 0.25, "inner_best_20_80": 0.20}
METHODS = (BASELINE, *RATIOS)
REPLAY = (*previous.original.METHODS, *previous.RATIOS)


def source_hashes():
    return {**previous.source_hashes(), SELF: sha256(ROOT / SELF)}


def load_reference(path, device):
    reference = previous.read_comparison(path)
    if reference["source_sha256"] != previous.source_hashes():
        raise ValueError("Previous comparison source changed; use its recorded revision.")
    base, manifest, files = previous.load_reference(Path(reference["baseline_recipe_path"]), device)
    if base != reference["baseline_recipe"]:
        raise ValueError("Previous baseline recipe changed.")
    files.update(previous.original.file_hashes([path, path.parent / "status.json"]))
    return reference, base, manifest, files


def unchanged(base, manifest, files, sources, device):
    previous.unchanged(base, manifest, files, previous.source_hashes(), device)
    if source_hashes() != sources:
        raise ValueError("Source changed during the comparison/export.")


def cached_members(args, frame, base, manifest, files, sources):
    # Construct the exact identity used by inner_best_refine.py. Never trust a
    # cache's declared input identity without checking the caller's current input.
    images = previous.original.image_digest(frame, args.image_root)
    identity = {
        "kind": previous.CACHE_KIND,
        "format_version": 1,
        "label_values": list(LABELS),
        "input_csv_sha256": files[str(args.input.resolve())],
        "input_images_sha256": images,
        "model_files": {
            **base["run_files"],
            **manifest["reference"]["base_recipe"]["provenance"]["files"],
        },
        "inference_source_sha256": base["source_sha256"],
        "runtime": manifest["runtime"],
    }
    if not (args.cache_dir / "status.json").is_file():
        raise ValueError("Completed probability cache required; this command never runs inference.")
    members, cache_files = previous.read_cache(args.cache_dir, identity, frame.sample_id.tolist())
    files.update(cache_files)
    unchanged(base, manifest, files, sources, args.device)
    if previous.original.image_digest(frame, args.image_root) != images:
        raise ValueError("Input images changed while reading the cache.")
    print("Using verified member probabilities; no training or image inference.", flush=True)
    return members, images


def candidates(neural, tree):
    values = validate_probabilities(np.stack([neural, tree], axis=1))
    result = previous.candidates(neural, tree)
    result.update(
        {
            name: weight * values[:, 0] + (1 - weight) * values[:, 1]
            for name, weight in RATIOS.items()
        }
    )
    return result


def changes(probabilities):
    baseline = probabilities[BASELINE].argmax(1)
    return {
        name: int(np.count_nonzero(probabilities[name].argmax(1) != baseline)) for name in METHODS
    }


def display(recipe, class_details=False):
    metrics = recipe["metrics"]
    baseline = checked_score(metrics[BASELINE])
    print("Ratios = inner_best neural : original ExtraTrees; baseline is 30:70.")
    print(
        "method\tvalidation_macro_f1\tdelta_vs_30_70\tchanged_rows_vs_30_70\t"
        + "\t".join(f"f1_{label}" for label in LABELS)
    )
    for name in METHODS:
        item = metrics[name]
        print(
            f"{name}\t{item['macro_f1']:.6f}\t{item['macro_f1'] - baseline:+.6f}\t"
            f"{recipe['changed_rows'][name]}\t"
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
                    f"{item['class_recall'][key]:.6f}\t{item['class_f1'][key]:.6f}\t{item['support'][key]}"
                )


def compare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    sources = source_hashes()
    reference, base, manifest, files = load_reference(args.baseline_recipe, args.device)
    files.update(previous.original.file_hashes([args.input]))
    digest = files[str(args.input.resolve())]
    if (
        digest != base["validation"]["csv_sha256"]
        or digest == manifest["inputs"]["train"]["csv_sha256"]
    ):
        raise ValueError("Compare requires the original official Validation CSV, not Train/Test.")
    frame = read_frame(args.input, labeled=True)
    members, images = cached_members(args, frame, base, manifest, files, sources)
    if images != base["validation"]["images_sha256"]:
        raise ValueError("Validation images differ from the original comparison.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        probabilities = candidates(*members)
        metrics = {
            name: classification_metrics(frame["label"], value.argmax(1) + 1)
            for name, value in probabilities.items()
        }
        replay = {name: metrics[name] == reference["metrics"][name] for name in REPLAY}
        write_json(
            args.output_dir / "validation_metrics.json",
            {"metrics": metrics, "replay_matches": replay},
        )
        if not all(replay.values()):
            raise ValueError("Previous Validation metrics do not replay, including 30:70.")
        unchanged(base, manifest, files, sources, args.device)
        recipe = {
            "kind": KIND,
            "format_version": 1,
            "status": "complete",
            "label_values": list(LABELS),
            "ratios_neural_weight": RATIOS,
            "reference_path": str(args.baseline_recipe.resolve()),
            "reference": reference,
            "source_sha256": sources,
            "metrics": metrics,
            "replay_matches": replay,
            "changed_rows": changes(probabilities),
            "validation_provenance": files,
            "selection": "two_predefined_ratios_no_fit_no_test_cache_only",
        }
        path = args.output_dir / "recipe.json"
        write_json(path, recipe)
        write_json(status, {"status": "complete", "recipe_sha256": sha256(path)})
        display(recipe)
        print(
            "Comparison complete. Previous results replayed; no training, inference or Test access."
        )
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
        or set(recipe.get("metrics", {})) != set((*REPLAY, *RATIOS))
    ):
        raise ValueError("Invalid cached-ratio recipe.")
    for name, metric in recipe["metrics"].items():
        checked_score(metric)
        if name in REPLAY and metric != recipe["reference"]["metrics"][name]:
            raise ValueError("Recorded replay metrics disagree.")
    return recipe


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    recipe = read_comparison(args.recipe)
    sources = source_hashes()
    if recipe["source_sha256"] != sources:
        raise ValueError("Comparison code changed; compare again in a new directory.")
    reference, base, manifest, files = load_reference(Path(recipe["reference_path"]), args.device)
    if reference != recipe["reference"]:
        raise ValueError("Baseline recipe changed after comparison.")
    files.update(
        previous.original.file_hashes(
            [args.recipe, args.recipe.parent / "status.json", args.input, args.sample_submission]
        )
    )
    frame = read_frame(args.input)
    ids = frame.sample_id.tolist()
    order = template_ids(args.sample_submission, ids)
    members, images = cached_members(args, frame, base, manifest, files, sources)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        probabilities = candidates(*members)
        differences = changes(probabilities)
        outputs = {}
        for name, weight in RATIOS.items():
            output = args.output_dir / f"submission_{name}.csv"
            values = (
                pd.Series(probabilities[name].argmax(1) + 1, index=ids)
                .loc[order]
                .to_numpy(dtype=np.int64)
            )
            with output.open("x", encoding="utf-8", newline="") as handle:
                pd.DataFrame({"sample_id": order, "label": values}).to_csv(handle, index=False)
            outputs[name] = check_csv(output, order)
            write_json(
                output.with_suffix(".manifest.json"),
                {
                    "kind": KIND,
                    "method": name,
                    "weights": {"inner_best_neural": weight, "extra_trees": 1 - weight},
                    "recipe": recipe,
                    "input_sha256": files,
                    "input_images_sha256": images,
                    "output": outputs[name],
                    "changed_rows_vs_30_70": differences[name],
                },
            )
            print(f"Saved {output}; changed rows vs 30:70 = {differences[name]}")
        names = tuple(RATIOS)
        different = int(
            np.count_nonzero(probabilities[names[0]].argmax(1) != probabilities[names[1]].argmax(1))
        )
        unchanged(base, manifest, files, sources, args.device)
        write_json(
            status,
            {
                "status": "complete",
                "outputs": outputs,
                "changed_rows_vs_30_70": differences,
                "candidate_disagreement_rows": different,
            },
        )
        print(f"Candidate disagreement rows: {different}. Zero means identical predictions.")
        print("Two CSVs exported from cached probabilities. Existing 30:70 is unchanged.")
        return outputs
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def report(args):
    display(read_comparison(args.recipe), args.class_details)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    comparison = commands.add_parser(
        "compare", help="Compare fixed ratios on cached Validation probabilities"
    )
    export = commands.add_parser(
        "predict", help="Export both candidates from cached Test probabilities"
    )
    for command in (comparison, export):
        for name in ("input", "image_root", "cache_dir", "output_dir"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument(
            "--device",
            choices=("cpu", "cuda"),
            default="cuda",
            help="Original inference device, checked for cache provenance only",
        )
    comparison.add_argument("--baseline_recipe", type=Path, required=True)
    comparison.set_defaults(handler=compare)
    export.add_argument("--recipe", type=Path, required=True)
    export.add_argument("--sample_submission", type=Path, required=True)
    export.set_defaults(handler=predict)
    saved = commands.add_parser("report", help="Read saved Validation results")
    saved.add_argument("--recipe", type=Path, required=True)
    saved.add_argument("--class_details", action="store_true")
    saved.set_defaults(handler=report)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
