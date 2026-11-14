"""Compare fixed 35:65/30:70 inner_best/ExtraTrees ratios without fitting.

Keep the successful 40:60 recipe and original inference code unchanged.
Optional server-local probability caches are bound to exact inputs and models.
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
from experiments.seungbeom import inner_best_ratios as original  # noqa: E402
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

KIND = "seungbeom_inner_best_refine"
SELF = "experiments/seungbeom/inner_best_refine.py"
BASELINE = "inner_best_40_60"
RATIOS = {"inner_best_35_65": 0.35, "inner_best_30_70": 0.30}
METHODS = ("inner_best_neural", "extra_trees", BASELINE, *RATIOS)
CACHE_KIND = "seungbeom_inner_best_member_probabilities"
CACHE_FILES = ("metadata.json", "probabilities.npz")
checkpoint = original.checkpoint
check_files = original.blend.midpoint.check_files


def source_hashes():
    return {**original.source_hashes(), SELF: sha256(ROOT / SELF)}


def load_reference(path, device):
    """Validate provenance without instantiating models on a cache hit."""
    files = original.file_hashes([path, path.parent / "status.json"])
    base = original.read_comparison(path)
    if base["source_sha256"] != original.source_hashes():
        raise ValueError("Original inference code changed; use the original ratio revision.")
    check_files(base["run_files"])
    manifest, result = checkpoint.read_result(Path(base["run_dir"]))
    compatibility = original.check_training_sources(
        manifest["source_sha256"], checkpoint.source_hashes()
    )
    target = checkpoint.ft.select_device(device)
    checkpoint.ft.seed_everything(manifest["config"]["seed"])
    if (
        compatibility != base["source_compatibility"]
        or manifest["runtime"] != base["runtime"]
        or checkpoint.runtime_signature(target) != base["runtime"]
        or base["validation"] != manifest["inputs"]["validation"]
        or any(base["metrics"][name] != result["metrics"][name] for name in original.REPLAY)
        or original.run_files(Path(base["run_dir"])) != base["run_files"]
    ):
        raise ValueError("Baseline models/runtime/replay provenance changed.")
    files.update(base["run_files"])
    files.update(manifest["reference"]["base_recipe"]["provenance"]["files"])
    check_files(files)
    return base, manifest, files


def unchanged(base, manifest, files, sources, device):
    original.check_unchanged(manifest, base["source_sha256"], files, device)
    if source_hashes() != sources:
        raise ValueError("Refinement code changed during execution.")


def validate_members(ids, neural, tree):
    if neural.shape != (len(ids), len(LABELS)) or tree.shape != neural.shape:
        raise ValueError("Cached probabilities must match current rows and six labels.")
    if neural.dtype.kind != "f" or tree.dtype.kind != "f":
        raise ValueError("Member probabilities must be floating point arrays.")
    validate_probabilities(np.stack([neural, tree], axis=1))


def read_cache(directory, identity, ids):
    status_path = directory / "status.json"
    if not status_path.is_file():
        raise ValueError("Cache incomplete; choose a new cache_dir.")
    status = read_json(status_path)
    if status.get("status") != "complete" or set(status.get("files", {})) != set(CACHE_FILES):
        raise ValueError("Cache incomplete; choose a new cache_dir.")
    hashes = {str((directory / name).resolve()): digest for name, digest in status["files"].items()}
    hashes[str(status_path.resolve())] = sha256(status_path)
    check_files(hashes)
    if read_json(directory / "metadata.json") != identity:
        raise ValueError("Cache inputs/models/runtime differ; choose a new cache_dir.")
    with np.load(directory / "probabilities.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"sample_ids", "neural", "extra_trees"}:
            raise ValueError("Invalid cache member arrays.")
        if saved["sample_ids"].tolist() != ids:
            raise ValueError("Cache sample order differs from current input.")
        neural, tree = saved["neural"], saved["extra_trees"]
    validate_members(ids, neural, tree)
    check_files(hashes)
    return (neural, tree), hashes


def save_cache(directory, identity, ids, neural, tree):
    validate_members(ids, neural, tree)
    directory.mkdir(parents=True, exist_ok=False)
    status = directory / "status.json"
    write_json(status, {"status": "running"})
    try:
        write_json(directory / "metadata.json", identity)
        with (directory / "probabilities.npz").open("xb") as handle:
            # Preserve the original member dtypes; never round or quantize cached values.
            np.savez(handle, sample_ids=np.asarray(ids, dtype=str), neural=neural, extra_trees=tree)
        write_json(
            status,
            {
                "status": "complete",
                "files": {name: sha256(directory / name) for name in CACHE_FILES},
            },
        )
        return original.file_hashes([status, *(directory / name for name in CACHE_FILES)])
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def member_probabilities(args, frame, base, manifest, files, sources, images):
    identity = {
        "kind": CACHE_KIND,
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
    cache = args.cache_dir
    ids = frame["sample_id"].tolist()
    if cache is not None and cache.exists():
        probabilities, cache_files = read_cache(cache, identity, ids)
        files.update(cache_files)
        print("Using verified member probability cache; no model inference.", flush=True)
    else:
        loaded_manifest, _, _, _, loaded = original.load_members(Path(base["run_dir"]), args.device)
        if loaded_manifest != manifest:
            raise ValueError("Models changed before inference.")
        probabilities = original.infer_members(frame, args.image_root, manifest, loaded)
        validate_members(ids, *probabilities)
        unchanged(base, manifest, files, sources, args.device)
        if original.image_digest(frame, args.image_root) != images:
            raise ValueError("Images changed during inference.")
        if cache is not None:
            files.update(save_cache(cache, identity, ids, *probabilities))
            print("Member probability cache saved:", cache, flush=True)
    unchanged(base, manifest, files, sources, args.device)
    if original.image_digest(frame, args.image_root) != images:
        raise ValueError("Images changed during inference.")
    return probabilities


def candidates(neural, tree):
    values = validate_probabilities(np.stack([neural, tree], axis=1))
    result = {"inner_best_neural": neural, "extra_trees": tree}
    result.update(
        {name: original.weighted_probabilities(neural, tree, name) for name in original.RATIOS}
    )
    result.update(
        {
            name: weight * values[:, 0] + (1 - weight) * values[:, 1]
            for name, weight in RATIOS.items()
        }
    )
    return result


def changes(probabilities):
    reference = probabilities[BASELINE].argmax(1)
    return {
        name: int(np.count_nonzero(value.argmax(1) != reference))
        for name, value in probabilities.items()
    }


def display(recipe, class_details=False):
    metrics = recipe["metrics"]
    reference = checked_score(metrics[BASELINE])
    print("Ratios = inner_best neural : original ExtraTrees; baseline is 40:60.")
    print(
        "method\tvalidation_macro_f1\tdelta_vs_40_60\tchanged_rows_vs_40_60\t"
        + "\t".join(f"f1_{i}" for i in LABELS)
    )
    for name in METHODS:
        item = metrics[name]
        print(
            f"{name}\t{item['macro_f1']:.6f}\t{item['macro_f1'] - reference:+.6f}\t"
            f"{recipe['changed_rows'][name]}\t"
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
                    f"{item['class_recall'][key]:.6f}\t{item['class_f1'][key]:.6f}\t{item['support'][key]}"
                )


def compare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    sources = source_hashes()
    base, manifest, files = load_reference(args.baseline_recipe, args.device)
    files.update(original.file_hashes([args.input]))
    digest = files[str(args.input.resolve())]
    if (
        digest != base["validation"]["csv_sha256"]
        or digest == manifest["inputs"]["train"]["csv_sha256"]
    ):
        raise ValueError("Compare requires the original official Validation CSV, not Train/Test.")
    frame = read_frame(args.input, labeled=True)
    images = original.image_digest(frame, args.image_root)
    if images != base["validation"]["images_sha256"]:
        raise ValueError("Validation images differ from the baseline run.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        probabilities = candidates(
            *member_probabilities(args, frame, base, manifest, files, sources, images)
        )
        metrics = {
            name: classification_metrics(frame["label"], value.argmax(1) + 1)
            for name, value in probabilities.items()
        }
        replay = {name: metrics[name] == base["metrics"][name] for name in original.METHODS}
        write_json(
            args.output_dir / "validation_metrics.json",
            {"metrics": metrics, "replay_matches": replay},
        )
        if not all(replay.values()):
            raise ValueError(
                "Original Validation replay differs, including 40:60; inspect before export."
            )
        unchanged(base, manifest, files, sources, args.device)
        recipe = {
            "kind": KIND,
            "format_version": 1,
            "status": "complete",
            "label_values": list(LABELS),
            "ratios_neural_weight": RATIOS,
            "baseline_recipe_path": str(args.baseline_recipe.resolve()),
            "baseline_recipe": base,
            "source_sha256": sources,
            "metrics": metrics,
            "replay_matches": replay,
            "changed_rows": changes(probabilities),
            "validation_provenance": files,
            "selection": "two_predefined_ratios_no_fit_no_test",
        }
        path = args.output_dir / "recipe.json"
        write_json(path, recipe)
        write_json(status, {"status": "complete", "recipe_sha256": sha256(path)})
        display(recipe)
        print("Comparison complete. Original members/40:60 replayed; no training or Test access.")
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
        or recipe.get("replay_matches") != dict.fromkeys(original.METHODS, True)
        or set(recipe.get("metrics", {})) != set((*original.METHODS, *RATIOS))
    ):
        raise ValueError("Invalid refinement recipe.")
    for name, metric in recipe["metrics"].items():
        checked_score(metric)
        if name in original.METHODS and metric != recipe["baseline_recipe"]["metrics"][name]:
            raise ValueError("Recorded baseline metrics disagree.")
    return recipe


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    recipe = read_comparison(args.recipe)
    sources = source_hashes()
    if recipe["source_sha256"] != sources:
        raise ValueError("Comparison code changed; compare again in a new directory.")
    base, manifest, files = load_reference(Path(recipe["baseline_recipe_path"]), args.device)
    if base != recipe["baseline_recipe"]:
        raise ValueError("Baseline recipe changed after comparison.")
    # Deliberately do not open the Validation CSV or cache for Test inference.
    files.update(
        original.file_hashes(
            [args.recipe, args.recipe.parent / "status.json", args.input, args.sample_submission]
        )
    )
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    order = template_ids(args.sample_submission, ids)
    images = original.image_digest(frame, args.image_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        probabilities = candidates(
            *member_probabilities(args, frame, base, manifest, files, sources, images)
        )
        differences = changes(probabilities)
        outputs = {}
        for name, weight in RATIOS.items():
            values = (
                pd.Series(probabilities[name].argmax(1) + 1, index=ids)
                .loc[order]
                .to_numpy(dtype=np.int64)
            )
            output = args.output_dir / f"submission_{name}.csv"
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
                    "changed_rows_vs_40_60": differences[name],
                },
            )
            print(f"Saved {output}; changed rows vs 40:60 = {differences[name]}")
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
                "changed_rows_vs_40_60": {name: differences[name] for name in RATIOS},
                "candidate_disagreement_rows": different,
            },
        )
        print(
            f"Candidate disagreement rows: {different}. "
            "Zero means the two CSVs predict identical labels."
        )
        print("Two CSVs exported. Existing 40:60/50:50 artifacts are unchanged; no training.")
        return outputs
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def report(args):
    display(read_comparison(args.recipe), args.class_details)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    comparison = commands.add_parser("compare", help="Replay 40:60 and compare two fixed ratios")
    export = commands.add_parser("predict", help="Export 35:65 and 30:70 together without fitting")
    for command in (comparison, export):
        for name in ("input", "image_root", "output_dir"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument(
            "--cache_dir", type=Path, help="Optional exact-input member probability cache"
        )
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    comparison.add_argument("--baseline_recipe", type=Path, required=True)
    comparison.set_defaults(handler=compare)
    export.add_argument("--recipe", type=Path, required=True)
    export.add_argument("--sample_submission", type=Path, required=True)
    export.set_defaults(handler=predict)
    saved = commands.add_parser("report", help="Read saved metrics without inference")
    saved.add_argument("--recipe", type=Path, required=True)
    saved.add_argument("--class_details", action="store_true")
    saved.set_defaults(handler=report)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
