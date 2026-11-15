"""Compare three fixed neural/ExtraTrees ratios using an already replayed midpoint recipe.

User-run organizer server commands only. No training, automated weight fitting,
or Test-based selection. Existing midpoint source files and recipes remain unchanged.
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
from experiments.seungbeom import midpoint_blend as midpoint  # noqa: E402
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

# Values are the neural weight, with the complement assigned to ExtraTrees.
CANDIDATES = {"blend_50_50": 0.5, "blend_75_25": 0.75, "blend_25_75": 0.25}
METHODS = ("neural", "extra_trees", *CANDIDATES)
KIND = "seungbeom_blend_ratios"
SELF = "experiments/seungbeom/blend_ratios.py"


def source_hashes():
    return {**midpoint.source_hashes(), SELF: sha256(ROOT / SELF)}


def weighted_probabilities(neural_prob, tree_prob, candidate):
    if candidate not in CANDIDATES:
        raise ValueError("Use one of the three predefined blend candidates.")
    if candidate == "blend_50_50":
        # Preserve the established candidate's exact arithmetic.
        return midpoint.blend_probabilities(neural_prob, tree_prob)
    left, right = np.asarray(neural_prob), np.asarray(tree_prob)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Members must have matching [rows, 6] probability shapes.")
    values = validate_probabilities(np.stack([left, right], axis=1))
    weight = CANDIDATES[candidate]
    return weight * values[:, 0] + (1.0 - weight) * values[:, 1]


def load_base(recipe, device):
    """Same compatibility checks as midpoint prediction; never read a Validation cache."""
    from runtime import seed_everything

    if (
        recipe.get("kind") != midpoint.KIND
        or recipe.get("format_version") != 2
        or recipe.get("status") != "complete"
        or recipe.get("rule") != midpoint.RULE
        or recipe.get("label_values") != list(LABELS)
        or recipe.get("image_precision") not in midpoint.POLICIES
        or not recipe.get("runtime_signature")
    ):
        raise ValueError("Use a completed v2 midpoint Validation recipe.")
    audit = recipe.get("precision_audit")
    policy = audit.get("image_precision") if isinstance(audit, dict) else "fp32"
    if recipe["image_precision"] != policy:
        raise ValueError("Image precision differs from the original audit.")
    if recipe.get("source_sha256") != midpoint.source_hashes():
        raise ValueError("Original source code changed; replay midpoint Validation first.")
    midpoint.check_files(recipe["provenance"]["files"])
    loaded, provenance = midpoint.load_sources(
        Path(recipe["neural_run"]), Path(recipe["ensemble_run"])
    )
    if provenance != recipe["provenance"]:
        raise ValueError("Source models differ from the midpoint recipe.")
    seed_everything(loaded["config"]["seed"])
    if midpoint.runtime_signature(device) != recipe["runtime_signature"]:
        raise ValueError("Runtime differs; use the environment of the midpoint recipe.")
    for name in midpoint.METHODS:
        checked_score(recipe["metrics"][name])
    return loaded


def check_unchanged(base, device, sources, files):
    midpoint.check_files(base["provenance"]["files"])
    midpoint.check_files(files)
    if (
        source_hashes() != sources
        or midpoint.runtime_signature(device) != base["runtime_signature"]
    ):
        raise ValueError("Source code or runtime changed during execution.")


def show_metrics(metrics):
    reference = checked_score(metrics["blend_50_50"])
    print("method\tvalidation_macro_f1\tdelta_vs_50_50\t" + "\t".join(f"f1_{i}" for i in LABELS))
    for name in METHODS:
        item = metrics[name]
        print(
            f"{name}\t{item['macro_f1']:.6f}\t{item['macro_f1'] - reference:+.6f}\t"
            + "\t".join(f"{item['class_f1'][str(i)]:.6f}" for i in LABELS)
        )


def compare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    files = {
        str(args.recipe.resolve()): sha256(args.recipe),
        str(args.input.resolve()): sha256(args.input),
    }
    base = read_json(args.recipe)
    loaded = load_base(base, args.device)
    digest = files[str(args.input.resolve())]
    if (
        digest != base["provenance"]["validation_csv_sha256"]
        or digest == base["provenance"]["train_csv_sha256"]
    ):
        raise ValueError("Compare requires the original held-out Validation CSV, not Train/Test.")
    frame = read_frame(args.input, labeled=True)
    sources = source_hashes()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        members = midpoint.method_probabilities(
            frame.drop(columns="label"),
            args.image_root,
            loaded,
            args.device,
            image_precision=base["image_precision"],
        )
        probabilities = {name: members[name] for name in ("neural", "extra_trees")}
        probabilities.update(
            {
                name: weighted_probabilities(members["neural"], members["extra_trees"], name)
                for name in CANDIDATES
            }
        )
        metrics = {
            name: classification_metrics(frame["label"], values.argmax(axis=1) + 1)
            for name, values in probabilities.items()
        }
        # The entire aggregate metric dict, including confusion counts, must replay.
        # This still does not prove identical per-row predictions to the old run.
        replay = {name: metrics[name] == base["metrics"][name] for name in midpoint.METHODS}
        write_json(
            args.output_dir / "validation_metrics.json",
            {"metrics": metrics, "replay_matches": replay},
        )
        if not all(replay.values()):
            raise ValueError("Member or 50:50 Validation replay differs from the original recipe.")
        check_unchanged(base, args.device, sources, files)
        recipe = {
            "kind": KIND,
            "format_version": 1,
            "status": "complete",
            "label_values": list(LABELS),
            "candidates": CANDIDATES.copy(),
            "base_recipe": base,
            "input_sha256": files,
            "source_sha256": sources,
            "metrics": metrics,
            "replay_matches": replay,
            "weight_selection": "three_predefined_ratios_compared_on_validation_no_fit_no_test",
        }
        write_json(args.output_dir / "recipe.json", recipe)
        write_json(status, {"status": "complete"})
        show_metrics(metrics)
        print(
            "Comparison saved. Validation is used for candidate comparison, "
            "not an independent final score."
        )
        return recipe
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def read_comparison(path):
    recipe = read_json(path)
    if (
        recipe.get("kind") != KIND
        or recipe.get("format_version") != 1
        or recipe.get("status") != "complete"
        or recipe.get("label_values") != list(LABELS)
        or recipe.get("candidates") != CANDIDATES
        or recipe.get("replay_matches") != dict.fromkeys(midpoint.METHODS, True)
    ):
        raise ValueError("Use a completed fixed-ratio comparison recipe.")
    for name in METHODS:
        checked_score(recipe["metrics"][name])
    if any(
        recipe["metrics"][name] != recipe["base_recipe"]["metrics"][name]
        for name in midpoint.METHODS
    ):
        raise ValueError("Recorded replay metrics disagree with the original recipe.")
    return recipe


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    candidates = args.candidates
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(name not in CANDIDATES for name in candidates)
    ):
        raise ValueError("Choose distinct predefined candidates.")
    files = {
        str(path.resolve()): sha256(path)
        for path in (args.recipe, args.input, args.sample_submission)
    }
    recipe = read_comparison(args.recipe)
    if recipe.get("source_sha256") != source_hashes():
        raise ValueError("Comparison source code changed; compare again before predicting.")
    base = recipe["base_recipe"]
    loaded = load_base(base, args.device)
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    ordered = template_ids(args.sample_submission, ids)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        members = midpoint.method_probabilities(
            frame.drop(columns="label", errors="ignore"),
            args.image_root,
            loaded,
            args.device,
            image_precision=base["image_precision"],
        )
        check_unchanged(base, args.device, recipe["source_sha256"], files)
        outputs = {}
        for candidate in candidates:
            labels = (
                weighted_probabilities(members["neural"], members["extra_trees"], candidate).argmax(
                    axis=1
                )
                + 1
            )
            values = pd.Series(labels, index=ids).loc[ordered].to_numpy(dtype=np.int64)
            output = args.output_dir / f"submission_{candidate}.csv"
            with output.open("x", encoding="utf-8", newline="") as handle:
                pd.DataFrame({"sample_id": ordered, "label": values}).to_csv(handle, index=False)
            outputs[candidate] = check_csv(output, ordered)
            weight = CANDIDATES[candidate]
            write_json(
                output.with_suffix(".manifest.json"),
                {
                    "kind": KIND,
                    "candidate": candidate,
                    "rule": {
                        "method": "probability_mean",
                        "neural": weight,
                        "extra_trees": 1 - weight,
                    },
                    "input_sha256": files,
                    "recipe": recipe,
                    "output": outputs[candidate],
                },
            )
            print("Submission CSV saved:", output)
        check_unchanged(base, args.device, recipe["source_sha256"], files)
        write_json(status, {"status": "complete", "outputs": outputs})
        return outputs
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def report(args):
    show_metrics(read_comparison(args.recipe)["metrics"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "compare", help="Replay the successful recipe and compare three ratios"
    )
    export = commands.add_parser(
        "predict", help="Export previously compared fixed candidates; no fitting"
    )
    for command in (check, export):
        for name in ("recipe", "input", "image_root", "output_dir"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    check.set_defaults(handler=compare)
    export.add_argument("--sample_submission", type=Path, required=True)
    export.add_argument(
        "--candidates", nargs="+", choices=tuple(CANDIDATES), default=["blend_75_25", "blend_25_75"]
    )
    export.set_defaults(handler=predict)
    saved = commands.add_parser("report", help="Print saved Validation scores without inference")
    saved.add_argument("--recipe", type=Path, required=True)
    saved.set_defaults(handler=report)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
