"""Compare saved neural-seed candidates with a fixed 50% ExtraTrees contribution."""

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
from experiments.seungbeom import seed_blend_core as core  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    checked_score,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from schema import LABELS, read_frame  # noqa: E402

KIND = "seungbeom_seed_blend"
CANDIDATES = ("baseline_50_50", "seed44_50_50", "seed_mean_50_50")
MEMBERS = (
    "original_neural",
    "extra_trees",
    "neural_seed42",
    "neural_seed43",
    "neural_seed44",
    "neural_mean",
)
REPLAY_MEMBERS = MEMBERS[:-1] + ("baseline_50_50",)
RULES = {
    "baseline_50_50": {"original_neural": 0.5, "extra_trees": 0.5},
    "seed44_50_50": {"neural_seed44": 0.5, "extra_trees": 0.5},
    "seed_mean_50_50": {
        "neural_seed42": 1 / 6,
        "neural_seed43": 1 / 6,
        "neural_seed44": 1 / 6,
        "extra_trees": 0.5,
    },
}
SOURCE_FILES = (
    *midpoint.SOURCE_FILES,
    "experiments/seungbeom/blend_ratios.py",
    "experiments/seungbeom/neural_seed_repeat.py",
    "experiments/seungbeom/neural_sweep.py",
    "experiments/seungbeom/seed_blend_core.py",
    "experiments/seungbeom/seed_blend.py",
)


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def check_unchanged(base, provenance, device, sources, files):
    midpoint.check_files(provenance["files"])
    midpoint.check_files(files)
    if (
        source_hashes() != sources
        or midpoint.runtime_signature(device) != base["runtime_signature"]
    ):
        raise ValueError("Source code or runtime changed during execution.")


def show_metrics(metrics, *, members=False):
    reference = checked_score(metrics["baseline_50_50"])
    print("method\tvalidation_macro_f1\tdelta_vs_baseline\t" + "\t".join(f"f1_{i}" for i in LABELS))
    for name in CANDIDATES + (MEMBERS if members else ()):
        item = metrics[name]
        print(
            f"{name}\t{item['macro_f1']:.6f}\t{item['macro_f1'] - reference:+.6f}\t"
            + "\t".join(f"{item['class_f1'][str(i)]:.6f}" for i in LABELS)
        )


def compare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    files = {str(path.resolve()): sha256(path) for path in (args.recipe, args.input)}
    base = read_json(args.recipe)
    loaded, provenance = core.load_members(base, args.experiment_dir, args.device)
    digest = files[str(args.input.resolve())]
    if digest != provenance["validation_csv_sha256"] or digest == provenance["train_csv_sha256"]:
        raise ValueError("Compare requires the original held-out Validation CSV, not Train/Test.")
    frame = read_frame(args.input, labeled=True)
    sources = source_hashes()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        print("Extracting images once for all saved neural members and ExtraTrees...", flush=True)
        probabilities = core.method_probabilities(
            frame.drop(columns="label"),
            args.image_root,
            loaded,
            args.device,
            image_precision=base["image_precision"],
        )
        if set(probabilities) != set(CANDIDATES + MEMBERS):
            raise ValueError("Missing or unexpected candidate probabilities.")
        metrics = {
            name: classification_metrics(frame["label"], values.argmax(axis=1) + 1)
            for name, values in probabilities.items()
        }
        expected = provenance["expected_metrics"]
        if set(expected) != set(REPLAY_MEMBERS):
            raise ValueError("Incomplete saved member metrics.")
        # Match all aggregate metrics including confusion counts, not only Macro F1.
        # This does not establish equality of every original per-row prediction.
        replay = {name: metrics[name] == expected[name] for name in REPLAY_MEMBERS}
        write_json(
            args.output_dir / "validation_metrics.json",
            {"metrics": metrics, "replay_matches": replay},
        )
        if not all(replay.values()):
            failed = ", ".join(name for name, ok in replay.items() if not ok)
            raise ValueError("Saved Validation replay differs for: " + failed)
        check_unchanged(base, provenance, args.device, sources, files)
        recipe = {
            "kind": KIND,
            "format_version": 1,
            "status": "complete",
            "label_values": list(LABELS),
            "candidates": list(CANDIDATES),
            "rules": RULES,
            "aggregation": "seed_probability_mean_then_50_50",
            "base_recipe": base,
            "experiment_dir": str(args.experiment_dir.resolve()),
            "provenance": provenance,
            "source_sha256": sources,
            "input_sha256": files,
            "metrics": metrics,
            "replay_matches": replay,
        }
        write_json(args.output_dir / "recipe.json", recipe)
        write_json(status, {"status": "complete"})
        show_metrics(metrics, members=getattr(args, "members", False))
        print("All saved members and the original baseline replayed successfully.")
        print("Comparison saved. No training or Test access; the original submission is unchanged.")
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
        or recipe.get("candidates") != list(CANDIDATES)
        or recipe.get("rules") != RULES
        or recipe.get("aggregation") != "seed_probability_mean_then_50_50"
        or recipe.get("replay_matches") != dict.fromkeys(REPLAY_MEMBERS, True)
    ):
        raise ValueError("Use a completed fixed seed-blend comparison recipe.")
    for name in CANDIDATES + MEMBERS:
        checked_score(recipe["metrics"][name])
    expected = recipe["provenance"]["expected_metrics"]
    if set(expected) != set(REPLAY_MEMBERS) or any(
        recipe["metrics"][name] != expected[name] for name in REPLAY_MEMBERS
    ):
        raise ValueError("Saved replay metrics disagree with member provenance.")
    return recipe


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    names = args.candidates
    if not names or len(names) != len(set(names)) or any(name not in CANDIDATES for name in names):
        raise ValueError("Choose distinct fixed seed-blend candidates.")
    files = {
        str(path.resolve()): sha256(path)
        for path in (args.recipe, args.input, args.sample_submission)
    }
    recipe = read_comparison(args.recipe)
    if recipe["source_sha256"] != source_hashes():
        raise ValueError("Source code changed; compare again before predicting.")
    midpoint.check_files(recipe["provenance"]["files"])
    base = recipe["base_recipe"]
    loaded, provenance = core.load_members(base, Path(recipe["experiment_dir"]), args.device)
    if provenance != recipe["provenance"]:
        raise ValueError("Saved models or experiment metadata changed after Validation.")
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    ordered = template_ids(args.sample_submission, ids)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        probabilities = core.method_probabilities(
            frame.drop(columns="label", errors="ignore"),
            args.image_root,
            loaded,
            args.device,
            image_precision=base["image_precision"],
        )
        check_unchanged(base, provenance, args.device, recipe["source_sha256"], files)
        outputs = {}
        for name in names:
            labels = probabilities[name].argmax(axis=1) + 1
            values = pd.Series(labels, index=ids).loc[ordered].to_numpy(dtype=np.int64)
            output = args.output_dir / f"submission_{name}.csv"
            with output.open("x", encoding="utf-8", newline="") as handle:
                pd.DataFrame({"sample_id": ordered, "label": values}).to_csv(handle, index=False)
            outputs[name] = check_csv(output, ordered)
            write_json(
                output.with_suffix(".manifest.json"),
                {
                    "kind": KIND,
                    "candidate": name,
                    "rule": RULES[name],
                    "input_sha256": files,
                    "recipe": recipe,
                    "output": outputs[name],
                },
            )
            print("Submission CSV saved:", output)
        check_unchanged(base, provenance, args.device, recipe["source_sha256"], files)
        write_json(status, {"status": "complete", "outputs": outputs})
        return outputs
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def report(args):
    show_metrics(read_comparison(args.recipe)["metrics"], members=getattr(args, "members", False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("compare", help="Compare saved seed blends on original Validation")
    export = commands.add_parser("predict", help="Apply fixed validated blends without fitting")
    for command in (check, export):
        for name in ("recipe", "input", "image_root", "output_dir"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    check.add_argument("--experiment_dir", type=Path, required=True)
    check.add_argument(
        "--members", action="store_true", help="Also show individual neural/tree scores"
    )
    check.set_defaults(handler=compare)
    export.add_argument("--sample_submission", type=Path, required=True)
    export.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES[1:]))
    export.set_defaults(handler=predict)
    saved = commands.add_parser("report", help="Print saved comparison without inference")
    saved.add_argument("--recipe", type=Path, required=True)
    saved.add_argument("--members", action="store_true")
    saved.set_defaults(handler=report)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
