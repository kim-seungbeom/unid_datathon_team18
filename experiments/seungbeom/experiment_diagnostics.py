"""Server-only group-size and Validation-error diagnosis of the saved frozen/tree blend.

No fitting, sampling changes, Test access, or submission export. Existing models
are replayed with their original inference paths. Only grouped aggregates are saved.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import classification_metrics  # noqa: E402
from experiments.seungbeom import frozen_tree_blend as blend  # noqa: E402
from experiments.seungbeom.ensemble_core import validate_probabilities  # noqa: E402
from experiments.seungbeom.export_submissions import read_json, sha256, write_json  # noqa: E402
from schema import LABELS, read_frame, validate_labels, validate_split  # noqa: E402

KIND = "seungbeom_experiment_diagnostics"
SELF = "experiments/seungbeom/experiment_diagnostics.py"
METHODS = ("frozen", "extra_trees", "frozen_tree_50_50")
TABLES = (
    "experiment_counts.csv",
    "class_coverage.csv",
    "validation_group_metrics.csv",
    "validation_group_classes.csv",
    "validation_confusions.csv",
)


def metadata(frame):
    required = ["sample_id", "experiment_id", "label"]
    if frame.empty or not set(required).issubset(frame.columns):
        raise ValueError("Nonempty labeled rows with sample_id and experiment_id are required.")
    result = frame[required].copy().reset_index(drop=True)
    for column in ("sample_id", "experiment_id"):
        if (
            result[column].isna().any()
            or not result[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError("Sample and experiment IDs must be nonempty strings.")
    if not result["sample_id"].is_unique:
        raise ValueError("Sample IDs must be unique within a split.")
    result["label"] = validate_labels(result["label"])
    return result


def population(frame, split):
    frame = metadata(frame)
    counts = pd.crosstab(frame["experiment_id"], frame["label"]).reindex(
        columns=LABELS, fill_value=0
    )
    rows = counts.sum(axis=1)
    table = counts.rename(columns={label: f"n_label_{label}" for label in LABELS}).reset_index()
    table.insert(0, "split", split)
    table["rows"] = rows.to_numpy()
    table["row_share"] = table["rows"] / len(frame)
    table["classes_present"] = (counts > 0).sum(axis=1).to_numpy()
    table = table.sort_values(["rows", "experiment_id"], ascending=[False, True])
    largest_n = max(1, math.ceil(0.1 * len(table)))
    summary = {
        "rows": len(frame),
        "experiments": len(table),
        "min_rows": int(rows.min()),
        "median_rows": float(rows.median()),
        "max_rows": int(rows.max()),
        "max_to_min_rows": float(rows.max() / rows.min()),
        "largest_10pct_groups_n": largest_n,
        "largest_10pct_groups_row_share": float(table.head(largest_n)["row_share"].sum()),
        "effective_groups_by_row_share": float(1 / np.square(table["row_share"]).sum()),
    }
    coverage = []
    for label in LABELS:
        support = int(counts[label].sum())
        coverage.append(
            {
                "split": split,
                "class": label,
                "rows": support,
                "row_share": support / len(frame),
                "experiments_present": int((counts[label] > 0).sum()),
                "largest_group_class_share": float(counts[label].max() / support)
                if support
                else None,
            }
        )
    return summary, table, pd.DataFrame(coverage)


def grouped_errors(frame, predictions, method):
    """All metrics are descriptive; missing true classes have undefined recall."""
    frame = metadata(frame)
    predicted = validate_labels(predictions)
    if len(predicted) != len(frame):
        raise ValueError("Prediction row count differs from Validation.")
    true = frame["label"].to_numpy()
    total_errors = int(np.count_nonzero(true != predicted))
    groups, classes, confusions = [], [], []
    for group, indices in frame.groupby("experiment_id", sort=True).indices.items():
        metrics = classification_metrics(true[indices], predicted[indices])
        matrix = np.asarray(metrics["confusion_matrix"])
        support, predicted_counts = matrix.sum(axis=1), matrix.sum(axis=0)
        errors = int(len(indices) - np.trace(matrix))
        present = support > 0
        group_f1 = np.asarray([metrics["class_f1"][str(c)] for c in LABELS])
        groups.append(
            {
                "method": method,
                "experiment_id": group,
                "rows": len(indices),
                "row_share": len(indices) / len(frame),
                "errors": errors,
                "error_rate": errors / len(indices),
                "error_share": errors / total_errors if total_errors else 0.0,
                "classes_present": int(present.sum()),
                "macro_f1_6": metrics["macro_f1"],
                "macro_f1_present": float(group_f1[present].mean()),
            }
        )
        for label in LABELS:
            i = label - 1
            classes.append(
                {
                    "method": method,
                    "experiment_id": group,
                    "class": label,
                    "support": int(support[i]),
                    "predicted": int(predicted_counts[i]),
                    "errors": int(support[i] - matrix[i, i]),
                    "recall": metrics["class_recall"][str(label)] if support[i] else None,
                    "precision": metrics["class_precision"][str(label)]
                    if predicted_counts[i]
                    else None,
                    "f1": metrics["class_f1"][str(label)]
                    if support[i] + predicted_counts[i]
                    else None,
                }
            )
            for predicted_label in LABELS:
                count = int(matrix[i, predicted_label - 1])
                if count:
                    confusions.append(
                        {
                            "method": method,
                            "experiment_id": group,
                            "true_class": label,
                            "predicted_class": predicted_label,
                            "count": count,
                        }
                    )
    groups = pd.DataFrame(groups).sort_values(["errors", "experiment_id"], ascending=[False, True])
    top = groups.head(3)
    summary = {
        "errors": total_errors,
        "error_rate": total_errors / len(frame),
        "top_error_groups_n": len(top),
        "top_error_groups_error_share": float(top["error_share"].sum()),
        "top_error_groups_row_share": float(top["row_share"].sum()),
    }
    return summary, groups, pd.DataFrame(classes), pd.DataFrame(confusions)


def show(summary, tables, *, method="frozen_tree_50_50", top=10):
    print("Train/Validation experiment sizes (shares are fractions, not percentages):")
    print("split\trows\texperiments\tmin\tmedian\tmax\tmax/min\ttop10%_groups_row_share")
    for split, value in summary["population"].items():
        print(
            f"{split}\t{value['rows']}\t{value['experiments']}\t{value['min_rows']}\t"
            f"{value['median_rows']:.1f}\t{value['max_rows']}\t{value['max_to_min_rows']:.2f}\t"
            f"{value['largest_10pct_groups_row_share']:.4f}"
        )
    counts = tables["experiment_counts.csv"]
    print("\nLargest Train experiments:")
    print(counts.loc[counts["split"] == "train"].head(top).to_string(index=False))
    print("\nClass rows and experiment coverage:")
    print(tables["class_coverage.csv"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved model replay and Validation errors:")
    print("method\tmacro_f1\terrors\terror_rate\ttop3_error_share\ttop3_groups_row_share")
    for name in METHODS:
        error = summary["errors"][name]
        print(
            f"{name}\t{summary['metrics'][name]['macro_f1']:.6f}\t{error['errors']}\t"
            f"{error['error_rate']:.4f}\t{error['top_error_groups_error_share']:.4f}\t"
            f"{error['top_error_groups_row_share']:.4f}"
        )
    print(f"\n{method}: most errors, then highest error rates (check rows/classes_present):")
    group = tables["validation_group_metrics.csv"]
    group = group.loc[group["method"] == method].drop(columns="method")
    for field in ("errors", "error_rate"):
        print(f"\nRanked by {field}:")
        print(
            group.sort_values([field, "experiment_id"], ascending=[False, True])
            .head(top)
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
    confusion = tables["validation_confusions.csv"]
    wrong = confusion.loc[
        (confusion["method"] == method) & (confusion["true_class"] != confusion["predicted_class"])
    ]
    pairs = wrong.groupby(["true_class", "predicted_class"])["count"].sum().reset_index()
    print("\nMost frequent class confusions across Validation:")
    print(
        pairs.sort_values(["count", "true_class", "predicted_class"], ascending=[False, True, True])
        .head(top)
        .to_string(index=False)
    )
    print(f"\n{method}: overall Validation class metrics:")
    print("class\tsupport\tprecision\trecall\tf1")
    selected = summary["metrics"][method]
    for label in LABELS:
        key = str(label)
        print(
            f"{label}\t{selected['support'][key]}\t{selected['class_precision'][key]:.4f}\t"
            f"{selected['class_recall'][key]:.4f}\t{selected['class_f1'][key]:.4f}"
        )
    print("\nAll saved aggregate metrics replayed. No training, Test access or submission export.")
    print("Group F1 depends on available classes; missing-class recall is NA, not zero.")
    print(
        "Error shares count mistakes, not contributions to Macro F1. "
        "Reused Validation is descriptive."
    )


def analyze(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    reference = blend.read_comparison(args.recipe)
    sources = {**blend.source_hashes(), SELF: sha256(ROOT / SELF)}
    if reference["source_sha256"] != blend.source_hashes():
        raise ValueError("Reference inference source changed; use its recorded version.")
    paths = (args.recipe, args.recipe.parent / "status.json", args.train_csv, args.validation_csv)
    files = {str(path.resolve()): sha256(path) for path in paths}
    provenance = reference["base_recipe"]["provenance"]
    for split in ("train", "validation"):
        if files[str(getattr(args, split + "_csv").resolve())] != provenance[split + "_csv_sha256"]:
            raise ValueError("Use exactly the recorded Train and Validation CSVs; no Test input.")
    train = metadata(
        pd.read_csv(
            args.train_csv,
            usecols=["sample_id", "experiment_id", "label"],
            dtype={"sample_id": str, "experiment_id": str},
            keep_default_na=False,
        )
    )
    validation = read_frame(args.validation_csv, labeled=True)
    validation_meta = metadata(validation)
    validate_split(train, validation_meta)
    populations, counts, coverage = {}, [], []
    for split, frame in (("train", train), ("validation", validation_meta)):
        populations[split], group, classes = population(frame, split)
        counts.append(group)
        coverage.append(classes)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        print(
            "Recorded CSVs verified. Counting experiments; replaying saved Validation models...",
            flush=True,
        )
        legacy, frozen, record = blend.load_members(
            reference["base_recipe"], Path(reference["frozen"]["run_dir"]), args.device
        )
        if record != reference["frozen"]:
            raise ValueError("Frozen model differs from the reference.")
        outputs = blend.infer_members(
            validation.drop(columns="label"),
            args.image_root,
            legacy,
            frozen,
            reference["base_recipe"],
            args.device,
        )
        predictions, metrics = {}, {}
        for name in blend.METHODS:
            probability = np.asarray(outputs[name])
            if probability.shape != (len(validation), 6):
                raise ValueError("Validation probabilities have the wrong row/class count.")
            validate_probabilities(probability[:, None, :])
            predictions[name] = probability.argmax(axis=1) + 1
            metrics[name] = classification_metrics(validation_meta["label"], predictions[name])
        replay = {name: metrics[name] == reference["metrics"][name] for name in blend.METHODS}
        write_json(args.output_dir / "replay.json", {"replay_matches": replay})
        if not all(replay.values()):
            raise ValueError(
                "Saved Validation metrics differ; inspect replay.json before analysis."
            )
        errors, groups, classes, confusions = {}, [], [], []
        for name in METHODS:
            errors[name], group, class_table, confusion = grouped_errors(
                validation_meta, predictions[name], name
            )
            groups.append(group)
            classes.append(class_table)
            confusions.append(confusion)
        tables = dict(
            zip(
                TABLES, map(pd.concat, (counts, coverage, groups, classes, confusions)), strict=True
            )
        )
        blend.check_unchanged(
            reference["base_recipe"], record, reference["source_sha256"], files, args.device
        )
        if sources != {**blend.source_hashes(), SELF: sha256(ROOT / SELF)}:
            raise ValueError("Diagnostic source changed during execution.")
        for filename, table in tables.items():
            table.to_csv(args.output_dir / filename, index=False)
        summary = {
            "kind": KIND,
            "format_version": 1,
            "population": populations,
            "metrics": {name: metrics[name] for name in METHODS},
            "errors": errors,
            "replay_matches": replay,
            "input_sha256": files,
            "source_sha256": sources,
            "scope": "train_counts_validation_errors_no_fit_no_test_no_row_exports",
            "notes": [
                "Experiment IDs/counts are diagnostic grouping metadata, never model inputs.",
                "No Train inference; Train fit accuracy is not held-out generalization.",
                "Per-group macro_f1_6 counts absent classes as zero; "
                "macro_f1_present uses true classes.",
                "Neither group F1 average nor error shares are the competition Macro F1.",
                "Validation has already informed model choices; this is not an independent test.",
            ],
        }
        write_json(args.output_dir / "summary.json", summary)
        write_json(
            status,
            {
                "status": "complete",
                "files": {
                    name: sha256(args.output_dir / name)
                    for name in (*TABLES, "summary.json", "replay.json")
                },
            },
        )
        show(summary, tables)
        print("Grouped reports saved:", args.output_dir)
        return summary
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def report(args):
    status = read_json(args.output_dir / "status.json")
    required = {*TABLES, "summary.json", "replay.json"}
    if status.get("status") != "complete" or set(status.get("files", {})) != required:
        raise ValueError("Analysis is incomplete; inspect the analyze log and replay.json.")
    for name, digest in status["files"].items():
        if sha256(args.output_dir / name) != digest:
            raise ValueError("Saved diagnostic output changed; use the original report files.")
    summary = read_json(args.output_dir / "summary.json")
    if (
        summary.get("kind") != KIND
        or summary.get("format_version") != 1
        or summary.get("replay_matches") != dict.fromkeys(blend.METHODS, True)
    ):
        raise ValueError("Report does not contain a successful model replay.")
    tables = {
        name: pd.read_csv(
            args.output_dir / name, dtype={"experiment_id": str}, keep_default_na=False
        )
        for name in TABLES
    }
    show(summary, tables, method=args.method, top=args.top)
    return summary


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("analyze", help="Count Train rows and replay Validation once")
    for name in ("recipe", "train_csv", "validation_csv", "image_root", "output_dir"):
        analysis.add_argument("--" + name, type=Path, required=True)
    analysis.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    analysis.set_defaults(handler=analyze)
    reporting = commands.add_parser("report", help="Read grouped results without model inference")
    reporting.add_argument("--output_dir", type=Path, required=True)
    reporting.add_argument("--method", choices=METHODS, default="frozen_tree_50_50")
    reporting.add_argument("--top", type=positive_int, default=10)
    reporting.set_defaults(handler=report)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
