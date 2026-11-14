"""Compare validation metrics from explicit run directories; run real results on the server."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def comparison_rows(run_dirs: list[Path]) -> list[dict]:
    rows = []
    reference_split = None
    for directory in run_dirs:
        with (directory / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        with (directory / "split.json").open(encoding="utf-8") as handle:
            split = json.load(handle)
        with (directory / "best_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        identity = (split["train_csv_sha256"], split["validation_csv_sha256"])
        if reference_split is not None and identity != reference_split:
            raise ValueError(
                "Run splits differ. Compare runs with identical Train/Validation CSVs."
            )
        reference_split = identity
        rows.append(
            {
                "run": directory.name,
                "mode": config.get("mode", "multimodal"),
                "imputation": config.get("sensor_imputation", "median"),
                "scaling": config.get("sensor_scaling", "standard"),
                "loss": config.get("loss", "cross_entropy"),
                "sampling": config.get("sampling", "shuffle"),
                "features": len(config["sensor_features"]) if "sensor_features" in config else 5,
                "seed": config["seed"],
                "best_epoch": metrics["epoch"],
                "macro_f1": f"{metrics['macro_f1']:.6f}",
                **{
                    f"f1_{label}": f"{metrics['class_f1'][str(label)]:.6f}" for label in range(1, 7)
                },
            }
        )
    return rows


def class_detail_rows(run_dirs: list[Path]) -> list[dict]:
    """Derive precision/recall from stored counts, including legacy metric files."""
    rows = []
    for directory in run_dirs:
        with (directory / "best_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
        if (
            matrix.shape != (6, 6)
            or not np.isfinite(matrix).all()
            or (matrix < 0).any()
            or not np.equal(matrix, np.floor(matrix)).all()
            or matrix.sum() == 0
        ):
            raise ValueError("Expected a nonempty 6x6 confusion matrix of nonnegative counts.")
        support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
        for index in range(6):
            tp = matrix[index, index]
            denominator = support[index] + predicted[index]
            rows.append(
                {
                    "run": directory.name,
                    "class": index + 1,
                    "precision": f"{tp / predicted[index] if predicted[index] else 0:.6f}",
                    "recall": f"{tp / support[index] if support[index] else 0:.6f}",
                    "f1": f"{2 * tp / denominator if denominator else 0:.6f}",
                    "support": int(support[index]),
                    "predicted": int(predicted[index]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--class_details", action="store_true")
    args = parser.parse_args()
    rows = comparison_rows(args.run_dirs)
    details = class_detail_rows(args.run_dirs) if args.class_details else []
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    if details:
        print("\nClass precision/recall at each run's best macro-F1 epoch:")
        writer = csv.DictWriter(sys.stdout, fieldnames=list(details[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(details)


if __name__ == "__main__":
    main()
