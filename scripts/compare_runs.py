"""Compare validation metrics from explicit run directories; run real results on the server."""

import argparse
import csv
import json
import sys
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    rows = comparison_rows(args.run_dirs)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
