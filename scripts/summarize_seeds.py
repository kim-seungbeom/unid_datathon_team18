"""Summarize explicit validation runs across seeds; use real results only on the server."""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import load_config  # noqa: E402


@dataclass
class RunResult:
    """One run's normalized configuration and unrounded validation metrics."""

    name: str
    config: dict
    split: tuple[str, str]
    macro_f1: float
    class_f1: dict[int, float]


@dataclass
class SeedGroup:
    """Runs sharing every normalized configuration setting except seed."""

    variant: str
    config: dict
    runs: dict[int, RunResult] = field(default_factory=dict)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def _score(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite score in [0, 1].")
    return float(value)


def load_runs(run_dirs: list[Path]) -> list[RunResult]:
    """Read only the three required metadata files in each explicitly supplied directory."""
    runs = []
    for directory in run_dirs:
        directory = Path(directory)
        config = load_config(directory / "config.json")
        split = _read_json(directory / "split.json")
        metrics = _read_json(directory / "best_metrics.json")
        try:
            identity = (split["train_csv_sha256"], split["validation_csv_sha256"])
            if any(not isinstance(value, str) or not value for value in identity):
                raise ValueError("Train/Validation CSV hashes must be nonempty strings.")
            class_f1 = metrics["class_f1"]
            if not isinstance(class_f1, dict):
                raise ValueError("class_f1 must contain scores for labels 1..6.")
            runs.append(
                RunResult(
                    name=directory.name,
                    config=config,
                    split=identity,
                    macro_f1=_score(metrics["macro_f1"], "macro_f1"),
                    class_f1={
                        label: _score(class_f1[str(label)], f"f1_{label}") for label in range(1, 7)
                    },
                )
            )
        except KeyError as error:
            raise ValueError(f"Missing required metadata field: {error.args[0]}.") from error
    return runs


def group_runs(runs: list[RunResult], require_seeds: list[int] | None = None) -> list[SeedGroup]:
    """Group in input order by full normalized config, after checking shared split hashes."""
    if not runs:
        raise ValueError("At least one run is required.")
    if require_seeds is not None:
        if not require_seeds or any(
            type(seed) is not int or not 0 <= seed < 2**32 for seed in require_seeds
        ):
            raise ValueError("Required seeds must be nonnegative integers below 2**32.")
        if len(set(require_seeds)) != len(require_seeds):
            raise ValueError("Required seeds must be unique.")
    groups = []
    used_names = set()
    for run in runs:
        if run.split != runs[0].split:
            raise ValueError(
                "Run splits differ. Compare runs with identical Train/Validation CSVs."
            )
        config = {key: value for key, value in run.config.items() if key != "seed"}
        group = next((group for group in groups if group.config == config), None)
        if group is None:
            base_name = re.sub(r"_seed\d+$", "", run.name) or run.name
            variant = base_name
            suffix = 2
            while variant in used_names:
                variant = f"{base_name}#{suffix}"
                suffix += 1
            used_names.add(variant)
            group = SeedGroup(variant=variant, config=config)
            groups.append(group)
        seed = run.config["seed"]
        if seed in group.runs:
            raise ValueError(f"Duplicate seed {seed} in variant {group.variant}.")
        group.runs[seed] = run
    if require_seeds is not None:
        expected = set(require_seeds)
        for group in groups:
            if set(group.runs) != expected:
                raise ValueError(
                    f"Variant {group.variant} has seeds {sorted(group.runs)}; "
                    f"required seeds are exactly {sorted(expected)}."
                )
    return groups


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def summary_rows(groups: list[SeedGroup]) -> list[dict]:
    """Return unrounded per-group means and sample standard deviations (ddof=1)."""
    rows = []
    for group in groups:
        seeds = sorted(group.runs)
        mean, std = _mean_std([group.runs[seed].macro_f1 for seed in seeds])
        row = {
            "variant": group.variant,
            "mode": group.config["mode"],
            "imputation": group.config["sensor_imputation"],
            "scaling": group.config["sensor_scaling"],
            "features": len(group.config["sensor_features"]),
            "loss": group.config["loss"],
            "sampling": group.config["sampling"],
            "seeds": tuple(seeds),
            "n": len(seeds),
            "macro_f1_mean": mean,
            "macro_f1_std": std,
        }
        for label in range(1, 7):
            mean, std = _mean_std([group.runs[seed].class_f1[label] for seed in seeds])
            row[f"f1_{label}_mean"] = mean
            row[f"f1_{label}_std"] = std
        rows.append(row)
    return rows


def paired_rows(groups: list[SeedGroup]) -> list[dict]:
    """Compare each later group with the first, matching exactly by seed."""
    if len(groups) < 2:
        raise ValueError("Paired comparison requires at least two configuration groups.")
    reference = groups[0]
    seeds = sorted(reference.runs)
    rows = []
    for candidate in groups[1:]:
        if set(candidate.runs) != set(seeds):
            raise ValueError("Paired comparison requires identical seed sets in every group.")
        deltas = {
            seed: candidate.runs[seed].macro_f1 - reference.runs[seed].macro_f1 for seed in seeds
        }
        mean, std = _mean_std(list(deltas.values()))
        rows.append(
            {
                "reference": reference.variant,
                "candidate": candidate.variant,
                "seeds": tuple(seeds),
                "n": len(seeds),
                "macro_f1_delta_mean": mean,
                "macro_f1_delta_std": std,
                "wins": sum(delta > 0 for delta in deltas.values()),
                "ties": sum(delta == 0 for delta in deltas.values()),
                "losses": sum(delta < 0 for delta in deltas.values()),
                "seed_deltas": deltas,
            }
        )
    return rows


def format_rows(rows: list[dict]) -> list[dict]:
    """Format only at display time; keep numeric calculation rows unchanged."""

    def display(value):
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.6f}"
        if isinstance(value, tuple):
            return ",".join(str(seed) for seed in value)
        if isinstance(value, dict):
            return ",".join(f"{seed}:{delta:+.6f}" for seed, delta in value.items())
        return value

    return [{key: display(value) for key, value in row.items()} for row in rows]


def _write_table(rows: list[dict]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(format_rows(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--require_seeds", type=int, nargs="+", help="Require this exact unique seed set per group."
    )
    parser.add_argument(
        "--paired", action="store_true", help="Compare later groups with the first input group."
    )
    args = parser.parse_args()
    try:
        groups = group_runs(load_runs(args.run_dirs), args.require_seeds)
        summaries = summary_rows(groups)
        comparisons = paired_rows(groups) if args.paired else []
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        "Sample standard deviation uses ddof=1; NA means only one seed. "
        "This describes seed variability, not a confidence interval or significance test.",
        file=sys.stderr,
    )
    _write_table(summaries)
    if comparisons:
        print()
        print("# Paired macro F1 delta = candidate - reference")
        _write_table(comparisons)


if __name__ == "__main__":
    main()
