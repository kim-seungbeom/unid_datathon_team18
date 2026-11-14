"""Bounded, sequential personal neural experiments; real runs are server-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom.neural_config import load_config, normalize_config  # noqa: E402
from experiments.seungbeom.neural_sensors import TEMPORAL_VARIANTS, sensor_variant  # noqa: E402

STAGES = ("lr", "regularization", "fusion", "encoders", "imbalance", "seeds", "temporal")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    else:
        temporary = path.with_name(path.name + ".writing")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


def _variations(stage, seeds, include_focal):
    if stage == "temporal":
        return [{"sensor_features": list(names)} for names in TEMPORAL_VARIANTS.values()]
    if stage == "lr":
        return [{"learning_rate": rate} for rate in (1e-4, 3e-4, 1e-3)]
    if stage == "regularization":
        return [
            {"dropout": drop, "weight_decay": decay}
            for drop in (0.1, 0.3)
            for decay in (1e-4, 1e-3)
        ]
    if stage == "fusion":
        return [{"fusion": name} for name in ("concat", "projected_concat", "gated")]
    if stage == "encoders":
        return [{"encoder": name, "fusion": "concat"} for name in ("small_cnn", "resnet18")]
    if stage == "imbalance":
        losses = ["cross_entropy", "weighted_cross_entropy"]
        return [{"loss": loss} for loss in losses + (["focal"] if include_focal else [])]
    if stage == "seeds":
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Seeds must be nonempty and unique.")
        return [{"seed": seed} for seed in seeds]
    raise ValueError("Unknown experiment stage.")


def create_plan(base_configs, stage, *, max_trials=9, seeds=(42, 43, 44), include_focal=False):
    """No data, torch, model construction, or training is needed to create a plan."""
    if type(max_trials) is not int or not 1 <= max_trials <= 64:
        raise ValueError("max_trials must be an integer between 1 and 64.")
    trials, seen = [], set()
    for index, config in enumerate(base_configs):
        base = normalize_config(config)
        if stage == "imbalance" and base["sampling"] != "shuffle":
            raise ValueError(
                "Loss comparisons require shuffle sampling; change one factor at a time."
            )
        for variation in _variations(stage, seeds, include_focal):
            candidate = normalize_config({**base, **variation})
            encoded = canonical(candidate)
            if encoded in seen:
                continue
            seen.add(encoded)
            digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
            trials.append(
                {
                    "id": (
                        f"{candidate['encoder']}_{candidate['fusion']}_s{candidate['seed']}_{digest}"
                    ),
                    "base_index": index,
                    "changes": {
                        key: value for key, value in candidate.items() if value != base[key]
                    },
                    "config": candidate,
                }
            )
    if not trials or len(trials) > max_trials:
        raise ValueError(f"Plan has {len(trials)} trials; explicit budget is {max_trials}.")
    return {"format_version": 1, "stage": stage, "max_trials": max_trials, "trials": trials}


def validate_plan(plan):
    if plan.get("format_version") != 1 or plan.get("stage") not in STAGES:
        raise ValueError("Unsupported plan.")
    trials = plan.get("trials", [])
    if not trials or not 1 <= len(trials) <= plan.get("max_trials", 0) <= 64:
        raise ValueError("Invalid plan budget.")
    ids, configs = set(), set()
    for trial in trials:
        config = normalize_config(trial["config"])
        if config != trial["config"]:
            raise ValueError("Plan configs must include all normalized defaults.")
        encoded = canonical(config)
        digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        expected = f"{config['encoder']}_{config['fusion']}_s{config['seed']}_{digest}"
        if trial["id"] != expected or expected in ids or encoded in configs:
            raise ValueError("Invalid or duplicate trial identity.")
        ids.add(expected)
        configs.add(encoded)
    return trials


def plan_command(args):
    bases = [load_config(path) for path in args.base_config] if args.base_config else [{}]
    plan = create_plan(
        bases,
        args.stage,
        max_trials=args.max_trials,
        seeds=args.seeds,
        include_focal=args.include_focal,
    )
    write_json(args.output, plan, exclusive=True)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print(f"Plan saved; {len(plan['trials'])} trials, no training started.")


def _cache_identity(train_embeddings, val_embeddings):
    """Bind reuse to explicit cache bytes as well as CSV/config/source identity."""
    if (train_embeddings is None) != (val_embeddings is None):
        raise ValueError("Provide both training and validation caches, or neither.")
    if train_embeddings is None:
        return None
    result = {}
    for split, directory in (("train", train_embeddings), ("validation", val_embeddings)):
        result[split] = {}
        for name in ("metadata.json", "sample_ids.csv", "features.npy"):
            digest = hashlib.sha256()
            with (Path(directory) / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[split][name] = digest.hexdigest()
    return result


def run_command(args, *, identity_fn=None, runner=None):
    if identity_fn is None:
        from experiments.seungbeom.neural_experiment import run_identity

        identity_fn = run_identity
    runner = subprocess.run if runner is None else runner
    plan = read_json(args.plan)
    trials = validate_plan(plan)
    if args.train_embeddings is not None and any(
        trial["config"]["encoder"] != "embedding" for trial in trials
    ):
        raise ValueError("Image caches are only accepted for embedding-encoder plans.")
    root = args.run_root
    manifest_path = root / "sweep_status.json"
    cache_identity = _cache_identity(args.train_embeddings, args.val_embeddings)
    context = {"plan": plan, "cached_embeddings_sha256": cache_identity}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["context"] != context:
            raise ValueError("Run root belongs to a different plan/cache; use a new run_root.")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Run root is not empty; choose a new directory.")
        manifest = {
            "context": context,
            "trials": {trial["id"]: {"status": "pending"} for trial in trials},
        }
        write_json(manifest_path, manifest, exclusive=True)
    for trial in trials:
        directory = root / trial["id"]
        expected = identity_fn(
            trial["config"],
            args.data_dir / "train" / "train.csv",
            args.data_dir / "validation" / "validation.csv",
            train_embeddings=args.train_embeddings,
            val_embeddings=args.val_embeddings,
        )
        state = manifest["trials"][trial["id"]]
        if directory.exists():
            status_path, identity_path = directory / "status.json", directory / "identity.json"
            if (
                status_path.is_file()
                and identity_path.is_file()
                and read_json(status_path).get("status") == "complete"
                and read_json(identity_path) == expected
                and read_json(directory / "config.json") == trial["config"]
            ):
                state.update(status="complete", reused=True)
                write_json(manifest_path, manifest)
                print("Reuse completed:", trial["id"], flush=True)
                continue
            raise ValueError(
                "Existing trial is incomplete or has different provenance; use a new root."
            )
        config_path = root / "planned_configs" / f"{trial['id']}.json"
        if config_path.exists() and read_json(config_path) != trial["config"]:
            raise ValueError("Stored trial config differs from plan.")
        if not config_path.exists():
            write_json(config_path, trial["config"], exclusive=True)
        command = [
            sys.executable,
            str(ROOT / "experiments/seungbeom/neural_experiment.py"),
            "train",
            "--config",
            str(config_path),
            "--data_dir",
            str(args.data_dir),
            "--run_dir",
            str(directory),
            "--device",
            args.device,
        ]
        if args.train_embeddings is not None and trial["config"]["encoder"] == "embedding":
            command += [
                "--train_embeddings",
                str(args.train_embeddings),
                "--val_embeddings",
                str(args.val_embeddings),
            ]
        state.update(status="running", reused=False)
        write_json(manifest_path, manifest)
        print("Starting:", trial["id"], flush=True)
        try:
            result = runner(command, cwd=ROOT, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Training process exited with code {result.returncode}.")
            if (
                read_json(directory / "status.json").get("status") != "complete"
                or read_json(directory / "identity.json") != expected
            ):
                raise RuntimeError("Trainer did not produce a complete matching run.")
        except BaseException as error:
            state.update(status="failed", error_type=type(error).__name__)
            write_json(manifest_path, manifest)
            raise
        state.update(status="complete")
        write_json(manifest_path, manifest)
    print("Sweep complete. Compare the saved Validation results; no Test data was used.")


def _context(identity):
    return {key: value for key, value in identity.items() if key not in ("config", "seed")}


def summarize(run_dirs, require_seeds=None, paired=False):
    groups, skipped = {}, []
    for directory in map(Path, run_dirs):
        status_file = directory / "status.json"
        if not status_file.is_file() or read_json(status_file).get("status") != "complete":
            skipped.append(str(directory))
            continue
        config = load_config(directory / "config.json")
        identity = read_json(directory / "identity.json")
        if identity.get("config") != config:
            raise ValueError("Saved config differs from the run identity.")
        metrics = read_json(directory / "best_metrics.json")
        score = metrics["macro_f1"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid validation score.")
        class_f1 = metrics.get("class_f1")
        if class_f1 is not None and (
            set(class_f1) != {str(label) for label in range(1, 7)}
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                for value in class_f1.values()
            )
        ):
            raise ValueError("Invalid six-class F1 scores.")
        variant = {key: value for key, value in config.items() if key != "seed"}
        key = canonical({"config": variant, "context": _context(identity)})
        group = groups.setdefault(
            key,
            {"config": variant, "context": _context(identity), "runs": {}, "name": directory.name},
        )
        seed = config["seed"]
        if seed in group["runs"]:
            raise ValueError("Duplicate seed in the same config/split/code group.")
        group["runs"][seed] = {
            "macro_f1": float(score),
            "metrics": metrics,
            "directory": str(directory),
            "cost": read_json(directory / "summary.json")
            if (directory / "summary.json").is_file()
            else {},
        }
    if not groups:
        raise ValueError("No complete runs to compare.")
    if require_seeds is not None:
        if not require_seeds or len(require_seeds) != len(set(require_seeds)):
            raise ValueError("Required seeds must be nonempty and unique.")
        for group in groups.values():
            if set(group["runs"]) != set(require_seeds):
                raise ValueError("A candidate does not have exactly the required seeds.")
    summaries = []
    for group in groups.values():
        ordered = sorted(group["runs"])
        values = [group["runs"][seed]["macro_f1"] for seed in ordered]
        summaries.append(
            {
                "name": group["name"],
                "config": group["config"],
                "context": group["context"],
                "seeds": ordered,
                "n": len(values),
                "macro_f1_mean": statistics.mean(values),
                "macro_f1_std": statistics.stdev(values) if len(values) > 1 else None,
                "seed_scores": {str(seed): group["runs"][seed]["macro_f1"] for seed in ordered},
            }
        )
        details = summaries[-1]
        if all("class_f1" in group["runs"][seed]["metrics"] for seed in ordered):
            details["class_f1_mean"] = {}
            details["class_f1_std"] = {}
            for label in map(str, range(1, 7)):
                scores = [group["runs"][seed]["metrics"]["class_f1"][label] for seed in ordered]
                details["class_f1_mean"][label] = statistics.mean(scores)
                details["class_f1_std"][label] = (
                    statistics.stdev(scores) if len(scores) > 1 else None
                )
        details["runs"] = [
            {
                "seed": seed,
                "directory": group["runs"][seed]["directory"],
                "best_epoch": group["runs"][seed]["metrics"].get("epoch"),
                **{
                    key: group["runs"][seed]["cost"].get(key)
                    for key in (
                        "parameter_count",
                        "training_seconds",
                        "inference_seconds",
                        "elapsed_epochs",
                        "stop_reason",
                    )
                },
            }
            for seed in ordered
        ]
    differences = []
    if paired:
        reference = summaries[0]
        for candidate in summaries[1:]:
            if (
                candidate["seeds"] != reference["seeds"]
                or candidate["context"] != reference["context"]
            ):
                raise ValueError("Paired comparison requires identical seeds, splits, and code.")
            deltas = {
                str(seed): candidate["seed_scores"][str(seed)] - reference["seed_scores"][str(seed)]
                for seed in reference["seeds"]
            }
            values = list(deltas.values())
            differences.append(
                {
                    "reference": reference["name"],
                    "candidate": candidate["name"],
                    "mean_delta": statistics.mean(values),
                    "std_delta": statistics.stdev(values) if len(values) > 1 else None,
                    "wins": sum(value > 0 for value in values),
                    "ties": sum(value == 0 for value in values),
                    "seed_deltas": deltas,
                }
            )
    return {
        "groups": summaries,
        "paired": differences,
        "skipped_incomplete": skipped,
        "note": "Sample std uses ddof=1; seed variation is not a confidence interval.",
    }


def compare_command(args):
    directories = list(args.run_dirs)
    if args.plan is not None:
        if args.run_root is None:
            raise ValueError("--plan requires --run_root.")
        for trial in validate_plan(read_json(args.plan)):
            directory = args.run_root / trial["id"]
            if (directory / "config.json").is_file() and load_config(
                directory / "config.json"
            ) != trial["config"]:
                raise ValueError("Run config differs from the planned trial.")
            directories.append(directory)
    result = summarize(directories, args.require_seeds, args.paired)
    compact = getattr(args, "compact", False)
    class_details = getattr(args, "class_details", False)
    header = "variant\tseeds\tn\tmacro_f1_mean\tmacro_f1_std"
    if compact:
        header += "\tsensors"
    if class_details:
        header += "\t" + "\t".join(f"f1_{label}" for label in range(1, 7))
    print(header)
    for group in result["groups"]:
        deviation = "NA" if group["macro_f1_std"] is None else f"{group['macro_f1_std']:.6f}"
        line = (
            f"{group['name']}\t{','.join(map(str, group['seeds']))}\t{group['n']}\t"
            f"{group['macro_f1_mean']:.6f}\t{deviation}"
        )
        if compact:
            line += f"\t{sensor_variant(group['config']['sensor_features'])}"
        if class_details:
            scores = group.get("class_f1_mean", {})
            line += "\t" + "\t".join(
                f"{scores[str(label)]:.6f}" if str(label) in scores else "NA"
                for label in range(1, 7)
            )
        print(line)
    if compact:
        for delta in result["paired"]:
            print(
                f"Paired {delta['candidate']} - {delta['reference']}: "
                f"mean={delta['mean_delta']:+.6f}, wins={delta['wins']}, ties={delta['ties']}"
            )
        if result["skipped_incomplete"]:
            print(f"Skipped incomplete runs: {len(result['skipped_incomplete'])}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Print and save full configs; never starts training")
    plan.add_argument("--base_config", type=Path, action="append")
    plan.add_argument("--stage", choices=STAGES, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--max_trials", type=int, default=9)
    plan.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    plan.add_argument("--include_focal", action="store_true")
    plan.set_defaults(handler=plan_command)
    run = commands.add_parser("run", help="Execute planned trials sequentially")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--data_dir", type=Path, required=True)
    run.add_argument("--run_root", type=Path, required=True)
    run.add_argument("--train_embeddings", type=Path)
    run.add_argument("--val_embeddings", type=Path)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run.set_defaults(handler=run_command)
    compare = commands.add_parser("compare", help="Compare explicit complete runs or a saved plan")
    compare.add_argument("run_dirs", type=Path, nargs="*")
    compare.add_argument("--plan", type=Path)
    compare.add_argument("--run_root", type=Path)
    compare.add_argument("--require_seeds", type=int, nargs="+")
    compare.add_argument("--paired", action="store_true")
    compare.add_argument(
        "--compact", action="store_true", help="Print tables without the full JSON"
    )
    compare.add_argument(
        "--class_details", action="store_true", help="Include six class F1 columns"
    )
    compare.set_defaults(handler=compare_command)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
