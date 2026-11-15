"""Repeat the saved neural configuration across seeds without altering the midpoint model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom import neural_experiment as neural  # noqa: E402
from experiments.seungbeom import neural_sweep as sweep  # noqa: E402
from experiments.seungbeom.export_submissions import checked_score  # noqa: E402
from experiments.seungbeom.neural_config import load_config  # noqa: E402
from schema import RAW_SENSOR_COLUMNS  # noqa: E402

KIND = "seungbeom_neural_seed_repeat"
SEEDS = (42, 43, 44)
ENVIRONMENT_KEYS = ("python", "os", "packages", "cuda_runtime", "cudnn_version", "device", "gpu")
HELPER_FILES = (
    "experiments/seungbeom/neural_seed_repeat.py",
    "experiments/seungbeom/neural_sweep.py",
)


def environment_signature(metadata):
    if not isinstance(metadata, dict) or any(key not in metadata for key in ENVIRONMENT_KEYS):
        return None
    return {key: metadata[key] for key in ENVIRONMENT_KEYS}


def current_environment(device):
    from runtime import environment_metadata, seed_everything, select_device

    seed_everything(42)
    return environment_metadata(select_device(device))


def helper_hashes():
    return {name: neural.file_sha(ROOT / name) for name in HELPER_FILES}


def check_files(files):
    if not files or any(neural.file_sha(Path(path)) != value for path, value in files.items()):
        raise ValueError("Reference run changed; keep it intact and prepare a new experiment.")


def capture_identity(config, paths):
    data = Path(paths["data_dir"])
    return neural.run_identity(
        config,
        data / "train/train.csv",
        data / "validation/validation.csv",
        train_embeddings=Path(paths["train_embeddings"]),
        val_embeddings=Path(paths["val_embeddings"]),
    )


def prepare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    reference = args.reference_run.resolve()
    required = ("status.json", "config.json", "identity.json", "best_metrics.json", "best.pt")
    files = {str(reference / name): neural.file_sha(reference / name) for name in required}
    if sweep.read_json(reference / "status.json").get("status") != "complete":
        raise ValueError("The reference neural run must be complete.")
    config = load_config(reference / "config.json")
    if (
        config["seed"] != 42
        or config["encoder"] != "embedding"
        or config["embedding_encoder"] != "resnet18"
        or config["fusion"] != "concat"
        or config["sensor_features"] != list(RAW_SENSOR_COLUMNS)
    ):
        raise ValueError("Use the original seed42 ResNet18/raw5/concat neural run.")
    reference_identity = sweep.read_json(reference / "identity.json")
    if reference_identity.get("config") != config:
        raise ValueError("Reference config and identity disagree.")
    checked_score(sweep.read_json(reference / "best_metrics.json"))
    paths = {
        key: str(getattr(args, key).resolve())
        for key in ("data_dir", "train_embeddings", "val_embeddings")
    }
    expected = capture_identity(config, paths)
    for key in ("train_csv_sha256", "validation_csv_sha256", "input_provenance"):
        if reference_identity.get(key) != expected[key]:
            raise ValueError(
                "CSV or embedding cache differs from the reference; use the original inputs."
            )
    environment = environment_signature(current_environment(args.device))
    if environment is None:
        raise ValueError("Current environment metadata is incomplete.")
    old_environment = None
    environment_file = reference / "environment.json"
    if environment_file.is_file():
        files[str(environment_file)] = neural.file_sha(environment_file)
        old_environment = environment_signature(sweep.read_json(environment_file))
    reasons = []
    if reference_identity != expected:
        reasons.append("source_identity_changed")
    if old_environment != environment:
        reasons.append("environment_changed_or_unrecorded")
    reuse_reference = not reasons
    seeds = SEEDS[1:] if reuse_reference else SEEDS
    plan = sweep.create_plan([config], "seeds", max_trials=len(seeds), seeds=seeds)
    sources = helper_hashes()
    manifest = {
        "kind": KIND,
        "format_version": 1,
        "reference_run": str(reference),
        "reference_files": files,
        "reference_config": config,
        "reference_identity": reference_identity,
        "reuse_reference": reuse_reference,
        "fresh_control_reasons": reasons,
        "paths": paths,
        "device": args.device,
        "expected_identity": expected,
        "environment": environment,
        "helper_sha256": sources,
        "plan": plan,
    }
    check_files(files)
    if capture_identity(config, paths) != expected or helper_hashes() != sources:
        raise ValueError("Inputs or code changed during preparation.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sweep.write_json(args.output_dir / "plan.json", plan, exclusive=True)
    sweep.write_json(args.output_dir / "experiment.json", manifest, exclusive=True)
    print("Prepared seeds:", ", ".join(map(str, seeds)))
    print("Original seed42:", "reuse" if reuse_reference else "preserved historical reference")
    if reasons:
        print("A fresh seed42 control is required:", ", ".join(reasons))
    print(
        f"Unchanged training settings: epochs<={config['epochs']}, "
        f"patience={config['early_stopping_patience']}, dropout={config['dropout']}, "
        f"weight_decay={config['weight_decay']}"
    )
    print("Plan saved; training has not started. Existing checkpoints remain unchanged.")
    return manifest


def read_experiment(directory):
    manifest = sweep.read_json(directory / "experiment.json")
    if (
        manifest.get("kind") != KIND
        or manifest.get("format_version") != 1
        or type(manifest.get("reuse_reference")) is not bool
    ):
        raise ValueError("Use a prepared neural seed-repeat experiment.")
    seeds = SEEDS[1:] if manifest["reuse_reference"] else SEEDS
    expected = sweep.create_plan(
        [manifest["reference_config"]], "seeds", max_trials=len(seeds), seeds=seeds
    )
    if manifest.get("plan") != expected or sweep.read_json(directory / "plan.json") != expected:
        raise ValueError("Seed plan changed; prepare a new experiment.")
    check_files(manifest["reference_files"])
    return manifest


def check_current(manifest):
    if helper_hashes() != manifest["helper_sha256"]:
        raise ValueError("Experiment helper code changed; prepare a new experiment.")
    if (
        capture_identity(manifest["reference_config"], manifest["paths"])
        != manifest["expected_identity"]
    ):
        raise ValueError("Training inputs or source code changed; prepare a new experiment.")
    if environment_signature(current_environment(manifest["device"])) != manifest["environment"]:
        raise ValueError("Runtime environment changed; use the prepared environment.")
    check_files(manifest["reference_files"])


def run(args):
    directory = args.experiment_dir
    manifest = read_experiment(directory)
    check_current(manifest)
    paths = manifest["paths"]
    sweep.run_command(
        SimpleNamespace(
            plan=directory / "plan.json",
            run_root=directory / "runs",
            data_dir=Path(paths["data_dir"]),
            train_embeddings=Path(paths["train_embeddings"]),
            val_embeddings=Path(paths["val_embeddings"]),
            encoder_embeddings=None,
            device=manifest["device"],
        )
    )
    check_current(manifest)
    print("Seed repeats complete. Run compare for per-seed and aggregate Validation results.")


def compare(args):
    directory = args.experiment_dir
    manifest = read_experiment(directory)
    runs = []
    if manifest["reuse_reference"]:
        runs.append(Path(manifest["reference_run"]))
    runs += [directory / "runs" / trial["id"] for trial in manifest["plan"]["trials"]]
    for path in runs:
        if (
            not (path / "status.json").is_file()
            or sweep.read_json(path / "status.json").get("status") != "complete"
        ):
            raise ValueError("All planned seeds must be complete before comparison.")
        config = load_config(path / "config.json")
        expected = {**manifest["expected_identity"], "config": config}
        if sweep.read_json(path / "identity.json") != expected:
            raise ValueError("A seed run has different input/source identity.")
        if {**config, "seed": 42} != manifest["reference_config"]:
            raise ValueError("Seed runs must differ only in seed.")
        if (
            environment_signature(sweep.read_json(path / "environment.json"))
            != manifest["environment"]
        ):
            raise ValueError("Seed runs have different training environments.")
    result = sweep.summarize(runs, require_seeds=list(SEEDS))
    if len(result["groups"]) != 1:
        raise ValueError("Expected a single controlled seed comparison.")
    group = result["groups"][0]
    if not manifest["reuse_reference"]:
        historical = sweep.read_json(Path(manifest["reference_run"]) / "best_metrics.json")
        print(
            f"Historical original seed42: {checked_score(historical):.6f} "
            "(excluded from seed summary)"
        )
    print("seed\tbest_epoch\tmacro_f1\t" + "\t".join(f"f1_{i}" for i in range(1, 7)))
    for item in group["runs"]:
        metrics = sweep.read_json(Path(item["directory"]) / "best_metrics.json")
        print(
            f"{item['seed']}\t{item['best_epoch']}\t{metrics['macro_f1']:.6f}\t"
            + "\t".join(f"{metrics['class_f1'][str(i)]:.6f}" for i in range(1, 7))
        )
    print(f"Macro F1 mean={group['macro_f1_mean']:.6f} sample_std={group['macro_f1_std']:.6f} n=3")
    print("Seed variability is not a confidence interval or an ensemble score.")
    print("Existing 50:50 submission is unchanged. No Test data was used.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Copy original settings; change only seeds")
    for name in ("reference_run", "data_dir", "train_embeddings", "val_embeddings", "output_dir"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prep.set_defaults(handler=prepare)
    for name, handler in (("run", run), ("compare", compare)):
        command = commands.add_parser(name)
        command.add_argument("--experiment_dir", type=Path, required=True)
        command.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
