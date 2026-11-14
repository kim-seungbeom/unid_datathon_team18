"""Seed planning and comparisons use only independent synthetic files and CPU metadata."""

import argparse
import copy
import hashlib
import json

import pytest

from experiments.seungbeom import neural_experiment as neural
from experiments.seungbeom import neural_seed_repeat as repeat
from experiments.seungbeom import neural_sweep as sweep
from experiments.seungbeom.neural_config import normalize_config


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


def metric(seed):
    score = {42: 0.7, 43: 0.72, 44: 0.71}[seed]
    return {
        "epoch": 2,
        "macro_f1": score,
        "class_f1": {str(label): score for label in range(1, 7)},
    }


@pytest.fixture
def synthetic_reference(tmp_path, monkeypatch):
    config = normalize_config(
        {"seed": 42, "epochs": 20, "early_stopping_patience": 3, "batch_size": 16}
    )
    data = tmp_path / "independent_csvs"
    caches = {}
    for split in ("train", "validation"):
        (data / split).mkdir(parents=True)
        (data / split / f"{split}.csv").write_text(f"independent {split} fixture\n")
        cache = tmp_path / f"invented_{split}_vectors"
        cache.mkdir()
        for name in ("metadata.json", "sample_ids.csv", "features.npy"):
            (cache / name).write_bytes(f"independent {split} {name}".encode())
        caches[split] = cache
    identity = neural.run_identity(
        config,
        data / "train/train.csv",
        data / "validation/validation.csv",
        train_embeddings=caches["train"],
        val_embeddings=caches["validation"],
    )
    environment = {
        "python": "independent Python 3.12 fixture",
        "python_executable": "/invented/python",
        "os": "independent OS fixture",
        "packages": {
            "torch": "fixture1",
            "numpy": "fixture1",
            "pandas": "fixture1",
            "Pillow": "fixture1",
        },
        "cuda_runtime": None,
        "cudnn_version": None,
        "device": "cpu",
        "gpu": None,
    }
    reference = tmp_path / "reference_neural"
    reference.mkdir()
    for filename, value in (
        ("status.json", {"status": "complete"}),
        ("config.json", config),
        ("identity.json", identity),
        ("best_metrics.json", metric(42)),
        ("environment.json", environment),
    ):
        dump(reference / filename, value)
    (reference / "best.pt").write_bytes(b"independent fixture, never deserialized")
    monkeypatch.setattr(repeat, "current_environment", lambda device: copy.deepcopy(environment))
    args = argparse.Namespace(
        reference_run=reference,
        data_dir=data,
        train_embeddings=caches["train"],
        val_embeddings=caches["validation"],
        output_dir=tmp_path / "seed_experiment",
        device="cpu",
    )
    return {"args": args, "config": config, "identity": identity, "environment": environment}


def prepared_plan(state):
    return read(state["args"].output_dir / "plan.json")


def run_args(state):
    return argparse.Namespace(experiment_dir=state["args"].output_dir)


def fake_sweep(state, monkeypatch):
    calls = []

    def run(sweep_args):
        calls.append(sweep_args)
        plan = read(sweep_args.plan)
        for trial in sweep.validate_plan(plan):
            config = trial["config"]
            directory = sweep_args.run_root / trial["id"]
            directory.mkdir(parents=True, exist_ok=False)
            identity = neural.run_identity(
                config,
                sweep_args.data_dir / "train/train.csv",
                sweep_args.data_dir / "validation/validation.csv",
                train_embeddings=sweep_args.train_embeddings,
                val_embeddings=sweep_args.val_embeddings,
            )
            for name, value in (
                ("config.json", config),
                ("identity.json", identity),
                ("status.json", {"status": "complete"}),
                ("best_metrics.json", metric(config["seed"])),
                ("environment.json", state["environment"]),
            ):
                dump(directory / name, value)
            (directory / "best.pt").write_bytes(b"independent fake repeat checkpoint")

    monkeypatch.setattr(sweep, "run_command", run)
    return calls


def test_prepare_reuses_identical_seed42_and_changes_only_seed(synthetic_reference):
    state = synthetic_reference
    original = hashes(state["args"].reference_run)
    repeat.prepare(state["args"])
    trials = sweep.validate_plan(prepared_plan(state))
    assert [trial["config"]["seed"] for trial in trials] == [43, 44]
    for trial in trials:
        assert trial["config"] == {**state["config"], "seed": trial["config"]["seed"]}
        assert trial["config"]["epochs"] == 20
        assert trial["config"]["early_stopping_patience"] == 3
    assert (state["args"].output_dir / "experiment.json").is_file()
    assert hashes(state["args"].reference_run) == original


@pytest.mark.parametrize("change", ["source", "environment", "missing_environment"])
def test_prepare_uses_fresh42_control_when_old_context_differs(synthetic_reference, change):
    state = synthetic_reference
    reference = state["args"].reference_run
    if change == "source":
        identity = read(reference / "identity.json")
        identity["source_sha256"][next(iter(identity["source_sha256"]))] = "older-source"
        dump(reference / "identity.json", identity)
    elif change == "environment":
        environment = read(reference / "environment.json")
        environment["packages"]["torch"] = "older-torch"
        dump(reference / "environment.json", environment)
    else:
        (reference / "environment.json").unlink()
    original = hashes(reference)
    repeat.prepare(state["args"])
    assert [trial["config"]["seed"] for trial in prepared_plan(state)["trials"]] == [42, 43, 44]
    assert hashes(reference) == original


@pytest.mark.parametrize("change", ["train", "validation", "train_cache", "validation_cache"])
def test_prepare_rejects_different_split_or_cache(synthetic_reference, change):
    state = synthetic_reference
    args = state["args"]
    if change.endswith("_cache"):
        folder = args.train_embeddings if change == "train_cache" else args.val_embeddings
        (folder / "features.npy").write_bytes(b"different independent vectors")
    else:
        (args.data_dir / change / f"{change}.csv").write_text("different independent split")
    with pytest.raises(ValueError):
        repeat.prepare(args)
    assert not (args.output_dir / "experiment.json").exists()


def test_prepare_does_not_overwrite_existing_experiment(synthetic_reference):
    args = synthetic_reference["args"]
    repeat.prepare(args)
    original = hashes(args.output_dir)
    with pytest.raises((ValueError, FileExistsError)):
        repeat.prepare(args)
    assert hashes(args.output_dir) == original


@pytest.mark.parametrize("change", ["incomplete", "wrong_seed", "config_identity"])
def test_prepare_requires_the_completed_original_reference(synthetic_reference, change):
    args = synthetic_reference["args"]
    if change == "incomplete":
        dump(args.reference_run / "status.json", {"status": "running"})
    elif change == "wrong_seed":
        config = read(args.reference_run / "config.json")
        config["seed"] = 43
        dump(args.reference_run / "config.json", config)
    else:
        identity = read(args.reference_run / "identity.json")
        identity["config"]["epochs"] = 50
        dump(args.reference_run / "identity.json", identity)
    with pytest.raises(ValueError):
        repeat.prepare(args)
    assert not args.output_dir.exists()


def test_run_dispatches_only_two_new_seeds_and_compare_prints_each(
    synthetic_reference, monkeypatch, capsys
):
    state = synthetic_reference
    repeat.prepare(state["args"])
    calls = fake_sweep(state, monkeypatch)
    repeat.run(run_args(state))
    assert len(calls) == 1
    assert calls[0].train_embeddings == state["args"].train_embeddings
    assert calls[0].val_embeddings == state["args"].val_embeddings
    capsys.readouterr()
    result = repeat.compare(run_args(state))
    output = capsys.readouterr().out
    assert all(str(seed) in output for seed in (42, 43, 44))
    assert "f1_1" in output and "f1_6" in output
    group = result["groups"][0]
    assert group["seeds"] == [42, 43, 44]
    assert group["macro_f1_mean"] == pytest.approx(0.71)
    assert group["macro_f1_std"] == pytest.approx(0.01)


@pytest.mark.parametrize("change", ["plan", "reference", "cache", "environment"])
def test_run_refuses_changed_inputs_before_training(synthetic_reference, monkeypatch, change):
    state = synthetic_reference
    args = state["args"]
    repeat.prepare(args)
    if change == "plan":
        plan = prepared_plan(state)
        plan["trials"][0]["config"]["epochs"] += 1
        dump(args.output_dir / "plan.json", plan)
    elif change == "reference":
        (args.reference_run / "best.pt").write_bytes(b"different independent checkpoint")
    elif change == "cache":
        (args.train_embeddings / "features.npy").write_bytes(b"different independent vectors")
    else:
        environment = copy.deepcopy(state["environment"])
        environment["packages"]["torch"] = "changed-runtime"
        monkeypatch.setattr(repeat, "current_environment", lambda device: environment)
    monkeypatch.setattr(sweep, "run_command", lambda *a, **kw: pytest.fail("Training started"))
    with pytest.raises(ValueError):
        repeat.run(run_args(state))


@pytest.mark.parametrize("change", ["missing", "incomplete", "config", "source", "environment"])
def test_compare_refuses_incomplete_or_incompatible_repeats(
    synthetic_reference, monkeypatch, change
):
    state = synthetic_reference
    repeat.prepare(state["args"])
    calls = fake_sweep(state, monkeypatch)
    repeat.run(run_args(state))
    trial = prepared_plan(state)["trials"][-1]
    run = calls[0].run_root / trial["id"]
    if change == "missing":
        (run / "status.json").unlink()
    elif change == "incomplete":
        dump(run / "status.json", {"status": "running"})
    elif change == "config":
        config = read(run / "config.json")
        config["dropout"] = 0.9
        dump(run / "config.json", config)
    elif change == "source":
        identity = read(run / "identity.json")
        identity["source_sha256"][next(iter(identity["source_sha256"]))] = "different-source"
        dump(run / "identity.json", identity)
    else:
        environment = read(run / "environment.json")
        environment["packages"]["torch"] = "different-torch"
        dump(run / "environment.json", environment)
    with pytest.raises((ValueError, FileNotFoundError)):
        repeat.compare(run_args(state))


def test_historical42_is_not_pooled_with_new_control(synthetic_reference, monkeypatch, capsys):
    state = synthetic_reference
    reference = state["args"].reference_run
    old_identity = read(reference / "identity.json")
    old_identity["source_sha256"][next(iter(old_identity["source_sha256"]))] = "old-source"
    dump(reference / "identity.json", old_identity)
    old_metrics = metric(42)
    old_metrics["macro_f1"] = 0.99
    dump(reference / "best_metrics.json", old_metrics)
    repeat.prepare(state["args"])
    fake_sweep(state, monkeypatch)
    repeat.run(run_args(state))
    capsys.readouterr()
    result = repeat.compare(run_args(state))
    assert result["groups"][0]["macro_f1_mean"] == pytest.approx(0.71)
    assert "0.990000" in capsys.readouterr().out


def test_saved_comparison_does_not_need_csvs_or_embedding_arrays(synthetic_reference, monkeypatch):
    state = synthetic_reference
    args = state["args"]
    repeat.prepare(args)
    fake_sweep(state, monkeypatch)
    repeat.run(run_args(state))
    for split in ("train", "validation"):
        (args.data_dir / split / f"{split}.csv").unlink()
    for directory in (args.train_embeddings, args.val_embeddings):
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
    result = repeat.compare(run_args(state))
    assert result["groups"][0]["seeds"] == [42, 43, 44]
