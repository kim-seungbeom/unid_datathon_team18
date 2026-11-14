"""Sweep tests use independent synthetic metadata, images, and CPU training only."""

import argparse
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.seungbeom import neural_sweep as sweep
from experiments.seungbeom.neural_config import normalize_config


def identity(config, train_csv, val_csv, **kwargs):
    del kwargs
    return {
        "config": config,
        "train_csv_sha256": hashlib.sha256(train_csv.read_bytes()).hexdigest(),
        "validation_csv_sha256": hashlib.sha256(val_csv.read_bytes()).hexdigest(),
        "source_sha256": {"synthetic.py": "source-v1"},
        "input_provenance": {"synthetic_images": "images-v1"},
    }


def test_plan_is_bounded_full_deterministic_and_deduplicated():
    plan = sweep.create_plan([{}, {}], "fusion", max_trials=3)
    assert len(plan["trials"]) == 3
    assert plan == sweep.create_plan([{}], "fusion", max_trials=3)
    assert {trial["config"]["fusion"] for trial in plan["trials"]} == {
        "concat",
        "projected_concat",
        "gated",
    }
    for trial in plan["trials"]:
        assert trial["config"] == normalize_config(trial["config"])
        assert set(trial["changes"]) <= {"fusion"}
    sweep.validate_plan(plan)
    with pytest.raises(ValueError, match="budget"):
        sweep.create_plan([{}], "fusion", max_trials=2)


@pytest.mark.parametrize(
    "stage, expected",
    [("lr", 3), ("regularization", 4), ("encoders", 2), ("imbalance", 2), ("seeds", 3)],
)
def test_stages_change_only_declared_factors(stage, expected):
    plan = sweep.create_plan([{}], stage)
    assert len(plan["trials"]) == expected
    for trial in plan["trials"]:
        if stage == "encoders":
            assert trial["config"]["encoder"] in ("small_cnn", "resnet18")
            assert trial["config"]["fusion"] == "concat"
        assert trial["config"]["epochs"] == 50


def test_plan_rejects_duplicate_seeds_mutation_and_path_traversal():
    with pytest.raises(ValueError, match="unique"):
        sweep.create_plan([{}], "seeds", seeds=(42, 42))
    plan = sweep.create_plan([{}], "fusion")
    invalid = copy.deepcopy(plan)
    invalid["trials"][0]["id"] = "../elsewhere"
    with pytest.raises(ValueError, match="identity"):
        sweep.validate_plan(invalid)
    invalid = copy.deepcopy(plan)
    invalid["trials"][0]["config"]["seed"] = 44
    with pytest.raises(ValueError, match="identity"):
        sweep.validate_plan(invalid)


def test_plan_cli_does_not_call_runner_or_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: pytest.fail("Training started"))
    target = tmp_path / "plan.json"
    sweep.main(["plan", "--stage", "lr", "--output", str(target), "--max_trials", "3"])
    assert "no training started" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        sweep.main(["plan", "--stage", "lr", "--output", str(target)])


def setup_run(tmp_path, stage="fusion"):
    plan_path = tmp_path / "plan.json"
    sweep.write_json(plan_path, sweep.create_plan([{}], stage), exclusive=True)
    data = tmp_path / "synthetic_data"
    for split in ("train", "validation"):
        (data / split).mkdir(parents=True)
        (data / split / f"{split}.csv").write_text("independent synthetic fixture\n")
    return argparse.Namespace(
        plan=plan_path,
        data_dir=data,
        run_root=tmp_path / "results",
        train_embeddings=None,
        val_embeddings=None,
        device="cpu",
    )


def fake_training(command, cwd, check):
    assert cwd == sweep.ROOT and check is False
    arguments = {command[i]: command[i + 1] for i in range(3, len(command), 2)}
    run = Path(arguments["--run_dir"])
    config = sweep.read_json(arguments["--config"])
    data = Path(arguments["--data_dir"])
    run.mkdir()
    sweep.write_json(run / "config.json", config)
    sweep.write_json(
        run / "identity.json",
        identity(config, data / "train/train.csv", data / "validation/validation.csv"),
    )
    sweep.write_json(run / "status.json", {"status": "complete"})
    sweep.write_json(run / "best_metrics.json", {"macro_f1": 0.5})
    return SimpleNamespace(returncode=0)


def test_runner_is_sequential_and_reuses_only_identical_completed_results(tmp_path):
    args = setup_run(tmp_path)
    calls = []

    def runner(command, **kwargs):
        manifest = sweep.read_json(args.run_root / "sweep_status.json")
        assert sum(state["status"] == "running" for state in manifest["trials"].values()) == 1
        calls.append(command)
        return fake_training(command, **kwargs)

    sweep.run_command(args, identity_fn=identity, runner=runner)
    assert len(calls) == 3
    sweep.run_command(
        args, identity_fn=identity, runner=lambda *a, **k: pytest.fail("Complete result retrained")
    )
    manifest = sweep.read_json(args.run_root / "sweep_status.json")
    assert all(
        state["status"] == "complete" and state["reused"] for state in manifest["trials"].values()
    )
    (args.data_dir / "train/train.csv").write_text("different independent fixture\n")
    with pytest.raises(ValueError, match="different provenance"):
        sweep.run_command(args, identity_fn=identity, runner=runner)


def test_runner_records_failure_and_preserves_partial_trial(tmp_path):
    args = setup_run(tmp_path)

    def failure(command, **kwargs):
        directory = Path(command[command.index("--run_dir") + 1])
        directory.mkdir()
        sweep.write_json(directory / "status.json", {"status": "failed"})
        return SimpleNamespace(returncode=7)

    with pytest.raises(RuntimeError, match="code 7"):
        sweep.run_command(args, identity_fn=identity, runner=failure)
    manifest = sweep.read_json(args.run_root / "sweep_status.json")
    assert [state["status"] for state in manifest["trials"].values()] == [
        "failed",
        "pending",
        "pending",
    ]
    with pytest.raises(ValueError, match="incomplete"):
        sweep.run_command(args, identity_fn=identity, runner=fake_training)


def test_runner_rejects_embedding_cache_for_scratch_plan_before_reading_it(tmp_path):
    args = setup_run(tmp_path, "encoders")
    args.train_embeddings = tmp_path / "does_not_exist_train"
    args.val_embeddings = tmp_path / "does_not_exist_validation"
    with pytest.raises(ValueError, match="only accepted"):
        sweep.run_command(args, identity_fn=identity, runner=fake_training)


def synthetic_run(root, name, seed, score, fusion="concat", source="source-v1", status="complete"):
    directory = root / name
    config = normalize_config({"seed": seed, "fusion": fusion})
    sweep.write_json(directory / "config.json", config)
    sweep.write_json(directory / "status.json", {"status": status})
    sweep.write_json(directory / "best_metrics.json", {"macro_f1": score})
    sweep.write_json(
        directory / "identity.json",
        {
            "config": config,
            "train_csv_sha256": "synthetic-train",
            "validation_csv_sha256": "synthetic-val",
            "source_sha256": {"synthetic.py": source},
            "input_provenance": {"synthetic": True},
        },
    )
    return directory


def test_seed_summary_pairs_full_configs_and_skips_failures(tmp_path):
    directories = [
        synthetic_run(tmp_path, f"base_{seed}", seed, value)
        for seed, value in ((42, 0.6), (43, 0.7), (44, 0.8))
    ]
    directories += [
        synthetic_run(tmp_path, f"gate_{seed}", seed, value, fusion="gated")
        for seed, value in ((42, 0.61), (43, 0.72), (44, 0.79))
    ]
    directories += [synthetic_run(tmp_path, "failed", 45, 0.99, status="failed")]
    result = sweep.summarize(directories, require_seeds=[42, 43, 44], paired=True)
    assert len(result["groups"]) == 2
    assert result["groups"][0]["macro_f1_mean"] == pytest.approx(0.7)
    assert result["groups"][0]["macro_f1_std"] == pytest.approx(0.1)
    assert result["paired"][0]["mean_delta"] == pytest.approx(0.02 / 3)
    assert result["paired"][0]["wins"] == 2
    assert len(result["skipped_incomplete"]) == 1


def test_seed_summary_does_not_conflate_missing_seeds_code_or_config(tmp_path):
    first = synthetic_run(tmp_path, "first", 42, 0.5)
    second = synthetic_run(tmp_path, "second", 42, 0.6, source="different-source")
    assert len(sweep.summarize([first, second])["groups"]) == 2
    with pytest.raises(ValueError, match="identical seeds, splits, and code"):
        sweep.summarize([first, second], paired=True)
    with pytest.raises(ValueError, match="required seeds"):
        sweep.summarize([first], require_seeds=[42, 43, 44])
    with pytest.raises(ValueError, match="Duplicate seed"):
        sweep.summarize([first, first])
    assert sweep.summarize([first])["groups"][0]["macro_f1_std"] is None


def test_compare_rejects_saved_config_mutation_and_wrong_planned_trial(tmp_path):
    directory = synthetic_run(tmp_path, "changed_config", 42, 0.5)
    sweep.write_json(directory / "config.json", normalize_config({"fusion": "gated"}))
    with pytest.raises(ValueError, match="run identity"):
        sweep.summarize([directory])
    plan = sweep.create_plan([{}], "seeds", seeds=[42], max_trials=1)
    plan_path = tmp_path / "plan.json"
    sweep.write_json(plan_path, plan)
    synthetic_run(tmp_path, plan["trials"][0]["id"], 42, 0.5, fusion="gated")
    with pytest.raises(ValueError, match="planned trial"):
        sweep.main(["compare", "--plan", str(plan_path), "--run_root", str(tmp_path)])


def test_actual_sweep_trainer_and_reuse_on_synthetic_cpu_images(tmp_path, capsys):
    pytest.importorskip("torch")
    import numpy as np
    import pandas as pd
    from PIL import Image

    data = tmp_path / "synthetic_data"
    rng = np.random.default_rng(2026)
    for split, count in (("train", 12), ("validation", 6)):
        folder = data / split
        (folder / "images").mkdir(parents=True)
        rows = []
        for index in range(count):
            sid = f"{split}_{index}"
            relative = f"images/{sid}.png"
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                folder / relative
            )
            rows.append(
                {
                    "sample_id": sid,
                    "experiment_id": f"{split}_experiment",
                    "relative_time_s": index,
                    "image_path": relative,
                    "label": index % 6 + 1,
                    **{
                        name: float(index + offset + 1)
                        for offset, name in enumerate(
                            ("meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p")
                        )
                    },
                }
            )
        pd.DataFrame(rows).to_csv(folder / f"{split}.csv", index=False)
    plan = sweep.create_plan(
        [
            {
                "encoder": "small_cnn",
                "epochs": 1,
                "image_size": 32,
                "batch_size": 6,
                "sensor_hidden": 4,
                "fusion_hidden": 8,
                "projection_dim": 4,
            }
        ],
        "seeds",
        seeds=[42],
        max_trials=1,
    )
    plan_path = tmp_path / "tiny_plan.json"
    sweep.write_json(plan_path, plan, exclusive=True)
    args = argparse.Namespace(
        plan=plan_path,
        data_dir=data,
        run_root=tmp_path / "tiny_results",
        train_embeddings=None,
        val_embeddings=None,
        device="cpu",
    )
    sweep.run_command(args)
    directory = args.run_root / plan["trials"][0]["id"]
    assert sweep.read_json(directory / "status.json")["status"] == "complete"
    assert (directory / "best.pt").is_file()
    sweep.run_command(args, runner=lambda *a, **k: pytest.fail("Exact completed run retrained"))
    result = sweep.summarize([directory], require_seeds=[42])
    assert result["groups"][0]["n"] == 1
    assert set(result["groups"][0]["class_f1_mean"]) == {str(label) for label in range(1, 7)}
    assert result["groups"][0]["runs"][0]["best_epoch"] == 1
    sweep.main(["compare", "--plan", str(plan_path), "--run_root", str(args.run_root)])
    assert "macro_f1_mean" in capsys.readouterr().out
