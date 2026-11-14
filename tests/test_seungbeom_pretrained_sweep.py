"""Pretrained sweep wiring tested with invented caches and mocked training only."""

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.seungbeom import neural_sweep as sweep
from experiments.seungbeom.neural_config import normalize_config

DIMENSIONS = {"resnet18": 512, "convnext_tiny": 768, "dinov2": 384}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_pair(root, model):
    paths = []
    for split in ("train", "validation"):
        directory = root / model / split
        directory.mkdir(parents=True)
        sweep.write_json(
            directory / "metadata.json",
            {
                "status": "complete",
                "model": model,
                "dimension": DIMENSIONS[model],
                "weights": f"INDEPENDENT_SYNTHETIC_{model}",
                "trained": False,
                "preprocessing": {"fixture": "invented", "size": 224},
                "l2_normalize": False,
                "rows": 2,
            },
        )
        (directory / "sample_ids.csv").write_text("sample_id\ninvented_1\ninvented_2\n")
        np.save(directory / "features.npy", np.zeros((2, DIMENSIONS[model]), dtype=np.float32))
        paths.append(directory)
    return paths


def setup_args(root, *, stage="pretrained"):
    plan_path = root / "plan.json"
    sweep.write_json(plan_path, sweep.create_plan([{}], stage, max_trials=3))
    data = root / "invented_data"
    for split in ("train", "validation"):
        directory = data / split
        directory.mkdir(parents=True)
        (directory / f"{split}.csv").write_text("independent mock CSV fixture\n")
    return argparse.Namespace(
        plan=plan_path,
        data_dir=data,
        run_root=root / "results",
        device="cpu",
        train_embeddings=None,
        val_embeddings=None,
        encoder_embeddings=None,
    )


def invented_identity(config, train_csv, val_csv, *, train_embeddings=None, val_embeddings=None):
    provenance = {}
    for split, directory in (("train", train_embeddings), ("validation", val_embeddings)):
        provenance[split] = (
            {
                name: digest(Path(directory) / name)
                for name in ("metadata.json", "sample_ids.csv", "features.npy")
            }
            if directory is not None
            else {"image_content_sha256": "invented-image-bytes"}
        )
    return {
        "config": config,
        "train_csv_sha256": digest(train_csv),
        "validation_csv_sha256": digest(val_csv),
        "source_sha256": {"invented.py": "synthetic-source-version"},
        "input_provenance": provenance,
    }


def command_arguments(command):
    return {command[index]: command[index + 1] for index in range(3, len(command), 2)}


def fake_train(command, cwd, check):
    assert cwd == sweep.ROOT and check is False
    args = command_arguments(command)
    run, data = Path(args["--run_dir"]), Path(args["--data_dir"])
    config = sweep.read_json(args["--config"])
    run.mkdir()
    sweep.write_json(run / "config.json", config)
    sweep.write_json(
        run / "identity.json",
        invented_identity(
            config,
            data / "train/train.csv",
            data / "validation/validation.csv",
            train_embeddings=args.get("--train_embeddings"),
            val_embeddings=args.get("--val_embeddings"),
        ),
    )
    sweep.write_json(run / "best_metrics.json", {"macro_f1": 0.5})
    sweep.write_json(run / "status.json", {"status": "complete"})
    return SimpleNamespace(returncode=0)


def never_train(*args, **kwargs):
    pytest.fail("A rejected or already completed trial must not start training.")


def test_pretrained_plan_changes_only_encoder_name_and_matching_dimension():
    base = normalize_config({"seed": 44, "learning_rate": 0.0003, "fusion": "gated"})
    plan = sweep.create_plan([base], "pretrained", max_trials=3)
    assert [
        (t["config"]["embedding_encoder"], t["config"]["embedding_dim"]) for t in plan["trials"]
    ] == list(DIMENSIONS.items())
    for trial in plan["trials"]:
        assert set(trial["changes"]) <= {"embedding_encoder", "embedding_dim"}
        for key, value in base.items():
            if key not in ("embedding_encoder", "embedding_dim"):
                assert trial["config"][key] == value
    assert sweep.create_plan([base, base], "pretrained", max_trials=3) == plan
    sweep.validate_plan(plan)
    with pytest.raises(ValueError, match="budget"):
        sweep.create_plan([base], "pretrained", max_trials=2)


@pytest.mark.parametrize("encoder", ["small_cnn", "resnet18"])
def test_pretrained_stage_rejects_a_scratch_encoder_base(encoder):
    with pytest.raises(ValueError):
        sweep.create_plan([{"encoder": encoder}], "pretrained", max_trials=3)


def test_model_specific_cache_routing_and_omitted_model_extracts_live(tmp_path):
    args = setup_args(tmp_path)
    resnet = cache_pair(tmp_path / "caches", "resnet18")
    convnext = cache_pair(tmp_path / "caches", "convnext_tiny")
    args.encoder_embeddings = [["convnext_tiny", *convnext], ["resnet18", *resnet]]
    calls, identities = [], []

    def identity(config, *paths, **kwargs):
        identities.append((config["embedding_encoder"], kwargs))
        return invented_identity(config, *paths, **kwargs)

    def run(command, **kwargs):
        calls.append(command_arguments(command))
        return fake_train(command, **kwargs)

    sweep.run_command(args, identity_fn=identity, runner=run)
    assert len(calls) == len(identities) == 3
    for call, (model, identity_paths) in zip(calls, identities, strict=True):
        assert sweep.read_json(call["--config"])["embedding_encoder"] == model
        pair = {"resnet18": resnet, "convnext_tiny": convnext}.get(model)
        if pair is None:
            assert "--train_embeddings" not in call and "--val_embeddings" not in call
            assert identity_paths == {"train_embeddings": None, "val_embeddings": None}
        else:
            assert Path(call["--train_embeddings"]) == pair[0]
            assert Path(call["--val_embeddings"]) == pair[1]
            assert Path(identity_paths["train_embeddings"]) == pair[0]
            assert Path(identity_paths["val_embeddings"]) == pair[1]
    # Cache provenance must be persisted, not merely validated once.
    context = sweep.canonical(sweep.read_json(args.run_root / "sweep_status.json")["context"])
    for directory in (*resnet, *convnext):
        for name in ("metadata.json", "sample_ids.csv", "features.npy"):
            assert digest(directory / name) in context
    assert not (args.data_dir / "test").exists()


def test_completed_model_trials_reuse_and_changed_cache_bytes_reject(tmp_path):
    args = setup_args(tmp_path)
    pair = cache_pair(tmp_path / "caches", "resnet18")
    args.encoder_embeddings = [["resnet18", *pair]]
    sweep.run_command(args, identity_fn=invented_identity, runner=fake_train)
    sweep.run_command(args, identity_fn=invented_identity, runner=never_train)
    states = sweep.read_json(args.run_root / "sweep_status.json")["trials"].values()
    assert all(state["status"] == "complete" and state["reused"] for state in states)
    values = np.load(pair[0] / "features.npy", allow_pickle=False)
    values[0, 0] = 0.25
    np.save(pair[0] / "features.npy", values)
    with pytest.raises(ValueError):
        sweep.run_command(args, identity_fn=invented_identity, runner=never_train)


@pytest.mark.parametrize("issue", ["mixed_global", "both_modes", "duplicate", "unused", "unknown"])
def test_invalid_cache_mapping_rejected_before_starting_a_trial(tmp_path, issue):
    args = setup_args(tmp_path, stage="fusion" if issue == "unused" else "pretrained")
    pair = [tmp_path / "unread_train", tmp_path / "unread_validation"]
    if issue in ("mixed_global", "both_modes"):
        args.train_embeddings, args.val_embeddings = pair
        if issue == "both_modes":
            args.encoder_embeddings = [["resnet18", *pair]]
    elif issue == "duplicate":
        args.encoder_embeddings = [["resnet18", *pair], ["resnet18", *pair]]
    elif issue == "unused":
        args.encoder_embeddings = [["dinov2", *pair]]
    else:
        args.encoder_embeddings = [["unknown_model", *pair]]
    with pytest.raises(ValueError):
        sweep.run_command(args, identity_fn=invented_identity, runner=never_train)
    assert not args.run_root.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"status": "running"},
        {"model": "dinov2"},
        {"dimension": 384},
        {"trained": True},
        {"weights": "DIFFERENT_SYNTHETIC_WEIGHTS"},
        {"preprocessing": {"fixture": "different"}},
        {"l2_normalize": True},
    ],
)
def test_incomplete_wrong_encoder_or_mismatched_validation_cache_rejected(tmp_path, change):
    args = setup_args(tmp_path)
    pair = cache_pair(tmp_path / "caches", "resnet18")
    args.encoder_embeddings = [["resnet18", *pair]]
    metadata = sweep.read_json(pair[1] / "metadata.json")
    sweep.write_json(pair[1] / "metadata.json", {**metadata, **change})
    with pytest.raises(ValueError):
        sweep.run_command(args, identity_fn=invented_identity, runner=never_train)
    assert not args.run_root.exists()


def test_interruption_before_next_trial_preserves_completed_control(tmp_path):
    args = setup_args(tmp_path)
    pair = cache_pair(tmp_path / "caches", "resnet18")
    args.encoder_embeddings = [["resnet18", *pair]]
    calls = []

    def interrupted(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return fake_train(command, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        sweep.run_command(args, identity_fn=invented_identity, runner=interrupted)
    completed = Path(command_arguments(calls[0])["--run_dir"])
    before = (completed / "identity.json").read_bytes()
    resumed = []

    def run_remaining(command, **kwargs):
        resumed.append(command)
        return fake_train(command, **kwargs)

    sweep.run_command(args, identity_fn=invented_identity, runner=run_remaining)
    assert len(resumed) == 2
    assert (completed / "identity.json").read_bytes() == before
    assert all(Path(command_arguments(command)["--run_dir"]) != completed for command in resumed)


def test_cli_accepts_repeated_model_cache_triples(tmp_path, monkeypatch):
    parsed = []
    monkeypatch.setattr(sweep, "run_command", lambda args: parsed.append(args))
    sweep.main(
        [
            "run",
            "--plan",
            str(tmp_path / "plan.json"),
            "--data_dir",
            str(tmp_path / "invented"),
            "--run_root",
            str(tmp_path / "results"),
            "--encoder_embeddings",
            "resnet18",
            "synthetic_resnet_train",
            "synthetic_resnet_val",
            "--encoder_embeddings",
            "dinov2",
            "synthetic_dino_train",
            "synthetic_dino_val",
        ]
    )
    assert parsed[0].encoder_embeddings == [
        ["resnet18", "synthetic_resnet_train", "synthetic_resnet_val"],
        ["dinov2", "synthetic_dino_train", "synthetic_dino_val"],
    ]
