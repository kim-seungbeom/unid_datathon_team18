"""Independent synthetic CPU tests for the isolated last-block experiment."""

# Optional runtime imports follow importorskip in code-only environments.
# ruff: noqa: E402

import argparse
import copy
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from torch import nn
from torch.utils.data import DataLoader

from experiments.seungbeom import resnet_finetune as experiment
from experiments.seungbeom import resnet_finetune_core as core
from experiments.seungbeom.neural_config import normalize_config as neural_config
from experiments.seungbeom.neural_sensors import NeuralSensorPreprocessor
from schema import RAW_SENSOR_COLUMNS


class TinyResNet(nn.Module):
    """Shape-compatible independent network; never load pretrained test weights."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.layer1 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(4, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(self.bn1(self.conv1(x)), 4)
        for block in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = block(x)
        return self.fc(x.mean(dim=(2, 3)))


@pytest.fixture
def tiny_backbone(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    calls = []

    def build(*, weights=None, **kwargs):
        calls.append(weights)
        return TinyResNet()

    vision = ModuleType("torchvision")
    models = ModuleType("torchvision.models")
    models.ResNet18_Weights = SimpleNamespace(IMAGENET1K_V1="synthetic_pretrained_requested")
    models.resnet18 = build
    vision.models = models
    monkeypatch.setitem(sys.modules, "torchvision", vision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)
    try:
        yield calls
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture
def synthetic_images(tmp_path):
    image_root = tmp_path / "invented_images"
    image_root.mkdir()
    frame = pd.DataFrame(
        {
            "sample_id": [f"invented_{i}" for i in range(6)],
            "experiment_id": [f"synthetic_group_{i // 2}" for i in range(6)],
            "image_path": [f"independent_{i}.png" for i in range(6)],
            "label": np.arange(1, 7),
            **{
                name: np.arange(6, dtype=float) + index
                for index, name in enumerate(RAW_SENSOR_COLUMNS)
            },
        }
    )
    frame.loc[0, "meas_v"] = np.nan
    for i, relative in enumerate(frame.image_path):
        pixels = np.zeros((12, 18, 3), dtype=np.uint8)
        pixels[..., i % 3] = 30 + 30 * i
        pixels[:5, :8, (i + 1) % 3] = 200
        Image.fromarray(pixels).save(image_root / relative)
    prep = NeuralSensorPreprocessor.fit(frame, RAW_SENSOR_COLUMNS)
    return frame, image_root, prep


@pytest.mark.parametrize("mode", ["frozen", "layer4"])
def test_training_updates_only_allowed_weights_and_preserves_all_bn(tiny_backbone, mode):
    torch.manual_seed(73)
    model = core.FineTuneModel(mode, sensor_hidden=4, fusion_hidden=8, dropout=0.0)
    optimizer = core.make_optimizer(model, 0.01, 0.005, 0.001)
    before = copy.deepcopy(model.state_dict())
    model.train()
    assert all(
        not part.training for part in model.backbone.modules() if isinstance(part, nn.BatchNorm2d)
    )
    logits = model(torch.randn(8, 3, 32, 32), torch.randn(8, 5))
    nn.functional.cross_entropy(logits, torch.arange(8) % 6).backward()
    optimizer.step()
    changed = {
        name for name, value in model.state_dict().items() if not torch.equal(before[name], value)
    }
    assert any(name.startswith("classifier.") for name in changed)
    assert any(name.startswith("sensor_encoder.") for name in changed)
    if mode == "layer4":
        assert any(name.startswith("backbone.layer4.") for name in changed)
    else:
        assert not any(name.startswith("backbone.") for name in changed)
    for name, value in model.backbone.state_dict().items():
        if (
            not name.startswith("layer4.")
            or mode == "frozen"
            or any(
                token in name for token in ("running_mean", "running_var", "num_batches_tracked")
            )
        ):
            assert torch.equal(before["backbone." + name], value), name
    for name, parameter in model.named_parameters():
        allowed = not name.startswith("backbone.") or (
            mode == "layer4" and name.startswith("backbone.layer4.")
        )
        assert parameter.requires_grad is allowed
        if not allowed:
            assert parameter.grad is None


def test_matched_controls_share_initial_weights_and_heads(tiny_backbone):
    torch.manual_seed(42)
    frozen = core.FineTuneModel("frozen")
    torch.manual_seed(42)
    fine = core.FineTuneModel("layer4")
    assert frozen.state_dict().keys() == fine.state_dict().keys()
    for name, value in frozen.state_dict().items():
        assert torch.equal(value, fine.state_dict()[name]), name
    assert all(weight == "synthetic_pretrained_requested" for weight in tiny_backbone)


def test_optimizer_rates_and_parameter_groups_are_disjoint(tiny_backbone):
    model = core.FineTuneModel("layer4")
    optimizer = core.make_optimizer(model, 0.001, 0.00001, 0.0001)
    assert sorted(group["lr"] for group in optimizer.param_groups) == [0.00001, 0.001]
    assigned = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    for group in optimizer.param_groups:
        if group["lr"] == 0.00001:
            assert {id(parameter) for parameter in group["params"]} == {
                id(parameter) for parameter in model.backbone.layer4.parameters()
            }


@pytest.mark.parametrize(
    "values", [(0, 0.00001, 0), (0.001, -1, 0), (float("nan"), 0.001, 0), (0.001, 0.001, -1)]
)
def test_invalid_optimizer_settings_fail(tiny_backbone, values):
    with pytest.raises(ValueError):
        core.make_optimizer(core.FineTuneModel("layer4"), *values)


def test_offline_state_reload_does_not_request_pretrained_weights(tiny_backbone, synthetic_images):
    frame, image_root, prep = synthetic_images
    torch.manual_seed(51)
    model = core.FineTuneModel("layer4", sensor_hidden=4, fusion_hidden=8)
    state = copy.deepcopy(model.state_dict())
    restored = core.FineTuneModel(**model.get_model_config(), pretrained=False)
    restored.load_state_dict(state, strict=True)
    assert tiny_backbone[-1] is None
    loader = DataLoader(core.FineTuneDataset(frame, image_root, prep), batch_size=4)
    np.testing.assert_array_equal(
        core.probabilities(model, loader, "cpu"), core.probabilities(restored, loader, "cpu")
    )


def test_predictions_do_not_depend_on_rows_order_batch_ids_or_labels(
    tiny_backbone, synthetic_images
):
    frame, image_root, prep = synthetic_images
    torch.manual_seed(19)
    model = core.FineTuneModel("layer4", sensor_hidden=4, fusion_hidden=8)

    def predict(part, batch_size):
        dataset = core.FineTuneDataset(part, image_root, prep, labeled=False)
        return core.probabilities(model, DataLoader(dataset, batch_size=batch_size), "cpu")

    reference = predict(frame, 4)
    assert reference.shape == (6, 6)
    np.testing.assert_allclose(reference.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(reference, predict(frame, 1))
    positions = [5, 1, 3]
    changed = frame.iloc[positions].copy()
    changed["sample_id"] = ["new_a", "new_b", "new_c"]
    changed["experiment_id"] = "unrelated_group"
    changed["relative_time_s"] = [-999, 0, 1e6]
    changed["label"] = 1
    np.testing.assert_array_equal(reference[positions], predict(changed, 2))


def test_dataset_transforms_without_fitting_or_exposing_identifiers(synthetic_images, monkeypatch):
    frame, image_root, prep = synthetic_images
    before = copy.deepcopy(prep.to_dict())
    observed = frame.copy()
    observed.loc[1, "meas_t1"] = 100000
    expected = prep.transform(observed)

    def no_fit(*args, **kwargs):
        pytest.fail("Dataset must never fit preprocessing")

    monkeypatch.setattr(NeuralSensorPreprocessor, "fit", no_fit)
    dataset = core.FineTuneDataset(observed, image_root, prep)
    item = dataset[0]
    assert set(item) == {"visual", "sensors", "targets"}
    assert item["visual"].shape == (3, 224, 224)
    assert item["targets"].item() == 0
    np.testing.assert_array_equal(dataset[1]["sensors"].numpy(), expected[1])
    assert prep.to_dict() == before
    assert set(core.FineTuneDataset(observed, image_root, prep, labeled=False)[0]) == {
        "visual",
        "sensors",
    }


@pytest.mark.parametrize("problem", ["traversal", "absolute", "symlink", "missing"])
def test_dataset_rejects_escaping_or_missing_images(synthetic_images, tmp_path, problem):
    frame, image_root, prep = synthetic_images
    outside = tmp_path / "outside.png"
    Image.new("RGB", (4, 4)).save(outside)
    frame = frame.copy()
    if problem == "traversal":
        frame.loc[0, "image_path"] = "../outside.png"
    elif problem == "absolute":
        frame.loc[0, "image_path"] = str(outside)
    elif problem == "symlink":
        (image_root / "escape.png").symlink_to(outside)
        frame.loc[0, "image_path"] = "escape.png"
    else:
        frame.loc[0, "image_path"] = "missing.png"
    with pytest.raises((ValueError, FileNotFoundError)):
        core.FineTuneDataset(frame, image_root, prep)[0]


def test_history_preprocessor_is_rejected(synthetic_images):
    frame, image_root, prep = synthetic_images
    prep = copy.deepcopy(prep)
    prep.imputation = "forward_fill"
    with pytest.raises(ValueError):
        core.FineTuneDataset(frame, image_root, prep)


def test_real_resnet_offline_constructor_has_only_last_block_trainable():
    pytest.importorskip("torchvision")
    model = core.FineTuneModel("layer4", pretrained=False)
    assert isinstance(model.backbone.fc, nn.Identity)
    for name, parameter in model.backbone.named_parameters():
        assert parameter.requires_grad == name.startswith("layer4.")
    model.train()
    assert all(
        not part.training for part in model.backbone.modules() if isinstance(part, nn.BatchNorm2d)
    )


@pytest.fixture
def paired_inputs(tmp_path, tiny_backbone):
    dataset = tmp_path / "invented_dataset"
    frames = []
    rng = np.random.default_rng(120)
    for split, groups in (("train", 5), ("validation", 2)):
        root = dataset / split
        root.mkdir(parents=True)
        n = groups * 6
        frame = pd.DataFrame(
            {
                "sample_id": [f"invented_{split}_{i}" for i in range(n)],
                "experiment_id": [f"invented_{split}_group_{i // 6}" for i in range(n)],
                "image_path": [f"independent_{i}.png" for i in range(n)],
                "label": np.tile(np.arange(1, 7), groups),
                **{name: rng.normal(size=n) for name in RAW_SENSOR_COLUMNS},
            }
        )
        frame.loc[::7, "meas_v"] = np.nan
        for i, name in enumerate(frame.image_path):
            pixels = np.zeros((12, 18, 3), dtype=np.uint8)
            pixels[..., i % 3] = 20 + i * 5
            pixels[:5, 3:10, (i + 1) % 3] = 230
            Image.fromarray(pixels).save(root / name)
        frame.to_csv(root / f"{split}.csv", index=False)
        frames.append(frame)
    # A deliberately unreadable synthetic Test CSV: training must never touch it.
    (dataset / "test").mkdir()
    (dataset / "test/test.csv").write_text("synthetic Test must not be read")
    config = experiment.normalize_config(
        {
            "epochs": 2,
            "patience": 1,
            "batch_size": 12,
            "sensor_hidden": 4,
            "fusion_hidden": 8,
            "dropout": 0.0,
            "head_lr": 0.01,
            "backbone_lr": 0.005,
        }
    )
    path = tmp_path / "synthetic_config.json"
    experiment.neural.write_json(path, config)
    reference = tmp_path / "historical_reference"
    reference.mkdir()
    original = neural_config(
        {
            **{key: config[key] for key in ("sensor_hidden", "fusion_hidden", "dropout")},
            "epochs": 2,
        }
    )
    experiment.neural.write_json(reference / "status.json", {"status": "complete"})
    experiment.neural.write_json(reference / "config.json", original)
    experiment.neural.write_json(
        reference / "identity.json",
        {
            "config": original,
            **{
                f"{split}_csv_sha256": experiment.neural.file_sha(dataset / split / f"{split}.csv")
                for split in ("train", "validation")
            },
        },
    )
    metrics = experiment.classification_metrics(frames[1].label, frames[1].label)
    experiment.neural.write_json(reference / "best_metrics.json", {"epoch": 2, **metrics})
    (reference / "best.pt").write_bytes(b"Independent synthetic historical artifact; never loaded")
    args = argparse.Namespace(
        reference_run=reference,
        data_dir=dataset,
        config=path,
        output_dir=tmp_path / "new_experiment",
        device="cpu",
        resume=False,
    )
    return args, frames, config


def trial_directory(args, mode):
    state = experiment.read_json(args.output_dir / "status.json")
    return args.output_dir / state["trials"][mode]


def test_inner_split_is_group_disjoint_shared_and_reproducible(paired_inputs):
    _, (frame, _), _ = paired_inputs
    part = experiment.inner_split(frame, 42)
    assert part == experiment.inner_split(frame, 42)
    assert set(part["fit"]).isdisjoint(part["stop"])
    assert sorted(part["fit"] + part["stop"]) == list(range(len(frame)))
    assert set(frame.iloc[part["fit"]].experiment_id).isdisjoint(
        frame.iloc[part["stop"]].experiment_id
    )
    assert set(frame.iloc[part["fit"]].label) == set(range(1, 7))


@pytest.mark.parametrize("problem", ["few_groups", "missing_class", "overlap_ids", "blank_group"])
def test_inner_split_rejects_bad_identity_or_missing_training_classes(paired_inputs, problem):
    frame = paired_inputs[1][0].copy()
    if problem == "few_groups":
        frame["experiment_id"] = "one"
    elif problem == "missing_class":
        frame.loc[frame.label == 6, "label"] = 5
    elif problem == "overlap_ids":
        frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    else:
        frame.loc[0, "experiment_id"] = " "
    with pytest.raises(ValueError):
        experiment.inner_split(frame, 42)


def test_paired_run_replays_checkpoints_preserves_reference_and_resumes(
    paired_inputs, monkeypatch, capsys
):
    args, (train, _), config = paired_inputs
    historical_before = {
        name: experiment.neural.file_sha(args.reference_run / name)
        for name in ("status.json", "config.json", "identity.json", "best_metrics.json", "best.pt")
    }
    read_paths = []
    original_read = experiment.read_frame

    def guarded_read(path, **kwargs):
        read_paths.append(path)
        assert path.parent.name in ("train", "validation")
        return original_read(path, **kwargs)

    monkeypatch.setattr(experiment, "read_frame", guarded_read)
    results = experiment.run(args)
    assert len(read_paths) == 2
    assert {row["mode"] for row in results} == {"frozen", "layer4"}
    assert results[0]["initial_state_sha256"] == results[1]["initial_state_sha256"]
    expected_prep = NeuralSensorPreprocessor.fit(train, RAW_SENSOR_COLUMNS).to_dict()
    for mode in ("frozen", "layer4"):
        model, prep, checkpoint = experiment.load_checkpoint(trial_directory(args, mode), "cpu")
        assert prep.to_dict() == expected_prep
        assert (
            checkpoint["selection"] == "train_inner_experiment_holdout_then_fresh_all_train_refit"
        )
        assert checkpoint["model_config"]["mode"] == mode
        assert checkpoint["config"] == config
        assert not model.training
    assert historical_before == {
        name: experiment.neural.file_sha(args.reference_run / name) for name in historical_before
    }
    report = experiment.report(argparse.Namespace(output_dir=args.output_dir, class_details=True))
    assert set(report) == {"historical_neural", "frozen", "layer4"}
    text = capsys.readouterr().out
    assert "different optimizer/selection protocol" in text
    assert "not 3-fold CV" in text
    before_state = experiment.read_json(args.output_dir / "status.json")

    def no_training(*args, **kwargs):
        pytest.fail("Completed trials must be reused on resume")

    monkeypatch.setattr(experiment, "fit_trial", no_training)
    args.resume = True
    experiment.run(args)
    assert experiment.read_json(args.output_dir / "status.json") == before_state


def test_official_validation_cannot_change_learned_weights_preprocessing_or_epoch(paired_inputs):
    args, (_, validation), _ = paired_inputs
    experiment.run(args)
    first = {
        mode: experiment.load_checkpoint(trial_directory(args, mode), "cpu")[2]
        for mode in ("frozen", "layer4")
    }
    validation = validation.copy()
    validation["label"] = 7 - validation["label"]
    validation[list(RAW_SENSOR_COLUMNS)] = 100000
    root = args.data_dir / "validation"
    for image in validation.image_path:
        Image.new("RGB", (12, 18), (255, 255, 255)).save(root / image)
    validation.to_csv(root / "validation.csv", index=False)
    # Make a separate synthetic historical run for the changed comparison rows.
    reference = args.reference_run.with_name("second_synthetic_reference")
    shutil.copytree(args.reference_run, reference)
    identity = experiment.read_json(reference / "identity.json")
    identity["validation_csv_sha256"] = experiment.neural.file_sha(root / "validation.csv")
    experiment.neural.write_json(reference / "identity.json", identity)
    args.reference_run = reference
    args.output_dir = args.output_dir.with_name("second_experiment")
    experiment.run(args)
    for mode in ("frozen", "layer4"):
        checkpoint = experiment.load_checkpoint(trial_directory(args, mode), "cpu")[2]
        assert checkpoint["selected_epoch"] == first[mode]["selected_epoch"]
        assert checkpoint["preprocessor"] == first[mode]["preprocessor"]
        for name, value in checkpoint["state_dict"].items():
            assert torch.equal(value, first[mode]["state_dict"][name]), name


def test_resume_restarts_missing_trial_status_without_overwriting_old_attempt(
    paired_inputs, monkeypatch
):
    args, _, _ = paired_inputs
    experiment.run(args)
    frozen = trial_directory(args, "frozen")
    interrupted = trial_directory(args, "layer4")
    saved_hash = experiment.neural.file_sha(interrupted / "model.pt")
    (interrupted / "status.json").unlink()
    old_fit = experiment.fit_trial
    modes = []

    def tracked(*values, **kwargs):
        modes.append(values[4])
        return old_fit(*values, **kwargs)

    monkeypatch.setattr(experiment, "fit_trial", tracked)
    args.resume = True
    experiment.run(args)
    assert modes == ["layer4"]
    assert trial_directory(args, "frozen") == frozen
    assert trial_directory(args, "layer4") != interrupted
    assert experiment.neural.file_sha(interrupted / "model.pt") == saved_hash
    assert experiment.read_json(args.output_dir / "status.json")["status"] == "complete"


@pytest.mark.parametrize("key", ["source_sha256", "runtime", "train_csv_sha256"])
def test_resume_rejects_rehashed_trial_identity_mismatch(paired_inputs, key):
    args, _, _ = paired_inputs
    experiment.run(args)
    trial = trial_directory(args, "frozen")
    result = experiment.read_json(trial / "result.json")
    result[key] = "different_synthetic_identity"
    experiment.neural.write_json(trial / "result.json", result)
    status = experiment.read_json(trial / "status.json")
    status["files"]["result.json"] = experiment.neural.file_sha(trial / "result.json")
    experiment.neural.write_json(trial / "status.json", status)
    args.resume = True
    with pytest.raises(ValueError):
        experiment.run(args)


def test_saved_model_predicts_after_all_training_material_removed(
    paired_inputs, tmp_path, monkeypatch
):
    args, (_, validation), _ = paired_inputs
    experiment.run(args)
    trial = trial_directory(args, "layer4")
    inference = tmp_path / "independent_inference"
    shutil.copytree(args.data_dir / "validation", inference)
    request = validation.iloc[[5, 1, 9]].copy()
    request["label"] = "ignored_non_label"
    request.to_csv(inference / "input.csv", index=False)
    template = request.iloc[::-1][["sample_id", "label"]]
    template.to_csv(inference / "template.csv", index=False)
    shutil.rmtree(args.data_dir)
    shutil.rmtree(args.reference_run)

    def no_fit(*args, **kwargs):
        pytest.fail("Saved model prediction must not fit or read training material")

    monkeypatch.setattr(NeuralSensorPreprocessor, "fit", no_fit)
    monkeypatch.setattr(experiment, "train_phase", no_fit)
    output = inference / "prediction.csv"
    prediction_args = argparse.Namespace(
        run_dir=trial,
        input=inference / "input.csv",
        image_root=inference,
        output=output,
        device="cpu",
        batch_size=2,
        sample_submission=inference / "template.csv",
    )
    experiment.predict(prediction_args)
    actual = pd.read_csv(output)
    assert list(actual.columns) == ["sample_id", "label"]
    assert actual.sample_id.tolist() == template.sample_id.tolist()
    assert actual.label.isin(range(1, 7)).all()
    with pytest.raises(ValueError, match="Output exists"):
        experiment.predict(prediction_args)
    (trial / "model.pt").write_bytes(b"tampered synthetic checkpoint")
    with pytest.raises(ValueError, match="artifacts changed"):
        experiment.load_checkpoint(trial, "cpu")
