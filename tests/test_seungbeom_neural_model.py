"""Synthetic neural fusion/config tests; no competition data or pretrained downloads."""

import copy
import io

import pytest

torch = pytest.importorskip("torch")

from experiments.seungbeom.neural_config import (  # noqa: E402
    DEFAULT_CONFIG,
    load_config,
    model_config,
    normalize_config,
)
from experiments.seungbeom.neural_model import NeuralFusionClassifier  # noqa: E402


@pytest.mark.parametrize("encoder", ["embedding", "small_cnn"])
@pytest.mark.parametrize("fusion", ["concat", "projected_concat", "gated"])
def test_shapes_and_both_paths_receive_gradients(encoder, fusion):
    torch.manual_seed(14)
    model = NeuralFusionClassifier(
        encoder=encoder, fusion=fusion, embedding_dim=12, projection_dim=8
    )
    shape = (4, 12) if encoder == "embedding" else (4, 3, 32, 32)
    visual = torch.randn(shape, requires_grad=True)
    sensors = torch.randn(4, 5, requires_grad=True)
    logits = model(visual, sensors)
    assert logits.shape == (4, 6)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 4, 5])).backward()
    assert visual.grad.abs().sum() > 0
    assert sensors.grad.abs().sum() > 0
    assert model.sensor_encoder[0].weight.grad.abs().sum() > 0
    if encoder == "small_cnn":
        assert model.image_encoder[0].weight.grad.abs().sum() > 0
    if fusion == "gated":
        assert model.gate[0].weight.grad.abs().sum() > 0
        gates = model.eval().gate_values(visual.detach(), sensors.detach())
        assert gates.shape == (4, 8)
        assert torch.all((gates >= 0) & (gates <= 1))


def test_projection_control_matches_the_gated_projections():
    torch.manual_seed(42)
    control = NeuralFusionClassifier(fusion="projected_concat", embedding_dim=12)
    torch.manual_seed(42)
    gated = NeuralFusionClassifier(fusion="gated", embedding_dim=12)
    for name in ("sensor_encoder", "image_projection", "sensor_projection"):
        left, right = getattr(control, name).state_dict(), getattr(gated, name).state_dict()
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert control.classifier[0].in_features == 128
    assert gated.classifier[0].in_features == 64
    with pytest.raises(ValueError, match="only supported"):
        control.gate_values(torch.randn(2, 12), torch.randn(2, 5))


@pytest.mark.parametrize("fusion", ["concat", "projected_concat", "gated"])
def test_restore_and_eval_batch_independence(fusion):
    torch.manual_seed(44)
    model = NeuralFusionClassifier(fusion=fusion, embedding_dim=12).eval()
    visual, sensors = torch.randn(5, 12), torch.randn(5, 5)
    expected = model(visual, sensors).detach()
    handle = io.BytesIO()
    torch.save({"config": model.get_model_config(), "state_dict": model.state_dict()}, handle)
    handle.seek(0)
    saved = torch.load(handle, weights_only=True)
    restored = NeuralFusionClassifier(**saved["config"]).eval()
    restored.load_state_dict(saved["state_dict"], strict=True)
    torch.testing.assert_close(restored(visual, sensors), expected, rtol=0, atol=0)
    per_row = torch.cat([restored(visual[i : i + 1], sensors[i : i + 1]) for i in range(5)])
    torch.testing.assert_close(per_row, expected, rtol=1e-5, atol=1e-7)
    assert torch.equal(per_row.argmax(dim=1), expected.argmax(dim=1))
    assert torch.all((per_row.argmax(dim=1) + 1 >= 1) & (per_row.argmax(dim=1) + 1 <= 6))


def test_small_cnn_concat_preserves_legacy_structure():
    from model import FusionClassifier

    torch.manual_seed(9)
    legacy = FusionClassifier().eval()
    new = NeuralFusionClassifier(encoder="small_cnn").eval()
    new.load_state_dict(legacy.state_dict(), strict=True)
    visual, sensors = torch.randn(3, 3, 32, 32), torch.randn(3, 5)
    torch.testing.assert_close(new(visual, sensors), legacy(visual, sensors))


def test_resnet18_is_scratch_and_eval_uses_frozen_batchnorm_statistics(monkeypatch):
    torchvision = pytest.importorskip("torchvision")
    original = torchvision.models.resnet18
    requested = []

    def checked(**kwargs):
        requested.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(torchvision.models, "resnet18", checked)
    model = NeuralFusionClassifier(encoder="resnet18", fusion="gated").eval()
    batchnorm = model.image_encoder.bn1
    mean = batchnorm.running_mean.clone()
    visual, sensors = torch.randn(2, 3, 32, 32), torch.randn(2, 5)
    logits = model(visual, sensors)
    assert logits.shape == (2, 6)
    assert requested == [{"weights": None}]
    torch.testing.assert_close(batchnorm.running_mean, mean, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder": "unknown"},
        {"fusion": "unknown"},
        {"sensor_dim": 0},
        {"projection_dim": True},
        {"dropout": float("nan")},
        {"dropout": 1},
    ],
)
def test_invalid_architectures_are_rejected(kwargs):
    with pytest.raises(ValueError):
        NeuralFusionClassifier(**kwargs)


def test_inputs_are_validated():
    model = NeuralFusionClassifier(embedding_dim=12)
    with pytest.raises(ValueError, match="Sensor dimension"):
        model(torch.randn(2, 12), torch.randn(2, 6))
    with pytest.raises(ValueError, match="embedding dimension"):
        model(torch.randn(2, 13), torch.randn(2, 5))
    with pytest.raises(ValueError, match="batch sizes"):
        model(torch.randn(3, 12), torch.randn(2, 5))
    with pytest.raises(ValueError, match="required"):
        model(None, torch.randn(2, 5))


def test_config_defaults_and_model_mapping_are_independent(tmp_path):
    supplied = {"fusion": "gated", "early_stopping_patience": 0}
    original = copy.deepcopy(supplied)
    config = normalize_config(supplied)
    assert config["epochs"] == 50
    assert config["early_stopping_patience"] == 0
    assert supplied == original
    config["sensor_features"].append("meas_co")
    assert len(DEFAULT_CONFIG["sensor_features"]) == 5
    assert model_config(config)["sensor_dim"] == 6
    path = tmp_path / "config.json"
    path.write_text('{"fusion":"gated"}')
    assert load_config(path) == normalize_config({"fusion": "gated"})


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": 1},
        {"pretrained": True},
        {"encoder": "unknown"},
        {"fusion": "invalid"},
        {"epochs": 0},
        {"epochs": True},
        {"seed": 2**32},
        {"early_stopping_patience": -1},
        {"min_delta": float("inf")},
        {"learning_rate": 0},
        {"dropout": 1},
        {"class_weight_power": 1.1},
        {"image_size": 8},
        {"sensor_features": ["label"]},
        {"sensor_features": ["meas_v", "meas_v_mean30s"]},
        {"sensor_imputation": "forward_fill"},
        {"sensor_scaling": "robust"},
        {"embedding_dim": 384},
        {"sampling": "invalid"},
        {"sampling": "balanced"},
        {"loss": "invalid"},
    ],
)
def test_invalid_or_mixed_comparisons_are_rejected(changes):
    with pytest.raises(ValueError):
        normalize_config(changes)


def test_dinov2_explicit_dimension_and_supported_loss():
    config = normalize_config(
        {"embedding_encoder": "dinov2", "embedding_dim": 384, "loss": "weighted_cross_entropy"}
    )
    assert model_config(config)["embedding_dim"] == 384
