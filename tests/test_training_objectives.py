"""Synthetic CPU tests for Train-only imbalance objectives and sampling."""

import inspect
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from training_objectives import FocalLoss, build_training_components, validate_training_options


def test_default_loss_preserves_cross_entropy_and_global_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    config = {"seed": 23}
    criterion, sampler, metadata = build_training_components([1, 1, 2, 6], config, "cpu")
    assert type(criterion) is nn.CrossEntropyLoss
    assert criterion.weight is None
    assert sampler is None
    logits = torch.tensor([[0.1, 0.2, 1.0, -0.2, 0.0, 0.9], [2.0, 1, 0, 0, 0, 0]])
    targets = torch.tensor([5, 0])
    assert torch.equal(criterion(logits, targets), nn.CrossEntropyLoss()(logits, targets))
    assert config == {"seed": 23}
    assert metadata["loss"] == "cross_entropy"
    assert metadata["sampling"] == "shuffle"
    assert metadata["class_weight_power"] == 0.5
    assert metadata["focal_gamma"] == 2.0
    assert random.getstate() == python_state
    actual_numpy_state = np.random.get_state()
    assert actual_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy_state[1], numpy_state[1])
    assert actual_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_weighted_loss_uses_train_counts_and_increases_minority_weight():
    criterion, sampler, metadata = build_training_components(
        [1] * 9 + [2], {"loss": "weighted_cross_entropy"}, "cpu"
    )
    torch.testing.assert_close(criterion.weight, torch.tensor([0.5, 1.5, 0, 0, 0, 0]))
    assert criterion.weight.dtype == torch.float32
    assert criterion.weight.device.type == "cpu"
    assert sampler is None
    assert metadata["train_class_counts"] == {"1": 9, "2": 1, "3": 0, "4": 0, "5": 0, "6": 0}
    assert metadata["class_weights"] == pytest.approx([0.5, 1.5, 0, 0, 0, 0])
    logits = torch.tensor([[0.7, 0.2, 0.1, -0.3, 1.0, -1], [0.1, 0.5, 1, 0, -1, -0.5]])
    targets = torch.tensor([0, 1])
    per_row = F.cross_entropy(logits, targets, reduction="none")
    expected = (0.5 * per_row[0] + 1.5 * per_row[1]) / 2
    torch.testing.assert_close(criterion(logits, targets), expected)


@pytest.mark.parametrize("power", [0, 0.5, 1])
def test_class_weights_are_finite_for_missing_train_classes(power):
    criterion, _, metadata = build_training_components(
        [6, 6, 6], {"loss": "weighted_cross_entropy", "class_weight_power": power}, "cpu"
    )
    torch.testing.assert_close(criterion.weight, torch.tensor([0, 0, 0, 0, 0, 1.0]))
    logits = torch.zeros(3, 6, requires_grad=True)
    criterion(logits, torch.full((3,), 5, dtype=torch.long)).backward()
    assert torch.isfinite(logits.grad).all()
    json.dumps(metadata, allow_nan=False)


def test_zero_power_weights_every_present_class_equally():
    criterion, _, _ = build_training_components(
        [1] * 10 + [2, 6],
        {"loss": "weighted_cross_entropy", "class_weight_power": 0},
        "cpu",
    )
    torch.testing.assert_close(criterion.weight, torch.tensor([1.0, 1, 0, 0, 0, 1]))


def test_balanced_sampler_has_equal_class_mass_and_private_deterministic_rng():
    labels = np.array([1] * 9 + [3] * 3 + [6])
    config = {"sampling": "balanced", "seed": 37}
    before = torch.get_rng_state().clone()
    _, sampler, metadata = build_training_components(labels, config, "cpu")
    _, repeat, _ = build_training_components(labels, config, "cpu")
    assert sampler.replacement is True
    assert sampler.num_samples == len(labels)
    probabilities = sampler.weights / sampler.weights.sum()
    for label in (1, 3, 6):
        assert probabilities[labels == label].sum().item() == pytest.approx(1 / 3)
    # Check exact draws across two epochs without flaky empirical-frequency assertions.
    for _ in range(2):
        indices = list(sampler)
        assert indices == list(repeat)
        assert len(indices) == len(labels)
        assert all(0 <= index < len(labels) for index in indices)
    assert torch.equal(before, torch.get_rng_state())
    assert metadata["sampler"]["replacement"] is True
    assert metadata["sampler"]["num_samples"] == len(labels)
    assert metadata["sampler"]["seed"] == 37


def test_focal_formula_and_gradient_match_scalar_definition():
    logits = torch.tensor([[1.3, -0.2, 0.7], [-0.5, 0.8, 0.1]], dtype=torch.float64)
    logits.requires_grad_()
    targets = torch.tensor([0, 2])
    gamma = 2.0
    actual = FocalLoss(gamma)(logits, targets)
    probabilities = logits.detach().softmax(dim=1)
    pt = probabilities[torch.arange(2), targets]
    expected = (-((1 - pt) ** gamma) * pt.log()).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    coefficient = (1 - pt) ** gamma - gamma * pt * (1 - pt) ** (gamma - 1) * pt.log()
    direction = probabilities - F.one_hot(targets, num_classes=3)
    expected_gradient = coefficient[:, None] * direction / len(targets)
    torch.testing.assert_close(logits.grad, expected_gradient)


def test_zero_gamma_matches_cross_entropy_value_and_gradient():
    logits = torch.tensor([[1.0, 0.3, -1], [-0.2, 0.7, 0.1]], requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    targets = torch.tensor([0, 2])
    actual = FocalLoss(0)(logits, targets)
    expected = F.cross_entropy(reference, targets)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


def test_focal_downweights_confident_examples_more():
    confident = torch.tensor([[5.0, 0.0, 0.0]])
    uncertain = torch.zeros(1, 3)
    target = torch.tensor([0])
    confident_ratio = FocalLoss()(confident, target) / F.cross_entropy(confident, target)
    uncertain_ratio = FocalLoss()(uncertain, target) / F.cross_entropy(uncertain, target)
    assert 0 < confident_ratio < uncertain_ratio < 1


@pytest.mark.parametrize("gamma", [0, 0.1, 0.5, 2, 10])
def test_focal_extreme_logits_have_finite_loss_and_gradients(gamma):
    logits = torch.tensor([[10000.0, -10000, 0], [-10000.0, 10000, 0]], requires_grad=True)
    value = FocalLoss(gamma)(logits, torch.tensor([0, 0]))
    assert value.ndim == 0
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("loss", ["cross_entropy", "weighted_cross_entropy", "focal"])
@pytest.mark.parametrize("sampling", ["shuffle", "balanced"])
def test_metadata_contains_only_aggregate_train_information(loss, sampling):
    _, _, metadata = build_training_components(
        [1, 2, 2, 5], {"loss": loss, "sampling": sampling, "seed": 1}, "cpu"
    )
    assert set(metadata) == {
        "loss",
        "sampling",
        "class_weight_power",
        "focal_gamma",
        "train_class_counts",
        "class_weights",
        "sampler",
    }
    assert metadata["train_class_counts"] == {"1": 1, "2": 2, "3": 0, "4": 0, "5": 1, "6": 0}
    assert metadata["loss"] == loss
    assert metadata["sampling"] == sampling
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    assert list(inspect.signature(build_training_components).parameters) == [
        "train_labels",
        "config",
        "device",
    ]


@pytest.mark.parametrize("labels", [[], [0], [7], [1.5], [None], [float("nan")]])
def test_empty_or_invalid_train_labels_fail(labels):
    with pytest.raises(ValueError):
        build_training_components(labels, {}, "cpu")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("loss", "unsupported"),
        ("loss", None),
        ("loss", True),
        ("sampling", "weighted"),
        ("sampling", None),
        ("sampling", True),
        ("class_weight_power", -0.1),
        ("class_weight_power", 1.1),
        ("class_weight_power", True),
        ("class_weight_power", "0.5"),
        ("class_weight_power", float("nan")),
        ("class_weight_power", float("inf")),
        ("focal_gamma", -0.1),
        ("focal_gamma", False),
        ("focal_gamma", "2.0"),
        ("focal_gamma", float("nan")),
        ("focal_gamma", float("inf")),
    ],
)
def test_training_options_reject_invalid_values(key, value):
    with pytest.raises(ValueError, match=key):
        validate_training_options({key: value})
    with pytest.raises(ValueError, match=key):
        build_training_components([1], {key: value}, "cpu")


def test_focal_constructor_validates_gamma():
    with pytest.raises(ValueError, match="focal_gamma"):
        FocalLoss(-1)
