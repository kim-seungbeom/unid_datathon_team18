"""Pure synthetic tests for Seungbeom's classwise mixture; no external data."""

import json

import numpy as np
import pytest

from experiments.seungbeom.ensemble_core import (
    blend_scores,
    classification_metrics,
    fit_classwise_weights,
    predict_labels,
    validate_probabilities,
    validate_weights,
)


def synthetic_probabilities(n_samples=60, n_models=3):
    rng = np.random.default_rng(123)
    return rng.dirichlet(np.ones(6), size=(n_samples, n_models))


def test_classwise_blending_does_not_mix_classes_or_rows():
    probabilities = synthetic_probabilities()
    weights = np.zeros((3, 6))
    for column in range(6):
        weights[column % 3, column] = 1
    scores = blend_scores(probabilities, weights)
    expected = np.column_stack([probabilities[:, c % 3, c] for c in range(6)])
    np.testing.assert_array_equal(scores, expected)
    np.testing.assert_array_equal(
        blend_scores(probabilities[[7, 2, 19]], weights), scores[[7, 2, 19]]
    )
    np.testing.assert_array_equal(predict_labels(probabilities, weights), scores.argmax(axis=1) + 1)
    assert not np.allclose(scores.sum(axis=1), 1.0)


def test_six_class_metrics_include_absent_classes_and_actual_rows():
    metrics = classification_metrics([1, 1, 2, 2], [1, 2, 2, 3])
    assert np.array(metrics["confusion_matrix"]).shape == (6, 6)
    assert metrics["confusion_matrix"][0] == [1, 1, 0, 0, 0, 0]
    assert metrics["class_precision"]["1"] == 1.0
    assert metrics["class_recall"]["1"] == 0.5
    assert metrics["class_f1"]["6"] == 0.0
    assert metrics["support"]["6"] == 0
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 6)


def test_search_is_reproducible_and_no_worse_than_oof_baselines():
    probabilities = synthetic_probabilities()
    labels = np.tile(np.arange(1, 7), 10)
    weights, report = fit_classwise_weights(probabilities, labels, seed=42)
    repeated, repeated_report = fit_classwise_weights(probabilities, labels, seed=42)
    np.testing.assert_array_equal(weights, repeated)
    assert report == repeated_report
    assert validate_weights(weights).shape == (3, 6)
    comparisons = report["oof_comparisons"]
    assert comparisons["classwise"]["macro_f1"] >= comparisons["equal_average"]["macro_f1"]
    assert comparisons["classwise"]["macro_f1"] >= max(
        entry["macro_f1"] for entry in comparisons["single_models"]
    )
    assert report["evaluations"] <= report["parameters"]["max_evaluations"]
    assert report["selection_data"] == "train_oof_only"
    json.dumps(report, allow_nan=False)


def test_classwise_specialists_can_beat_every_individual_and_uniform_average():
    # Both predictors overassign another class outside their expertise.
    # Each is informative for three classes; the mixture must use that evidence.
    labels = np.tile(np.arange(1, 7), 8)
    probabilities = np.zeros((len(labels), 2, 6))
    for row, label in enumerate(labels):
        true = label - 1
        specialist = 0 if label <= 3 else 1
        wrong = (true + 1) % 3 + 3 * specialist
        probabilities[row, specialist] = 0.08
        probabilities[row, specialist, true] = 0.6
        probabilities[row, 1 - specialist] = 0.06
        probabilities[row, 1 - specialist, wrong] = 0.7
    weights, report = fit_classwise_weights(probabilities, labels)
    comparisons = report["oof_comparisons"]
    assert comparisons["classwise"]["macro_f1"] > comparisons["equal_average"]["macro_f1"]
    assert comparisons["classwise"]["macro_f1"] > max(
        entry["macro_f1"] for entry in comparisons["single_models"]
    )
    assert classification_metrics(labels, predict_labels(probabilities, weights))["macro_f1"] == 1.0


def test_single_model_and_missing_classes_are_finite():
    probabilities = synthetic_probabilities(n_samples=5, n_models=1)
    weights, report = fit_classwise_weights(probabilities, [1, 1, 1, 2, 2])
    np.testing.assert_array_equal(weights, np.ones((1, 6)))
    assert report["evaluations"] == 0
    assert report["selected_strategy"] == "equal_average"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.zeros((3, 2, 6)),
        np.full((3, 2, 6), np.nan),
        np.full((3, 2, 6), np.inf),
        np.full((3, 2, 6), -1.0),
        np.full((3, 2, 6), 2.0),
        np.zeros((0, 2, 6)),
        np.zeros((3, 0, 6)),
        np.zeros((3, 2, 5)),
        np.ones((3, 6)),
    ],
)
def test_reject_invalid_probabilities(probabilities):
    with pytest.raises(ValueError, match="Probabilities|probability"):
        validate_probabilities(probabilities)


@pytest.mark.parametrize(
    "weights",
    [np.zeros((2, 6)), np.full((2, 6), np.nan), np.ones((2, 5)), [[-1] * 6, [2] * 6]],
)
def test_reject_invalid_weights(weights):
    with pytest.raises(ValueError, match="Weights"):
        validate_weights(weights)


def test_reject_model_count_mismatch_and_bad_labels():
    probabilities = synthetic_probabilities()
    with pytest.raises(ValueError, match="number of models"):
        blend_scores(probabilities, np.full((2, 6), 0.5))
    with pytest.raises(ValueError, match="number of samples"):
        fit_classwise_weights(probabilities, [1, 2])
    with pytest.raises(ValueError, match="labels 1..6"):
        fit_classwise_weights(probabilities, [0] * len(probabilities))
    with pytest.raises(ValueError, match="same length"):
        classification_metrics([1], [1, 2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"max_rounds": 0},
        {"max_rounds": 21},
        {"regularization": -0.1},
        {"regularization": float("nan")},
    ],
)
def test_reject_invalid_search_options(kwargs):
    with pytest.raises(ValueError):
        fit_classwise_weights(synthetic_probabilities(), np.tile(np.arange(1, 7), 10), **kwargs)
