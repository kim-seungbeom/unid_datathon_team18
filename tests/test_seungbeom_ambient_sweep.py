"""Ambient ablations checked with independent, invented sensor rows only."""

import json

import numpy as np
import pandas as pd
import pytest

from experiments.seungbeom import neural_sweep as sweep
from experiments.seungbeom.neural_config import model_config, normalize_config
from experiments.seungbeom.neural_sensors import (
    AMBIENT_VARIANTS,
    NeuralSensorPreprocessor,
    neural_sensor_values,
    sensor_variant,
)
from schema import RAW_SENSOR_COLUMNS


def invented_train():
    # Deliberately no labels, IDs, images, or time: these are same-row features.
    return pd.DataFrame(
        {
            "meas_v": [1.0, 3.0, 5.0],
            "meas_a": [0.0, 0.0, 0.0],
            "meas_t1": [10.0, 40.0, np.nan],
            "meas_t2": [30.0, 8.0, 60.0],
            "meas_p": [2.0, 4.0, 6.0],
            "atmp_t": [20.0, 10.0, 20.0],
        },
        index=[19, 3, 42],
    )


@pytest.mark.parametrize(
    "base_overrides",
    [
        {},
        {
            "embedding_encoder": "convnext_tiny",
            "embedding_dim": 768,
            "seed": 44,
            "fusion": "gated",
            "learning_rate": 0.0003,
        },
    ],
)
def test_ambient_plan_changes_only_sensor_input(base_overrides):
    base = normalize_config(base_overrides)
    plan = sweep.create_plan([base], "ambient", max_trials=3)
    trials = sweep.validate_plan(plan)
    expected = [
        tuple(RAW_SENSOR_COLUMNS),
        (*RAW_SENSOR_COLUMNS, "atmp_t"),
        (*RAW_SENSOR_COLUMNS, "atmp_t", "temp1_minus_ambient", "temp2_minus_ambient"),
    ]
    assert list(AMBIENT_VARIANTS.values()) == expected
    assert [tuple(trial["config"]["sensor_features"]) for trial in trials] == expected
    assert [sensor_variant(names) for names in expected] == ["raw", "ambient", "ambient_delta"]
    for trial, dimension in zip(trials, (5, 6, 8), strict=True):
        candidate = trial["config"]
        assert {k: v for k, v in candidate.items() if k != "sensor_features"} == {
            k: v for k, v in base.items() if k != "sensor_features"
        }
        assert model_config(candidate)["sensor_dim"] == dimension
        assert set(trial["changes"]) <= {"sensor_features"}


def test_ambient_plan_preserves_old_raw_identity_and_honors_budget():
    plan = sweep.create_plan([{}], "ambient", max_trials=3)
    assert plan["trials"][0]["id"] == "embedding_concat_s42_3daac9bb72d0ba69"
    assert plan["trials"][0]["changes"] == {}
    with pytest.raises(ValueError, match="budget"):
        sweep.create_plan([{}], "ambient", max_trials=2)


def test_ambient_differences_are_signed_observed_current_row_values():
    frame = invented_train()
    names = AMBIENT_VARIANTS["ambient_delta"]
    actual = neural_sensor_values(frame, names)
    np.testing.assert_allclose(actual[:, -2:], [[-10.0, 10.0], [30.0, -2.0], [np.nan, 40.0]])
    invalid = frame.copy()
    invalid["atmp_t"] = [np.inf, "not-a-number", np.nan]
    assert np.isnan(neural_sensor_values(invalid, names)[:, -3:]).all()
    with pytest.raises(ValueError, match="atmp_t"):
        neural_sensor_values(frame.drop(columns="atmp_t"), names)
    # The reference condition does not require the optional ambient source.
    assert neural_sensor_values(frame.drop(columns="atmp_t"), AMBIENT_VARIANTS["raw"]).shape == (
        3,
        5,
    )


def test_ambient_preprocessing_uses_frozen_train_statistics_and_featurewise_fallback():
    names = AMBIENT_VARIANTS["ambient_delta"]
    train = invented_train()
    prep = NeuralSensorPreprocessor.fit(train, names)
    assert not prep.uses_history
    raw = neural_sensor_values(train, names)
    medians = np.nanmedian(raw, axis=0)
    filled = np.where(np.isnan(raw), medians, raw)
    scales = filled.std(axis=0)
    scales[scales < 1e-8] = 1.0
    np.testing.assert_allclose(prep.medians, medians)
    np.testing.assert_allclose(prep.means, filled.mean(axis=0))
    np.testing.assert_allclose(prep.scales, scales)
    frozen = prep.to_dict()

    validation = train.iloc[[0, 1]].copy()
    validation["meas_t1"] = [80.0, np.nan]
    validation["meas_t2"] = [1000.0, -5.0]
    validation["atmp_t"] = [np.nan, 0.0]
    # Derive first; missing differences use their own Train medians. They are
    # not computed by subtracting an imputed ambient value from a temperature.
    np.testing.assert_allclose(
        prep.impute(validation)[:, -3:], [[20.0, 10.0, 10.0], [0.0, 10.0, -5.0]]
    )
    transformed = prep.transform(validation)
    np.testing.assert_allclose(
        transformed, ((prep.impute(validation) - filled.mean(axis=0)) / scales).astype(np.float32)
    )
    assert prep.to_dict() == frozen
    restored = NeuralSensorPreprocessor.from_dict(json.loads(json.dumps(frozen)))
    np.testing.assert_array_equal(restored.transform(validation), transformed)
    np.testing.assert_array_equal(restored.transform(validation.iloc[[1, 0]]), transformed[[1, 0]])
    np.testing.assert_array_equal(restored.transform(validation.iloc[[1]]), transformed[[1]])
