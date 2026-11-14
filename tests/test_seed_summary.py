"""Seed summaries use only independently invented metadata in temporary directories."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.summarize_seeds import (
    format_rows,
    group_runs,
    load_runs,
    paired_rows,
    summary_rows,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def make_run(tmp_path):
    base = json.loads((ROOT / "configs" / "sensor_raw.json").read_text(encoding="utf-8"))

    def create(name, seed=42, score=0.5, config_updates=None, split_updates=None, explicit=False):
        directory = tmp_path / name
        directory.mkdir()
        config = {**base, "seed": seed, **(config_updates or {})}
        if explicit:
            config = {
                "sensor_imputation": "median",
                "sensor_scaling": "standard",
                "loss": "cross_entropy",
                "sampling": "shuffle",
                "class_weight_power": 0.5,
                "focal_gamma": 2.0,
                **config,
            }
        files = {
            "config.json": config,
            "split.json": {
                "train_csv_sha256": "invented_train_hash",
                "validation_csv_sha256": "invented_validation_hash",
                **(split_updates or {}),
            },
            "best_metrics.json": {
                "epoch": 1,
                "macro_f1": score,
                "class_f1": {str(label): score / label for label in range(1, 7)},
            },
        }
        for filename, contents in files.items():
            (directory / filename).write_text(json.dumps(contents), encoding="utf-8")
        return directory

    return create


def test_full_configuration_groups_same_size_features_separately(make_run):
    raw = ["meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p"]
    voltage = raw + ["meas_v_mean30s", "meas_v_delta30s"]
    pressure = raw + ["meas_p_mean30s", "meas_p_delta30s"]
    paths = [
        make_run("voltage_seed42", config_updates={"sensor_features": voltage}),
        make_run("unrelated_name_seed43", seed=43, config_updates={"sensor_features": voltage}),
        make_run("pressure_seed42", config_updates={"sensor_features": pressure}),
    ]
    groups = group_runs(load_runs(paths))
    assert [group.variant for group in groups] == ["voltage", "pressure"]
    assert [sorted(group.runs) for group in groups] == [[42, 43], [42]]
    assert [row["features"] for row in summary_rows(groups)] == [7, 7]


@pytest.mark.parametrize(
    "change",
    [
        {"learning_rate": 0.002},
        {"epochs": 11},
        {"sensor_hidden": 64},
        {"sensor_scaling": "robust"},
        {"loss": "focal"},
        {"sampling": "balanced"},
        {"class_weight_power": 0.75},
        {"focal_gamma": 1.5},
    ],
)
def test_hyperparameters_separate_groups_and_disambiguate_names(make_run, change):
    groups = group_runs(
        load_runs(
            [
                make_run("variant_seed42"),
                make_run("variant_seed43", seed=43, config_updates=change),
            ]
        )
    )
    assert [group.variant for group in groups] == ["variant", "variant#2"]


def test_legacy_defaults_group_with_explicit_defaults(make_run):
    groups = group_runs(
        load_runs([make_run("old_seed42"), make_run("new_seed43", seed=43, explicit=True)])
    )
    assert len(groups) == 1
    row = summary_rows(groups)[0]
    assert row["seeds"] == (42, 43)
    assert row["loss"] == "cross_entropy"
    assert row["sampling"] == "shuffle"


def test_duplicate_seed_rejected_even_when_directory_names_differ(make_run):
    with pytest.raises(ValueError, match="Duplicate seed 42"):
        group_runs(load_runs([make_run("first"), make_run("second")]))


@pytest.mark.parametrize("key", ["train_csv_sha256", "validation_csv_sha256"])
def test_every_group_must_use_same_train_and_validation_hashes(make_run, key):
    paths = [
        make_run("reference"),
        make_run("candidate", config_updates={"epochs": 11}, split_updates={key: "different_hash"}),
    ]
    with pytest.raises(ValueError, match="splits differ"):
        group_runs(load_runs(paths))


@pytest.mark.parametrize("required", [[42, 43, 44], [42], [42, 44], [42, 43, 43], []])
def test_require_seeds_rejects_incomplete_extra_or_nonunique_sets(make_run, required):
    runs = load_runs([make_run("a_seed42"), make_run("a_seed43", seed=43)])
    with pytest.raises(ValueError, match="[Rr]equired seeds"):
        group_runs(runs, required)


def test_require_seeds_applies_to_every_group(make_run):
    paths = [
        make_run("a_seed42"),
        make_run("a_seed43", seed=43),
        make_run("b_seed42", config_updates={"epochs": 11}),
    ]
    with pytest.raises(ValueError, match="Variant b has seeds"):
        group_runs(load_runs(paths), [42, 43])


def test_single_seed_std_is_missing_and_displayed_na(make_run):
    groups = group_runs(load_runs([make_run("one_seed42", score=0.3333333333)]))
    row = summary_rows(groups)[0]
    assert row["macro_f1_mean"] == 0.3333333333
    assert row["macro_f1_std"] is None
    assert row["f1_6_std"] is None
    display = format_rows([row])[0]
    assert display["macro_f1_mean"] == "0.333333"
    assert display["macro_f1_std"] == "NA"
    assert row["macro_f1_mean"] == 0.3333333333


def test_mean_and_sample_std_use_full_precision(make_run):
    scores = [0.10000004, 0.20000005, 0.30000009]
    groups = group_runs(
        load_runs(
            [
                make_run(f"variant_seed{seed}", seed=seed, score=score)
                for seed, score in zip([42, 43, 44], scores, strict=True)
            ]
        ),
        [44, 42, 43],
    )
    row = summary_rows(groups)[0]
    mean = sum(scores) / 3
    std = math.sqrt(sum((score - mean) ** 2 for score in scores) / 2)
    assert row["seeds"] == (42, 43, 44)
    assert row["n"] == 3
    assert row["macro_f1_mean"] == pytest.approx(mean, abs=1e-15)
    assert row["macro_f1_std"] == pytest.approx(std, abs=1e-15)
    for label in range(1, 7):
        assert row[f"f1_{label}_mean"] == pytest.approx(mean / label, abs=1e-15)
        assert row[f"f1_{label}_std"] == pytest.approx(std / label, abs=1e-15)


def test_paired_deltas_match_seeds_and_first_input_group(make_run):
    paths = [
        make_run("reference_seed43", seed=43, score=0.5),
        make_run("candidate_seed44", seed=44, score=0.6, config_updates={"epochs": 11}),
        make_run("reference_seed42", seed=42, score=0.2),
        make_run("candidate_seed42", seed=42, score=0.3, config_updates={"epochs": 11}),
        make_run("reference_seed44", seed=44, score=0.8),
        make_run("candidate_seed43", seed=43, score=0.5, config_updates={"epochs": 11}),
    ]
    groups = group_runs(load_runs(paths), [42, 43, 44])
    row = paired_rows(groups)[0]
    assert row["reference"] == "reference"
    assert row["candidate"] == "candidate"
    assert row["seed_deltas"] == pytest.approx({42: 0.1, 43: 0.0, 44: -0.2})
    assert row["macro_f1_delta_mean"] == pytest.approx(-0.1 / 3)
    assert row["macro_f1_delta_std"] == pytest.approx(math.sqrt(7 / 300))
    assert (row["wins"], row["ties"], row["losses"]) == (1, 1, 1)
    assert format_rows([row])[0]["seed_deltas"] == "42:+0.100000,43:+0.000000,44:-0.200000"


def test_paired_counts_use_unrounded_deltas(make_run):
    paths = [
        make_run("reference_seed42", score=0.5),
        make_run("candidate_seed42", score=0.500000001, config_updates={"epochs": 11}),
    ]
    row = paired_rows(group_runs(load_runs(paths)))[0]
    assert row["wins"] == 1 and row["ties"] == 0
    assert row["macro_f1_delta_std"] is None
    assert format_rows([row])[0]["macro_f1_delta_std"] == "NA"


def test_paired_rejects_unequal_seed_sets(make_run):
    paths = [
        make_run("reference_seed42"),
        make_run("candidate_seed43", seed=43, config_updates={"epochs": 11}),
    ]
    with pytest.raises(ValueError, match="identical seed sets"):
        paired_rows(group_runs(load_runs(paths)))


def test_direct_cli_import_and_paired_output(make_run, tmp_path):
    paths = [
        make_run("reference_seed42", score=0.4),
        make_run("candidate_seed42", score=0.6, config_updates={"epochs": 11}),
    ]
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_seeds.py"),
            *(str(path) for path in paths),
            "--require_seeds",
            "42",
            "--paired",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "macro_f1_mean\tmacro_f1_std" in result.stdout
    assert "0.400000\tNA" in result.stdout
    assert "macro_f1_delta_mean\tmacro_f1_delta_std" in result.stdout
    assert "0.200000\tNA" in result.stdout
    assert "not a confidence interval or significance test" in result.stderr


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_metric_values_fail_before_summary(make_run, score):
    with pytest.raises(ValueError, match="finite score"):
        load_runs([make_run("invalid", score=score)])
