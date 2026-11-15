"""Independent synthetic CPU checks for Train-only experiment-weighted learning."""

import argparse

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from test_seungbeom_resnet_finetune import synthetic_images, tiny_backbone  # noqa: F401

from experiments.seungbeom import experiment_balance as experiment
from experiments.seungbeom import experiment_balance_core as core
from experiments.seungbeom import resnet_finetune as ft
from experiments.seungbeom.export_submissions import read_json, sha256, write_json
from schema import RAW_SENSOR_COLUMNS


def test_weights_equalize_experiment_totals_without_using_labels_or_holdout_rows():
    frame = pd.DataFrame({"experiment_id": ["A"] * 8 + ["B"] * 2, "label": [1] * 10})
    weights = core.row_weights(frame)
    np.testing.assert_allclose(weights[:8], 0.625)
    np.testing.assert_allclose(weights[8:], 2.5)
    assert weights.mean() == 1
    assert weights[:8].sum() == weights[8:].sum() == 5
    np.testing.assert_array_equal(weights, core.row_weights(frame.assign(label=6)))
    # Each phase receives only its fit rows; no full-Train group size lookup.
    inner = frame.iloc[[0, 1, 8, 9]]
    np.testing.assert_array_equal(core.row_weights(inner), np.ones(4))
    order = [9, 0, 8, 1, 2, 3, 4, 5, 6, 7]
    np.testing.assert_array_equal(core.row_weights(frame.iloc[order]), weights[order])


@pytest.mark.parametrize("ids", [[], [None], [" "], [1]])
def test_invalid_training_groups_rejected(ids):
    with pytest.raises(ValueError):
        core.row_weights(pd.DataFrame({"experiment_id": ids}))


def test_weighted_loss_preserves_weight_for_singleton_batch_and_correct_gradient():
    logits = torch.tensor([[0.2, 0.8, -0.4]], requires_grad=True)
    target = torch.tensor([1])
    plain = torch.nn.functional.cross_entropy(logits, target)
    original_grad = torch.autograd.grad(plain, logits, retain_graph=True)[0]
    weighted = core.objective(logits, target, torch.tensor([2.5]))
    torch.testing.assert_close(weighted, 2.5 * plain)
    torch.testing.assert_close(torch.autograd.grad(weighted, logits)[0], original_grad * 2.5)
    with pytest.raises(ValueError):
        core.objective(logits, target, torch.tensor([0.0]))


def test_uniform_path_exactly_matches_original_training_and_frozen_backbone(
    tiny_backbone,  # noqa: F811
    synthetic_images,  # noqa: F811
):
    frame, root, _ = synthetic_images
    config = ft.normalize_config(
        {"epochs": 2, "batch_size": 2, "sensor_hidden": 2, "fusion_hidden": 4, "dropout": 0.1}
    )
    original, prep, _, _, initial = ft.train_phase(frame, root, config, "frozen", "cpu", epochs=2)
    same, same_prep, _, _, same_initial = core.train_phase(
        frame,
        root,
        config,
        "cpu",
        epochs=2,
        balanced=False,
        expected_initial=initial,
        expected_backbone=ft.state_digest(original.backbone),
    )
    assert same_initial == initial
    assert ft.state_digest(same) == ft.state_digest(original)
    assert same_prep.to_dict() == prep.to_dict()
    assert ft.state_digest(same.backbone) == ft.state_digest(original.backbone)
    with pytest.raises(ValueError, match="Initialization/backbone"):
        core.train_phase(frame, root, config, "cpu", epochs=1, expected_initial="wrong")


def test_pair_errors_report_rates_reverse_errors_and_absent_classes():
    frame = pd.DataFrame(
        {"experiment_id": ["A"] * 4 + ["B"] * 4, "label": [3, 3, 5, 2, 3, 5, 5, 4]}
    )
    before = np.array([2, 3, 4, 3, 2, 4, 3, 5])
    after = np.array([3, 3, 5, 3, 2, 5, 3, 5])
    summary, groups = core.pair_errors(frame, before)
    assert summary["3->2"]["rate"] == 2 / 3
    assert summary["3->2"]["groups_with_errors"] == 2
    assert summary["4->5"]["groups_with_support"] == 1
    assert pd.isna(groups[(groups.experiment_id == "A") & (groups.pair == "4->5")].iloc[0].rate)
    _, new = core.pair_errors(frame, after)
    change = core.pair_changes(groups, new)
    assert change["3->2"] == {
        "groups": 2,
        "improved": 1,
        "tied": 1,
        "worse": 0,
        "error_count_delta": -1,
    }
    assert change["5->4"]["improved"] == 2
    with pytest.raises(ValueError, match="identical"):
        core.pair_changes(groups, new.iloc[:-1])


@pytest.fixture
def case(tmp_path, monkeypatch, tiny_backbone):  # noqa: F811
    config = ft.normalize_config(
        {
            "epochs": 2,
            "patience": 1,
            "batch_size": 8,
            "sensor_hidden": 2,
            "fusion_hidden": 4,
            "dropout": 0.1,
        }
    )
    data = tmp_path / "invented_data"
    frames = {}
    for split, sizes in (("train", [6, 6, 6, 12, 12]), ("validation", [6, 6])):
        root = data / split
        root.mkdir(parents=True)
        groups = [f"{split}_group_{g}" for g, size in enumerate(sizes) for _ in range(size)]
        n = len(groups)
        frame = pd.DataFrame(
            {
                "sample_id": [f"{split}_{i:04}" for i in range(n)],
                "experiment_id": groups,
                "image_path": [f"invented_{i}.png" for i in range(n)],
                "label": np.tile(np.arange(1, 7), n // 6),
                **{
                    name: (np.arange(n) % 6 + j).astype(float)
                    for j, name in enumerate(RAW_SENSOR_COLUMNS)
                },
            }
        )
        for i, name in enumerate(frame.image_path):
            Image.fromarray(np.full((8, 12, 3), 40 + i % 6 * 20, dtype=np.uint8)).save(root / name)
        frame.to_csv(root / f"{split}.csv", index=False)
        frames[split] = frame
    original, prep, _, _, initial = ft.train_phase(
        frames["train"], data / "train", config, "frozen", "cpu", epochs=1
    )
    reference_file = tmp_path / "original_model_marker"
    reference_file.write_text("independent synthetic original; do not overwrite")
    record = {
        "run_dir": str(tmp_path / "original_frozen"),
        "config": config,
        "selected_epoch": 1,
        "files": {str(reference_file): sha256(reference_file)},
    }
    base = {
        "provenance": {
            "files": record["files"],
            **{f"{split}_csv_sha256": sha256(data / split / f"{split}.csv") for split in frames},
        },
        "runtime_signature": experiment.runtime_signature("cpu"),
        "image_precision": "fp32",
    }
    saved = {"model": original, "preprocessor": prep, "checkpoint": {"config": config}}
    calls = []

    def tree_probability(current):
        targets = current["meas_v"].to_numpy(dtype=int) % 6
        p = np.full((len(current), 6), 0.1, dtype=np.float32)
        p[np.arange(len(current)), targets] = 0.5
        return p

    def old_infer(current, image_root, legacy, frozen, old_base, device):
        assert "label" not in current
        calls.append(current["sample_id"].tolist())
        p = experiment.probability(original, prep, current, image_root, config, device)
        tree = tree_probability(current)
        mix = experiment.blend.midpoint.blend_probabilities(p, tree)
        return {
            "frozen": p,
            "extra_trees": tree,
            "frozen_tree_50_50": mix,
            "historical_neural": p,
            "baseline_50_50": mix,
        }

    outputs = old_infer(
        frames["validation"].drop(columns="label"), data / "validation", {}, saved, base, "cpu"
    )
    reference = {
        "frozen": record,
        "base_recipe": base,
        "source_sha256": experiment.blend.source_hashes(),
        "metrics": {
            name: experiment.classification_metrics(
                frames["validation"]["label"], value.argmax(1) + 1
            )
            for name, value in outputs.items()
        },
    }
    recipe_dir = tmp_path / "original_recipe"
    recipe_dir.mkdir()
    write_json(recipe_dir / "recipe.json", reference)
    write_json(recipe_dir / "status.json", {"status": "complete"})
    monkeypatch.setattr(experiment.blend, "read_comparison", lambda path: read_json(path))
    monkeypatch.setattr(experiment.blend, "load_members", lambda *a: ({}, saved, record))
    monkeypatch.setattr(ft, "check_trial", lambda *a: {"initial_state_sha256": initial})
    monkeypatch.setattr(experiment.blend, "infer_members", old_infer)
    monkeypatch.setattr(experiment.blend.ratios, "load_base", lambda *a: {})
    monkeypatch.setattr(
        experiment.blend.midpoint,
        "method_probabilities",
        lambda current, *a, **kw: {"extra_trees": tree_probability(current)},
    )
    args = argparse.Namespace(
        recipe=recipe_dir / "recipe.json",
        data_dir=data,
        output_dir=tmp_path / "new_balanced",
        device="cpu",
    )
    calls.clear()
    return args, frames, calls, reference_file


def test_full_train_flow_weights_and_preprocessing_never_fit_validation(case, monkeypatch):
    args, frames, calls, original = case
    before = sha256(original)
    fit_calls, weight_calls = [], []
    original_fit = ft.NeuralSensorPreprocessor.fit
    original_weights = core.row_weights

    def track_fit(frame, *a, **kw):
        fit_calls.append(set(frame.sample_id))
        return original_fit(frame, *a, **kw)

    def track_weights(frame):
        weight_calls.append(set(frame.sample_id))
        return original_weights(frame)

    monkeypatch.setattr(ft.NeuralSensorPreprocessor, "fit", track_fit)
    monkeypatch.setattr(core, "row_weights", track_weights)
    result = experiment.run(args)
    manifest, saved = experiment.read_result(args.output_dir)
    assert result == saved and result["checkpoint_replay_matches"]
    expected = [
        set(frames["train"].iloc[manifest["inner_split"]["fit"]].sample_id),
        set(frames["train"].sample_id),
    ]
    assert fit_calls == weight_calls == expected
    assert not (set(frames["validation"].sample_id) & set.union(*fit_calls))
    assert calls == [frames["validation"].sample_id.tolist()]
    assert sha256(original) == before
    table = pd.read_csv(args.output_dir / "group_pair_errors.csv")
    assert set(table.method) == set(experiment.METHODS)
    assert not {"sample_id", "image_path", "label"} & set(table.columns)
    assert result["selected_epoch"] <= manifest["config"]["epochs"]
    experiment.report(argparse.Namespace(output_dir=args.output_dir, class_details=True))


def test_saved_candidate_predicts_subsets_after_removing_train_validation_csvs(case, tmp_path):
    args, frames, _, _ = case
    experiment.run(args)
    args.recipe.unlink()
    for split in frames:
        (args.data_dir / split / f"{split}.csv").unlink()
    predictions = {}
    for method in ("balanced_neural", "balanced_50_50"):
        for name, indexes in (("all", list(range(12))), ("subset", [9, 1, 7])):
            current = frames["validation"].iloc[indexes].copy()
            current["label"] = "not an input"
            current["experiment_id"] = "unknown_group"
            input_path = tmp_path / f"invented_{method}_{name}.csv"
            current.to_csv(input_path, index=False)
            output = tmp_path / f"predictions_{method}_{name}.csv"
            template = tmp_path / f"template_{method}_{name}.csv"
            pd.DataFrame({"sample_id": current.sample_id.iloc[::-1], "label": 1}).to_csv(
                template, index=False
            )
            experiment.predict(
                argparse.Namespace(
                    run_dir=args.output_dir,
                    input=input_path,
                    image_root=args.data_dir / "validation",
                    output=output,
                    device="cpu",
                    sample_submission=template,
                    method=method,
                )
            )
            values = pd.read_csv(output, dtype={"sample_id": str}).set_index("sample_id").label
            assert values.index.tolist() == current.sample_id.iloc[::-1].tolist()
            assert values.between(1, 6).all()
            if name == "all":
                predictions[method] = values
            else:
                pd.testing.assert_series_equal(values, predictions[method].loc[values.index])


@pytest.mark.parametrize("problem", ["csv", "output", "replay", "initial"])
def test_guards_stop_before_unsafe_or_unmatched_training(case, monkeypatch, problem):
    args, _, _, _ = case
    if problem == "csv":
        (args.data_dir / "train/train.csv").write_text("changed input")
    elif problem == "output":
        args.output_dir.mkdir()
    elif problem == "replay":
        recipe = read_json(args.recipe)
        recipe["metrics"]["frozen"]["macro_f1"] = -1
        write_json(args.recipe, recipe)
    else:
        monkeypatch.setattr(ft, "check_trial", lambda *a: {"initial_state_sha256": None})

    def forbidden(*a, **kw):
        pytest.fail("Guard must stop before fitting a candidate")

    monkeypatch.setattr(core, "train_phase", forbidden)
    with pytest.raises(ValueError):
        experiment.run(args)


def test_artifact_corruption_blocks_report_and_export(case):
    args, _, _, _ = case
    experiment.run(args)
    path = args.output_dir / "model.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        experiment.read_result(args.output_dir)
