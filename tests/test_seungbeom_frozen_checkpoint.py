"""Independent CPU-only tests; no organizer data, weights or network access."""

import argparse
import ast
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_seungbeom_experiment_balance import case  # noqa: F401
from test_seungbeom_resnet_finetune import synthetic_images, tiny_backbone  # noqa: F401

from experiments.seungbeom import frozen_checkpoint as experiment
from experiments.seungbeom import frozen_checkpoint_core as core
from experiments.seungbeom import resnet_finetune as ft
from experiments.seungbeom.export_submissions import read_json, sha256, write_json


def test_scripts_support_server_python310():
    for module in (experiment, core):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))


def test_save_callback_does_not_change_original_selection_trajectory(
    tmp_path,
    tiny_backbone,  # noqa: F811
    synthetic_images,  # noqa: F811
):
    frame, root, _ = synthetic_images
    fit, stop = frame.iloc[:4].copy(), frame.iloc[4:].copy()
    config = ft.normalize_config(
        {"epochs": 3, "patience": 2, "batch_size": 2, "sensor_hidden": 2, "fusion_hidden": 4}
    )
    old, prep, selected, history, initial = ft.train_phase(
        fit, root, config, "frozen", "cpu", epochs=3, stopping=(stop, root)
    )
    saves = []

    def save(model, fitted, epoch, score, digest):
        saves.append((epoch, score))
        core.save_checkpoint(tmp_path / "inner_best.pt", model, fitted, {"epoch": epoch})

    new, new_prep, new_selected, new_history, new_initial = core.train_phase(
        fit,
        root,
        config,
        "cpu",
        epochs=3,
        stopping=(stop, root),
        on_best=save,
        expected_initial=initial,
        expected_backbone=ft.state_digest(old.backbone),
    )
    assert new_initial == initial
    assert ft.state_digest(new) == ft.state_digest(old)
    assert new_prep.to_dict() == prep.to_dict()
    assert new_selected == selected == saves[-1][0]
    assert [{k: v for k, v in r.items() if k != "elapsed_seconds"} for r in history] == [
        {k: v for k, v in r.items() if k != "elapsed_seconds"} for r in new_history
    ]


def test_snapshot_keeps_strict_best_not_tie_or_final_state(
    tmp_path,
    monkeypatch,
    tiny_backbone,  # noqa: F811
    synthetic_images,  # noqa: F811
):
    frame, root, _ = synthetic_images
    config = ft.normalize_config(
        {"epochs": 4, "patience": 3, "batch_size": 2, "sensor_hidden": 2, "fusion_hidden": 4}
    )
    scores = iter([0.5, 0.8, 0.8, 0.6])
    monkeypatch.setattr(core, "classification_metrics", lambda *a: {"macro_f1": next(scores)})
    snapshots = []
    path = tmp_path / "best.pt"

    def save(model, prep, epoch, score, digest):
        snapshots.append((epoch, ft.state_digest(model)))
        core.save_checkpoint(path, model, prep, {"epoch": epoch, "score": score})

    last, _, selected, _, _ = core.train_phase(
        frame.iloc[:4],
        root,
        config,
        "cpu",
        epochs=4,
        stopping=(frame.iloc[4:], root),
        on_best=save,
    )
    assert [epoch for epoch, _ in snapshots] == [1, 2]
    assert selected == 2
    final_digest = ft.state_digest(last)
    saved = torch.load(path, weights_only=True)
    assert saved["epoch"] == 2 and saved["score"] == 0.8
    last.load_state_dict(saved["state_dict"])
    assert ft.state_digest(last) == snapshots[-1][1] != final_digest
    assert not path.with_suffix(".pt.tmp").exists()


def test_inner_checkpoint_survives_later_training_interruption(
    tmp_path,
    tiny_backbone,  # noqa: F811
    synthetic_images,  # noqa: F811
):
    frame, root, _ = synthetic_images
    config = ft.normalize_config({"epochs": 2, "batch_size": 2})
    path = tmp_path / "inner_best.pt"

    def save(model, prep, epoch, score, digest):
        core.save_checkpoint(path, model, prep, {"epoch": epoch})

    def interrupt(row):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        core.train_phase(
            frame.iloc[:4],
            root,
            config,
            "cpu",
            epochs=2,
            stopping=(frame.iloc[4:], root),
            on_best=save,
            progress=interrupt,
        )
    assert torch.load(path, weights_only=True)["epoch"] == 1


def test_two_distinct_checkpoints_fit_train_only_and_replay(case, monkeypatch):  # noqa: F811
    args, frames, calls, original = case
    before = sha256(original)
    fitted, phases = [], []
    original_fit = ft.NeuralSensorPreprocessor.fit
    original_phase = core.train_phase

    def track_fit(frame, *a, **kw):
        fitted.append(set(frame.sample_id))
        return original_fit(frame, *a, **kw)

    def track_phase(frame, *a, **kw):
        phases.append((set(frame.sample_id), kw.get("stopping"), kw.get("on_best")))
        return original_phase(frame, *a, **kw)

    monkeypatch.setattr(ft.NeuralSensorPreprocessor, "fit", track_fit)
    monkeypatch.setattr(core, "train_phase", track_phase)
    result = experiment.run(args)
    manifest, restored = experiment.read_result(args.output_dir)
    assert restored == result
    assert result["checkpoint_replay_matches"] == {"inner_best": True, "refit": True}
    inner_ids = set(frames["train"].iloc[manifest["inner_split"]["fit"]].sample_id)
    stop_ids = set(frames["train"].iloc[manifest["inner_split"]["stop"]].sample_id)
    assert fitted == [inner_ids, set(frames["train"].sample_id)]
    assert phases[0][0] == inner_ids
    assert set(phases[0][1][0].sample_id) == stop_ids
    assert phases[0][2] is not None and phases[1][1:] == (None, None)
    assert not set(frames["validation"].sample_id) & set.union(*fitted)
    inner = torch.load(args.output_dir / "inner_best.pt", weights_only=True)
    refit = torch.load(args.output_dir / "refit.pt", weights_only=True)
    assert inner["fit_partition"] == "train_inner_fit"
    assert refit["fit_partition"] == "all_train"
    assert inner["selected_epoch"] == refit["selected_epoch"] == result["selected_epoch"]
    assert inner["selection_score"] == result["inner_best_macro_f1"]
    # Identical distributions can yield identical statistics despite independent fits.
    assert (
        inner["preprocessor"]
        == original_fit(
            frames["train"].iloc[manifest["inner_split"]["fit"]], ft.RAW_SENSOR_COLUMNS
        ).to_dict()
    )
    assert refit["preprocessor"] == original_fit(frames["train"], ft.RAW_SENSOR_COLUMNS).to_dict()
    assert sha256(original) == before
    assert calls == [frames["validation"].sample_id.tolist()]
    assert set(result["metrics"]) == set(experiment.METHODS)
    experiment.report(argparse.Namespace(output_dir=args.output_dir, class_details=True))


def test_all_four_exports_preserve_subset_predictions_without_training_csvs(case, tmp_path):  # noqa: F811
    args, frames, _, _ = case
    experiment.run(args)
    args.recipe.unlink()
    for split in frames:
        (args.data_dir / split / f"{split}.csv").unlink()
    for method in experiment.CANDIDATES:
        predictions = None
        for name, indexes in (("all", list(range(12))), ("subset", [9, 1, 7])):
            frame = frames["validation"].iloc[indexes].copy()
            frame["label"] = "not an input"
            frame["experiment_id"] = "not an input"
            input_path = tmp_path / f"invented_{method}_{name}.csv"
            frame.to_csv(input_path, index=False)
            output = tmp_path / f"predictions_{method}_{name}.csv"
            template = tmp_path / f"template_{method}_{name}.csv"
            pd.DataFrame({"sample_id": frame.sample_id.iloc[::-1], "label": 1}).to_csv(
                template, index=False
            )
            predict_args = argparse.Namespace(
                run_dir=args.output_dir,
                input=input_path,
                image_root=args.data_dir / "validation",
                output=output,
                device="cpu",
                sample_submission=template,
                method=method,
            )
            experiment.predict(predict_args)
            values = pd.read_csv(output, dtype={"sample_id": str}).set_index("sample_id").label
            assert values.index.tolist() == frame.sample_id.iloc[::-1].tolist()
            assert values.between(1, 6).all()
            sidecar = read_json(output.with_suffix(".manifest.json"))
            assert sidecar["identity"]["stage"] == experiment.CANDIDATES[method]
            if predictions is None:
                predictions = values
            else:
                pd.testing.assert_series_equal(values, predictions.loc[values.index])
            with pytest.raises(ValueError, match="Output exists"):
                experiment.predict(predict_args)


@pytest.mark.parametrize("problem", ["csv", "output", "replay", "initial"])
def test_guards_stop_before_training(case, monkeypatch, problem):  # noqa: F811
    args, _, _, _ = case
    if problem == "csv":
        (args.data_dir / "train/train.csv").write_text("changed")
    elif problem == "output":
        args.output_dir.mkdir()
    elif problem == "replay":
        recipe = read_json(args.recipe)
        recipe["metrics"]["frozen"]["macro_f1"] = -1
        write_json(args.recipe, recipe)
    else:
        monkeypatch.setattr(ft, "check_trial", lambda *a: {"initial_state_sha256": None})

    def forbidden(*a, **kw):
        pytest.fail("Guard must stop before fitting")

    monkeypatch.setattr(core, "train_phase", forbidden)
    with pytest.raises(ValueError):
        experiment.run(args)


@pytest.mark.parametrize("name", ["inner_best.pt", "refit.pt"])
def test_checkpoint_corruption_blocks_reading_and_stage_swap(case, name):  # noqa: F811
    args, _, _, _ = case
    result = experiment.run(args)
    with pytest.raises(ValueError, match="identity"):
        experiment.restore(args.output_dir / "inner_best.pt", result["identities"]["refit"], "cpu")
    path = args.output_dir / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        experiment.read_result(args.output_dir)
