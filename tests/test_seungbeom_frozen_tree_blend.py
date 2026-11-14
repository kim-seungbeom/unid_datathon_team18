"""Independent synthetic CPU tests; never read competition inputs or trained artifacts."""

import argparse
import copy
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401
from test_seungbeom_seed_blend import synthetic_case  # noqa: F401

from experiments.seungbeom import frozen_tree_blend as blend
from experiments.seungbeom import resnet_finetune as finetune


def dump(path, value):
    blend.write_json(path, value)


def read(path):
    return blend.read_json(path)


def refresh_trial(directory):
    names = ("model.pt", "result.json", "selection_history.json", "refit_history.json")
    dump(
        directory / "status.json",
        {"status": "complete", "files": {name: blend.sha256(directory / name) for name in names}},
    )


@pytest.fixture
def case(synthetic_case, tiny_backbone, tmp_path, monkeypatch):  # noqa: F811
    import torch

    state = synthetic_case
    frame = state["frame"]
    root = state["args"].image_root
    for index, name in enumerate(frame["image_path"]):
        pixels = np.zeros((16, 24, 3), dtype=np.uint8)
        pixels[..., index % 3] = 50 + 20 * index
        pixels[:8, 4:14, (index + 1) % 3] = 225
        Image.fromarray(pixels).save(root / name)
    config = finetune.normalize_config({"sensor_hidden": 2, "fusion_hidden": 4, "dropout": 0.0})
    finetune.seed_everything(config["seed"])
    model = finetune.FineTuneModel("frozen", **finetune.model_kwargs(config), pretrained=False)
    prep = finetune.NeuralSensorPreprocessor.fit(frame, finetune.RAW_SENSOR_COLUMNS)
    probability = finetune.probabilities(
        model, finetune.evaluation_loader(frame, root, prep, config), "cpu"
    )
    base = read(state["args"].recipe)
    frozen_run = tmp_path / "independent_frozen_trial"
    frozen_run.mkdir()
    checkpoint = {
        "kind": finetune.CHECKPOINT_KIND,
        "format_version": 1,
        "config": config,
        "model_config": model.get_model_config(),
        "mode": "frozen",
        "selected_epoch": 18,
        "label_values": list(range(1, 7)),
        "preprocessor": prep.to_dict(),
        "image_contract": finetune.IMAGE_CONTRACT,
        "state_dict": model.state_dict(),
        "source_sha256": finetune.source_hashes(),
        "runtime": finetune.runtime_signature("cpu"),
        **{
            name: base["provenance"][name] for name in ("train_csv_sha256", "validation_csv_sha256")
        },
    }
    torch.save(checkpoint, frozen_run / "model.pt")
    result = {
        key: value
        for key, value in checkpoint.items()
        if key
        in (
            "mode",
            "config",
            "source_sha256",
            "runtime",
            "selected_epoch",
            "train_csv_sha256",
            "validation_csv_sha256",
        )
    }
    result["metrics"] = blend.classification_metrics(frame["label"], probability.argmax(1) + 1)
    result["checkpoint_replay_matches"] = True
    dump(frozen_run / "result.json", result)
    for name in ("selection_history.json", "refit_history.json"):
        dump(frozen_run / name, {"epochs": []})
    refresh_trial(frozen_run)
    args = argparse.Namespace(
        baseline_recipe=state["args"].recipe,
        frozen_run=frozen_run,
        input=state["args"].input,
        image_root=root,
        output_dir=tmp_path / "new_comparison",
        device="cpu",
    )
    frozen_policies = []
    original = finetune.probabilities

    def checked_probabilities(model, loader, device, **kwargs):
        frozen_policies.append(
            (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        )
        assert "label" not in loader.dataset.frame
        return original(model, loader, device, **kwargs)

    monkeypatch.setattr(finetune, "probabilities", checked_probabilities)
    return {
        **state,
        "compare_args": args,
        "frozen_probability": probability,
        "frozen_policies": frozen_policies,
    }


def prediction_args(case, tmp_path, rows=None, suffix="one", template=True):
    rows = case["frame"] if rows is None else rows
    path = tmp_path / f"invented_input_{suffix}.csv"
    # Even a supplied label column is never used for inference.
    rows.assign(label="not_an_input_feature").to_csv(path, index=False)
    form = tmp_path / f"invented_template_{suffix}.csv"
    pd.DataFrame({"sample_id": rows["sample_id"].iloc[::-1], "label": 1}).to_csv(form, index=False)
    return argparse.Namespace(
        recipe=case["compare_args"].output_dir / "recipe.json",
        input=path,
        image_root=case["compare_args"].image_root,
        sample_submission=form if template else None,
        output=tmp_path / f"new_prediction_{suffix}.csv",
        device="cpu",
    )


def forbid_fit(monkeypatch):
    from sklearn.pipeline import Pipeline

    def fail(*args, **kwargs):
        pytest.fail("Inference/comparison must never fit a model or preprocessing")

    monkeypatch.setattr(Pipeline, "fit", fail)
    monkeypatch.setattr(finetune.NeuralSensorPreprocessor, "fit", fail)


def test_compare_preserves_saved_models_replays_and_records_equal_probability_rule(
    case, monkeypatch, capsys
):
    forbid_fit(monkeypatch)
    args = case["compare_args"]
    frozen_files = list(read(args.frozen_run / "status.json")["files"])
    paths = [args.baseline_recipe, args.frozen_run / "status.json"]
    paths += [args.frozen_run / name for name in frozen_files]
    before = {str(p): blend.sha256(p) for p in paths}
    result = blend.compare(args)
    assert result["rule"] == {"method": "probability_mean", "frozen": 0.5, "extra_trees": 0.5}
    assert result["replay_matches"] == dict.fromkeys(blend.REPLAY_NAMES, True)
    expected = (case["frozen_probability"] + case["tree_prob"]) * 0.5
    assert result["metrics"]["frozen_tree_50_50"] == blend.classification_metrics(
        case["frame"]["label"], expected.argmax(1) + 1
    )
    assert before == {str(p): blend.sha256(p) for p in paths}
    assert case["calls"] == [case["frame"]["sample_id"].tolist()]
    assert case["frozen_policies"] == [(False, False)]
    assert blend.read_comparison(args.output_dir / "recipe.json") == result
    blend.report(argparse.Namespace(recipe=args.output_dir / "recipe.json", class_details=True))
    output = capsys.readouterr().out
    assert "delta_vs_frozen" in output and "precision" in output
    assert "No training or Test access" in output


def test_predict_after_removing_training_validation_caches_uses_current_rows_only(
    case, tmp_path, monkeypatch
):
    forbid_fit(monkeypatch)
    args = case["compare_args"]
    blend.compare(args)
    args.input.unlink()
    (args.input.parent.parent / "train/train.csv").unlink()
    args.baseline_recipe.unlink()
    for folder in case["cache_dirs"].values():
        shutil.rmtree(folder)
    case["calls"].clear()
    expected = ((case["frozen_probability"] + case["tree_prob"]) * 0.5).argmax(1) + 1
    expected = pd.Series(expected, index=case["frame"]["sample_id"])
    for index, rows in enumerate((case["frame"], case["frame"].iloc[[5, 0, 2]])):
        prediction = prediction_args(case, tmp_path, rows, str(index), template=index == 0)
        blend.predict(prediction)
        output = pd.read_csv(prediction.output, dtype={"sample_id": str}, keep_default_na=False)
        order = rows["sample_id"].iloc[::-1].tolist() if index == 0 else rows["sample_id"].tolist()
        assert list(output) == ["sample_id", "label"]
        assert output["sample_id"].tolist() == order
        assert output["label"].tolist() == expected.loc[order].tolist()
        assert pd.api.types.is_integer_dtype(output["label"])
        assert output["label"].between(1, 6).all()
        assert read(prediction.output.with_suffix(".manifest.json"))["rule"] == blend.RULE
    assert case["calls"] == [
        case["frame"]["sample_id"].tolist(),
        case["frame"].iloc[[5, 0, 2]]["sample_id"].tolist(),
    ]


def test_precision_is_kept_separate_for_new_frozen_and_old_tree(case, tmp_path):
    args = case["compare_args"]
    base = read(args.baseline_recipe)
    base["image_precision"] = "cudnn_tf32"
    base["precision_audit"] = {
        "image_precision": "cudnn_tf32",
        "reference_files": {"never_open_this_cache": "synthetic"},
    }
    dump(args.baseline_recipe, base)
    blend.compare(args)
    blend.predict(prediction_args(case, tmp_path))
    assert case["frozen_policies"] == [(False, False), (False, False)]
    assert case["policies"] == [(True, False), (True, False)]


@pytest.mark.parametrize("name", ["frozen", "historical_neural", "extra_trees", "baseline_50_50"])
def test_full_aggregate_replay_including_confusion_counts_is_required(case, name):
    args = case["compare_args"]
    if name == "frozen":
        result = read(args.frozen_run / "result.json")
        result["metrics"]["confusion_matrix"][0][0] += 1
        dump(args.frozen_run / "result.json", result)
        refresh_trial(args.frozen_run)
    else:
        base = read(args.baseline_recipe)
        old = next(key for key, new in blend.OLD_NAMES.items() if new == name)
        base["metrics"][old]["confusion_matrix"][0][0] += 1
        dump(args.baseline_recipe, base)
    with pytest.raises(ValueError, match="did not replay"):
        blend.compare(args)
    assert read(args.output_dir / "status.json")["status"] == "failed"
    assert not (args.output_dir / "recipe.json").exists()


@pytest.mark.parametrize("problem", ["layer4", "split", "different_validation", "train", "output"])
def test_wrong_member_split_and_output_are_rejected_before_image_inference(case, problem):
    args = case["compare_args"]
    if problem in ("layer4", "split"):
        result = read(args.frozen_run / "result.json")
        result["mode" if problem == "layer4" else "train_csv_sha256"] = "different"
        dump(args.frozen_run / "result.json", result)
        refresh_trial(args.frozen_run)
    elif problem == "different_validation":
        args.input.write_text(args.input.read_text() + "\n")
    elif problem == "train":
        args.input = args.input.parent.parent / "train/train.csv"
    else:
        args.output_dir.mkdir()
    with pytest.raises(ValueError):
        blend.compare(args)
    assert not case["frozen_policies"] and not case["calls"]


@pytest.mark.parametrize(
    "problem",
    [
        "frozen_model",
        "old_model",
        "rule",
        "source",
        "runtime",
        "status",
        "template",
        "output",
        "sidecar",
    ],
)
def test_prediction_guards_before_reading_images(case, tmp_path, problem):
    args = case["compare_args"]
    blend.compare(args)
    case["calls"].clear()
    case["frozen_policies"].clear()
    prediction = prediction_args(case, tmp_path)
    recipe = read(prediction.recipe)
    if problem == "frozen_model":
        (args.frozen_run / "model.pt").write_bytes(b"changed independent checkpoint")
    elif problem == "old_model":
        path = next(iter(recipe["base_recipe"]["provenance"]["files"]))
        from pathlib import Path

        Path(path).write_bytes(b"changed independent historical artifact")
    elif problem in ("rule", "source", "runtime"):
        if problem == "rule":
            recipe["rule"]["frozen"] = 0.6
        elif problem == "source":
            recipe["source_sha256"][blend.SELF] = "changed"
        else:
            recipe["base_recipe"]["runtime_signature"]["packages"]["torch"] = "changed"
        dump(prediction.recipe, recipe)
        dump(
            prediction.recipe.parent / "status.json",
            {"status": "complete", "recipe_sha256": blend.sha256(prediction.recipe)},
        )
    elif problem == "status":
        dump(prediction.recipe.parent / "status.json", {"status": "failed"})
    elif problem == "template":
        pd.DataFrame({"sample_id": ["NA", "NA"], "label": [1, 1]}).to_csv(
            prediction.sample_submission, index=False
        )
    elif problem == "output":
        prediction.output.write_bytes(b"keep original output")
    else:
        prediction.output.with_suffix(".manifest.json").write_bytes(b"keep original manifest")
    with pytest.raises(ValueError):
        blend.predict(prediction)
    assert not case["calls"] and not case["frozen_policies"]


def test_labels_are_averaged_as_probabilities_not_integer_stage_numbers():
    new = np.array([[0.01, 0.01, 0.01, 0.01, 0.56, 0.4]])
    tree = np.array([[0.01, 0.01, 0.01, 0.01, 0.4, 0.56]])
    value = blend.midpoint.blend_probabilities(new, tree)
    np.testing.assert_allclose(value, (new + tree) / 2)
    assert int(value.argmax(1)[0] + 1) in range(1, 7)
    assert not np.isin((new.argmax(1) + tree.argmax(1)) / 2 + 1, range(1, 7)).all()


def test_new_frozen_loader_rejects_both_changed_checkpoint_split_fields(case):
    import torch

    args = case["compare_args"]
    checkpoint = torch.load(args.frozen_run / "model.pt", weights_only=True)
    result = read(args.frozen_run / "result.json")
    checkpoint["train_csv_sha256"] = result["train_csv_sha256"] = "other_training_split"
    torch.save(checkpoint, args.frozen_run / "model.pt")
    dump(args.frozen_run / "result.json", result)
    refresh_trial(args.frozen_run)
    with pytest.raises(ValueError, match="identical Train/Validation"):
        blend.compare(args)
    assert not case["calls"]


def test_frozen_validation_scores_remain_read_only_under_blend(case):
    args = case["compare_args"]
    original = copy.deepcopy(read(args.frozen_run / "result.json"))
    blend.compare(args)
    assert read(args.frozen_run / "result.json") == original
