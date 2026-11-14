"""Seed blends use independent synthetic models, vectors and CSVs on CPU only."""

import argparse
import copy
import json

import numpy as np
import pandas as pd
import pytest

from evaluate import classification_metrics
from experiments.seungbeom import midpoint_blend as midpoint
from experiments.seungbeom import neural_experiment as neural
from experiments.seungbeom import neural_seed_repeat as repeat
from experiments.seungbeom import seed_blend as blend
from experiments.seungbeom import seed_blend_core as core
from experiments.seungbeom.export_submissions import sha256
from experiments.seungbeom.neural_config import model_config, normalize_config
from schema import RAW_SENSOR_COLUMNS


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def invented_probabilities():
    rng = np.random.default_rng(901)
    values = rng.uniform(0.01, 1, (5, 8, 6))
    values /= values.sum(axis=2, keepdims=True)
    return values[0], values[1], dict(zip((42, 43, 44), values[2:], strict=True))


def test_seed_mean_is_probability_mean_with_half_total_neural_weight():
    original, tree, seeds = invented_probabilities()
    outputs = core.combine_probabilities(original, tree, seeds)
    expected_mean = np.stack([seeds[42], seeds[43], seeds[44]]).mean(axis=0)
    np.testing.assert_array_equal(outputs["neural_mean"], expected_mean)
    np.testing.assert_allclose(outputs["seed_mean_50_50"], expected_mean * 0.5 + tree * 0.5)
    np.testing.assert_allclose(outputs["seed44_50_50"], seeds[44] * 0.5 + tree * 0.5)
    np.testing.assert_array_equal(
        outputs["baseline_50_50"], midpoint.blend_probabilities(original, tree)
    )
    for values in outputs.values():
        np.testing.assert_allclose(values.sum(axis=1), 1)
    # The tree keeps half rather than one quarter of a four-model average.
    assert not np.allclose(outputs["seed_mean_50_50"], (sum(seeds.values()) + tree) / 4)


@pytest.mark.parametrize("problem", ["missing", "extra", "shape", "nan", "sum", "negative"])
def test_invalid_seed_probabilities_are_rejected(problem):
    original, tree, seeds = invented_probabilities()
    if problem == "missing":
        del seeds[43]
    elif problem == "extra":
        seeds[45] = seeds[44]
    elif problem == "shape":
        seeds[44] = seeds[44][:-1]
    elif problem == "nan":
        seeds[43][0, 0] = np.nan
    elif problem == "sum":
        seeds[42] *= 0.5
    else:
        tree[0, 0] = -0.1
    with pytest.raises(ValueError):
        core.combine_probabilities(original, tree, seeds)


@pytest.fixture
def synthetic_case(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    from experiments.seungbeom import multimodal as mm
    from experiments.seungbeom.neural_model import NeuralFusionClassifier
    from experiments.seungbeom.neural_sensors import NeuralSensorPreprocessor
    from sensor_features import build_sensor_features

    frame = pd.DataFrame(
        {
            "sample_id": ["00001", "NA", "invented_c", "invented_d", "invented_e", "invented_f"],
            "image_path": [f"invented_{i}.png" for i in range(6)],
            "experiment_id": ["independent_validation"] * 6,
            "label": np.arange(1, 7),
            **{name: np.linspace(1 + i, 6 + i, 6) for i, name in enumerate(RAW_SENSOR_COLUMNS)},
        }
    )
    train = frame.assign(
        sample_id=lambda value: "train_" + value["sample_id"], experiment_id="independent_train"
    )
    data = tmp_path / "synthetic_csvs"
    for split, rows in (("train", train), ("validation", frame)):
        (data / split).mkdir(parents=True)
        rows.to_csv(data / split / f"{split}.csv", index=False)
    caches = {}
    for split in ("train", "validation"):
        caches[split] = tmp_path / f"invented_{split}_cache"
        caches[split].mkdir()
        for name in ("features.npy", "metadata.json", "sample_ids.csv"):
            (caches[split] / name).write_bytes(f"independent {split}/{name}".encode())
    config = normalize_config({"sensor_hidden": 2, "fusion_hidden": 4, "epochs": 2})
    prep = NeuralSensorPreprocessor.fit(train)
    vectors = np.random.default_rng(301).normal(size=(6, 512)).astype(np.float32)
    by_id = dict(zip(frame["sample_id"], vectors, strict=True))
    contract = {
        "model": "resnet18",
        "weights": "INDEPENDENT_SYNTHETIC_WEIGHTS",
        "dimension": 512,
        "preprocessing": {"fixture": "independent synthetic vectors"},
        "l2_normalize": False,
        "trained": False,
    }
    calls, policies = [], []

    def features(current):
        sensors = build_sensor_features(current).reset_index(drop=True).add_prefix("sensor_")
        visual = pd.DataFrame(
            np.stack([by_id[sid] for sid in current["sample_id"]]),
            columns=[f"img_{i:04d}" for i in range(512)],
        )
        return pd.concat([sensors, visual], axis=1)

    def extract(current, csv_path, image_root, spec, **kwargs):
        assert csv_path is None
        assert kwargs.get("embeddings_dir") is None
        assert kwargs["batch_size"] == 1 and kwargs["workers"] == 0
        calls.append(current["sample_id"].tolist())
        policies.append((torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32))
        return features(current), copy.deepcopy(contract)

    monkeypatch.setattr(mm, "make_features", extract)
    full_features = features(frame)
    pipeline = mm.build_pipeline("extra_trees", full_features.columns, pca_dim=2, seed=42)
    pipeline.set_params(
        classifier__estimator__n_estimators=3,
        classifier__estimator__n_jobs=1,
        classifier__estimator__min_samples_leaf=1,
    )
    pipeline.fit(full_features, frame["label"])
    tree_prob = mm.predict_proba_six_stable(pipeline, full_features)
    bundle = {
        "kind": "seungbeom_classwise_multimodal",
        "format_version": 1,
        "label_values": list(range(1, 7)),
        "models": ["extra_trees"],
        "pipelines": [pipeline],
        "weights": np.ones((1, 6)),
        "sensor_spec": mm.sensor_spec(),
        "columns": list(full_features.columns),
        "encoder": "resnet18",
        "embedding": contract,
        "seed": 42,
        "refit_train_val": False,
        "weight_fit_split": "train_group_oof_only",
    }
    tree_run = tmp_path / "tree_run"
    tree_run.mkdir()
    joblib.dump(bundle, tree_run / "ensemble.joblib")
    split = {
        "train_csv_sha256": sha256(data / "train/train.csv"),
        "validation_csv_sha256": sha256(data / "validation/validation.csv"),
    }
    dump(tree_run / "status.json", {"status": "complete"})
    dump(tree_run / "split.json", split)
    dump(
        tree_run / "metrics.json",
        {
            "validation": {
                "extra_trees": classification_metrics(frame["label"], tree_prob.argmax(axis=1) + 1)
            }
        },
    )
    saved_probs = {}

    def write_neural(directory, seed, legacy=False):
        directory.mkdir(parents=True)
        current = {**config, "seed": seed}
        torch.manual_seed(seed + 100 if legacy else seed)
        model = NeuralFusionClassifier(**model_config(current)).eval()
        transformed = np.ascontiguousarray(prep.transform(frame), dtype=np.float32)
        with torch.inference_mode():
            probabilities = np.concatenate(
                [
                    torch.softmax(
                        model(
                            torch.from_numpy(vectors[i : i + 1]),
                            torch.from_numpy(transformed[i : i + 1]),
                        ),
                        dim=1,
                    ).numpy()
                    for i in range(len(frame))
                ]
            )
        saved_probs[seed] = probabilities
        identity = neural.run_identity(
            current,
            data / "train/train.csv",
            data / "validation/validation.csv",
            train_embeddings=caches["train"],
            val_embeddings=caches["validation"],
        )
        if legacy:
            identity["source_sha256"][next(iter(identity["source_sha256"]))] = "legacy-code"
        dump(directory / "config.json", current)
        dump(directory / "identity.json", identity)
        dump(directory / "status.json", {"status": "complete"})
        dump(directory / "environment.json", repeat.current_environment("cpu"))
        dump(
            directory / "best_metrics.json",
            {
                "epoch": 1,
                "ce_loss": 2.0,
                **classification_metrics(frame["label"], probabilities.argmax(axis=1) + 1),
            },
        )
        torch.save(
            {
                "kind": "seungbeom_neural_fusion",
                "format_version": 1,
                "config": current,
                "model_config": model_config(current),
                "state_dict": model.state_dict(),
                "preprocessor": prep.to_dict(),
                "visual_contract": contract,
                "label_values": list(range(1, 7)),
                "epoch": 1,
            },
            directory / "best.pt",
        )

    original = tmp_path / "original_neural"
    write_neural(original, 42, legacy=True)
    original_probabilities = saved_probs[42].copy()
    baseline_dir = tmp_path / "baseline_validation"
    base_args = argparse.Namespace(
        neural_run=original,
        ensemble_run=tree_run,
        input=data / "validation/validation.csv",
        image_root=tmp_path,
        output_dir=baseline_dir,
        device="cpu",
        reference_embeddings=None,
    )
    midpoint.validate(base_args)
    experiment = tmp_path / "seed_experiment"
    manifest = repeat.prepare(
        argparse.Namespace(
            reference_run=original,
            data_dir=data,
            train_embeddings=caches["train"],
            val_embeddings=caches["validation"],
            output_dir=experiment,
            device="cpu",
        )
    )
    seed_runs = {}
    for trial in manifest["plan"]["trials"]:
        seed = trial["config"]["seed"]
        seed_runs[seed] = experiment / "runs" / trial["id"]
        write_neural(seed_runs[seed], seed)
    args = argparse.Namespace(
        recipe=baseline_dir / "recipe.json",
        experiment_dir=experiment,
        input=base_args.input,
        image_root=tmp_path,
        output_dir=tmp_path / "seed_blend_validation",
        device="cpu",
    )
    calls.clear()
    policies.clear()
    return {
        "args": args,
        "frame": frame,
        "base_args": base_args,
        "seed_runs": seed_runs,
        "probabilities": saved_probs,
        "original_probabilities": original_probabilities,
        "tree_prob": tree_prob,
        "calls": calls,
        "policies": policies,
        "cache_dirs": caches,
        "contract": contract,
        "by_id": by_id,
    }


def prediction_args(state, tmp_path, frame=None, suffix="one"):
    frame = state["frame"] if frame is None else frame
    input_csv = tmp_path / f"synthetic_input_{suffix}.csv"
    template = tmp_path / f"synthetic_template_{suffix}.csv"
    frame.drop(columns="label").to_csv(input_csv, index=False)
    pd.DataFrame({"sample_id": frame["sample_id"].iloc[::-1], "label": 6}).to_csv(
        template, index=False
    )
    return argparse.Namespace(
        recipe=state["args"].output_dir / "recipe.json",
        input=input_csv,
        image_root=state["args"].image_root,
        sample_submission=template,
        output_dir=tmp_path / f"seed_blend_submissions_{suffix}",
        device="cpu",
        candidates=["baseline_50_50", "seed44_50_50", "seed_mean_50_50"],
    )


def test_actual_core_shares_extraction_and_matches_current_row_inference(synthetic_case):
    torch = pytest.importorskip("torch")
    state = synthetic_case
    args = state["args"]
    loaded, provenance = core.load_members(read(args.recipe), args.experiment_dir, "cpu")
    assert set(loaded["seeds"]) == {42, 43, 44}
    assert set(provenance["seed_runs"]) == {"42", "43", "44"}
    head_calls = []

    def assert_head_policy(module, inputs):
        assert not module.training
        assert all(value.is_contiguous() and value.dtype == torch.float32 for value in inputs)
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        head_calls.append(inputs[0].shape[0])

    for item in [loaded["baseline"], *loaded["seeds"].values()]:
        item["model"].register_forward_pre_hook(assert_head_policy)
    outputs = core.method_probabilities(
        state["frame"].drop(columns="label"), args.image_root, loaded, "cpu", "fp32"
    )
    assert state["calls"] == [state["frame"]["sample_id"].tolist()]
    assert head_calls == [1] * (4 * len(state["frame"]))
    assert not np.array_equal(state["original_probabilities"], state["probabilities"][42])
    expected = core.combine_probabilities(
        state["original_probabilities"], state["tree_prob"], state["probabilities"]
    )
    for method in expected:
        np.testing.assert_array_equal(outputs[method], expected[method])
    for positions in ([5, 1, 0, 4, 3, 2], [1]):
        smaller = core.method_probabilities(
            state["frame"].iloc[positions].drop(columns="label"),
            args.image_root,
            loaded,
            "cpu",
            "fp32",
        )
        for method in outputs:
            np.testing.assert_array_equal(smaller[method], outputs[method][positions])
    legacy_precision = core.method_probabilities(
        state["frame"].drop(columns="label"), args.image_root, loaded, "cpu", "cudnn_tf32"
    )
    assert state["policies"][-1] == (True, False)
    for method in outputs:
        np.testing.assert_array_equal(legacy_precision[method], outputs[method])


def test_compare_replays_every_member_and_preserves_original(synthetic_case):
    state = synthetic_case
    parent_digest = sha256(state["args"].recipe)
    blend.compare(state["args"])
    recipe = read(state["args"].output_dir / "recipe.json")
    assert recipe["status"] == "complete"
    assert recipe["kind"] == "seungbeom_seed_blend"
    assert all(recipe["replay_matches"].values())
    assert len(state["calls"]) == 1
    assert sha256(state["args"].recipe) == parent_digest
    assert (
        recipe["metrics"]["baseline_50_50"] == read(state["args"].recipe)["metrics"]["blend_50_50"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "seed_incomplete",
        "config",
        "identity",
        "environment",
        "weights",
        "reference",
        "cache_identity",
        "preprocessor",
        "image_contract",
    ],
)
def test_loading_rejects_incompatible_or_incomplete_seed_runs(synthetic_case, change):
    state = synthetic_case
    run = state["seed_runs"][44]
    if change == "seed_incomplete":
        dump(run / "status.json", {"status": "running"})
    elif change == "config":
        value = read(run / "config.json")
        value["dropout"] = 0.8
        dump(run / "config.json", value)
    elif change in ("identity", "cache_identity"):
        value = read(run / "identity.json")
        if change == "identity":
            value["source_sha256"][next(iter(value["source_sha256"]))] = "different-code"
        else:
            value["input_provenance"]["train"]["features.npy"] = "different-vectors"
        dump(run / "identity.json", value)
    elif change == "environment":
        value = read(run / "environment.json")
        value["packages"]["torch"] = "different-runtime"
        dump(run / "environment.json", value)
    elif change in ("weights", "preprocessor", "image_contract"):
        torch = pytest.importorskip("torch")
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        if change == "weights":
            checkpoint["label_values"] = [0, 1, 2, 3, 4, 5]
        elif change == "preprocessor":
            checkpoint["preprocessor"]["means"][0] += 1
        else:
            checkpoint["visual_contract"]["weights"] = "different-independent-image-weights"
        torch.save(checkpoint, run / "best.pt")
    else:
        value = read(state["args"].experiment_dir / "experiment.json")
        value["reference_run"] = str(state["seed_runs"][42])
        dump(state["args"].experiment_dir / "experiment.json", value)
    with pytest.raises(ValueError):
        core.load_members(read(state["args"].recipe), state["args"].experiment_dir, "cpu")
    assert not state["calls"]


@pytest.mark.parametrize("change", ["train", "changed_validation", "existing_output"])
def test_compare_rejects_wrong_split_or_existing_output(synthetic_case, change):
    state = synthetic_case
    args = state["args"]
    if change == "train":
        args.input = args.input.parent.parent / "train/train.csv"
    elif change == "changed_validation":
        args.input.write_text(args.input.read_text() + "\n")
    else:
        args.output_dir.mkdir()
        (args.output_dir / "keep.txt").write_text("preserve")
    with pytest.raises(ValueError):
        blend.compare(args)
    assert not state["calls"]
    assert not (args.output_dir / "recipe.json").exists()


@pytest.mark.parametrize(
    "member", ["neural_seed42", "neural_seed43", "neural_seed44", "baseline_50_50"]
)
def test_compare_rejects_replay_confusion_changes(synthetic_case, member):
    state = synthetic_case
    if member == "baseline_50_50":
        value = read(state["args"].recipe)
        value["metrics"]["blend_50_50"]["confusion_matrix"][0][0] += 1
        dump(state["args"].recipe, value)
    else:
        seed = int(member[-2:])
        path = state["seed_runs"][seed] / "best_metrics.json"
        value = read(path)
        value["confusion_matrix"][0][0] += 1
        dump(path, value)
    with pytest.raises(ValueError):
        blend.compare(state["args"])
    assert not (state["args"].output_dir / "recipe.json").exists()


def test_predict_preserves_template_subset_and_original_midpoint_csv(synthetic_case, tmp_path):
    state = synthetic_case
    blend.compare(state["args"])
    state["calls"].clear()
    full_values = {}
    for index, frame in enumerate((state["frame"], state["frame"].iloc[[5, 1, 0]])):
        args = prediction_args(state, tmp_path, frame, suffix=str(index))
        blend.predict(args)
        for candidate in args.candidates:
            path = args.output_dir / f"submission_{candidate}.csv"
            output = pd.read_csv(path, dtype={"sample_id": str}, keep_default_na=False)
            assert output["sample_id"].tolist() == frame["sample_id"].iloc[::-1].tolist()
            assert list(output.columns) == ["sample_id", "label"]
            assert pd.api.types.is_integer_dtype(output["label"])
            assert output["label"].between(1, 6).all()
            if index == 0:
                full_values[candidate] = output.set_index("sample_id")["label"]
            else:
                assert (
                    output["label"].tolist()
                    == full_values[candidate].loc[output["sample_id"]].tolist()
                )
        if index == 0:
            original = argparse.Namespace(
                recipe=state["args"].recipe,
                input=args.input,
                image_root=args.image_root,
                sample_submission=args.sample_submission,
                output=tmp_path / "old_midpoint.csv",
                device="cpu",
            )
            midpoint.predict(original)
            assert (
                original.output.read_bytes()
                == (args.output_dir / "submission_baseline_50_50.csv").read_bytes()
            )
    assert len(state["calls"]) == 3


def test_predict_does_not_require_validation_csv_or_embedding_caches(synthetic_case, tmp_path):
    state = synthetic_case
    blend.compare(state["args"])
    state["args"].input.unlink()
    for cache in state["cache_dirs"].values():
        for path in cache.iterdir():
            path.unlink()
        cache.rmdir()
    args = prediction_args(state, tmp_path)
    blend.predict(args)
    assert (args.output_dir / "submission_seed_mean_50_50.csv").is_file()


@pytest.mark.parametrize("change", ["code", "weights", "rule", "candidate", "output", "template"])
def test_prediction_guards_fail_before_image_inference(synthetic_case, tmp_path, change):
    state = synthetic_case
    blend.compare(state["args"])
    state["calls"].clear()
    args = prediction_args(state, tmp_path)
    recipe = read(args.recipe)
    if change == "code":
        recipe["source_sha256"][next(iter(recipe["source_sha256"]))] = "changed-source"
        dump(args.recipe, recipe)
    elif change == "weights":
        (state["seed_runs"][44] / "best.pt").write_bytes(b"changed synthetic checkpoint")
    elif change == "rule":
        recipe["rules"] = {"seed_mean_50_50": "unapproved weights"}
        dump(args.recipe, recipe)
    elif change == "candidate":
        args.candidates = ["unapproved_candidate"]
    elif change == "output":
        args.output_dir.mkdir()
        (args.output_dir / "keep.csv").write_text("preserve")
    else:
        pd.DataFrame({"sample_id": ["NA", "NA"], "label": [1, 1]}).to_csv(
            args.sample_submission, index=False
        )
    with pytest.raises(ValueError):
        blend.predict(args)
    assert not state["calls"]


def test_report_reads_saved_metrics_without_loading_models(synthetic_case, monkeypatch, capsys):
    state = synthetic_case
    blend.compare(state["args"])
    capsys.readouterr()
    monkeypatch.setattr(core, "load_members", lambda *a, **k: pytest.fail("Loaded models"))
    blend.report(argparse.Namespace(recipe=state["args"].output_dir / "recipe.json"))
    output = capsys.readouterr().out
    assert "seed44_50_50" in output and "seed_mean_50_50" in output
