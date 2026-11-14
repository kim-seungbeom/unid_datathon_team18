"""Fixed neural/tree blending checked with independent synthetic inputs only."""

import argparse
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from evaluate import classification_metrics
from experiments.seungbeom import midpoint_blend as blend
from schema import RAW_SENSOR_COLUMNS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def independent_probabilities():
    neural = np.full((6, 6), 0.04)
    tree = np.full((6, 6), 0.04)
    np.fill_diagonal(neural, 0.8)
    np.fill_diagonal(tree, 0.8)
    neural[5] = [0.0125, 0.0125, 0.0125, 0.0125, 0.55, 0.4]
    tree[4] = [0.0125, 0.0125, 0.0125, 0.55, 0.4, 0.0125]
    return neural, tree


def test_fixed_probability_blend_can_recover_complementary_errors():
    neural, tree = independent_probabilities()
    actual = blend.blend_probabilities(neural, tree)
    np.testing.assert_allclose(actual, (neural + tree) / 2)
    np.testing.assert_array_equal(actual.argmax(axis=1) + 1, np.arange(1, 7))
    # Averaging the two integer class predictions would instead produce 4.5/5.5.
    assert neural.argmax(axis=1)[-1] != actual.argmax(axis=1)[-1]
    assert tree.argmax(axis=1)[-2] != actual.argmax(axis=1)[-2]
    np.testing.assert_allclose(actual.sum(axis=1), 1)


@pytest.mark.parametrize("problem", ["shape", "empty", "nan", "negative", "sum", "classes"])
def test_malformed_probabilities_are_rejected(problem):
    neural, tree = independent_probabilities()
    if problem == "shape":
        tree = tree[:-1]
    elif problem == "empty":
        neural, tree = neural[:0], tree[:0]
    elif problem == "nan":
        neural[0, 0] = np.nan
    elif problem == "negative":
        neural[0, :2] = [-0.1, 0.94]
    elif problem == "sum":
        tree *= 0.9
    else:
        neural, tree = neural[:, :-1], tree[:, :-1]
    with pytest.raises(ValueError):
        blend.blend_probabilities(neural, tree)


@pytest.fixture
def invented_sources(tmp_path, monkeypatch):
    neural_run, ensemble_run = tmp_path / "neural", tmp_path / "tree"
    neural_run.mkdir()
    ensemble_run.mkdir()
    checkpoints = [neural_run / "best.pt", ensemble_run / "ensemble.joblib"]
    for path in checkpoints:
        path.write_bytes(b"independent synthetic checkpoint; never deserialized")
    frame = pd.DataFrame(
        {
            "sample_id": ["00001", "NA", "fixture_3", "fixture_4", "fixture_5", "fixture_6"],
            "image_path": [f"invented_{index}.png" for index in range(6)],
            "label": np.arange(1, 7),
            **{
                name: np.arange(6, dtype=float) + index
                for index, name in enumerate(RAW_SENSOR_COLUMNS)
            },
        }
    )
    validation_csv = tmp_path / "independent_validation.csv"
    frame.to_csv(validation_csv, index=False)
    train_csv = tmp_path / "independent_train.csv"
    frame.assign(sample_id=lambda x: "train_" + x["sample_id"]).to_csv(train_csv, index=False)
    neural, tree = independent_probabilities()
    probabilities = {"neural": neural, "extra_trees": tree, "blend_50_50": (neural + tree) / 2}
    provenance = {
        "files": {str(path.resolve()): digest(path) for path in checkpoints},
        "train_csv_sha256": digest(train_csv),
        "validation_csv_sha256": digest(validation_csv),
        "saved_macro_f1": {
            name: classification_metrics(frame["label"], probabilities[name].argmax(axis=1) + 1)[
                "macro_f1"
            ]
            for name in ("neural", "extra_trees")
        },
    }
    calls = []
    policies = []
    loaded = {"synthetic_fixture": True, "config": {"seed": 42}}

    def load_sources(*args, **kwargs):
        return loaded, copy.deepcopy(provenance)

    def predict_probabilities(current, image_root, models, device, image_precision="fp32"):
        assert models is loaded
        assert device == "cpu"
        calls.append(current["sample_id"].tolist())
        policies.append(image_precision)
        order = pd.Index(frame["sample_id"]).get_indexer(current["sample_id"])
        assert (order >= 0).all()
        return {name: value[order].copy() for name, value in probabilities.items()}

    monkeypatch.setattr(blend, "load_sources", load_sources)
    monkeypatch.setattr(blend, "method_probabilities", predict_probabilities)
    args = argparse.Namespace(
        neural_run=neural_run,
        ensemble_run=ensemble_run,
        input=validation_csv,
        image_root=tmp_path / "invented_images",
        output_dir=tmp_path / "validation_report",
        device="cpu",
        reference_embeddings=None,
    )
    return {
        "args": args,
        "frame": frame,
        "provenance": provenance,
        "checkpoints": checkpoints,
        "probabilities": probabilities,
        "calls": calls,
        "policies": policies,
        "loaded": loaded,
        "train_csv": train_csv,
    }


def test_validation_replays_both_models_and_records_fixed_candidate(invented_sources):
    state = invented_sources
    blend.validate(state["args"])
    recipe = json.loads((state["args"].output_dir / "recipe.json").read_text())
    assert recipe["kind"] == "seungbeom_midpoint_blend"
    assert recipe["status"] == "complete"
    assert recipe["format_version"] == 2
    assert recipe["image_precision"] == "fp32"
    assert recipe["precision_audit"] is None
    assert recipe["runtime_signature"]
    assert recipe["metrics"]["blend_50_50"]["macro_f1"] == 1.0
    assert recipe["provenance"] == state["provenance"]
    assert recipe["source_sha256"]
    assert state["calls"] == [state["frame"]["sample_id"].tolist()]
    assert state["policies"] == ["fp32"]


@pytest.mark.parametrize("problem", ["different_split", "train_split", "saved_score", "output"])
def test_validation_rejects_wrong_split_replay_or_output(invented_sources, problem):
    state = invented_sources
    args = state["args"]
    if problem == "different_split":
        args.input.write_text(args.input.read_text() + "\n")
    elif problem == "train_split":
        args.input = state["train_csv"]
    elif problem == "saved_score":
        state["provenance"]["saved_macro_f1"]["neural"] = 0.0
    else:
        args.output_dir.mkdir()
        (args.output_dir / "preserve.txt").write_text("keep")
    with pytest.raises(ValueError):
        blend.validate(args)
    assert not (args.output_dir / "recipe.json").exists()
    if problem != "saved_score":
        assert not state["calls"]
    if problem == "output":
        assert (args.output_dir / "preserve.txt").read_text() == "keep"


def predict_args(state, tmp_path, frame=None, *, suffix="one"):
    frame = state["frame"] if frame is None else frame
    input_csv, template = tmp_path / f"input_{suffix}.csv", tmp_path / f"template_{suffix}.csv"
    frame.drop(columns="label").to_csv(input_csv, index=False)
    pd.DataFrame({"sample_id": frame["sample_id"].iloc[::-1], "label": 6}).to_csv(
        template, index=False
    )
    return argparse.Namespace(
        recipe=state["args"].output_dir / "recipe.json",
        input=input_csv,
        image_root=state["args"].image_root,
        sample_submission=template,
        output=tmp_path / f"submission_{suffix}.csv",
        device="cpu",
    )


def test_prediction_preserves_template_ids_and_subset_values(invented_sources, tmp_path):
    state = invented_sources
    blend.validate(state["args"])
    for index, frame in enumerate((state["frame"], state["frame"].iloc[[5, 1, 0]])):
        args = predict_args(state, tmp_path, frame, suffix=str(index))
        blend.predict(args)
        output = pd.read_csv(args.output, dtype={"sample_id": str}, keep_default_na=False)
        assert list(output.columns) == ["sample_id", "label"]
        assert output["sample_id"].tolist() == frame["sample_id"].iloc[::-1].tolist()
        assert output["label"].tolist() == frame["label"].iloc[::-1].tolist()
        assert pd.api.types.is_integer_dtype(output["label"])


@pytest.mark.parametrize(
    "problem",
    [
        "model_changed",
        "code_changed",
        "status",
        "rule",
        "template",
        "output",
        "missing_policy",
        "unknown_policy",
        "policy_without_audit",
        "old_recipe",
    ],
)
def test_prediction_guards_fail_before_inference(invented_sources, tmp_path, problem):
    state = invented_sources
    blend.validate(state["args"])
    args = predict_args(state, tmp_path)
    state["calls"].clear()
    recipe = json.loads(args.recipe.read_text())
    if problem == "model_changed":
        state["checkpoints"][0].write_bytes(b"changed independent model")
    elif problem == "code_changed":
        name = next(iter(recipe["source_sha256"]))
        recipe["source_sha256"][name] = "different-hash"
        dump(args.recipe, recipe)
    elif problem == "status":
        recipe["status"] = "running"
        dump(args.recipe, recipe)
    elif problem == "rule":
        recipe["rule"]["neural"] = 0.7
        recipe["rule"]["extra_trees"] = 0.3
        dump(args.recipe, recipe)
    elif problem == "missing_policy":
        del recipe["image_precision"]
        dump(args.recipe, recipe)
    elif problem == "unknown_policy":
        recipe["image_precision"] = "unknown_precision"
        dump(args.recipe, recipe)
    elif problem == "policy_without_audit":
        recipe["image_precision"] = "cudnn_tf32"
        dump(args.recipe, recipe)
    elif problem == "old_recipe":
        recipe["format_version"] = 1
        dump(args.recipe, recipe)
    elif problem == "template":
        pd.DataFrame({"sample_id": ["NA", "NA"], "label": [1, 1]}).to_csv(
            args.sample_submission, index=False
        )
    else:
        args.output.write_text("preserve existing submission")
    with pytest.raises(ValueError):
        blend.predict(args)
    assert not state["calls"]
    if problem == "output":
        assert args.output.read_text() == "preserve existing submission"


def test_runtime_drift_blocks_prediction_before_member_inference(
    invented_sources, tmp_path, monkeypatch
):
    state = invented_sources
    monkeypatch.setattr(blend, "runtime_signature", lambda *a, **k: {"torch": "invented_v1"})
    blend.validate(state["args"])
    args = predict_args(state, tmp_path)
    state["calls"].clear()
    monkeypatch.setattr(blend, "runtime_signature", lambda *a, **k: {"torch": "invented_v2"})
    with pytest.raises(ValueError, match="[Rr]untime|[Ee]nvironment"):
        blend.predict(args)
    assert not state["calls"]
    assert not args.output.exists()


def test_audited_precision_is_saved_and_reused_without_cache_access_at_prediction(
    invented_sources, tmp_path, monkeypatch
):
    state = invented_sources
    reference = tmp_path / "independent_cache"
    reference.mkdir()
    reference_file = reference / "fixture.txt"
    reference_file.write_text("independent synthetic cache marker")
    audit = {
        "image_precision": "cudnn_tf32",
        "reference_files": {str(reference_file): digest(reference_file)},
        "probe": {"exact": True, "rows": 6},
        "cache_environment": {"fixture": "independent"},
    }
    state["args"].reference_embeddings = reference

    def audit_cache(frame, image_root, loaded, reference_dir, device):
        assert loaded is state["loaded"]
        assert reference_dir == reference
        assert device == "cpu"
        assert frame["sample_id"].tolist() == state["frame"]["sample_id"].tolist()
        return copy.deepcopy(audit)

    monkeypatch.setattr(blend, "audit_cache", audit_cache)
    recipe = blend.validate(state["args"])
    assert recipe["image_precision"] == "cudnn_tf32"
    assert recipe["precision_audit"] == audit
    assert state["policies"] == ["cudnn_tf32"]
    # Prediction must reconstruct live images, not consult the Validation cache.
    reference_file.unlink()
    reference.rmdir()
    monkeypatch.setattr(
        blend, "audit_cache", lambda *a, **k: pytest.fail("Prediction attempted cache audit")
    )
    args = predict_args(state, tmp_path)
    blend.predict(args)
    assert state["policies"] == ["cudnn_tf32", "cudnn_tf32"]
    assert args.output.is_file()


def test_precision_audit_does_not_relax_saved_metric_replay_guard(
    invented_sources, tmp_path, monkeypatch
):
    state = invented_sources
    state["args"].reference_embeddings = tmp_path / "synthetic_cache"
    monkeypatch.setattr(
        blend,
        "audit_cache",
        lambda *a, **k: {
            "image_precision": "cudnn_tf32",
            "reference_files": {},
            "probe": {"exact": True},
            "cache_environment": {},
        },
    )
    state["provenance"]["saved_macro_f1"]["neural"] += 0.00001
    with pytest.raises(ValueError, match="replay differs"):
        blend.validate(state["args"])
    assert not (state["args"].output_dir / "recipe.json").exists()
    assert state["policies"] == ["cudnn_tf32"]


def test_live_member_integration_uses_saved_transforms_one_extraction_and_ordered_classes(
    tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    torch = pytest.importorskip("torch")
    from sklearn.pipeline import Pipeline

    from experiments.seungbeom import multimodal as mm
    from experiments.seungbeom.neural_config import model_config, normalize_config
    from experiments.seungbeom.neural_model import NeuralFusionClassifier
    from experiments.seungbeom.neural_sensors import NeuralSensorPreprocessor
    from sensor_features import build_sensor_features

    frame = pd.DataFrame(
        {
            "sample_id": ["00001", "NA", "fixture_c", "fixture_d", "fixture_e", "fixture_f"],
            "image_path": [f"unused_invented_{i}.png" for i in range(6)],
            "label": np.arange(1, 7),
            **{
                name: np.linspace(10 + index, 40 + index, 6)
                for index, name in enumerate(RAW_SENSOR_COLUMNS)
            },
        }
    )
    config = normalize_config({"sensor_hidden": 2, "fusion_hidden": 3, "dropout": 0.3})
    prep = NeuralSensorPreprocessor.fit(frame)
    model = NeuralFusionClassifier(**model_config(config))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.sensor_encoder[0].weight[0, 0] = 1
        model.sensor_encoder[0].bias[0] = 0.5
        model.classifier[0].bias.fill_(1)
        model.classifier[0].weight[0, 0] = 1
        model.classifier[0].weight[1, 1] = 1
        model.classifier[0].weight[2, 512] = 1
        model.classifier[-1].weight[0, 0] = 1
        model.classifier[-1].weight[1, 1] = 1
        model.classifier[-1].weight[2, 2] = 1
    vectors = np.random.default_rng(100).normal(size=(6, 512)).astype(np.float32)
    vectors[:, :2] = np.linspace(0.1, 0.6, 6)[:, None]
    by_id = dict(zip(frame["sample_id"], vectors, strict=True))
    contract = {
        "model": "resnet18",
        "dimension": 512,
        "weights": "SYNTHETIC_TEST_RANDOM_WEIGHTS",
        "l2_normalize": False,
        "preprocessing": {"fixture": "independent synthetic vectors"},
        "trained": False,
    }
    calls = []
    extraction_precision = []

    def matrix(current):
        sensors = build_sensor_features(current).reset_index(drop=True).add_prefix("sensor_")
        images = pd.DataFrame(
            np.stack([by_id[sid] for sid in current["sample_id"]]),
            columns=[f"img_{i:04d}" for i in range(512)],
        )
        return pd.concat([sensors, images], axis=1)

    features = matrix(frame)
    pipeline = mm.build_pipeline("extra_trees", features.columns, pca_dim=2, seed=42)
    pipeline.set_params(
        classifier__estimator__n_estimators=5,
        classifier__estimator__n_jobs=1,
        classifier__estimator__min_samples_leaf=1,
    )
    pipeline.fit(features, frame["label"])
    bundle = {
        "kind": "seungbeom_classwise_multimodal",
        "format_version": 1,
        "label_values": list(range(1, 7)),
        "models": ["extra_trees"],
        "pipelines": [pipeline],
        "weights": np.ones((1, 6)),
        "sensor_spec": mm.sensor_spec(),
        "columns": list(features.columns),
        "encoder": "resnet18",
        "embedding": contract,
        "seed": 42,
        "refit_train_val": False,
        "weight_fit_split": "train_group_oof_only",
    }
    neural_run, tree_run = tmp_path / "independent_neural", tmp_path / "independent_tree"
    neural_run.mkdir()
    tree_run.mkdir()
    split = {"train_csv_sha256": "synthetic-train", "validation_csv_sha256": "synthetic-val"}
    for run in (neural_run, tree_run):
        dump(run / "status.json", {"status": "complete"})
    dump(neural_run / "config.json", config)
    dump(neural_run / "identity.json", {"config": config, **split})
    dump(neural_run / "best_metrics.json", {"macro_f1": 0.5})
    torch.save(
        {
            "kind": "seungbeom_neural_fusion",
            "format_version": 1,
            "config": config,
            "model_config": model_config(config),
            "state_dict": model.state_dict(),
            "preprocessor": prep.to_dict(),
            "visual_contract": contract,
            "label_values": list(range(1, 7)),
            "epoch": 1,
        },
        neural_run / "best.pt",
    )
    dump(tree_run / "split.json", split)
    dump(tree_run / "metrics.json", {"validation": {"extra_trees": {"macro_f1": 0.5}}})
    joblib.dump(bundle, tree_run / "ensemble.joblib")

    def extract_once(current, csv_path, image_root, spec, **kwargs):
        assert csv_path is None
        assert spec == mm.sensor_spec()
        assert "embeddings_dir" not in kwargs
        assert kwargs["batch_size"] == 1 and kwargs["workers"] == 0
        assert kwargs["encoder"] == "resnet18"
        calls.append(current["sample_id"].tolist())
        extraction_precision.append(
            (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        )
        return matrix(current), contract.copy()

    monkeypatch.setattr(mm, "make_features", extract_once)
    monkeypatch.setattr(Pipeline, "fit", lambda *a, **k: pytest.fail("Inference called fit"))
    loaded, provenance = blend.load_sources(neural_run, tree_run)
    head_calls = []

    def check_head_precision_and_layout(module, inputs):
        visual, sensors = inputs
        assert visual.is_contiguous()
        assert sensors.is_contiguous()
        assert visual.dtype == torch.float32 and sensors.dtype == torch.float32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        head_calls.append(visual.shape[0])

    loaded["model"].register_forward_pre_hook(check_head_precision_and_layout)
    assert provenance["files"][str((neural_run / "best.pt").resolve())] == digest(
        neural_run / "best.pt"
    )
    original = blend.method_probabilities(frame, tmp_path, loaded, "cpu")
    assert calls == [frame["sample_id"].tolist()]
    assert extraction_precision == [(False, False)]
    assert head_calls == [1] * len(frame)
    with torch.inference_mode():
        expected = torch.softmax(
            model.eval()(torch.from_numpy(vectors), torch.from_numpy(prep.transform(frame))),
            dim=1,
        ).numpy()
    np.testing.assert_allclose(original["neural"], expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(
        original["extra_trees"], mm.predict_proba_six_stable(pipeline, features)
    )
    for positions in ([5, 3, 1, 4, 0, 2], [1]):
        subset = blend.method_probabilities(frame.iloc[positions], tmp_path, loaded, "cpu")
        for method in original:
            np.testing.assert_array_equal(subset[method], original[method][positions])
    same_vectors_legacy_precision = blend.method_probabilities(
        frame, tmp_path, loaded, "cpu", image_precision="cudnn_tf32"
    )
    assert extraction_precision[-1] == (True, False)
    for method in original:
        np.testing.assert_array_equal(same_vectors_legacy_precision[method], original[method])
    # Alter the current image vector while keeping this row's sensors unchanged.
    by_id["00001"] = np.zeros(512, dtype=np.float32)
    changed = blend.method_probabilities(frame.iloc[[0]], tmp_path, loaded, "cpu")
    assert not np.allclose(changed["neural"], original["neural"][[0]])
    # Both members must reject extraction metadata from different image weights.
    contract["weights"] = "different-independent-synthetic-source"
    with pytest.raises(ValueError, match="image contract"):
        blend.method_probabilities(frame.iloc[[0]], tmp_path, loaded, "cpu")
