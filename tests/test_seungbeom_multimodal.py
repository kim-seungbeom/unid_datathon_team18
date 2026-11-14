"""Personal multimodal adapters tested only with independently generated inputs."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from experiments.seungbeom import multimodal as M  # noqa: E402


def metadata():
    return {
        "status": "complete",
        "model": "resnet18",
        "weights": "SYNTHETIC_TEST_RANDOM_WEIGHTS",
        "dimension": 512,
        "l2_normalize": False,
        "preprocessing": {"size": 224, "augmentation": "none"},
        "trained": False,
    }


def cache(folder, ids, values=None, changes=None):
    folder.mkdir(parents=True)
    pd.DataFrame({"sample_id": ids}).to_csv(folder / "sample_ids.csv", index=False)
    if values is None:
        values = np.repeat(np.arange(len(ids), dtype=np.float32)[:, None], 512, axis=1)
    np.save(folder / "features.npy", values)
    meta = {**metadata(), "rows": len(ids), **(changes or {})}
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def sensor_frame():
    return pd.DataFrame(
        {
            "sample_id": ["001", "NA", "last"],
            "image_path": ["images/001.png", "images/NA.png", "images/last.png"],
            "experiment_id": ["a", "a", "a"],
            "relative_time_s": [0.0, 10.0, 40.0],
            **{name: [1.0, 2.0, 3.0] for name in M.RAW_SENSOR_COLUMNS},
        }
    )


def test_cache_alignment_preserves_ids_and_subset_order(tmp_path):
    folder = tmp_path / "embeddings"
    cache(folder, ["last", "001", "NA"])
    frame = sensor_frame().iloc[[2, 0]]
    matrix, meta = M.make_features(frame, None, None, M.sensor_spec(), embeddings_dir=folder)
    assert matrix.columns[:5].tolist() == [f"sensor_{c}" for c in M.RAW_SENSOR_COLUMNS]
    assert matrix.shape == (2, 517)
    np.testing.assert_array_equal(matrix["img_0000"], [0, 1])
    assert meta["trained"] is False


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"trained": True}, "frozen"),
        ({"trained": None}, "frozen"),
        ({"dimension": 7}, "dimension"),
        ({"rows": 2}, "rows"),
        ({"status": "running"}, "complete"),
        ({"sample_order_sha256": "incorrect"}, "hash"),
    ],
)
def test_cache_rejects_invalid_provenance(tmp_path, changes, message):
    folder = tmp_path / "embeddings"
    cache(folder, ["001", "NA", "last"], changes=changes)
    with pytest.raises(ValueError, match=message):
        M.load_embeddings(folder, ["001"])


def test_cache_rejects_duplicate_ids_bad_dimension_and_nonfinite(tmp_path):
    duplicate = tmp_path / "duplicate"
    cache(duplicate, ["same", "same"])
    with pytest.raises(ValueError, match="unique"):
        M.load_embeddings(duplicate, ["same"])
    wrong = tmp_path / "wrong"
    cache(wrong, ["one"], values=np.zeros((1, 10)))
    with pytest.raises(ValueError, match="shape"):
        M.load_embeddings(wrong, ["one"])
    bad = tmp_path / "nonfinite"
    cache(bad, ["one"], values=np.full((1, 512), np.nan))
    with pytest.raises(ValueError, match="nonfinite"):
        M.load_embeddings(bad, ["one"])


def test_cache_contract_rejects_changed_encoder_settings(tmp_path):
    folder = tmp_path / "embeddings"
    saved = cache(folder, ["001", "NA", "last"])
    changed = {**M.embedding_info(saved), "l2_normalize": True}
    with pytest.raises(ValueError, match="l2_normalize"):
        M.make_features(
            sensor_frame(),
            None,
            None,
            M.sensor_spec(),
            embeddings_dir=folder,
            embedding_spec=changed,
        )


def test_online_manifest_includes_only_supplied_rows(tmp_path, monkeypatch):
    observed = []

    def fake_extract(**kwargs):
        manifest = pd.read_csv(kwargs["csv_path"], dtype=str, keep_default_na=False)
        observed.append(manifest)
        cache(kwargs["output_dir"], manifest["sample_id"].tolist())

    monkeypatch.setitem(
        sys.modules, "image_features.extract_features", SimpleNamespace(extract=fake_extract)
    )
    frame = sensor_frame().iloc[[1]].copy()
    frame["label"] = 6
    matrix, _ = M.make_features(
        frame, tmp_path / "not_read.csv", tmp_path, M.sensor_spec(), device="cpu"
    )
    assert len(matrix) == 1
    assert observed[0].columns.tolist() == ["sample_id", "image_path"]
    assert observed[0]["sample_id"].tolist() == ["NA"]


def training_matrix():
    rng = np.random.default_rng(40)
    data = rng.normal(size=(36, 7))
    data[0, 0] = np.nan
    return pd.DataFrame(
        data,
        columns=["sensor_meas_v", "sensor_meas_t1", *[f"img_{index:04d}" for index in range(5)]],
    )


@pytest.mark.parametrize("model", ["extra_trees", "logistic", "xgboost", "lightgbm"])
@pytest.mark.parametrize("classes", [[1, 3, 6], [2, 5], [4]])
def test_missing_fold_classes_keep_correct_six_probability_columns(model, classes):
    if model in ("xgboost", "lightgbm"):
        pytest.importorskip(model)
    X = training_matrix()
    y = np.resize(classes, len(X))
    pipeline = M.build_pipeline(model, X.columns, pca_dim=3).fit(X, y)
    probabilities = M.predict_proba_six(pipeline, X.iloc[:5])
    assert probabilities.shape == (5, 6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    absent = [label - 1 for label in M.LABELS if label not in classes]
    np.testing.assert_array_equal(probabilities[:, absent], 0)
    np.testing.assert_array_equal(pipeline.classes_, classes)
    assert np.isin(pipeline.predict(X.iloc[:5]), classes).all()


def test_preprocessing_uses_training_only_and_serializes(tmp_path):
    import joblib

    train = training_matrix()
    pipeline = M.build_pipeline("logistic", train.columns, pca_dim=3).fit(
        train, np.resize([1, 3, 6], len(train))
    )
    transform = pipeline.named_steps["features"]
    sensor_medians = transform.named_transformers_["sensor"].statistics_.copy()
    image_means = transform.named_transformers_["image"].named_steps["scale"].mean_.copy()
    probe = train.iloc[:4].copy() + 10000
    probe.iloc[0, 0] = np.nan
    expected = M.predict_proba_six(pipeline, probe)
    np.testing.assert_array_equal(
        transform.named_transformers_["sensor"].statistics_, sensor_medians
    )
    np.testing.assert_array_equal(
        transform.named_transformers_["image"].named_steps["scale"].mean_, image_means
    )
    path = tmp_path / "synthetic_model.joblib"
    joblib.dump(pipeline, path)
    np.testing.assert_allclose(M.predict_proba_six(joblib.load(path), probe), expected)


def test_sensor_contract_and_pca_bounds(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"sensor_features": list(M.RAW_SENSOR_COLUMNS), "loss": "focal"}))
    spec = M.sensor_spec(path)
    assert "loss" not in spec
    M.check_versions(spec)
    with pytest.raises(ValueError, match="version"):
        M.check_versions({**spec, "sensor_feature_version": -1})
    assert M.effective_pca_dim(32, 5, 512) == 4
    assert M.effective_pca_dim(32, 1, 512) == 0
    assert M.effective_pca_dim(0, 100, 512) == 0
    with pytest.raises(ValueError):
        M.effective_pca_dim(-1, 5, 512)


@pytest.mark.parametrize("model", ["extra_trees", "logistic", "mlp", "xgboost", "lightgbm"])
def test_stable_predictions_match_after_shuffle_and_subsetting(model):
    if model in ("xgboost", "lightgbm"):
        pytest.importorskip(model)
    train = training_matrix()
    pipeline = M.build_pipeline(model, train.columns, pca_dim=3)
    if model == "mlp":
        pipeline.set_params(classifier__estimator__max_iter=5)
    pipeline.fit(train, np.resize([1, 3, 6], len(train)))
    classifier = pipeline.named_steps["classifier"].estimator_
    original_jobs = classifier.get_params().get("n_jobs")
    full = M.predict_proba_six_stable(pipeline, train)
    indices = np.array([35, 0, 2, 24, 19])
    subset = M.predict_proba_six_stable(pipeline, train.iloc[indices])
    np.testing.assert_array_equal(subset, full[indices])
    np.testing.assert_array_equal(subset.argmax(axis=1), full[indices].argmax(axis=1))
    assert classifier.get_params().get("n_jobs") == original_jobs
    np.testing.assert_array_equal(full[:, [1, 3, 4]], 0)
    assert M.predict_proba_six_stable(pipeline, train.iloc[:0]).shape == (0, 6)


def test_stable_tree_classifier_is_batched_and_thread_setting_restored(monkeypatch):
    train = training_matrix()
    pipeline = M.build_pipeline("extra_trees", train.columns, pca_dim=3).fit(
        train, np.resize([1, 3, 6], len(train))
    )
    classifier = pipeline.named_steps["classifier"]
    original = classifier.predict_proba
    calls = []

    def predict(values):
        calls.append((len(values), classifier.estimator_.n_jobs))
        return original(values)

    monkeypatch.setattr(classifier, "predict_proba", predict)
    M.predict_proba_six_stable(pipeline, train)
    assert calls == [(len(train), 1)]
    assert classifier.estimator_.n_jobs == 4

    def fail(values):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(classifier, "predict_proba", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        M.predict_proba_six_stable(pipeline, train)
    assert classifier.estimator_.n_jobs == 4
