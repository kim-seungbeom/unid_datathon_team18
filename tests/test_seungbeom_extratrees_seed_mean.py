"""Synthetic CPU-only checks; never open competition material."""

import argparse
import ast
import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.seungbeom import extratrees_seed_core as core
from experiments.seungbeom import extratrees_seed_mean as run
from experiments.seungbeom import multimodal as mm
from experiments.seungbeom.export_submissions import sha256, write_json
from schema import RAW_SENSOR_COLUMNS


@pytest.fixture
def case(tmp_path, monkeypatch):
    rng = np.random.default_rng(123)
    meta = dict(
        status="complete",
        model="resnet18",
        weights="synthetic-only",
        dimension=512,
        preprocessing={"synthetic": True},
        trained=False,
        l2_normalize=False,
    )
    frames, dirs, paths, hashes, matrices = {}, {}, {}, {}, {}
    columns = [f"sensor_{n}" for n in RAW_SENSOR_COLUMNS] + [f"img_{i:04d}" for i in range(512)]
    for split, rows in (("train", 72), ("validation", 24)):
        f = pd.DataFrame(rng.normal(size=(rows, 5)), columns=RAW_SENSOR_COLUMNS)
        f.iloc[0, 0] = np.nan
        f["sample_id"] = [f"{split}_{i}" for i in range(rows)]
        f["experiment_id"] = [f"{split}_group_{i // 6}" for i in range(rows)]
        f["image_path"] = [f"image_{i}.png" for i in range(rows)]
        f["label"] = np.tile(np.arange(1, 7), rows // 6)
        v = rng.normal(size=(rows, 512)).astype(np.float32)
        path, folder = tmp_path / f"{split}.csv", tmp_path / split
        f.to_csv(path, index=False)
        folder.mkdir()
        np.save(folder / "features.npy", v)
        f[["sample_id"]].to_csv(folder / "sample_ids.csv", index=False)
        write_json(folder / "metadata.json", meta)
        hashes[split] = {n: sha256(folder / n) for n in run.inputs.FILES}
        frames[split], dirs[split], paths[split] = f, folder, path
        matrices[split] = run.features(f, v, mm.sensor_spec(), columns)
    pipeline = mm.build_pipeline("extra_trees", columns, pca_dim=3, seed=42)
    pipeline.set_params(classifier__estimator__n_jobs=1)
    with threadpool_limits(limits=1):
        pipeline.fit(matrices["train"], frames["train"].label)
    bundle = dict(
        kind="seungbeom_classwise_multimodal",
        format_version=run.legacy.FORMAT_VERSION,
        label_values=list(range(1, 7)),
        sensor_spec=mm.sensor_spec(),
        encoder="resnet18",
        refit_train_val=False,
        weight_fit_split="train_group_oof_only",
        models=["extra_trees"],
        pipelines=[pipeline],
        weights=np.ones((1, 6)),
        embedding=meta,
        columns=columns,
    )
    model = tmp_path / "ensemble.joblib"
    joblib.dump(bundle, model)
    identity = tmp_path / "identity.json"
    write_json(identity, {"input_provenance": hashes})
    files = {str(p): sha256(p) for p in (model, identity)}
    trees = mm.predict_proba_six_stable(pipeline, matrices["validation"])
    neural = rng.dirichlet(np.ones(6), 24)
    metrics = {
        n: run.prior30.classification_metrics(frames["validation"].label, p.argmax(1) + 1)
        for n, p in run.cache.previous.candidates(neural, trees).items()
    }
    manifest = {
        "inputs": {s: {"csv_sha256": sha256(paths[s])} for s in frames},
        "reference": {
            "base_recipe": {
                "ensemble_run": str(tmp_path),
                "neural_run": str(tmp_path),
                "image_precision": "fp32",
            }
        },
    }
    monkeypatch.setattr(
        run.previous,
        "load_reference",
        lambda *a: ({"metrics": metrics}, {"metrics": metrics}, {}, manifest, files.copy()),
    )
    monkeypatch.setattr(
        run.previous, "unchanged", lambda b, m, f, *a: run.original.blend.midpoint.check_files(f)
    )
    monkeypatch.setattr(
        run.cache, "cached_members", lambda *a: ((neural, trees), "synthetic-images")
    )
    args = argparse.Namespace(
        baseline_recipe=tmp_path / "baseline.json",
        train_csv=paths["train"],
        input=paths["validation"],
        train_embeddings=dirs["train"],
        validation_embeddings=dirs["validation"],
        image_root=tmp_path,
        cache_dir=tmp_path,
        output_dir=tmp_path / "result",
        device="cpu",
    )
    return args, pipeline, matrices, frames, files


def test_train_only_classifiers_and_exact_roundtrip(case, monkeypatch):
    args, original, matrices, frames, files = case
    for module in (run, core):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))
    before = joblib.hash(original)

    def forbidden(*a, **kw):
        raise AssertionError("Preprocessing must never be fitted again")

    for cls in (ColumnTransformer, PCA, SimpleImputer, StandardScaler):
        monkeypatch.setattr(cls, "fit", forbidden)
        monkeypatch.setattr(cls, "fit_transform", forbidden)
    fit = mm.EncodedClassifier.fit
    called = []

    def checked(self, x, y):
        np.testing.assert_array_equal(y, frames["train"].label)
        assert len(x) == 72
        called.append(self.estimator.random_state)
        return fit(self, x, y)

    monkeypatch.setattr(mm.EncodedClassifier, "fit", checked)
    record = run.train(args)
    assert called == [43, 44] and joblib.hash(original) == before
    assert run.read_recipe(args.output_dir / "recipe.json") == record
    model = joblib.load(args.output_dir / "members.joblib")
    prep = model["preprocessing"]
    full_x = core.stable_transform(prep, matrices["validation"])
    full = {s: core.predict_classifier(c, full_x) for s, c in model["classifiers"].items()}
    np.testing.assert_array_equal(
        full[42], mm.predict_proba_six_stable(original, matrices["validation"])
    )
    for ids in ([23, 0, 6], [1]):
        x = core.stable_transform(prep, matrices["validation"].iloc[ids])
        for s, c in model["classifiers"].items():
            np.testing.assert_array_equal(core.predict_classifier(c, x), full[s][ids])
    assert all(sha256(Path(p)) == h for p, h in files.items())
    with pytest.raises(ValueError, match="exists"):
        run.train(args)
    with (args.output_dir / "members.joblib").open("ab") as out:
        out.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        run.read_recipe(args.output_dir / "recipe.json")


def test_wrong_original_probability_stops_before_training(case, monkeypatch):
    args, *_ = case
    p = np.full((24, 6), 1 / 6)
    monkeypatch.setattr(run.cache, "cached_members", lambda *a: ((p, p), "synthetic-images"))
    monkeypatch.setattr(core, "fit_members", lambda *a: pytest.fail("Should not fit"))
    with pytest.raises(ValueError, match="Stop before fitting"):
        run.train(args)
    assert not args.output_dir.exists()


def test_changed_embedding_cache_rejected(case):
    args, *_ = case
    with (args.train_embeddings / "features.npy").open("ab") as out:
        out.write(b"changed")
    with pytest.raises(ValueError, match="exact original embedding"):
        run.train(args)


def test_fitted_hash_survives_copy_but_detects_changed_tree(case):
    _, original, *_ = case
    classifier = original.named_steps["classifier"]
    restored = copy.deepcopy(classifier)
    expected = core.fitted_classifier_hash(classifier)
    assert core.fitted_classifier_hash(restored) == expected
    restored.estimator_.estimators_[0].tree_.threshold[0] += 0.25
    assert core.fitted_classifier_hash(restored) != expected


def test_fixed_mean_formula_and_invalid_members():
    rng = np.random.default_rng(5)
    neural = rng.dirichlet(np.ones(6), 12)
    trees = {s: rng.dirichlet(np.ones(6), 12) for s in core.SEEDS}
    actual = core.candidates(neural, trees)
    mean = (trees[42] + trees[43] + trees[44]) / 3
    np.testing.assert_array_equal(actual[core.CANDIDATE], 0.3 * neural + 0.7 * mean)
    np.testing.assert_array_equal(actual[core.BASELINE], 0.3 * neural + 0.7 * trees[42])
    sub = core.candidates(neural[[9, 1]], {s: p[[9, 1]] for s, p in trees.items()})
    for n, p in actual.items():
        np.testing.assert_array_equal(sub[n], p[[9, 1]])
    bad = copy.deepcopy(trees)
    bad.pop(44)
    with pytest.raises(ValueError, match="exactly once"):
        core.candidates(neural, bad)
