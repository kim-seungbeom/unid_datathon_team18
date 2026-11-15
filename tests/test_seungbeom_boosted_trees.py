"""Only invented local CPU features; real competition inputs are never opened."""

import argparse
import ast
import copy
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.seungbeom import boosted_trees as run
from experiments.seungbeom import boosted_trees_core as core
from experiments.seungbeom.export_submissions import sha256, write_json
from schema import RAW_SENSOR_COLUMNS


def require_libraries():
    for name in core.MODELS:
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as error:
            pytest.skip(f"Optional native {name} unavailable: {type(error).__name__}")


def invented(rows=120, seed=12):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(1, 7), rows // 6)
    frame = pd.DataFrame(
        {n: labels * (i + 1) + rng.normal(0, 0.3, rows) for i, n in enumerate(RAW_SENSOR_COLUMNS)}
    )
    vectors = rng.normal(size=(rows, 512))
    vectors[:, :3] += labels[:, None]
    return frame, vectors, labels


def test_train_medians_all_missing_no_holdout_fit_row_invariance():
    for module in (run, core):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))
    frame, image, _ = invented()
    frame.iloc[0, 0] = np.inf
    frame.iloc[:, 1] = np.nan
    prep = core.fit_preprocessor(frame)
    assert prep["medians"][1] == 0
    assert prep["medians"][0] == np.median(frame.iloc[1:, 0])
    saved = copy.deepcopy(prep)
    full = core.transform(frame, image, prep)
    assert full.shape == (120, 517) and np.isfinite(full).all()
    for rows in ([119, 0, 3], [1]):
        np.testing.assert_array_equal(
            core.transform(frame.iloc[rows], image[rows], prep), full[rows]
        )
    assert prep == saved
    with pytest.raises(ValueError, match="contract"):
        core.transform(frame, image[:, :511], prep)


@pytest.mark.parametrize("name", core.MODELS)
def test_native_training_best_checkpoint_roundtrip_and_subset(name, tmp_path):
    require_libraries()
    frame, image, y = invented()
    vf, vi, vy = invented(60, 23)
    prep = core.fit_preprocessor(frame)
    x, val = core.transform(frame, image, prep), core.transform(vf, vi, prep)
    record, p = core.fit_model(name, x, y, val, vy, tmp_path, iterations=8, threads=1, seconds=30)
    assert 1 <= record["best_iteration"] <= 8
    assert (
        record["validation_macro_f1"]
        == run.prior30.classification_metrics(vy, p.argmax(1) + 1)["macro_f1"]
    )
    for rows in ([59, 0, 7], [1]):
        np.testing.assert_array_equal(
            core.predict_model(name, tmp_path / record["file"], val[rows], threads=1), p[rows]
        )
    # Budget interruption is graceful and still stores a best, loadable checkpoint.
    folder = tmp_path / "budget"
    folder.mkdir()
    r, _ = core.fit_model(name, x, y, val, vy, folder, iterations=8, threads=1, seconds=1e-12)
    assert r["budget_reached"] and r["best_iteration"] == 1


def test_light_budget_preserves_best_not_latest():
    require_libraries()
    from lightgbm.callback import EarlyStopException

    budget = core.LightBudget(100)
    budget(
        argparse.Namespace(iteration=0, evaluation_result_list=[("valid_0", "macro_f1", 0.8, True)])
    )
    budget.seconds = 0
    with pytest.raises(EarlyStopException) as caught:
        budget(
            argparse.Namespace(
                iteration=1, evaluation_result_list=[("valid_0", "macro_f1", 0.7, True)]
            )
        )
    assert caught.value.best_iteration == 0


def test_candidates_have_fixed_ratios_and_no_cross_row_dependence():
    rng = np.random.default_rng(4)
    members = [rng.dirichlet(np.ones(6), 20) for _ in range(4)]
    full = core.candidates(*members)
    np.testing.assert_array_equal(
        full["inner_best_cat_lgbm_30_35_35"],
        0.3 * members[0] + 0.35 * members[2] + 0.35 * members[3],
    )
    for name, p in full.items():
        np.testing.assert_allclose(p.sum(1), 1)
        np.testing.assert_array_equal(
            core.candidates(*(x[[7, 1]] for x in members))[name], p[[7, 1]]
        )


def test_cached_train_and_live_export_without_fit_inputs(tmp_path, monkeypatch):
    require_libraries()
    splits = {}
    paths = {}
    embedding_dirs = {}
    vectors = {}
    meta = {
        "status": "complete",
        "model": "resnet18",
        "weights": "synthetic-only",
        "dimension": 512,
        "preprocessing": {"synthetic": True},
        "trained": False,
        "l2_normalize": False,
    }
    hashes = {}
    for split, n, seed in [("train", 120, 12), ("validation", 60, 23), ("test", 60, 31)]:
        f, v, y = invented(n, seed)
        f["sample_id"] = [f"{split}_{i}" for i in range(n)]
        f["experiment_id"] = [f"{split}_group_{i // 12}" for i in range(n)]
        f["image_path"] = [f"{split}_{i}.png" for i in range(n)]
        if split != "test":
            f["label"] = y
        path = tmp_path / f"{split}.csv"
        f.to_csv(path, index=False)
        folder = tmp_path / f"{split}_vectors"
        folder.mkdir()
        np.save(folder / "features.npy", v.astype(np.float32))
        pd.DataFrame({"sample_id": f.sample_id}).to_csv(folder / "sample_ids.csv", index=False)
        write_json(folder / "metadata.json", meta)
        splits[split], paths[split], embedding_dirs[split], vectors[split] = (
            f,
            path,
            folder,
            v.astype(np.float32),
        )
        hashes[split] = {n: sha256(folder / n) for n in run.FILES}
    identity = tmp_path / "identity.json"
    write_json(identity, {"input_provenance": hashes})
    marker = tmp_path / "original_marker"
    marker.write_text("unchanged")
    files = {str(identity): sha256(identity), str(marker): sha256(marker)}
    p = np.full((60, 6), 1 / 6)
    old_metrics = {
        n: run.prior30.classification_metrics(splits["validation"].label, v.argmax(1) + 1)
        for n, v in run.cache.previous.candidates(p, p).items()
    }
    old = {"metrics": old_metrics}
    ref = {"metrics": old_metrics}
    base = {}
    manifest = {
        "inputs": {s: {"csv_sha256": sha256(paths[s])} for s in ("train", "validation")},
        "reference": {"base_recipe": {"neural_run": str(tmp_path), "image_precision": "fp32"}},
    }
    monkeypatch.setattr(
        run.previous, "load_reference", lambda *a: (old, ref, base, manifest, files.copy())
    )
    monkeypatch.setattr(
        run.previous, "unchanged", lambda b, m, f, *a: run.original.blend.midpoint.check_files(f)
    )
    monkeypatch.setattr(run.cache, "cached_members", lambda *a: ((p, p), "synthetic-images"))
    monkeypatch.setattr(run.original, "image_digest", lambda *a: "synthetic-images")
    args = argparse.Namespace(
        baseline_recipe=tmp_path / "baseline.json",
        train_csv=paths["train"],
        input=paths["validation"],
        train_embeddings=embedding_dirs["train"],
        validation_embeddings=embedding_dirs["validation"],
        image_root=tmp_path,
        cache_dir=tmp_path,
        output_dir=tmp_path / "model",
        device="cpu",
        iterations=8,
        threads=1,
        minutes_per_model=1,
    )
    fit_calls = []
    fit = core.fit_preprocessor

    def checked(frame):
        fit_calls.append(frame.sample_id.tolist())
        return fit(frame)

    monkeypatch.setattr(core, "fit_preprocessor", checked)
    recipe = run.train(args)
    assert fit_calls == [splits["train"].sample_id.tolist()]
    assert recipe["selection_split"] == "official_validation"
    run.display(run.read_recipe(args.output_dir / "recipe.json"))
    with pytest.raises(ValueError, match="exists"):
        run.train(args)
    paths["train"].unlink()
    paths["validation"].unlink()

    def forbidden(*a, **kw):
        raise AssertionError("No fitting during Test inference")

    monkeypatch.setattr(core, "fit_preprocessor", forbidden)
    monkeypatch.setattr(core, "fit_model", forbidden)

    def fake_live(frame, *a, **kw):
        assert "label" not in frame
        v = vectors["test"]
        return pd.DataFrame(v, columns=[f"img_{i:04d}" for i in range(512)]), meta

    monkeypatch.setattr(run.mm, "make_features", fake_live)
    template = tmp_path / "template.csv"
    pd.DataFrame({"sample_id": splits["test"].sample_id.iloc[::-1], "label": 1}).to_csv(
        template, index=False
    )
    export = argparse.Namespace(
        recipe=args.output_dir / "recipe.json",
        input=paths["test"],
        image_root=tmp_path,
        cache_dir=tmp_path,
        output_dir=tmp_path / "csvs",
        device="cpu",
        sample_submission=template,
        methods=list(core.METHODS),
    )
    outputs = run.predict(export)
    assert outputs
    for name in outputs:
        out = pd.read_csv(export.output_dir / f"submission_{name}.csv")
        assert out.sample_id.tolist() == splits["test"].sample_id.iloc[::-1].tolist()
        assert out.label.between(1, 6).all()
    assert sha256(marker) == files[str(marker)]
    model_file = args.output_dir / recipe["models"]["catboost"]["file"]
    with model_file.open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="model changed"):
        run.read_recipe(args.output_dir / "recipe.json")
