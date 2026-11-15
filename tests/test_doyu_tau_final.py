"""tau_final bundle on synthetic objects only (no competition data or real weights)."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from experiments.doyu import prior_correction as pc
from experiments.doyu import tau_final as tf
from experiments.seungbeom import frozen_checkpoint as checkpoint
from experiments.seungbeom import inner_best_ratios as ibr


def fake_export(tmp_path, monkeypatch, tau=0.5):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "inner_best.pt").write_bytes(b"weights")
    prior = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]
    correction = tmp_path / "correction.json"
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    correction.write_text("{}")
    monkeypatch.setattr(pc, "read_correction", lambda path: {"adopt": True, "prior": prior, "tau": tau})
    bundle = {"sensor_spec": {"s": 1}, "encoder": "resnet18", "embedding": {"e": 1},
              "columns": ["a", "b"], "models": ["xgboost", "extra_trees"], "pipelines": ["x", "ET"]}
    manifest = {"config": {"seed": 42}, "reference": {"base_recipe": {"image_precision": "fp32"}}}
    loaded = (None, None, {"bundle": bundle, "config": {"seed": 7}}, "cpu")
    monkeypatch.setattr(pc, "load_saved_members",
                        lambda path, device: ({"run_dir": str(run)}, manifest, {}, loaded))
    monkeypatch.setattr(checkpoint, "read_result",
                        lambda d: (manifest, {"identities": {"inner_best": {"config": {"seed": 42}}}}))
    monkeypatch.setattr(checkpoint, "runtime_signature", lambda target: {"device": "cpu"})
    out = tmp_path / "weights" / "tau_final"
    config = tf.main(["export", "--recipe", str(recipe), "--correction", str(correction),
                      "--out", str(out), "--device", "cpu"])
    return out, config, prior


def test_export_writes_self_contained_bundle(tmp_path, monkeypatch):
    out, config, prior = fake_export(tmp_path, monkeypatch)
    assert tf.is_bundle(out) and not tf.is_bundle(tmp_path)
    saved = json.loads((out / "final.json").read_text())
    assert saved["tau"] == 0.5 and saved["prior"] == prior and saved["neural_weight"] == 0.4
    assert saved["tree"]["seed"] == 7 and saved["uses_history"] is False
    import joblib

    assert joblib.load(out / "extra_trees.joblib")["pipeline"] == "ET"
    with pytest.raises(ValueError, match="exists"):
        fake_export(tmp_path, monkeypatch)


def test_predict_matches_original_rule_and_test_py(tmp_path, monkeypatch):
    out, config, prior = fake_export(tmp_path, monkeypatch, tau=1.0)
    rng = np.random.default_rng(0)
    n = 40
    neural = rng.dirichlet(np.ones(6), n)
    tree = rng.dirichlet(np.ones(6), n)
    frame = pd.DataFrame({"sample_id": [f"s{i:03d}" for i in range(n)], "experiment_id": "E",
                          "relative_time_s": np.arange(n), "image_path": "x.png"})
    csv = tmp_path / "input.csv"
    frame.iloc[::-1].to_csv(csv, index=False)
    lookup = dict(zip(frame.sample_id, range(n)))
    monkeypatch.setattr(tf, "load", lambda model_dir, device="auto": (config, None))
    monkeypatch.setattr(tf, "read_frame", lambda path: pd.read_csv(path, dtype={"sample_id": str}))

    def members(f, image_root, cfg, loaded):
        rows = [lookup[s] for s in f.sample_id]
        return neural[rows], tree[rows]

    monkeypatch.setattr(tf, "probabilities", members)
    expected = pc.corrected_labels(ibr.weighted_probabilities(neural, tree, "inner_best_40_60"),
                                   prior, 1.0)
    import test as submission

    result = tmp_path / "sub.csv"
    monkeypatch.setattr("sys.argv", ["test.py", "--input", str(csv), "--image_root", str(tmp_path),
                                     "--output", str(result), "--checkpoint", str(out)])
    submission.main()
    got = pd.read_csv(result, dtype={"sample_id": str})
    assert got.sample_id.tolist() == frame.sample_id.tolist()[::-1]  # input order
    np.testing.assert_array_equal(got.set_index("sample_id").loc[frame.sample_id, "label"], expected)
    other = tmp_path / "other.csv"
    got.iloc[::-1].to_csv(other, index=False)
    assert tf.compare(argparse.Namespace(a=result, b=other)) == 0


def test_load_rejects_changed_weights(tmp_path, monkeypatch):
    out, _config, _prior = fake_export(tmp_path, monkeypatch)
    (out / "extra_trees.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        tf.load(out, "cpu")
