"""Synthetic tests for the prior-correction rule and cached-feature head training."""

# ruff: noqa: E402

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from experiments.doyu import prior_correction as pc
from schema import RAW_SENSOR_COLUMNS
from tests.test_seungbeom_resnet_finetune import tiny_backbone  # noqa: F401  (fixture)


def test_tau_zero_is_plain_argmax_and_tau_favors_rare_classes():
    p = np.array([[0.45, 0.35, 0.05, 0.05, 0.05, 0.05]])
    prior = np.array([0.4, 0.1, 0.2, 0.1, 0.1, 0.1])
    assert pc.corrected_labels(p, prior, 0.0).tolist() == [1]
    assert pc.corrected_labels(p, prior, 0.5).tolist() == [2]


def test_prior_requires_every_class():
    with pytest.raises(ValueError):
        pc.class_prior([1, 2, 3, 4, 5])
    np.testing.assert_allclose(pc.class_prior([1, 2, 3, 4, 5, 6, 6, 6]).sum(), 1.0)


def test_best_tau_prefers_zero_on_ties():
    labels = np.arange(1, 7)
    p = np.eye(6) * 0.9 + 0.1 / 6
    tau, scores = pc.best_tau(p, labels, np.full(6, 1 / 6))
    assert tau == 0.0 and max(scores) == scores[0]


def write_oof(directory, labels, fold, neural, tree):
    directory.mkdir()
    np.savez(directory / "oof.npz", sample_ids=np.array([f"s{i}" for i in range(len(labels))]),
             labels=labels, fold=fold, neural=neural, extra_trees=tree)
    pc.write_json(directory / "oof.json", {"oof_sha256": pc.sha256(directory / "oof.npz")})
    pc.write_json(directory / "status.json", {"status": "complete"})


def test_select_adopts_only_when_nested_oof_improves(tmp_path):
    rng = np.random.default_rng(0)
    # Invented data: class 1 is common, model systematically under-predicts rare classes.
    labels = np.concatenate([np.ones(600, int), np.repeat(np.arange(2, 7), 80)])
    n = len(labels)
    logits = rng.normal(size=(n, 6))
    logits[np.arange(n), labels - 1] += 2.5
    logits[:, 0] += 2.0
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    fold = rng.integers(0, 5, size=n)
    write_oof(tmp_path / "oof", labels, fold, p, p)
    c = pc.select(argparse_ns(oof_dir=tmp_path / "oof", output_dir=tmp_path / "sel"))
    assert c["tau"] > 0 and c["adopt"] is True
    assert json.loads((tmp_path / "sel/correction.json").read_text())["tau"] == c["tau"]
    # A perfectly calibrated, balanced invented case should not adopt a correction.
    labels2 = np.tile(np.arange(1, 7), 100)
    p2 = np.full((600, 6), 0.02)
    p2[np.arange(600), labels2 - 1] = 0.9
    write_oof(tmp_path / "oof2", labels2, np.arange(600) % 5, p2, p2)
    c2 = pc.select(argparse_ns(oof_dir=tmp_path / "oof2", output_dir=tmp_path / "sel2"))
    assert c2["adopt"] is False and c2["tau"] == 0.0


def argparse_ns(**kw):
    import argparse

    return argparse.Namespace(**kw)


def test_cached_head_training_is_deterministic(tiny_backbone):
    rng = np.random.default_rng(3)
    n = 60
    frame = pd.DataFrame({name: rng.normal(size=n) for name in RAW_SENSOR_COLUMNS})
    frame["label"] = np.tile(np.arange(1, 7), n // 6)
    feats = rng.normal(size=(n, 512)).astype(np.float32)
    config = pc.ft.normalize_config({"sensor_hidden": 4, "fusion_hidden": 8, "batch_size": 16,
                                     "head_lr": 0.01})
    device = torch.device("cpu")
    m1, p1 = pc.train_head(feats, frame, config, 2, device)
    m2, p2 = pc.train_head(feats, frame, config, 2, device)
    a = pc.head_probabilities(m1, p1, feats, frame, device)
    b = pc.head_probabilities(m2, p2, feats, frame, device)
    assert a.shape == (n, 6)
    np.testing.assert_allclose(a.sum(1), 1, atol=1e-6)
    np.testing.assert_array_equal(a, b)
    assert all(not p.requires_grad for p in m1.backbone.parameters())
