"""Frozen-vector provenance and training checks using independent temporary fixtures."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from experiments.seungbeom import neural_data as D  # noqa: E402
from experiments.seungbeom import neural_experiment as E  # noqa: E402
from experiments.seungbeom.neural_config import normalize_config  # noqa: E402
from schema import RAW_SENSOR_COLUMNS  # noqa: E402


def write_cache(folder, ids, *, seed=1):
    folder.mkdir(parents=True)
    values = np.random.default_rng(seed).normal(size=(len(ids), 512)).astype(np.float32)
    np.save(folder / "features.npy", values)
    pd.DataFrame({"sample_id": ids}).to_csv(folder / "sample_ids.csv", index=False)
    digest = hashlib.sha256()
    for sid in ids:
        digest.update(sid.encode("utf-8") + b"\0")
    metadata = {
        "status": "complete",
        "model": "resnet18",
        "dimension": 512,
        "weights": "SYNTHETIC_TEST_RANDOM_WEIGHTS",
        "trained": False,
        "l2_normalize": False,
        "preprocessing": {"synthetic_fixture": True},
        "rows": len(ids),
        "sample_order_sha256": digest.hexdigest(),
    }
    (folder / "metadata.json").write_text(json.dumps(metadata))
    return values, metadata


def test_cached_vectors_align_exact_ids_in_requested_order(tmp_path):
    values, metadata = write_cache(tmp_path / "cache", ["third", "NA", "00001"])
    result, contract = D.load_vectors(tmp_path / "cache", ["00001", "third", "NA"])
    np.testing.assert_array_equal(result, values[[2, 0, 1]])
    assert result.dtype == np.float32
    assert contract == D.embedding_contract(metadata)


@pytest.mark.parametrize(
    "changes",
    [
        {"trained": True},
        {"dimension": 64},
        {"preprocessing": {}},
        {"weights": ""},
        {"l2_normalize": "false"},
        {"model": "unknown"},
    ],
)
def test_invalid_frozen_provenance_is_rejected(tmp_path, changes):
    _, metadata = write_cache(tmp_path / "cache", ["a", "b"])
    metadata.update(changes)
    (tmp_path / "cache/metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="contract"):
        D.load_vectors(tmp_path / "cache", ["a"])


def test_wrong_array_shape_unknown_or_duplicate_ids_and_nonfinite_values(tmp_path):
    folder = tmp_path / "cache"
    values, _ = write_cache(folder, ["a", "b"])
    with pytest.raises(ValueError, match="do not have"):
        D.load_vectors(folder, ["missing"])
    with pytest.raises(ValueError, match="unique"):
        D.load_vectors(folder, ["a", "a"])
    np.save(folder / "features.npy", values[:, :-1])
    with pytest.raises(ValueError, match="dimensions"):
        D.load_vectors(folder, ["a"])
    values[0, 0] = np.nan
    np.save(folder / "features.npy", values)
    with pytest.raises(ValueError, match="finite"):
        D.load_vectors(folder, ["a"])


def test_cached_metadata_rows_and_order_hash_are_verified(tmp_path):
    folder = tmp_path / "cache"
    _, metadata = write_cache(folder, ["a", "b"])
    invalid = {**metadata, "rows": 3}
    (folder / "metadata.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="rows|count"):
        D.load_vectors(folder, ["a"])
    (folder / "metadata.json").write_text(json.dumps(metadata))
    pd.DataFrame({"sample_id": ["b", "a"]}).to_csv(folder / "sample_ids.csv", index=False)
    with pytest.raises(ValueError, match="order|hash"):
        D.load_vectors(folder, ["a"])


def test_visual_vectors_reject_changed_encoder_or_weights(tmp_path):
    folder = tmp_path / "cache"
    _, metadata = write_cache(folder, ["a", "b"])
    frame = pd.DataFrame({"sample_id": ["b", "a"]})
    config = normalize_config({})
    vectors, contract = D.visual_vectors(frame, None, config, cache=folder)
    assert vectors.shape == (2, 512)
    with pytest.raises(ValueError, match="encoder"):
        D.visual_vectors(frame, None, {**config, "embedding_dim": 384}, cache=folder)
    changed = {**contract, "weights": "DIFFERENT_SYNTHETIC_WEIGHTS"}
    with pytest.raises(ValueError, match="weights/preprocessing"):
        D.visual_vectors(frame, None, config, cache=folder, expected=changed)
    metadata["status"] = "running"
    (folder / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not complete"):
        D.visual_vectors(frame, None, config, cache=folder)


def create_split(root, split, groups, embeddings):
    directory = root / split
    directory.mkdir(parents=True)
    rows = []
    for group in groups:
        for label in range(1, 7):
            rows.append(
                {
                    "sample_id": f"{split}_{group}_{label}",
                    "experiment_id": f"{split}_group_{group}",
                    "image_path": f"intentionally_missing/{group}_{label}.png",
                    "relative_time_s": float(label),
                    "label": label,
                    **{
                        name: float(label + index * 0.1)
                        for index, name in enumerate(RAW_SENSOR_COLUMNS)
                    },
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / f"{split}.csv", index=False)
    write_cache(embeddings / split, frame["sample_id"].tolist(), seed=len(split))
    return directory / f"{split}.csv"


def test_cached_embedding_cpu_training_and_source_identity_without_image_access(tmp_path):
    data, embeddings, run = tmp_path / "synthetic", tmp_path / "embeddings", tmp_path / "run"
    train_csv = create_split(data, "train", range(2), embeddings)
    val_csv = create_split(data, "validation", range(1), embeddings)
    config = normalize_config(
        {
            "encoder": "embedding",
            "fusion": "gated",
            "epochs": 2,
            "batch_size": 6,
            "sensor_hidden": 4,
            "fusion_hidden": 8,
            "projection_dim": 4,
            "early_stopping_patience": 0,
            "dropout": 0,
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = E.parser().parse_args(
        [
            "train",
            "--config",
            str(config_path),
            "--data_dir",
            str(data),
            "--run_dir",
            str(run),
            "--train_embeddings",
            str(embeddings / "train"),
            "--val_embeddings",
            str(embeddings / "validation"),
            "--device",
            "cpu",
        ]
    )
    E.train(args)
    assert not (data / "test").exists()
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    summary = json.loads((run / "summary.json").read_text())
    assert summary["elapsed_epochs"] == 2
    assert summary["best_epoch"] in (1, 2)
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    assert saved["config"] == config
    assert saved["model_config"]["fusion"] == "gated"
    assert saved["label_values"] == [1, 2, 3, 4, 5, 6]
    assert saved["visual_contract"]["trained"] is False
    identity = json.loads((run / "identity.json").read_text())
    vector_path = embeddings / "train/features.npy"
    vectors = np.load(vector_path, allow_pickle=False)
    vectors[0, 0] += 1
    np.save(vector_path, vectors)
    changed = E.run_identity(
        config,
        train_csv,
        val_csv,
        train_embeddings=embeddings / "train",
        val_embeddings=embeddings / "validation",
    )
    assert identity["input_provenance"] != changed["input_provenance"]
    assert "experiments/seungbeom/neural_data.py" in changed["source_sha256"]
