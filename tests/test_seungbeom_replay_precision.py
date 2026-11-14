"""Precision audit tests use invented cache bytes and CPU flags only."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from experiments.seungbeom import replay_precision as replay
from schema import RAW_SENSOR_COLUMNS


@pytest.mark.parametrize("policy", replay.POLICIES)
def test_context_restores_flags_after_normal_and_exception_exit(policy):
    torch = pytest.importorskip("torch")
    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        for fail in (False, True):
            try:
                with replay.precision_context(policy):
                    assert torch.backends.cuda.matmul.allow_tf32 is False
                    assert torch.backends.cudnn.allow_tf32 == (policy == "cudnn_tf32")
                    if fail:
                        raise RuntimeError("independent synthetic failure")
            except RuntimeError:
                assert fail
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is False
        with pytest.raises(ValueError, match="Unsupported"):
            with replay.precision_context("unknown"):
                pytest.fail("Invalid policy entered")
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


def test_runtime_signature_cpu_does_not_probe_a_gpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: pytest.fail("CPU queried GPU"))
    signature = replay.runtime_signature("cpu")
    assert signature["device"] == "cpu"
    assert signature["gpu"] is None
    assert {"torch", "numpy", "scikit-learn", "joblib"} <= signature["packages"].keys()
    json.dumps(signature, allow_nan=False)


@pytest.mark.parametrize("length", [1, 2, 31, 32, 33, 103])
def test_probe_positions_are_deterministic_unique_and_bounded(length):
    positions = replay.probe_positions(length)
    assert len(positions) == min(length, 32)
    assert len(np.unique(positions)) == len(positions)
    assert positions[0] == 0
    assert positions[-1] == length - 1
    np.testing.assert_array_equal(positions, replay.probe_positions(length))


def test_policy_requires_exact_vectors_and_prefers_fp32_only_for_a_tie():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    almost = np.nextafter(values, np.float32(np.inf))
    selected, report = replay.matching_policy(values, {"fp32": almost, "cudnn_tf32": values.copy()})
    assert selected == "cudnn_tf32"
    assert report["exact_match_by_policy"] == {"fp32": False, "cudnn_tf32": True}
    assert report["max_abs_difference_by_policy"]["fp32"] > 0
    selected, _ = replay.matching_policy(values, dict.fromkeys(replay.POLICIES, values))
    assert selected == "fp32"
    with pytest.raises(ValueError, match="Neither"):
        replay.matching_policy(values, dict.fromkeys(replay.POLICIES, almost))


@pytest.fixture
def independent_cache(tmp_path):
    pytest.importorskip("sklearn")
    from experiments.seungbeom import multimodal as mm

    directory = tmp_path / "independent_reference"
    directory.mkdir()
    ids = ["NA", "00001", *[f"synthetic_{i}" for i in range(38)]]
    vectors = np.arange(40 * 512, dtype=np.float32).reshape(40, 512) / 1024
    np.save(directory / "features.npy", vectors)
    pd.DataFrame({"sample_id": ids}).to_csv(directory / "sample_ids.csv", index=False)
    metadata = {
        "status": "complete",
        "model": "resnet18",
        "dimension": 512,
        "weights": "SYNTHETIC_REFERENCE_WEIGHTS",
        "preprocessing": {"fixture": "invented vectors, no real images"},
        "trained": False,
        "l2_normalize": False,
        "rows": len(ids),
        "batch_size": 1,
        "workers": 0,
        "device": "synthetic-cpu",
        "packages": {"torch": "synthetic"},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    frame = pd.DataFrame(
        {
            "sample_id": ids[::-1],
            "image_path": [f"unused_{i}.png" for i in range(40)],
            "label": [1] * 40,
            **{name: np.arange(40, dtype=float) for name in RAW_SENSOR_COLUMNS},
        }
    )
    contract = mm.embedding_info(metadata)
    loaded = {
        "identity": {
            "input_provenance": {
                "validation": {name: replay.sha256(directory / name) for name in replay.CACHE_FILES}
            }
        },
        "bundle": {
            "encoder": "resnet18",
            "sensor_spec": mm.sensor_spec(),
            "embedding": copy.deepcopy(contract),
        },
        "checkpoint": {"visual_contract": copy.deepcopy(contract)},
    }
    return directory, frame, loaded, dict(zip(ids, vectors, strict=True))


def test_audit_uses_exact_vectors_without_labels_or_validation_scores(
    independent_cache, tmp_path, monkeypatch
):
    directory, frame, loaded, vectors = independent_cache
    calls = []

    def extract(current, image_root, state, policy, device):
        assert state is loaded and device == "cpu"
        calls.append((policy, current["sample_id"].tolist()))
        expected = np.stack([vectors[sid] for sid in current["sample_id"]])
        return expected if policy == "cudnn_tf32" else np.nextafter(expected, np.float32(np.inf))

    monkeypatch.setattr(replay, "extract_probe", extract)
    audit = replay.audit_cache(frame, tmp_path, loaded, directory, "cpu")
    assert audit["image_precision"] == "cudnn_tf32"
    assert audit["probe"]["rows"] == 32
    assert [policy for policy, _ in calls] == list(replay.POLICIES)
    assert calls[0][1] == calls[1][1]
    assert len(audit["reference_files"]) == 3
    assert audit["cache_environment"]["batch_size"] == 1
    loaded["unused_validation_metrics"] = {"fp32": 1.0, "cudnn_tf32": 0.0}
    assert replay.audit_cache(frame.assign(label=6), tmp_path, loaded, directory, "cpu") == audit
    assert "sample_id" not in audit["probe"]


@pytest.mark.parametrize("problem", ["hash", "no_cache_identity", "contract", "no_exact_match"])
def test_audit_rejects_unproven_reference_and_numerical_match(
    independent_cache, tmp_path, monkeypatch, problem
):
    directory, frame, loaded, vectors = independent_cache
    calls = []

    def extract(current, *args):
        calls.append(True)
        expected = np.stack([vectors[sid] for sid in current["sample_id"]])
        return np.nextafter(expected, np.float32(np.inf))

    monkeypatch.setattr(replay, "extract_probe", extract)
    if problem == "hash":
        with (directory / "metadata.json").open("a") as stream:
            stream.write(" ")
    elif problem == "no_cache_identity":
        loaded["identity"]["input_provenance"]["validation"] = {"image_content_sha256": "unused"}
    elif problem == "contract":
        loaded["checkpoint"]["visual_contract"]["weights"] = "different synthetic source"
    with pytest.raises(ValueError):
        replay.audit_cache(frame, tmp_path, loaded, directory, "cpu")
    if problem != "no_exact_match":
        assert not calls


def test_audit_rechecks_reference_hash_after_extraction(independent_cache, tmp_path, monkeypatch):
    directory, frame, loaded, vectors = independent_cache

    def extract(current, image_root, state, policy, device):
        if policy == "fp32":
            with (directory / "metadata.json").open("a") as stream:
                stream.write(" ")
        return np.stack([vectors[sid] for sid in current["sample_id"]])

    monkeypatch.setattr(replay, "extract_probe", extract)
    with pytest.raises(ValueError, match="Reference cache differs"):
        replay.audit_cache(frame, tmp_path, loaded, directory, "cpu")


def test_probe_extraction_pins_math_without_caches_and_restores_flags(
    independent_cache, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from experiments.seungbeom import multimodal as mm

    _, frame, loaded, vectors = independent_cache
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    def make_features(current, csv_path, image_root, spec, **kwargs):
        assert csv_path is None
        assert "embeddings_dir" not in kwargs and "work_dir" not in kwargs
        assert kwargs["batch_size"] == 1 and kwargs["workers"] == 0
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is True
        return pd.DataFrame(
            np.stack([vectors[sid] for sid in current["sample_id"]]),
            columns=[f"img_{index:04d}" for index in range(512)],
        ), loaded["bundle"]["embedding"]

    monkeypatch.setattr(mm, "make_features", make_features)
    chosen = frame.iloc[[3, 1]]
    value = replay.extract_probe(chosen, tmp_path, loaded, "cudnn_tf32", "cpu")
    np.testing.assert_array_equal(value, np.stack([vectors[sid] for sid in chosen["sample_id"]]))
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
