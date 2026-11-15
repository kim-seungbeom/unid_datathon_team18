"""Independent synthetic replay checks; no competition data, weights, or network."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from experiments.seungbeom import final_prior40_verify as audit


def test_compare_aligns_ids_and_preserves_original_order(tmp_path):
    path = tmp_path / "original.csv"
    pd.DataFrame({"sample_id": ["0002", "NA"], "label": [3, 1]}).to_csv(path, index=False)
    frame, changed = audit.compare_reference(path, ["NA", "0002"], [1, 3])
    assert changed == 0 and frame.sample_id.tolist() == ["0002", "NA"]
    assert audit.compare_reference(path, ["NA", "0002"], [2, 3])[1] == 1
    with pytest.raises(ValueError, match="same unique IDs"):
        audit.compare_reference(path, ["other", "0002"], [1, 3])


def test_infer_strips_labels_and_uses_saved_40_60_tau(monkeypatch):
    neural = np.array([[0.8, 0.1, 0.025, 0.025, 0.025, 0.025]])
    tree = neural[:, [1, 0, 2, 3, 4, 5]]

    def members(frame, *args):
        assert "label" not in frame
        return neural, tree

    monkeypatch.setattr(audit.pc.ibr, "infer_members", members)
    correction = {"prior": [1 / 6] * 6, "tau": 0.35}
    labels = audit.infer(pd.DataFrame({"label": [6]}), None, correction, {}, {}, None)
    assert labels.tolist() == [2]  # tree receives 60%, not 30:70 or a label vote


@pytest.mark.parametrize("wrong_subset", [False, True])
def test_replay_checks_full_and_partial_input_without_touching_original(
    tmp_path,
    monkeypatch,
    wrong_subset,
):
    csv = tmp_path / "original.csv"
    pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "label": [1, 2, 3]}).to_csv(csv, index=False)
    original_bytes = csv.read_bytes()
    source = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "image_path": ["a", "b", "c"],
            "meas_v": [1, 2, 3],
            "meas_a": [0] * 3,
            "meas_t1": [1] * 3,
            "meas_t2": [1] * 3,
            "meas_p": [1] * 3,
        }
    ).to_csv(source, index=False)
    csv.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "input_sha256": audit.pc.sha256(source),
            }
        )
    )
    candidate = {"csv": str(csv), "correction": "synthetic", "recipe": "synthetic"}
    monkeypatch.setattr(audit, "checked_candidate", lambda *args: candidate)
    monkeypatch.setattr(
        audit.pc,
        "read_correction",
        lambda *args: {
            "prior": [1 / 6] * 6,
            "tau": 0.35,
        },
    )
    monkeypatch.setattr(
        audit.pc,
        "load_saved_members",
        lambda *args: (
            {"source_sha256": {}},
            {"runtime": {"synthetic": True}},
            {},
            ["cpu"],
        ),
    )
    monkeypatch.setattr(audit.pc.ibr, "image_digest", lambda *args: "invented_digest")
    monkeypatch.setattr(audit.pc.ibr, "check_unchanged", lambda *args: None)
    calls = []

    def infer(frame, *args):
        calls.append(frame.sample_id.tolist())
        labels = frame.meas_v.to_numpy(dtype=int)
        return labels if len(calls) == 1 or not wrong_subset else np.ones(len(frame), dtype=int)

    monkeypatch.setattr(audit, "infer", infer)
    args = argparse.Namespace(
        location=tmp_path / "location.json",
        candidate=1,
        input=source,
        image_root=tmp_path,
        output_dir=tmp_path / "audit",
        device="cpu",
    )
    if wrong_subset:
        with pytest.raises(ValueError, match="Replay mismatch"):
            audit.verify(args)
    else:
        result = audit.verify(args)
        assert result["status"] == "replay_passed"
        assert result["byte_identical"]
        assert not result["leaderboard_identity_verified"]
        assert not result["portable_package_verified"]
    assert calls == [["s1", "s2", "s3"], ["s3", "s2", "s1"]]
    assert csv.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="Output directory exists"):
        audit.verify(args)
