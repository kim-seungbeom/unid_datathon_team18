"""Synthetic-only export without an original CSV, receipt, or OOF files."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.seungbeom import final_prior40_export as export


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def test_discovery_selects_only_prior40_and_never_opens_original_csv(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    correction = put(
        runs / "prior" / "correction.json",
        {
            "kind": "doyu_prior_correction",
            "format_version": 1,
            "base_method": "inner_best_40_60",
            "neural_weight": 0.4,
            "label_values": list(range(1, 7)),
            "prior": [1 / 6] * 6,
            "tau": 0.35,
            "rule": "argmax(p_blend / prior**tau)",
            "this_file_sha256": export.locator.SOURCE_SHA,
        },
    )
    wrong = json.loads(correction.read_text())
    wrong["base_method"] = "inner_best_30_70"
    put(runs / "wrong" / "correction.json", wrong)
    recipe = put(runs / "ratio" / "recipe.json", {"synthetic": True})
    monkeypatch.setattr(export.pc.ibr, "read_comparison", lambda path: {})
    original = runs / export.locator.FILENAME
    original.write_text("Must never be read")
    real_hash = export.pc.sha256

    def guarded(path):
        assert Path(path) != original
        return real_hash(path)

    monkeypatch.setattr(export.pc, "sha256", guarded)
    output = tmp_path / "found.json"
    report = export.discover(argparse.Namespace(runs_root=[runs], team_home=None, output=output))
    assert report["correction_files_seen"] == 2
    assert report["rejected_corrections"][0]["reason"] == "not_40_60_correction"
    args = argparse.Namespace(discovery=output, correction=None, recipe=None)
    assert export.choose_paths(args) == (correction, recipe)
    correction.write_text("changed")
    with pytest.raises(ValueError, match="changed since discovery"):
        export.choose_paths(args)


def test_missing_source_hash_is_reported_without_printing_private_values(tmp_path, capsys):
    runs = tmp_path / "runs"
    put(
        runs / "prior" / "correction.json",
        {
            "base_method": "inner_best_40_60",
            "kind": export.pc.KIND,
            "private_example": "SYNTHETIC_CONTENT_MUST_NOT_PRINT",
        },
    )
    result = export.discover(
        argparse.Namespace(
            runs_root=[runs],
            team_home=None,
            output=tmp_path / "discovery.json",
        )
    )
    assert result["corrections"] == []
    assert result["rejected_corrections"][0]["reason"] == (
        "generator_source_hash_missing_or_different"
    )
    assert "SYNTHETIC_CONTENT_MUST_NOT_PRINT" not in capsys.readouterr().out


def test_no_correction_files_distinguished_from_rejected_files(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    result = export.discover(
        argparse.Namespace(
            runs_root=[runs],
            team_home=None,
            output=tmp_path / "discovery.json",
        )
    )
    assert result["correction_files_seen"] == 0
    assert result["rejected_corrections"] == []


def test_ambiguous_discovery_never_guesses(tmp_path):
    report = put(
        tmp_path / "report.json",
        {
            "kind": export.KIND,
            "corrections": [{}, {}],
            "recipes": [],
        },
    )
    with pytest.raises(ValueError, match="ambiguous"):
        export.choose_paths(argparse.Namespace(discovery=report, correction=None, recipe=None))


@pytest.mark.parametrize("subset_bad", [False, True])
def test_export_template_order_no_original_and_no_export_on_subset_failure(
    tmp_path,
    monkeypatch,
    subset_bad,
):
    recipe = put(tmp_path / "recipe.json", {"synthetic": True})
    correction = put(tmp_path / "correction.json", {"prior": [1 / 6] * 6, "tau": 0.35})
    monkeypatch.setattr(export, "correction_at", lambda path: json.loads(path.read_text()))
    source, template = tmp_path / "input.csv", tmp_path / "template.csv"
    pd.DataFrame(
        {
            "sample_id": ["0001", "NA", "0003"],
            "image_path": ["a", "b", "c"],
            "meas_v": [1, 2, 3],
            "meas_a": [0] * 3,
            "meas_t1": [1] * 3,
            "meas_t2": [1] * 3,
            "meas_p": [1] * 3,
        }
    ).to_csv(source, index=False)
    pd.DataFrame({"sample_id": ["0003", "NA", "0001"], "label": [6] * 3}).to_csv(
        template,
        index=False,
    )
    monkeypatch.setattr(
        export.pc,
        "load_saved_members",
        lambda *args: (
            {"source_sha256": {}},
            {"runtime": {}, "reference": {"base_recipe": {"provenance": {"files": {}}}}},
            {},
            ["cpu"],
        ),
    )
    monkeypatch.setattr(export.pc.ibr, "check_unchanged", lambda *args: None)
    monkeypatch.setattr(export.pc.ibr, "image_digest", lambda *args: "synthetic")
    calls = []

    def infer(frame, *args):
        calls.append(frame.sample_id.tolist())
        if subset_bad and len(calls) == 2:
            return np.ones(len(frame), dtype=int)
        return frame.meas_v.to_numpy(dtype=int)

    monkeypatch.setattr(export.audit, "infer", infer)
    args = argparse.Namespace(
        discovery=None,
        correction=correction,
        recipe=recipe,
        input=source,
        sample_submission=template,
        image_root=tmp_path,
        output_dir=tmp_path / "output",
        device="cpu",
    )
    if subset_bad:
        with pytest.raises(ValueError, match="subset predictions differ"):
            export.predict(args)
        assert not list(args.output_dir.glob("*.csv"))
    else:
        record = export.predict(args)
        (csv,) = args.output_dir.glob("*.csv")
        result = pd.read_csv(csv, dtype={"sample_id": str}, keep_default_na=False)
        assert result.label.tolist() == [3, 2, 1]
        assert result.sample_id.tolist() == ["0003", "NA", "0001"]
        assert record["tau"] == 0.35
        assert record["public_score"] is None
        assert not record["original_submission_match_verified"]
        with pytest.raises(ValueError, match="Output directory exists"):
            export.predict(args)
    assert calls == [["0001", "NA", "0003"], ["0003", "NA", "0001"]]
