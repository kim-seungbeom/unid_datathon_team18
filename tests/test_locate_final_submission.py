"""Synthetic files only: locate exact prior40 provenance without loading models."""

import json

import pytest

from scripts import locate_final_submission as tool


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)
    return path


@pytest.fixture
def case(tmp_path, monkeypatch):
    home = tmp_path / "team"
    repo = home / "another_member" / "unid_datathon_team18"
    runs = repo / "runs"
    source = put(repo / tool.SOURCE_FILE, "# invented generator source\n")
    monkeypatch.setattr(tool, "SOURCE_SHA", tool.sha256(source))
    legacy_file = put(runs / "tree" / "ensemble.joblib", "NOT A PICKLE; NEVER LOAD ME")
    run = runs / "checkpoint"
    for name in tool.CHECKPOINT_FILES:
        put(run / name, "invented artifact")
    put(
        run / "manifest.json",
        {
            "reference": {
                "base_recipe": {
                    "provenance": {
                        "files": {str(legacy_file): tool.sha256(legacy_file)},
                    }
                }
            }
        },
    )
    recipe = put(
        runs / "ratios" / "recipe.json",
        {
            "kind": "seungbeom_inner_best_ratios",
            "format_version": 1,
            "status": "complete",
            "label_values": list(range(1, 7)),
            "ratios_neural_weight": {"inner_best_40_60": 0.4},
            "replay_matches": dict.fromkeys(
                ("inner_best_neural", "extra_trees", "inner_best_50_50"),
                True,
            ),
            "run_dir": str(run),
            "run_files": {str(run / n): tool.sha256(run / n) for n in tool.CHECKPOINT_FILES},
            "source_sha256": {},
        },
    )
    put(recipe.parent / "status.json", {"status": "complete", "recipe_sha256": tool.sha256(recipe)})
    correction = put(
        runs / "prior40" / "correction.json",
        {
            "kind": tool.KIND,
            "format_version": 1,
            "base_method": "inner_best_40_60",
            "neural_weight": 0.4,
            "label_values": list(range(1, 7)),
            "prior": [1 / 6] * 6,
            "tau": 0.35,
            "this_file_sha256": tool.SOURCE_SHA,
            "rule": "argmax(p_blend / prior**tau)",
        },
    )
    csv = put(runs / "submissions" / tool.FILENAME, "sample_id,label\ninvented-a,3\n")
    receipt = put(
        csv.with_suffix(".manifest.json"),
        {
            "kind": tool.KIND,
            "this_file_sha256": tool.SOURCE_SHA,
            "output_sha256": tool.sha256(csv),
            "correction": 0.35,
            "correction_sha256": tool.sha256(correction),
            "recipe_sha256": tool.sha256(recipe),
        },
    )
    return home, repo, runs, csv, receipt, recipe, correction


def locate_case(case):
    return tool.locate([], case[0])


def test_finds_teammates_actual_tau_and_never_claims_score_or_replay(case):
    report = locate_case(case)
    assert report["score_source"] == "user_report_only"
    (candidate,) = report["candidates"]
    assert candidate["status"] == "artifacts_matched"
    assert candidate["tau"] == 0.35  # never substitute prior30's 0.15
    assert candidate["neural_weight"] == 0.4
    assert candidate["missing_sources"] == []
    assert not candidate["replay_verified"]
    assert not candidate["leaderboard_identity_verified"]


def test_private_backup_checks_bytes_and_preserves_originals(case, tmp_path):
    report = locate_case(case)
    original = case[3].read_bytes()
    output = tmp_path / "backup"
    tool.archive(report, 1, output)
    result = tool.read_json(output / "backup.json")
    assert result["status"] == "private_backup_complete"
    assert not result["portable_submission_package"]
    assert not result["replay_verified"]
    for record in result["files"]:
        assert tool.sha256(output / record["backup"]) == record["sha256"]
        assert (output / record["backup"]).stat().st_mode & 0o777 == 0o600
    assert case[3].read_bytes() == original
    with pytest.raises(FileExistsError):
        tool.archive(report, 1, output)


def test_same_name_different_bytes_is_not_accepted(case):
    case[3].write_text("modified synthetic predictions")
    (candidate,) = locate_case(case)["candidates"]
    assert candidate["status"] == "unverified"
    assert candidate["reason"] == "recorded_file_hash_mismatch"


def test_metadata_hash_not_just_filename(case):
    correction = tool.read_json(case[-1])
    correction["tau"] = 0.5
    put(case[-1], correction)
    (candidate,) = locate_case(case)["candidates"]
    assert candidate["reason"] == "matching_correction.json_missing"


@pytest.mark.parametrize("change", ["neural_weight", "prior", "tau"])
def test_wrong_30_70_or_invalid_prior_is_rejected_even_with_updated_hash(case, change):
    correction = tool.read_json(case[-1])
    correction[change] = {"neural_weight": 0.3, "prior": [0.0] * 6, "tau": 9}[change]
    put(case[-1], correction)
    receipt = tool.read_json(case[4])
    receipt["correction_sha256"] = tool.sha256(case[-1])
    receipt["correction"] = correction["tau"]
    put(case[4], receipt)
    (candidate,) = locate_case(case)["candidates"]
    assert candidate["reason"] == "wrong_40_60_correction_contract"


def test_missing_receipt_and_duplicate_names_are_not_silently_selected(case):
    copied = put(case[2] / "different_run" / tool.FILENAME, case[3].read_text())
    report = locate_case(case)
    assert len(report["candidates"]) == 2
    assert (
        next(c for c in report["candidates"] if c["csv"] == str(copied))["status"] == "unverified"
    )
    assert sum(c["status"] == "artifacts_matched" for c in report["candidates"]) == 1


def test_changed_models_block_backup(case, tmp_path):
    report = locate_case(case)
    (case[2] / "checkpoint" / "inner_best.pt").write_text("changed synthetic model")
    output = tmp_path / "backup"
    with pytest.raises(ValueError, match="incomplete or changed"):
        tool.archive(report, 1, output)
    assert not output.exists()


def test_symlinks_and_other_projects_not_searched(case, tmp_path):
    outside = put(tmp_path / "private" / tool.FILENAME, "not a candidate")
    (case[2] / "linked").symlink_to(outside.parent, target_is_directory=True)
    put(case[0] / "other_project" / "runs" / tool.FILENAME, "not a candidate")
    report = locate_case(case)
    assert len(report["candidates"]) == 1


def test_dependencies_cannot_escape_explicit_run_roots(case, tmp_path):
    outside = put(tmp_path / "elsewhere" / "inner_best.pt", "private synthetic object")
    checkpoint = case[2] / "checkpoint" / "inner_best.pt"
    checkpoint.unlink()
    checkpoint.symlink_to(outside)
    (candidate,) = locate_case(case)["candidates"]
    assert candidate["reason"] == "dependency_outside_search_roots"


def test_cli_zero_candidates_is_honest_and_report_not_overwritten(tmp_path, capsys):
    runs = tmp_path / "repo" / "runs"
    runs.mkdir(parents=True)
    output = tmp_path / "report.json"
    argv = ["locate", "--runs_root", str(runs), "--output", str(output)]
    tool.main(argv)
    assert "candidates: 0" in capsys.readouterr().out
    assert tool.read_json(output)["status"] == "not_found_in_searched_roots"
    with pytest.raises(ValueError, match="Report exists"):
        tool.main(argv)


def test_missing_source_is_reported_not_replaced(case, tmp_path):
    (case[1] / tool.SOURCE_FILE).write_text("# different source revision")
    report = locate_case(case)
    (candidate,) = report["candidates"]
    assert candidate["status"] == "artifacts_matched"
    assert candidate["missing_sources"] == [tool.SOURCE_FILE]
    tool.archive(report, 1, tmp_path / "backup")
    assert tool.read_json(tmp_path / "backup" / "backup.json")["missing_sources"] == [
        tool.SOURCE_FILE
    ]
