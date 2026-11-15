"""Human-run server locator/private backup for the reported best 40:60+prior CSV.

Standard library only. Never trains, unpickles weights, parses CSV rows, or sends
files anywhere. A matching filename is NOT proof of a leaderboard submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

FILENAME = "submission_inner_best_40_60_prior_tau.csv"
KIND = "doyu_prior_correction"
SOURCE_FILE = "experiments/doyu/prior_correction.py"
# Reviewed source at origin/feat/doyu-prior-correction, commit 2dae970.
SOURCE_SHA = "dfd6a8e604602d8bbaa7fc5174b94a1bd11ac9077839cff87dfbf1e2e0f25873"
CHECKPOINT_FILES = {
    "status.json",
    "inner_best.pt",
    "refit.pt",
    "manifest.json",
    "result.json",
    "selection_history.json",
    "refit_history.json",
    "group_pair_errors.csv",
}
LEGACY_FILES = {
    "status.json",
    "config.json",
    "identity.json",
    "best_metrics.json",
    "best.pt",
    "split.json",
    "metrics.json",
    "ensemble.joblib",
}
SKIP_DIRS = {
    "data",
    "datasets",
    "open",
    "images",
    "embeddings",
    "venv",
    "__pycache__",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    # These are small metadata files, not prediction caches or model objects.
    if Path(path).stat().st_size > 8 << 20:
        raise ValueError("metadata_too_large")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata_not_object")
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def checked_path(path, roots):
    path = Path(path)
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError("dependency_outside_search_roots")
    if not resolved.is_file():
        raise ValueError("dependency_missing")
    return resolved


def roots_for(runs_roots, team_home=None):
    roots = list(runs_roots)
    if team_home is not None:
        home = Path(team_home).resolve()
        # Probe only the agreed team clone layout; never recursively search home,
        # credentials, dataset folders, or arbitrary other projects.
        roots.append(home / "unid_datathon_team18" / "runs")
        for member in sorted(home.iterdir()):
            if member.name.startswith(".") or member.is_symlink() or not member.is_dir():
                continue
            repo = member / "unid_datathon_team18"
            if not repo.is_symlink():
                roots.append(repo / "runs")
    return sorted({p.resolve() for p in map(Path, roots) if p.is_dir() and not p.is_symlink()})


def inventory(roots):
    files, errors = [], []
    for root in roots:

        def onerror(error):
            errors.append({"path": error.filename, "error_type": type(error).__name__})

        for directory, dirs, names in os.walk(root, followlinks=False, onerror=onerror):
            dirs[:] = sorted(
                name
                for name in dirs
                if not name.startswith(".")
                and name not in SKIP_DIRS
                and not name.endswith("_cache")
                and not (Path(directory) / name).is_symlink()
            )
            for name in sorted(names):
                if name not in {FILENAME, "recipe.json", "correction.json"}:
                    continue
                path = Path(directory) / name
                if not path.is_symlink():
                    files.append(path)
    return sorted(set(files)), errors


def matching_path(files, name, digest):
    matches = [p for p in files if p.name == name and sha256(p) == digest]
    if not matches:
        raise ValueError(f"matching_{name}_missing")
    # Duplicated metadata with identical bytes is interchangeable; CSV candidates
    # remain separate and are never automatically ranked by timestamp or score.
    return matches[0]


def inspect_candidate(csv_path, files, roots):
    result = {"csv": str(csv_path), "status": "unverified", "evidence": []}
    evidence = {}

    def include(path, digest=None):
        path = checked_path(path, roots)
        actual = sha256(path)
        if digest is not None and actual != digest:
            raise ValueError("recorded_file_hash_mismatch")
        evidence[str(path)] = actual
        return path

    try:
        include(csv_path)
        result["csv_sha256"] = evidence[str(csv_path.resolve())]
        receipt_path = include(csv_path.with_suffix(".manifest.json"))
        receipt = read_json(receipt_path)
        if receipt.get("kind") != KIND or receipt.get("this_file_sha256") != SOURCE_SHA:
            raise ValueError("unknown_submission_generator")
        include(csv_path, receipt.get("output_sha256", ""))
        correction_path = include(
            matching_path(files, "correction.json", receipt["correction_sha256"])
        )
        recipe_path = include(matching_path(files, "recipe.json", receipt["recipe_sha256"]))
        correction, recipe = read_json(correction_path), read_json(recipe_path)
        prior, tau = correction.get("prior", []), correction.get("tau")
        if (
            correction.get("kind") != KIND
            or correction.get("format_version") != 1
            or correction.get("base_method") != "inner_best_40_60"
            or correction.get("neural_weight") != 0.4
            or correction.get("label_values") != list(range(1, 7))
            or correction.get("this_file_sha256") != SOURCE_SHA
            or correction.get("rule") != "argmax(p_blend / prior**tau)"
            or tau not in [round(i * 0.05, 2) for i in range(41)]
            or receipt.get("correction") != tau
            or len(prior) != 6
            or not all(math.isfinite(p) and p > 0 for p in prior)
            or not math.isclose(sum(prior), 1.0, rel_tol=0, abs_tol=1e-8)
        ):
            raise ValueError("wrong_40_60_correction_contract")
        status_path = include(recipe_path.parent / "status.json")
        status = read_json(status_path)
        if (
            status.get("status") != "complete"
            or status.get("recipe_sha256") != sha256(recipe_path)
            or recipe.get("kind") != "seungbeom_inner_best_ratios"
            or recipe.get("format_version") != 1
            or recipe.get("status") != "complete"
            or recipe.get("label_values") != list(range(1, 7))
            or recipe.get("ratios_neural_weight", {}).get("inner_best_40_60") != 0.4
            or recipe.get("replay_matches")
            != dict.fromkeys(("inner_best_neural", "extra_trees", "inner_best_50_50"), True)
        ):
            raise ValueError("wrong_or_incomplete_ratio_recipe")
        run = Path(recipe["run_dir"]).resolve()
        expected = {str(run / name) for name in CHECKPOINT_FILES}
        if set(recipe["run_files"]) != expected:
            raise ValueError("unexpected_checkpoint_file_set")
        for path, digest in recipe["run_files"].items():
            include(path, digest)
        manifest = read_json(run / "manifest.json")
        base = manifest["reference"]["base_recipe"]
        legacy = base["provenance"]["files"]
        if not legacy or not any(Path(p).name == "ensemble.joblib" for p in legacy):
            raise ValueError("missing_original_tree_bundle")
        for path, digest in legacy.items():
            if Path(path).name not in LEGACY_FILES:
                raise ValueError("unexpected_legacy_file")
            include(path, digest)
        sources = dict(recipe["source_sha256"])
        sources[SOURCE_FILE] = SOURCE_SHA
        missing_sources = []
        source_evidence = {}
        repos = sorted({root.parent for root in roots})
        for relative, digest in sources.items():
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.suffix != ".py"
            ):
                raise ValueError("invalid_source_path")
            matches = [
                repo / relative
                for repo in repos
                if (repo / relative).is_file()
                and (repo / relative).resolve().is_relative_to(repo)
                and sha256(repo / relative) == digest
            ]
            if matches:
                source_evidence[str(matches[0].resolve())] = digest
            else:
                missing_sources.append(relative)
        result.update(
            status="artifacts_matched",
            correction=str(correction_path),
            recipe=str(recipe_path),
            tau=tau,
            neural_weight=0.4,
            tree_weight=0.6,
            sources=source_evidence,
            missing_sources=missing_sources,
            evidence=[{"path": p, "sha256": d} for p, d in sorted(evidence.items())],
            replay_verified=False,
            leaderboard_identity_verified=False,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        # Never echo JSON/CSV contents in an error. Keep the full investigation on
        # the organizer server; output only paths and structural error categories.
        result["reason"] = str(error) if type(error) is ValueError else type(error).__name__
        if isinstance(error, json.JSONDecodeError):
            result["reason"] = "invalid_json"
    return result


def locate(runs_roots, team_home=None):
    roots = roots_for(runs_roots, team_home)
    files, errors = inventory(roots)
    candidates = [inspect_candidate(p, files, roots) for p in files if p.name == FILENAME]
    return {
        "kind": "final_submission_location",
        "format_version": 1,
        "target_filename": FILENAME,
        "reported_public_score": "0.9093891677",
        "score_source": "user_report_only",
        "search_roots": list(map(str, roots)),
        "scan_errors": errors,
        "candidates": candidates,
        "status": "candidates_found" if candidates else "not_found_in_searched_roots",
        "replay_verified": False,
        "leaderboard_identity_verified": False,
    }


def archive(report, candidate_number, output_dir):
    if report.get("kind") != "final_submission_location":
        raise ValueError("Use the locator report.")
    if not 1 <= candidate_number <= len(report["candidates"]):
        raise ValueError("Candidate number is out of range.")
    selected = report["candidates"][candidate_number - 1]
    roots = list(map(Path, report["search_roots"]))
    # Rebuild provenance now, rather than trusting a stale or edited report.
    files, _ = inventory(roots)
    current = inspect_candidate(Path(selected["csv"]), files, roots)
    if current != selected or current["status"] != "artifacts_matched":
        raise ValueError(
            "Candidate incomplete or changed. Run locate again; do not substitute another model."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    records = [
        *current["evidence"],
        *({"path": p, "sha256": d} for p, d in sorted(current["sources"].items())),
    ]
    try:
        copied = []
        for index, item in enumerate(records):
            source = Path(item["path"])
            target = output_dir / "files" / f"{index:03d}_{source.name}"
            target.parent.mkdir(mode=0o700, exist_ok=True)
            shutil.copyfile(source, target)
            os.chmod(target, 0o600)
            if sha256(target) != item["sha256"] or sha256(source) != item["sha256"]:
                raise ValueError("File changed during backup.")
            copied.append({**item, "backup": str(target.relative_to(output_dir))})
        for item in records:
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("Source changed before backup completed.")
        write_new(
            output_dir / "backup.json",
            {
                "status": "private_backup_complete",
                "candidate": current,
                "files": copied,
                "missing_sources": current["missing_sources"],
                "replay_verified": False,
                "leaderboard_identity_verified": False,
                "portable_submission_package": False,
                "notice": (
                    "Private backup only. Contains predictions and diagnostics; "
                    "do not publish to Git."
                ),
            },
        )
    except BaseException:
        write_new(output_dir / "FAILED.json", {"status": "incomplete_backup_do_not_use"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    find = commands.add_parser("locate")
    find.add_argument("--runs_root", type=Path, action="append", default=[])
    find.add_argument("--team_home", type=Path)
    find.add_argument("--output", type=Path, required=True)
    save = commands.add_parser("archive")
    save.add_argument("--report", type=Path, required=True)
    save.add_argument(
        "--candidate", type=int, required=True, help="1-based printed candidate number"
    )
    save.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "archive":
        archive(read_json(args.report), args.candidate, args.output_dir)
        print(
            "Private backup complete. "
            "Leaderboard identity and inference replay still need verification."
        )
        return
    if args.output.exists():
        raise ValueError("Report exists; choose a new --output path.")
    report = locate(args.runs_root or [Path.cwd() / "runs"], args.team_home)
    write_new(args.output, report)
    print(
        f"Searched run roots: {len(report['search_roots'])}; "
        f"candidates: {len(report['candidates'])}"
    )
    for index, candidate in enumerate(report["candidates"], 1):
        print(f"[{index}] {candidate['csv']}")
        print(f"    status={candidate['status']} reason={candidate.get('reason', '-')}")
        if candidate["status"] == "artifacts_matched":
            print(
                f"    neural:tree=40:60 tau={candidate['tau']} "
                f"missing_sources={len(candidate['missing_sources'])}"
            )
    if report["scan_errors"]:
        print(f"Unreadable paths: {len(report['scan_errors'])}; scan is incomplete.")
    if not report["candidates"]:
        print(
            "Not found in these folders. Ask the teammate who submitted it "
            "for the original file and generation paths."
        )
    print(
        "No model loaded/inference/training. "
        "Matching provenance does not verify the leaderboard upload."
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
