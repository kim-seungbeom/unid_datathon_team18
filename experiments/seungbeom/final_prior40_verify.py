"""Human-run final replay audit; never tune/refit or replace the original submission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.doyu import prior_correction as pc  # noqa: E402
from schema import read_frame, validate_labels  # noqa: E402
from scripts import locate_final_submission as locator  # noqa: E402


def checked_candidate(location, number):
    report = locator.read_json(location)
    if report.get("kind") != "final_submission_location":
        raise ValueError("Use the final submission locator report.")
    if not 1 <= number <= len(report["candidates"]):
        raise ValueError("Invalid candidate number.")
    selected = report["candidates"][number - 1]
    roots = list(map(Path, report["search_roots"]))
    files, _ = locator.inventory(roots)
    current = locator.inspect_candidate(Path(selected["csv"]), files, roots)
    if current != selected or current["status"] != "artifacts_matched":
        raise ValueError("Candidate changed or incomplete; rerun locate.")
    if current["missing_sources"]:
        raise ValueError(
            "Recorded sources are missing. Restore the recorded source revision first."
        )
    if pc.sha256(ROOT / pc.THIS_FILE) != locator.SOURCE_SHA:
        raise ValueError("Prior40 generator source changed.")
    return current


def infer(frame, image_root, correction, recipe, manifest, loaded):
    neural, tree = pc.ibr.infer_members(
        frame.drop(columns="label", errors="ignore"), image_root, manifest, loaded
    )
    probabilities = pc.ibr.weighted_probabilities(neural, tree, pc.BASE_METHOD)
    return validate_labels(
        pc.corrected_labels(probabilities, correction["prior"], correction["tau"])
    )


def compare_reference(reference_path, ids, labels):
    reference = pd.read_csv(reference_path, dtype={"sample_id": str}, keep_default_na=False)
    if list(reference.columns) != ["sample_id", "label"]:
        raise ValueError("Original CSV columns must be sample_id,label.")
    validate_labels(reference["label"])
    if reference.sample_id.duplicated().any() or set(reference.sample_id) != set(ids):
        raise ValueError("Original CSV and input must contain exactly the same unique IDs.")
    predicted = pd.Series(validate_labels(labels), index=ids).loc[reference.sample_id].to_numpy()
    output = pd.DataFrame({"sample_id": reference.sample_id, "label": predicted})
    return output, int(np.count_nonzero(predicted != reference.label.to_numpy()))


def verify(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; use a new output_dir.")
    candidate = checked_candidate(args.location, args.candidate)
    original = Path(candidate["csv"])
    receipt = locator.read_json(original.with_suffix(".manifest.json"))
    input_digest = pc.sha256(args.input)
    if input_digest != receipt["input_sha256"]:
        raise ValueError("Input differs from the CSV used for the original export.")
    correction = pc.read_correction(Path(candidate["correction"]))
    recipe, manifest, files, loaded = pc.load_saved_members(Path(candidate["recipe"]), args.device)
    frame = read_frame(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pc.write_json(args.output_dir / "status.json", {"status": "running"})
    try:
        print(
            "Replaying exact saved prior40 models from images; no probability cache or fitting.",
            flush=True,
        )
        image_digest = pc.ibr.image_digest(frame, args.image_root)
        labels = infer(frame, args.image_root, correction, recipe, manifest, loaded)
        # The reference is read only AFTER predictions, solely for equality checks/order.
        output, changed = compare_reference(original, frame.sample_id.tolist(), labels)
        positions = np.unique(np.linspace(0, len(frame) - 1, min(32, len(frame)), dtype=int))[::-1]
        subset = frame.iloc[positions].reset_index(drop=True)
        subset_labels = infer(subset, args.image_root, correction, recipe, manifest, loaded)
        subset_changed = int(np.count_nonzero(subset_labels != labels[positions]))
        pc.ibr.check_unchanged(manifest, recipe["source_sha256"], files, loaded[-1])
        if (
            pc.sha256(args.input) != input_digest
            or pc.ibr.image_digest(frame, args.image_root) != image_digest
            or checked_candidate(args.location, args.candidate) != candidate
        ):
            raise ValueError("Inputs, sources, or saved artifacts changed during replay.")
        replay = args.output_dir / "replayed_submission.csv"
        with replay.open("x", encoding="utf-8", newline="") as handle:
            output.to_csv(handle, index=False)
        passed = changed == 0 and subset_changed == 0
        result = {
            "status": "replay_passed" if passed else "replay_mismatch",
            "changed_predictions": changed,
            "reordered_subset_rows": len(positions),
            "reordered_subset_changed": subset_changed,
            "original_sha256": pc.sha256(original),
            "replayed_sha256": pc.sha256(replay),
            "byte_identical": pc.sha256(original) == pc.sha256(replay),
            "input_sha256": input_digest,
            "images_sha256": image_digest,
            "neural_weight": 0.4,
            "tree_weight": 0.6,
            "tau": correction["tau"],
            "candidate": candidate,
            "runtime": manifest["runtime"],
            "reported_public_score": "0.9093891677",
            "score_source": "user_report_only",
            "leaderboard_identity_verified": False,
            "portable_package_verified": False,
            "note": "Subset check is a 32-row spot check, not every possible subset. "
            "Submission owner must confirm the original is the uploaded file.",
        }
        pc.write_json(args.output_dir / "verification.json", result)
        pc.write_json(args.output_dir / "status.json", {"status": result["status"]})
        print(
            f"Full replay changed={changed}; reordered subset changed={subset_changed}; "
            f"byte_identical={result['byte_identical']}"
        )
        print(
            "Original models/CSV unchanged. "
            "Leaderboard identity and portable submission packaging still require confirmation."
        )
        if not passed:
            raise ValueError("Replay mismatch. Keep the original; do not designate the replay CSV.")
        return result
    except BaseException as error:
        pc.write_json(
            args.output_dir / "status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("location", "input", "image_root", "output_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return verify(parser.parse_args(argv))


if __name__ == "__main__":
    main()
