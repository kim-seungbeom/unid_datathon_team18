"""Team self-check based on the organizer guide; run real-data checks manually on the server.

This is a team implementation, not the organizer's downloadable check_submission.py.
It checks consistency and CSV format; it does not establish model quality or all rule compliance.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocessing import SensorPreprocessor
from schema import read_frame, validate_labels


def read_submission(path: Path, expected_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str}, keep_default_na=False)
    if list(frame.columns) != ["sample_id", "label"]:
        raise ValueError("Submission columns must be sample_id,label in that order.")
    if frame["sample_id"].tolist() != expected_ids:
        raise ValueError("Submission IDs/order do not match the expected input or template.")
    labels = validate_labels(frame["label"])
    # Competition output must contain integer tokens, not floats such as 1.0.
    if not pd.api.types.is_integer_dtype(frame["label"]):
        raise ValueError("Submission labels must be stored as integers.")
    frame["label"] = labels
    if frame["sample_id"].duplicated().any():
        raise ValueError("Submission contains duplicate IDs.")
    return frame


def compare_predictions(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    expected = reference.set_index("sample_id").loc[candidate["sample_id"], "label"].to_numpy()
    if not np.array_equal(expected, candidate["label"].to_numpy()):
        raise ValueError("Predictions changed for one or more of the same input samples.")


def check_arbitrary_subset(reference, candidate, *, uses_history=False, diagnose_history=False):
    try:
        compare_predictions(reference, candidate)
    except ValueError:
        if not uses_history or not diagnose_history:
            raise
        print("DIAGNOSTIC: arbitrary subset changed after removing past observations.")
        print("Confirm the organizer's partial-input contract before final submission.")
        return False
    print("PASS: arbitrary subset")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--sample_submission", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Default: weights/tau_final (final model), else weights/model.pt",
    )
    parser.add_argument(
        "--diagnose_history_subset",
        action="store_true",
        help="Research only: report history-dependent subset differences without failing.",
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    script = here / "test.py"
    if args.checkpoint is None:
        final = here / "weights/tau_final"
        args.checkpoint = final if final.is_dir() else here / "weights/model.pt"
    from experiments.doyu import tau_final

    if tau_final.is_bundle(args.checkpoint):
        uses_history = False  # current row and its image only
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        uses_history = SensorPreprocessor.from_dict(checkpoint["preprocessor"]).uses_history
    test_root = args.data_dir.resolve() / "test"
    frame = read_frame(test_root / "test.csv")
    for column in ("experiment_id", "relative_time_s"):
        if column not in frame:
            raise ValueError(f"Self-check requires {column}.")
    if frame["experiment_id"].str.strip().eq("").any():
        raise ValueError("Self-check requires nonempty experiment IDs.")
    times = pd.to_numeric(frame["relative_time_s"], errors="coerce")
    if not np.isfinite(times.to_numpy(dtype=float)).all():
        raise ValueError("Self-check requires finite relative_time_s values.")
    frame["relative_time_s"] = times
    template_path = args.sample_submission or args.data_dir / "sample_submission.csv"
    if not template_path.is_file():
        raise ValueError("Template not found. Set its location with --sample_submission.")
    template = pd.read_csv(template_path, dtype={"sample_id": str}, keep_default_na=False)
    if list(template.columns) != ["sample_id", "label"]:
        raise ValueError("Template columns must be sample_id,label.")
    expected_ids = frame["sample_id"].tolist()
    if template["sample_id"].tolist() != expected_ids:
        raise ValueError("Template order differs from test.csv. Resolve before final submission.")
    final = read_submission(args.final, expected_ids)

    with tempfile.TemporaryDirectory(prefix="unid-selfcheck-") as directory:
        work = Path(directory)
        counter = 0

        def run(candidate: pd.DataFrame) -> pd.DataFrame:
            nonlocal counter
            counter += 1
            input_path = work / f"input_{counter}.csv"
            output_path = work / f"output_{counter}.csv"
            candidate.to_csv(input_path, index=False)
            process = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(input_path),
                    "--image_root",
                    str(test_root),
                    "--output",
                    str(output_path),
                    "--checkpoint",
                    str(args.checkpoint.resolve()),
                ],
                capture_output=True,
                text=True,
            )
            if process.returncode:
                raise RuntimeError(
                    "test.py failed. Check the default weights/model.pt and environment "
                    "by running test.py directly on the server."
                )
            return read_submission(output_path, candidate["sample_id"].tolist())

        original = run(frame)
        compare_predictions(final, original)
        print("PASS: final submission reproduced")
        compare_predictions(original, run(frame))
        print("PASS: repeated inference")
        compare_predictions(original, run(frame.sample(frac=1, random_state=42)))
        print("PASS: shuffled input")
        for _, group in frame.groupby("experiment_id", sort=False):
            compare_predictions(original, run(group))
        print("PASS: separate experiments")
        prefixes = []
        for _, group in frame.groupby("experiment_id", sort=False):
            ordered = group.sort_values("relative_time_s", kind="stable")
            count = max(1, int(np.ceil(len(ordered) * 0.6)))
            cutoff = ordered["relative_time_s"].iloc[count - 1]
            prefixes.append(ordered.loc[ordered["relative_time_s"] <= cutoff])
        compare_predictions(original, run(pd.concat(prefixes, ignore_index=True)))
        print("PASS: future rows removed")
        subset_same = check_arbitrary_subset(
            original,
            run(frame.iloc[::2]),
            uses_history=uses_history,
            diagnose_history=args.diagnose_history_subset,
        )
    if subset_same:
        print("All consistency checks passed; accuracy and overall compliance are not certified.")
    else:
        print("Causal checks passed; arbitrary-subset invariance was NOT established.")


if __name__ == "__main__":
    main()
