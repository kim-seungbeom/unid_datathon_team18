"""Export saved multimodal tree candidates using one live image extraction.

Each candidate is a fixed member or fixed blend of an existing Train-fitted bundle.
This command never fits models, preprocessing, or weights on the inference input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom import classwise_ensemble as ensemble  # noqa: E402
from schema import LABELS, read_frame  # noqa: E402

DEFAULT_METHODS = ("classwise_ensemble", "extra_trees", "equal_average", "xgboost")
METHODS = ("classwise_ensemble", "equal_average", *ensemble.MODEL_NAMES)


def validate_methods(methods):
    if isinstance(methods, str):
        raise ValueError("methods must be a sequence of distinct method names.")
    methods = tuple(methods)
    if (
        not methods
        or any(not isinstance(method, str) or method not in METHODS for method in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("Specify distinct supported candidate methods.")
    return methods


def template_ids(template_path, expected_ids):
    template = pd.read_csv(template_path, dtype=str, keep_default_na=False)
    if list(template.columns) != ["sample_id", "label"]:
        raise ValueError("Submission template columns must be sample_id,label in that order.")
    ids = template["sample_id"]
    if (
        not ids.is_unique
        or ids.str.strip().eq("").any()
        or len(ids) != len(expected_ids)
        or set(ids) != set(expected_ids)
    ):
        raise ValueError("Submission template IDs must match the supplied input exactly.")
    return ids.tolist()


def candidate_labels(probabilities, bundle, method):
    """Apply only the requested, already fixed decision rule to each input row."""
    if method == "classwise_ensemble":
        scores = ensemble.blend_scores(probabilities, bundle["weights"])
    elif method == "equal_average":
        weights = np.full((len(bundle["models"]), len(LABELS)), 1.0 / len(bundle["models"]))
        scores = ensemble.blend_scores(probabilities, weights)
    else:
        scores = probabilities[:, bundle["models"].index(method), :]
    return np.asarray(scores.argmax(axis=1) + 1, dtype=np.int64)


def predict_candidates(
    *,
    checkpoint: Path,
    input_csv: Path,
    image_root: Path,
    template_path: Path,
    output_dir: Path,
    methods=DEFAULT_METHODS,
    device="cuda",
):
    """Return method/filename/hash/row metadata, never raw predictions, for each CSV."""
    methods = validate_methods(methods)
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda.")
    output_dir = Path(output_dir)
    paths = {method: output_dir / f"submission_{method}.csv" for method in methods}
    if any(path.exists() for path in paths.values()):
        raise ValueError("A candidate output already exists; choose a new output directory.")
    frame = read_frame(input_csv)
    ids = frame["sample_id"].tolist()
    ordered_ids = template_ids(template_path, ids)
    bundle = ensemble.load_bundle(Path(checkpoint))
    missing = set(methods) - {"classwise_ensemble", "equal_average", *bundle["models"]}
    if missing:
        raise ValueError("Requested candidate member is absent from the saved bundle.")

    from experiments.seungbeom import multimodal as mm

    # No persistent cache is accepted: modified images and arbitrary input subsets
    # must be processed from this invocation's explicit manifest and image root.
    features, _ = mm.make_features(
        frame,
        Path(input_csv),
        Path(image_root),
        bundle["sensor_spec"],
        encoder=bundle["encoder"],
        embedding_spec=bundle["embedding"],
        device=device,
        batch_size=1,
        workers=0,
    )
    if list(features.columns) != bundle["columns"]:
        raise ValueError("Input feature columns differ from training.")
    print("Image features ready; predicting with the saved tree models...", flush=True)
    probabilities = ensemble.prediction_cube(bundle["pipelines"], features, stable_rows=True)
    if probabilities.shape[0] != len(frame):
        raise ValueError("Prediction rows differ from the supplied input.")
    outputs = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        labels = candidate_labels(probabilities, bundle, method)
        ordered = pd.Series(labels, index=ids).loc[ordered_ids].to_numpy(dtype=np.int64)
        if not np.isin(ordered, LABELS).all():
            raise ValueError("Candidate labels must be integers in 1..6.")
        path = paths[method]
        with path.open("x", encoding="utf-8", newline="") as handle:
            pd.DataFrame({"sample_id": ordered_ids, "label": ordered}).to_csv(handle, index=False)
        outputs.append(
            {
                "method": method,
                "filename": path.name,
                "sha256": ensemble.digest_file(path),
                "rows": len(frame),
            }
        )
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--sample_submission", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    outputs = predict_candidates(
        checkpoint=args.checkpoint,
        input_csv=args.input,
        image_root=args.image_root,
        template_path=args.sample_submission,
        output_dir=args.output_dir,
        methods=args.methods,
        device=args.device,
    )
    print(json.dumps(outputs, indent=2))
    print("Candidate CSVs saved; no training or leaderboard upload was performed.")


if __name__ == "__main__":
    main()
