"""Diagnose a failed leaf replay using saved trees and the original Validation cache.

No fitting, raw-image inference, Test access, CSV export or replay bypass. This
is a cache-based diagnostic, not a substitute for live-image Validation replay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom import extratrees_leaf_sweep as sweep  # noqa: E402
from experiments.seungbeom.export_submissions import read_json, sha256, write_json  # noqa: E402
from schema import read_frame  # noqa: E402


def array_difference(left, right):
    """Aggregate numerical diagnostics only; never expose individual observations."""
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        return {"same_shape": False, "exact": False, "max_abs": None}
    exact = bool(np.array_equal(a, b, equal_nan=True))
    if a.dtype.kind in "biuf" and b.dtype.kind in "biuf":
        if np.array_equal(np.isfinite(a), np.isfinite(b)):
            finite = np.isfinite(a)
            delta = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
            maximum = float(delta.max()) if delta.size else 0.0
        else:
            maximum = None
    else:
        maximum = None
    return {"same_shape": True, "exact": exact, "max_abs": maximum}


def fitted_state_difference(original, refitted):
    left, right = (model.named_steps["features"] for model in (original, refitted))
    preprocessing = {}
    for name in ("sensor", "image"):
        a, b = left.named_transformers_[name], right.named_transformers_[name]
        components = (
            [(name, a, b)]
            if name == "sensor"
            else [
                (f"image.{step}", a.named_steps[step], b.named_steps[step])
                for step in a.named_steps
            ]
        )
        for prefix, a, b in components:
            for attr in (
                "statistics_",
                "mean_",
                "scale_",
                "var_",
                "components_",
                "explained_variance_",
                "singular_values_",
                "n_samples_seen_",
            ):
                if hasattr(a, attr) or hasattr(b, attr):
                    if not hasattr(a, attr) or not hasattr(b, attr):
                        preprocessing[f"{prefix}.{attr}"] = {
                            "same_shape": False,
                            "exact": False,
                            "max_abs": None,
                        }
                    else:
                        preprocessing[f"{prefix}.{attr}"] = array_difference(
                            getattr(a, attr), getattr(b, attr)
                        )
    a, b = (
        model.named_steps["classifier"].estimator_.estimators_ for model in (original, refitted)
    )
    fields = (
        "children_left",
        "children_right",
        "feature",
        "threshold",
        "n_node_samples",
        "weighted_n_node_samples",
        "value",
        "impurity",
        "missing_go_to_left",
    )
    trees = {"same_count": len(a) == len(b), "different_tree_count": 0, "fields": {}}
    for one, two in zip(a, b, strict=False):
        changed = False
        for field in fields:
            if not hasattr(one.tree_, field) and not hasattr(two.tree_, field):
                continue
            if not hasattr(one.tree_, field) or not hasattr(two.tree_, field):
                diff = {"same_shape": False, "exact": False, "max_abs": None}
            else:
                diff = array_difference(getattr(one.tree_, field), getattr(two.tree_, field))
            item = trees["fields"].setdefault(
                field, {"different_trees": 0, "shape_mismatches": 0, "max_abs": 0.0}
            )
            item["different_trees"] += int(not diff["exact"])
            item["shape_mismatches"] += int(not diff["same_shape"])
            if diff["max_abs"] is not None:
                item["max_abs"] = max(item["max_abs"], diff["max_abs"])
            changed |= not diff["exact"]
        trees["different_tree_count"] += int(changed)
    return {"preprocessing": preprocessing, "trees": trees}


def probability_difference(original, refitted):
    if original.shape != refitted.shape or original.ndim != 2 or original.shape[1] != 6:
        raise ValueError("Expected matching [rows, 6] probability arrays.")
    if not np.isfinite(original).all() or not np.isfinite(refitted).all():
        raise ValueError("Probabilities must be finite.")
    delta = np.abs(original - refitted)
    return {
        "exact": bool(np.array_equal(original, refitted)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "changed_probability_elements": int(np.count_nonzero(delta)),
        "changed_tree_predictions": int(np.count_nonzero(original.argmax(1) != refitted.argmax(1))),
    }


def audit(args):
    import joblib

    if args.output_dir.exists():
        raise ValueError("Choose a new output_dir; existing runs are preserved.")
    status = read_json(args.failed_run / "status.json")
    if status != {"status": "failed", "error_type": "ValueError"}:
        raise ValueError("Use the failed ValueError run, not a running/completed run.")
    prior = read_json(args.failed_run / "validation_metrics.json")
    reference = sweep.frozen_blend.read_comparison(args.recipe)
    legacy, _ = sweep.load_reference(reference, args.device)
    expected = reference["base_recipe"]["provenance"]["validation_csv_sha256"]
    if sha256(args.validation_csv) != expected:
        raise ValueError("Use the recorded official Validation CSV, not Train/Test.")
    recorded = legacy["identity"]["input_provenance"]["validation"]
    if {name: sha256(args.validation_embeddings / name) for name in sweep.CACHE_FILES} != recorded:
        raise ValueError("Use the original recorded Validation embedding cache.")
    paths = [
        args.recipe,
        args.recipe.parent / "status.json",
        args.validation_csv,
        args.failed_run / "status.json",
        args.failed_run / "validation_metrics.json",
        args.failed_run / "leaf2.joblib",
        *(args.validation_embeddings / name for name in sweep.CACHE_FILES),
    ]
    files = {str(path.resolve()): sha256(path) for path in paths}
    sources = {
        **sweep.source_hashes(),
        str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
    }
    original = sweep.tree_template(legacy)
    refitted = joblib.load(args.failed_run / "leaf2.joblib")
    if (
        sweep.parameter_fingerprint(original) != sweep.parameter_fingerprint(refitted)
        or list(refitted.feature_names_in_) != legacy["bundle"]["columns"]
    ):
        raise ValueError("The saved leaf2 parameters/feature contract differ from the original.")
    frame = read_frame(args.validation_csv, labeled=True)
    features = sweep.feature_matrix(
        frame, legacy["bundle"], cache=args.validation_embeddings, device=args.device
    )
    a = sweep.mm.predict_proba_six_stable(original, features)
    b = sweep.mm.predict_proba_six_stable(refitted, features)
    metrics = {
        name: sweep.classification_metrics(frame["label"], values.argmax(1) + 1)
        for name, values in (("original", a), ("refitted", b))
    }
    cache_matches_live_metrics = {
        "original": metrics["original"] == reference["metrics"]["extra_trees"],
        "refitted": metrics["refitted"] == prior["metrics"]["extra_trees_leaf2"],
    }
    report = {
        "kind": "seungbeom_leaf_replay_audit",
        "scope": "validation_cache_diagnostic_only",
        "prior_replay_matches": prior["replay_matches"],
        "cache_matches_saved_live_metrics": cache_matches_live_metrics,
        "probabilities": probability_difference(a, b),
        "fitted_state": fitted_state_difference(original, refitted),
        "input_sha256": files,
        "source_sha256": sources,
        "note": (
            "No replay guard was bypassed. Cached metrics do not verify per-row live inference."
        ),
    }
    sweep.check_unchanged(reference, sweep.source_hashes(), files, args.device)
    if sources != {
        **sweep.source_hashes(),
        str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
    }:
        raise ValueError("Source changed during audit.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "audit.json", report)
    print("Prior replay:", report["prior_replay_matches"])
    print("Cache reproduces saved live aggregate metrics:", cache_matches_live_metrics)
    print("Probability differences:", report["probabilities"])
    print("Preprocessing differences:")
    for name, item in report["fitted_state"]["preprocessing"].items():
        print(name, item)
    print("Tree state differences:", report["fitted_state"]["trees"])
    print("Audit saved:", args.output_dir / "audit.json")
    print("No fitting/image extraction/Test access/export. Original failed run is unchanged.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "failed_run", "validation_csv", "validation_embeddings", "output_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    audit(parser.parse_args(argv))


if __name__ == "__main__":
    main()
