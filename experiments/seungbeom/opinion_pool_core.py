"""Fixed arithmetic/geometric probability pools; no fitting or data access.

Logarithmic opinion pool: normalized p_neural**0.3 * p_tree**0.7.
Reference: Heskes, Selecting Weighting Factors in Logarithmic Opinion Pools (1997).
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

from experiments.seungbeom.probability_fusion_core import members, score
from schema import validate_labels

BASELINE = "inner_best_30_70"
METHODS = ("geometric_30_70", "arithmetic_geometric_half")
CONTRACT = {
    "format_version": 1,
    "classes": [1, 2, 3, 4, 5, 6],
    "member_order": ["neural", "extra_trees"],
    "neural_weight": 0.3,
    "log_clip": 1e-12,
    "hybrid_geometric_weight": 0.5,
    "temperature": None,
    "prior_tau": None,
    "selection": "fixed_before_oof_and_validation_comparison",
}


def apply(neural, tree, state):
    if state != {"contract": CONTRACT}:
        raise ValueError("Opinion pool contract mismatch.")
    p = members(neural, tree)
    arithmetic = 0.3 * p[:, 0] + 0.7 * p[:, 1]
    logs = np.log(np.clip(p, CONTRACT["log_clip"], 1.0))
    geometric = softmax(0.3 * logs[:, 0] + 0.7 * logs[:, 1], axis=1)
    return {
        BASELINE: arithmetic,
        METHODS[0]: geometric,
        METHODS[1]: 0.5 * arithmetic + 0.5 * geometric,
    }


def evaluate_oof(neural, tree, labels, folds):
    """Evaluate fixed rules on existing proxy OOF. Never fit/select a parameter."""
    y = validate_labels(labels)
    folds = np.asarray(folds)
    if folds.shape != y.shape or folds.dtype.kind not in "iu" or len(np.unique(folds)) < 3:
        raise ValueError("At least three aligned integer folds required.")
    state = {"contract": CONTRACT}
    values = apply(neural, tree, state)
    if len(values[BASELINE]) != len(y):
        raise ValueError("OOF labels and probabilities differ in length.")
    per_fold = [
        {
            "fold": int(f),
            "rows": int(np.count_nonzero(folds == f)),
            "metrics": {n: score(y[folds == f], p[folds == f]) for n, p in values.items()},
        }
        for f in np.unique(folds)
    ]
    metrics = {n: score(y, p) for n, p in values.items()}
    return state, {
        "pooled_oof": metrics,
        "per_fold": per_fold,
        "fold_mean": {
            n: float(np.mean([f["metrics"][n]["classification"]["macro_f1"] for f in per_fold]))
            for n in values
        },
        "limitations": "Fixed rules; no fusion fitting. Existing proxy-head OOF and reused "
        "epoch/architecture choices are not an independent blind evaluation. Fold means "
        "and pooled OOF F1 are distinct; neither predicts Public F1.",
    }
