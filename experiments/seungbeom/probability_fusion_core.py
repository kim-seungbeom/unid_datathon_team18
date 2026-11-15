"""Small OOF-only fusion learners; six classes, neural probabilities then tree.

No dataset or filesystem access. Frozen base models never train here.
Temperature scaling uses clipped log probabilities because logits are not cached.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from experiments.seungbeom.ensemble_core import classification_metrics, validate_probabilities
from schema import LABELS, validate_labels

EPS = 1e-12
BOUNDS = (0.25, 4.0)
STACK_C = 0.1
METHODS = ("temperature_30_70", "logistic_stacking")
BASELINE = "inner_best_30_70"
CONTRACT = {
    "format_version": 1,
    "classes": list(LABELS),
    "feature_order": [f"{member}_{c}" for member in ("neural", "extra_trees") for c in LABELS],
    "neural_weight": 0.3,
    "temperature_bounds": list(BOUNDS),
    "temperature_objective": "each_member_train_oof_log_loss",
    "log_clip": EPS,
    "stacking": {
        "C": STACK_C,
        "solver": "lbfgs",
        "penalty": "l2",
        "max_iter": 2000,
        "tol": 1e-7,
        "class_weight": None,
        "seed": 42,
    },
    "prior_tau": None,
}


def members(neural, tree):
    return validate_probabilities(np.stack([neural, tree], axis=1))


def log_loss(labels, probabilities):
    y = validate_labels(labels)
    p = validate_probabilities(np.asarray(probabilities)[:, None, :])[:, 0]
    if len(y) != len(p):
        raise ValueError("Labels/probabilities differ in length.")
    return float(-np.log(np.clip(p[np.arange(len(y)), y - 1], EPS, 1)).mean())


def temperature(p, value):
    p = validate_probabilities(np.asarray(p)[:, None, :])[:, 0]
    if not np.isfinite(value) or not BOUNDS[0] <= value <= BOUNDS[1]:
        raise ValueError("Temperature outside fixed bounds.")
    if value == 1:
        return p.copy()  # exact identity baseline, including zero probabilities
    return softmax(np.log(np.clip(p, EPS, 1)) / float(value), axis=1)


def fit_temperature(p, y):
    result = minimize_scalar(
        lambda log_t: log_loss(y, temperature(p, float(np.exp(log_t)))),
        bounds=tuple(np.log(BOUNDS)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Temperature optimization did not converge.")
    options = (1.0, *BOUNDS, float(np.clip(np.exp(result.x), *BOUNDS)))
    scored = [(log_loss(y, temperature(p, t)), abs(np.log(t)), t) for t in options]
    best_loss, _, best = min(scored)
    baseline = log_loss(y, p)
    if best_loss >= baseline - 1e-12:
        best, best_loss = 1.0, baseline
    return {
        "T": best,
        "before_nll": baseline,
        "fit_nll": best_loss,
        "at_bound": bool(min(abs(best - b) for b in BOUNDS) < 1e-4),
    }


def stack_probabilities(neural, tree, state):
    p = members(neural, tree)
    coef = np.asarray(state["coef"], dtype=float)
    intercept = np.asarray(state["intercept"], dtype=float)
    if (
        state.get("classes") != list(LABELS)
        or coef.shape != (6, 12)
        or intercept.shape != (6,)
        or not np.isfinite(coef).all()
        or not np.isfinite(intercept).all()
    ):
        raise ValueError("Invalid six-class stacking state.")
    features = p.reshape(len(p), 12)
    # Rowwise sum avoids batch-size-dependent GEMM choices at decision boundaries.
    logits = np.sum(features[:, None, :] * coef[None, :, :], axis=2) + intercept
    return softmax(logits, axis=1)


def fit(neural, tree, labels):
    p = members(neural, tree)
    y = validate_labels(labels)
    if len(p) != len(y) or set(y) != set(LABELS):
        raise ValueError("Fit requires aligned rows and all six classes.")
    temperatures = [fit_temperature(p[:, i], y) for i in (0, 1)]
    # Default L2 is stable across the project's sklearn versions; avoid newer-only APIs.
    model = LogisticRegression(
        C=STACK_C, solver="lbfgs", max_iter=2000, tol=1e-7, class_weight=None, random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(p.reshape(len(p), 12), y)
    state = {
        "classes": model.classes_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }
    replay = stack_probabilities(p[:, 0], p[:, 1], state)
    if not np.allclose(replay, model.predict_proba(p.reshape(len(p), 12)), rtol=0, atol=1e-10):
        raise ValueError("Serialized logistic coefficients do not reproduce sklearn probabilities.")
    return {
        "contract": CONTRACT,
        "temperatures": temperatures,
        "stack": state,
        "stack_iterations": model.n_iter_.tolist(),
    }


def apply(neural, tree, state):
    if state.get("contract") != CONTRACT or len(state.get("temperatures", [])) != 2:
        raise ValueError("Fusion contract mismatch.")
    p = members(neural, tree)
    left = temperature(p[:, 0], state["temperatures"][0]["T"])
    right = temperature(p[:, 1], state["temperatures"][1]["T"])
    return {
        BASELINE: 0.3 * p[:, 0] + 0.7 * p[:, 1],
        METHODS[0]: 0.3 * left + 0.7 * right,
        METHODS[1]: stack_probabilities(p[:, 0], p[:, 1], state["stack"]),
    }


def score(labels, probabilities):
    return {
        "classification": classification_metrics(labels, probabilities.argmax(1) + 1),
        "log_loss": log_loss(labels, probabilities),
    }


def crossfit(neural, tree, labels, folds):
    """Second-level holdout diagnostic, NOT fully nested base-model validation."""
    p = members(neural, tree)
    y = validate_labels(labels)
    folds = np.asarray(folds)
    if (
        len(p) != len(y)
        or folds.shape != y.shape
        or folds.dtype.kind not in "iu"
        or len(np.unique(folds)) < 3
    ):
        raise ValueError("At least three aligned integer folds required.")
    predictions = {name: np.full((len(p), 6), np.nan) for name in (BASELINE, *METHODS)}
    per_fold = []
    for f in np.unique(folds):
        held = folds == f
        print(f"Fitting small fusion models on OOF folds excluding fold {f}...", flush=True)
        state = fit(p[~held, 0], p[~held, 1], y[~held])
        values = apply(p[held, 0], p[held, 1], state)
        for name, probs in values.items():
            predictions[name][held] = probs
        per_fold.append(
            {
                "fold": int(f),
                "rows": int(held.sum()),
                "temperatures": state["temperatures"],
                "metrics": {n: score(y[held], v) for n, v in values.items()},
            }
        )
    diagnostic = {n: score(y, v) for n, v in predictions.items()}
    print("Fitting final small fusion models on all Train OOF rows...", flush=True)
    final = fit(p[:, 0], p[:, 1], y)
    fitted = apply(p[:, 0], p[:, 1], final)
    return final, {
        "fold_excluded_diagnostic": diagnostic,
        "per_fold": per_fold,
        "full_fit_diagnostic": {n: score(y, v) for n, v in fitted.items()},
        "limitations": "Base OOF proxy models were reused, not refit inside outer folds. "
        "These are fitting diagnostics, not an independent nested CV score.",
    }
