"""Bounded row-local mixture around the existing 30:70 prior-corrected model.

Entropy is a confidence proxy, not a correctness probability. A single signed
strength is chosen on Train OOF; tau remains the raw30 anchor's OOF choice.
"""

from __future__ import annotations

import numpy as np

from experiments.seungbeom import inner_best_prior30 as prior
from experiments.seungbeom.ensemble_core import validate_probabilities
from schema import LABELS, validate_labels

BASELINE = prior.METHOD
METHOD = "inner_best_confidence_prior_tau"
STRENGTHS = (0.0, -0.5, 0.5, -1.0, 1.0)
CONTRACT = {
    "classes": list(LABELS),
    "base_neural_weight": 0.3,
    "max_weight_shift": 0.1,
    "strength_grid": list(STRENGTHS),
    "entropy": "shannon_div_log6",
    "rule": "alpha=clip(0.3+0.1*strength*(H_tree-H_neural),0.2,0.4)",
    "objective": "train_oof_macro_f1_minus_0.001_strength_squared",
    "tau": "selected_from_raw30_train_oof_then_fixed_for_strength_search",
    "penalty": 0.001,
    "recommended_gain": 0.001,
    "worst_fold_delta_min": -0.01,
    "entropy_clip": 1e-12,
}


def members(neural, tree):
    return validate_probabilities(np.stack([neural, tree], axis=1))


def normalized_entropy(probabilities):
    p = validate_probabilities(np.asarray(probabilities)[:, None, :])[:, 0]
    return np.clip(
        -np.sum(p * np.log(np.clip(p, CONTRACT["entropy_clip"], 1)), axis=1) / np.log(6.0), 0.0, 1.0
    )


def mixture(neural, tree, strength):
    p = members(neural, tree)
    if strength not in STRENGTHS:
        raise ValueError("Strength must be chosen from the predefined grid.")
    if strength == 0:
        return prior.mixture(neural, tree), np.full(len(p), 0.3)
    h_neural, h_tree = normalized_entropy(p[:, 0]), normalized_entropy(p[:, 1])
    alpha = np.clip(0.3 + 0.1 * strength * (h_tree - h_neural), 0.2, 0.4)
    return alpha[:, None] * p[:, 0] + (1 - alpha[:, None]) * p[:, 1], alpha


def check_state(state):
    if state.get("contract") != CONTRACT or state.get("strength") not in STRENGTHS:
        raise ValueError("Confidence blend state/contract changed.")
    prior.corrected(np.full((1, 6), 1 / 6), state["prior"], state["tau"])


def apply(neural, tree, state):
    check_state(state)
    raw = prior.mixture(neural, tree)
    adjusted, _ = mixture(neural, tree, state["strength"])
    return {
        BASELINE: prior.corrected(raw, state["prior"], state["tau"]),
        METHOD: prior.corrected(adjusted, state["prior"], state["tau"]),
    }


def fit(neural, tree, labels):
    p = members(neural, tree)
    y = validate_labels(labels)
    if len(y) != len(p) or set(y) != set(LABELS):
        raise ValueError("Train OOF with aligned rows and all six classes required.")
    raw = prior.mixture(neural, tree)
    prior_values = prior.prior_for(y)
    tau, _ = prior.choose(raw, y, prior_values)
    best, chosen, grid = -np.inf, 0.0, []
    for strength in STRENGTHS:
        adjusted, _ = mixture(neural, tree, strength)
        prediction = prior.corrected(adjusted, prior_values, tau)
        f1 = prior.classification_metrics(y, prediction)["macro_f1"]
        objective = f1 - CONTRACT["penalty"] * strength**2
        grid.append({"strength": strength, "macro_f1": f1, "objective": objective})
        if objective > best + 1e-12:
            best, chosen = objective, strength
    return {
        "contract": CONTRACT,
        "prior": prior_values.tolist(),
        "tau": tau,
        "strength": chosen,
        "fit_grid": grid,
    }


def crossfit(neural, tree, labels, folds):
    p = members(neural, tree)
    y, folds = validate_labels(labels), np.asarray(folds)
    if (
        len(y) != len(p)
        or folds.shape != y.shape
        or folds.dtype.kind not in "iu"
        or len(np.unique(folds)) < 3
    ):
        raise ValueError("At least three aligned integer OOF folds required.")
    predictions = {n: np.zeros(len(y), dtype=int) for n in (BASELINE, METHOD)}
    per_fold = []
    for fold in np.unique(folds):
        held = folds == fold
        print(f"Selecting confidence strength excluding OOF fold {fold}...", flush=True)
        state = fit(p[~held, 0], p[~held, 1], y[~held])
        results = apply(p[held, 0], p[held, 1], state)
        metrics = {n: prior.classification_metrics(y[held], v) for n, v in results.items()}
        for n, v in results.items():
            predictions[n][held] = v
        per_fold.append(
            {
                "fold": int(fold),
                "state": state,
                "metrics": metrics,
                "delta": metrics[METHOD]["macro_f1"] - metrics[BASELINE]["macro_f1"],
            }
        )
    diagnostic = {n: prior.classification_metrics(y, v) for n, v in predictions.items()}
    final = fit(neural, tree, y)
    fitted = {n: prior.classification_metrics(y, v) for n, v in apply(neural, tree, final).items()}
    delta = diagnostic[METHOD]["macro_f1"] - diagnostic[BASELINE]["macro_f1"]
    wins = sum(f["delta"] > 1e-12 for f in per_fold)
    return final, {
        "fold_excluded_diagnostic": diagnostic,
        "full_fit_diagnostic": fitted,
        "per_fold": per_fold,
        "delta": delta,
        "wins": wins,
        "recommended": bool(
            delta >= CONTRACT["recommended_gain"]
            and wins > len(per_fold) / 2
            and min(f["delta"] for f in per_fold) >= CONTRACT["worst_fold_delta_min"]
            and final["strength"] != 0
        ),
        "limitations": "Prior/tau and confidence strength fitted excluding held OOF fold; "
        "base proxy models and earlier epoch/architecture choices are reused, not nested CV. "
        "Entropy may reflect miscalibration rather than correctness. Validation is reused.",
    }
