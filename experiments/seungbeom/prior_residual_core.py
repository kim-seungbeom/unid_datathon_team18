"""Small bounded class offsets on top of the existing 30:70 + OOF prior rule.

No filesystem/data access. All fitting callers must supply Train OOF only.
This is a constrained experiment, not an implementation of a paper's guarantee.
"""

from __future__ import annotations

import numpy as np

from experiments.seungbeom import inner_best_prior30 as prior
from experiments.seungbeom.ensemble_core import validate_probabilities
from schema import LABELS, validate_labels

BASELINE = prior.METHOD
METHOD = "inner_best_30_70_prior_residual"
GRID = (0.0, -0.05, 0.05, -0.10, 0.10, -0.15, 0.15)
BOUND = 0.15
CONTRACT = {
    "classes": list(LABELS),
    "neural_weight": 0.3,
    "anchor": "train_oof_selected_prior_tau",
    "offset_grid": list(GRID),
    "coordinate_passes": 2,
    "penalty": 0.01,
    "objective": "macro_f1_minus_mean_squared_normalized_offset_penalty",
    "min_held_gain": 0.001,
    "worst_fold_delta_min": -0.01,
    "require_majority_fold_wins": True,
    "log_clip": 1e-12,
}


def fast_f1(y, labels):
    confusion = np.bincount((y - 1) * 6 + labels - 1, minlength=36).reshape(6, 6)
    denom = confusion.sum(0) + confusion.sum(1)
    return float(
        np.divide(2.0 * np.diag(confusion), denom, out=np.zeros(6), where=denom > 0).mean()
    )


def check_state(state):
    if state.get("contract") != CONTRACT:
        raise ValueError("Residual correction contract changed.")
    offset = np.asarray(state["offsets"], dtype=float)
    if offset.shape != (6,) or not np.isfinite(offset).all() or not np.isin(offset, GRID).all():
        raise ValueError("Offsets must use the fixed bounded six-class grid.")
    prior.corrected(np.full((1, 6), 1 / 6), state["prior"], state["tau"])
    return offset


def logits(p, prior_values, tau):
    p = validate_probabilities(np.asarray(p)[:, None, :])[:, 0]
    prior.corrected(p[:1], prior_values, tau)  # Validate prior/tau contract.
    return np.log(np.clip(p, CONTRACT["log_clip"], 1)) - tau * np.log(prior_values)


def apply(p, state):
    offsets = check_state(state)
    anchor = prior.corrected(p, state["prior"], state["tau"])
    # Preserve the exact existing ratio/prior argmax for a no-op correction.
    candidate = (
        anchor.copy()
        if not np.any(offsets)
        else (logits(p, state["prior"], state["tau"]) + offsets).argmax(1) + 1
    )
    return {BASELINE: anchor, METHOD: candidate}


def fit(p, labels):
    p = validate_probabilities(np.asarray(p)[:, None, :])[:, 0]
    y = validate_labels(labels)
    if len(p) != len(y) or set(y) != set(LABELS):
        raise ValueError("Aligned Train OOF with all six classes required.")
    counts = prior.prior_for(y)
    tau, _ = prior.choose(p, y, counts)
    base = logits(p, counts, tau)
    anchor = prior.corrected(p, counts, tau)
    offsets = np.zeros(6)

    def objective(values):
        pred = anchor if not np.any(values) else (base + values).argmax(1) + 1
        return fast_f1(y, pred) - CONTRACT["penalty"] * np.mean((values / BOUND) ** 2)

    best = objective(offsets)
    for _ in range(CONTRACT["coordinate_passes"]):
        changed = False
        for c in range(6):
            selected = offsets.copy()
            for value in GRID:
                proposal = offsets.copy()
                proposal[c] = value
                score = objective(proposal)
                if score > best + 1e-12:
                    best, selected = score, proposal
            if not np.array_equal(selected, offsets):
                changed = True
                offsets = selected
        if not changed:
            break
    state = {
        "contract": CONTRACT,
        "prior": counts.tolist(),
        "tau": tau,
        "offsets": offsets.tolist(),
    }
    check_state(state)
    return state


def crossfit(neural, tree, labels, folds):
    p = prior.mixture(neural, tree)
    y, folds = validate_labels(labels), np.asarray(folds)
    if (
        len(p) != len(y)
        or folds.shape != y.shape
        or folds.dtype.kind not in "iu"
        or len(np.unique(folds)) < 3
    ):
        raise ValueError("At least three aligned integer folds required.")
    held_predictions = {n: np.zeros(len(y), dtype=int) for n in (BASELINE, METHOD)}
    per_fold = []
    for f in np.unique(folds):
        held = folds == f
        print(f"Fitting bounded residual correction excluding OOF fold {f}...", flush=True)
        state = fit(p[~held], y[~held])
        predictions = apply(p[held], state)
        metrics = {n: prior.classification_metrics(y[held], v) for n, v in predictions.items()}
        for n, values in predictions.items():
            held_predictions[n][held] = values
        per_fold.append(
            {
                "fold": int(f),
                "state": state,
                "metrics": metrics,
                "delta": metrics[METHOD]["macro_f1"] - metrics[BASELINE]["macro_f1"],
            }
        )
    metrics = {n: prior.classification_metrics(y, v) for n, v in held_predictions.items()}
    delta = metrics[METHOD]["macro_f1"] - metrics[BASELINE]["macro_f1"]
    final = fit(p, y)
    final_metrics = {n: prior.classification_metrics(y, v) for n, v in apply(p, final).items()}
    wins = sum(f["delta"] > 1e-12 for f in per_fold)
    adopt = (
        delta >= CONTRACT["min_held_gain"]
        and wins > len(per_fold) / 2
        and min(f["delta"] for f in per_fold) >= CONTRACT["worst_fold_delta_min"]
        and any(final["offsets"])
    )
    return final, {
        "fold_excluded_diagnostic": metrics,
        "full_fit_diagnostic": final_metrics,
        "per_fold": per_fold,
        "delta": delta,
        "wins": wins,
        "adopt": bool(adopt),
        "limitations": "Prior/tau and offsets refitted excluding each held OOF fold. "
        "Base proxy heads were not refitted in an outer nested CV; reused architecture/epoch "
        "and OOF mean these are diagnostics, not independent generalization estimates.",
    }
