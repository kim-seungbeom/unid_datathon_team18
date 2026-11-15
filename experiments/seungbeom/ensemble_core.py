"""Classwise mixtures fitted exclusively to caller-supplied Train OOF predictions.

The class axis is always labels 1..6. Each class has a separate convex mixture
over models. The six resulting scores do not necessarily sum to one, so they
must not be described as calibrated probabilities. Prediction is argmax + 1.
"""

from __future__ import annotations

import numpy as np

LABELS = (1, 2, 3, 4, 5, 6)
_N_CLASSES = len(LABELS)
_STEPS = (0.5, 0.25, 0.1, 0.05)
_MAX_EVALUATIONS = 4096
_EPS = 1e-12


def validate_probabilities(probabilities) -> np.ndarray:
    """Require [samples, models, 6] probabilities with an aligned class axis."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or min(values.shape[:2]) < 1 or values.shape[2] != _N_CLASSES:
        raise ValueError("Probabilities must have nonempty shape [samples, models, 6].")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite and between zero and one.")
    if not np.allclose(values.sum(axis=2), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("Each sample/model probability vector must sum to one.")
    return values


def validate_weights(weights, n_models: int | None = None) -> np.ndarray:
    """Require a convex mixture of models separately for each of six classes."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != _N_CLASSES:
        raise ValueError("Weights must have nonempty shape [models, 6].")
    if n_models is not None and values.shape[0] != n_models:
        raise ValueError("Weights do not match the number of models.")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Weights must be finite and between zero and one.")
    if not np.allclose(values.sum(axis=0), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Weights for each class must sum to one across models.")
    return values


def blend_scores(probabilities, weights) -> np.ndarray:
    """Return six class scores, using no other samples when combining a row."""
    values = validate_probabilities(probabilities)
    mixture = validate_weights(weights, values.shape[1])
    return np.einsum("nmc,mc->nc", values, mixture, optimize=False)


def predict_labels(probabilities, weights) -> np.ndarray:
    """Predict 1..6 labels; exact ties consistently select the smaller label."""
    return blend_scores(probabilities, weights).argmax(axis=1) + 1


def _labels(values, name: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or numeric.size == 0 or not np.isin(numeric, LABELS).all():
        raise ValueError(f"{name} must be a nonempty one-dimensional array of labels 1..6.")
    return numeric.astype(np.int64)


def _confusion(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount((true - 1) * _N_CLASSES + pred - 1, minlength=36).reshape(6, 6)


def _f1(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return np.divide(2.0 * np.diag(matrix), denominator, out=np.zeros(6), where=denominator != 0)


def classification_metrics(y_true, y_pred) -> dict:
    """Root-compatible six-class metrics; confusion rows are actual classes."""
    true, pred = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError("Labels and predictions must have the same length.")
    matrix = _confusion(true, pred)
    tp, predicted, support = np.diag(matrix), matrix.sum(axis=0), matrix.sum(axis=1)
    f1 = _f1(matrix)
    precision = np.divide(tp, predicted, out=np.zeros(6), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(6), where=support != 0)
    return {
        "macro_f1": float(f1.mean()),
        "class_f1": {str(label): float(f1[label - 1]) for label in LABELS},
        "class_precision": {str(label): float(precision[label - 1]) for label in LABELS},
        "class_recall": {str(label): float(recall[label - 1]) for label in LABELS},
        "support": {str(label): int(support[label - 1]) for label in LABELS},
        "confusion_matrix": matrix.tolist(),
    }


def fit_classwise_weights(
    probabilities,
    labels,
    *,
    seed: int = 42,
    regularization: float = 0.001,
    max_rounds: int = 3,
) -> tuple[np.ndarray, dict]:
    """Optimize Macro F1 on Train OOF predictions, with a bounded local search.

    Caller must provide predictions made without fitting the corresponding
    sample's experiment, including preprocessing. No Validation/Test arguments
    are accepted. These OOF scores are fitting diagnostics, not unbiased scores
    of the fitted ensemble.

    The objective is Macro F1 minus regularization * mean_over_classes(sum_over_
    models((weight - 1 / n_models)**2)). Each move transfers weight between two
    models within one class. Search starts at equal averaging and the best
    single model, trying fixed step sizes with a seeded traversal order.
    A fallback ensures the returned weights never have lower OOF Macro F1 than
    both equal averaging and the best single model. This is not a guarantee of
    better performance on unseen experiments.
    """
    values = validate_probabilities(probabilities)
    true = _labels(labels, "labels")
    if len(true) != len(values):
        raise ValueError("Labels and probabilities must have the same number of samples.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if type(max_rounds) is not int or not 1 <= max_rounds <= 20:
        raise ValueError("max_rounds must be an integer in [1, 20].")
    if (
        type(regularization) not in (int, float)
        or not np.isfinite(regularization)
        or regularization < 0
    ):
        raise ValueError("regularization must be a finite nonnegative number.")

    n_models = values.shape[1]
    equal = np.full((n_models, 6), 1.0 / n_models)
    equal_scores = values.mean(axis=1)
    equal_metrics = classification_metrics(true, equal_scores.argmax(axis=1) + 1)
    single_metrics = [
        classification_metrics(true, values[:, index].argmax(axis=1) + 1)
        for index in range(n_models)
    ]
    best_single = int(np.argmax([metric["macro_f1"] for metric in single_metrics]))
    single_weights = np.zeros_like(equal)
    single_weights[best_single, :] = 1.0
    minimum_f1 = max(equal_metrics["macro_f1"], single_metrics[best_single]["macro_f1"])

    def penalty(weights):
        return float(np.square(weights - equal).sum(axis=0).mean())

    def objective(score, weights):
        return float(score - regularization * penalty(weights))

    if equal_metrics["macro_f1"] >= single_metrics[best_single]["macro_f1"]:
        chosen = equal.copy()
    else:
        chosen = single_weights.copy()
    chosen_f1 = minimum_f1
    chosen_objective = objective(chosen_f1, chosen)
    rng = np.random.default_rng(seed)
    evaluations = 0
    starts = [equal] if n_models == 1 else [equal, single_weights]

    for start in starts:
        current = start.copy()
        scores = np.einsum("nmc,mc->nc", values, current, optimize=False)
        current_f1 = float(_f1(_confusion(true, scores.argmax(axis=1) + 1)).mean())
        current_objective = objective(current_f1, current)
        for step in _STEPS:
            for _ in range(max_rounds):
                changed = False
                for column in rng.permutation(6):
                    for donor in rng.permutation(n_models):
                        for receiver in rng.permutation(n_models):
                            if donor == receiver or current[donor, column] <= _EPS:
                                continue
                            if evaluations >= _MAX_EVALUATIONS:
                                break
                            amount = min(step, current[donor, column])
                            candidate = current.copy()
                            candidate[donor, column] -= amount
                            candidate[receiver, column] += amount
                            candidate[:, column] = np.clip(candidate[:, column], 0.0, 1.0)
                            candidate[:, column] /= candidate[:, column].sum()
                            candidate_scores = scores.copy()
                            candidate_scores[:, column] = np.einsum(
                                "nm,m->n",
                                values[:, :, column],
                                candidate[:, column],
                                optimize=False,
                            )
                            candidate_f1 = float(
                                _f1(_confusion(true, candidate_scores.argmax(axis=1) + 1)).mean()
                            )
                            candidate_objective = objective(candidate_f1, candidate)
                            evaluations += 1
                            if candidate_objective > current_objective + _EPS:
                                current = candidate
                                scores = candidate_scores
                                current_f1 = candidate_f1
                                current_objective = candidate_objective
                                changed = True
                                if (
                                    current_f1 + _EPS >= minimum_f1
                                    and current_objective > chosen_objective + _EPS
                                ):
                                    chosen = current.copy()
                                    chosen_f1 = current_f1
                                    chosen_objective = current_objective
                        if evaluations >= _MAX_EVALUATIONS:
                            break
                    if evaluations >= _MAX_EVALUATIONS:
                        break
                if not changed or evaluations >= _MAX_EVALUATIONS:
                    break
            if evaluations >= _MAX_EVALUATIONS:
                break
        if evaluations >= _MAX_EVALUATIONS:
            break

    # Recompute with the exact inference arithmetic instead of accumulated score
    # updates. Near a decision tie, rounding can otherwise change a prediction.
    final_metrics = classification_metrics(true, predict_labels(values, chosen))
    if final_metrics["macro_f1"] + _EPS < minimum_f1:
        chosen = equal.copy() if equal_metrics["macro_f1"] >= minimum_f1 else single_weights
        final_metrics = classification_metrics(true, predict_labels(values, chosen))
    if np.array_equal(chosen, equal):
        selected_strategy = "equal_average"
    elif np.array_equal(chosen, single_weights):
        selected_strategy = "best_single"
    else:
        selected_strategy = "classwise"
    report = {
        "parameters": {
            "seed": seed,
            "regularization": float(regularization),
            "max_rounds": max_rounds,
            "coordinate_steps": list(_STEPS),
            "max_evaluations": _MAX_EVALUATIONS,
        },
        "oof_comparisons": {
            "equal_average": equal_metrics,
            "single_models": single_metrics,
            "classwise": final_metrics,
        },
        "best_single_model_index": best_single,
        "selected_strategy": selected_strategy,
        "objective": objective(final_metrics["macro_f1"], chosen),
        "evaluations": evaluations,
        "search_budget_exhausted": evaluations >= _MAX_EVALUATIONS,
        "selection_data": "train_oof_only",
        "interpretation": (
            "OOF metrics here were used to fit/select ensemble weights; evaluate the fixed "
            "ensemble on independent held-out experiments. Class scores are not calibrated "
            "probabilities. Improvement on unseen data is not guaranteed."
        ),
    }
    return validate_weights(chosen, n_models), report
