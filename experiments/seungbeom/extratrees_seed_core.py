"""Reuse fitted preprocessing/PCA, fit only two new ExtraTrees classifiers."""

from __future__ import annotations

import copy

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from experiments.seungbeom import multimodal as mm
from experiments.seungbeom.ensemble_core import validate_probabilities
from schema import LABELS, validate_labels

SEEDS = (42, 43, 44)
BASELINE = "inner_best_30_70"
CANDIDATE = "inner_best_et_seed_mean_30_70"
CONTRACT = {
    "seeds": list(SEEDS),
    "neural_weight": 0.3,
    "tree_aggregation": "equal_probability_mean",
    "fit_split": "train",
    "preprocessing": "original_fitted_state_never_refit",
    "seed42": "original_saved_classifier_never_refit",
    "prior_tau": None,
    "temperature": None,
    "classes": list(LABELS),
}


def classifier_hash(classifier, *, normalize_seed=False):
    obj = clone(classifier)
    if normalize_seed:
        obj.set_params(estimator__random_state=42)
    return joblib.hash(obj)


def fitted_classifier_hash(classifier):
    """Hash semantic tree state, excluding uninitialized structured-array padding.

    sklearn Tree node arrays contain alignment padding. Copy/pickle round trips
    may change those bytes without changing any node field or prediction.
    """
    forest = classifier.estimator_
    trees = []
    for estimator in forest.estimators_:
        state = estimator.tree_.__getstate__()
        nodes = state["nodes"]
        trees.append(
            {
                "estimator": {k: v for k, v in vars(estimator).items() if k != "tree_"},
                "tree": {**state, "nodes": {n: nodes[n].copy() for n in nodes.dtype.names}},
                "n_features": estimator.tree_.n_features,
                "n_classes": estimator.tree_.n_classes.copy(),
                "n_outputs": estimator.tree_.n_outputs,
            }
        )
    return joblib.hash(
        {
            "wrapper": {k: v for k, v in vars(classifier).items() if k != "estimator_"},
            "forest": {k: v for k, v in vars(forest).items() if k != "estimators_"},
            "trees": trees,
        }
    )


def validate_original(pipeline):
    if [n for n, _ in pipeline.steps] != ["features", "classifier"]:
        raise ValueError("Use the original feature-transform + classifier pipeline.")
    c = pipeline.named_steps["classifier"]
    estimator = c.estimator_
    if (
        not isinstance(estimator, ExtraTreesClassifier)
        or estimator.random_state != 42
        or estimator.min_samples_leaf != 2
        or estimator.n_estimators != 400
        or estimator.max_features != 0.8
        or list(c.classes_) != list(LABELS)
        or estimator.get_params() != c.estimator.get_params()
    ):
        raise ValueError("Expected original seed42, leaf2, 400-tree ExtraTrees.")
    image = pipeline.named_steps["features"].named_transformers_["image"]
    if "pca" not in image.named_steps or not hasattr(image.named_steps["pca"], "components_"):
        raise ValueError("Original fitted PCA is required.")


def stable_transform(preprocessing, features):
    """Match existing stable inference's rowwise, single-BLAS-thread transform."""
    if len(features) == 0 or list(features.columns) != list(preprocessing.feature_names_in_):
        raise ValueError("Features must match the saved preprocessing columns.")
    before = joblib.hash(preprocessing)
    parts = []
    with threadpool_limits(limits=1):
        for i in range(len(features)):
            parts.append(preprocessing.transform(features.iloc[i : i + 1]))
            if (i + 1) % 4096 == 0:
                print(f"Frozen preprocessing: {i + 1}/{len(features)} rows", flush=True)
    result = np.concatenate(parts, axis=0)
    if not np.isfinite(result).all() or before != joblib.hash(preprocessing):
        raise ValueError("Nonfinite transformed features or preprocessing state mutation.")
    return result


def predict_classifier(classifier, transformed):
    estimator = classifier.estimator_
    jobs = estimator.n_jobs
    try:
        estimator.set_params(n_jobs=1)
        with threadpool_limits(limits=1):
            p = classifier.predict_proba(transformed)
    finally:
        estimator.set_params(n_jobs=jobs)
    return mm._align_probabilities(p, classifier.classes_, len(transformed))


def fit_members(original_pipeline, transformed, labels):
    validate_original(original_pipeline)
    y = validate_labels(labels)
    if len(y) != len(transformed) or set(y) != set(LABELS):
        raise ValueError("Aligned Train rows with all six classes required.")
    original_hash = joblib.hash(original_pipeline)
    original_classifier = original_pipeline.named_steps["classifier"]
    parameter_hash = classifier_hash(original_classifier)
    classifiers = {42: copy.deepcopy(original_classifier)}
    for seed in (43, 44):
        print(
            f"Fitting ExtraTrees seed={seed}; frozen preprocessing/PCA, Train only...", flush=True
        )
        classifier = clone(original_classifier).set_params(estimator__random_state=seed)
        classifier.fit(transformed, y)
        if classifier_hash(classifier, normalize_seed=True) != parameter_hash:
            raise ValueError("An estimator parameter other than random_state changed.")
        classifiers[seed] = classifier
    if joblib.hash(original_pipeline) != original_hash:
        raise ValueError("Original model was modified.")
    return classifiers


def candidates(neural, trees):
    if set(trees) != set(SEEDS):
        raise ValueError("Use original seed42 and new seeds43/44 exactly once.")
    p = validate_probabilities(np.stack([neural, *(trees[s] for s in SEEDS)], axis=1))
    mean = (p[:, 1] + p[:, 2] + p[:, 3]) / 3
    result = {"inner_best_neural": p[:, 0], BASELINE: 0.3 * p[:, 0] + 0.7 * p[:, 1]}
    result.update({f"extra_trees_seed{s}": p[:, i + 1] for i, s in enumerate(SEEDS)})
    result["extra_trees_seed_mean"] = mean
    result[CANDIDATE] = 0.3 * p[:, 0] + 0.7 * mean
    return result
