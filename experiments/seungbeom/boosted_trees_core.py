"""Bounded CPU CatBoost/LightGBM experiments on cached multimodal features."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from experiments.seungbeom.ensemble_core import validate_probabilities
from schema import RAW_SENSOR_COLUMNS, validate_labels

MODELS = ("lightgbm", "catboost")
BASELINE = "inner_best_30_70"
METHODS = (
    "catboost",
    "lightgbm",
    "inner_best_catboost_30_70",
    "inner_best_lightgbm_30_70",
    "inner_best_cat_lgbm_30_35_35",
)
CONTRACT = {
    "sensors": list(RAW_SENSOR_COLUMNS),
    "image_dimension": 512,
    "sensor_imputation": "train_median_all_missing_zero",
    "pca": None,
    "scaling": None,
    "classes": [1, 2, 3, 4, 5, 6],
    "early_stop_metric": "official_validation_macro_f1",
    "patience": 50,
    "seed": 42,
    "learning_rate": 0.05,
    "depth": 6,
    "neural_weight": 0.3,
    "combined_tree_weights": [0.35, 0.35],
    "prior_tau": None,
    "temperature": None,
}


def sensor_values(frame):
    return (
        frame[list(RAW_SENSOR_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=float)
    )


def fit_preprocessor(train):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(sensor_values(train), axis=0)
    return {"contract": CONTRACT, "medians": np.nan_to_num(medians, nan=0.0).tolist()}


def transform(frame, vectors, state):
    medians = np.asarray(state.get("medians"), dtype=float)
    vectors = np.asarray(vectors)
    if (
        state.get("contract") != CONTRACT
        or medians.shape != (5,)
        or not np.isfinite(medians).all()
        or vectors.shape != (len(frame), 512)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Invalid multimodal feature/preprocessing contract.")
    sensors = sensor_values(frame)
    sensors = np.where(np.isnan(sensors), medians, sensors)
    return np.ascontiguousarray(np.concatenate([sensors, vectors], axis=1), dtype=np.float64)


def macro_eval(y, probabilities):
    values = np.asarray(probabilities)
    if values.shape != (len(y), 6):
        raise ValueError("Expected LightGBM six-class probabilities.")
    return (
        "macro_f1",
        float(f1_score(y, values.argmax(1), labels=range(6), average="macro", zero_division=0)),
        True,
    )


class CatBudget:
    def __init__(self, seconds):
        self.start, self.seconds, self.stopped = time.monotonic(), seconds, False

    def after_iteration(self, info):
        self.stopped = time.monotonic() - self.start >= self.seconds
        if self.stopped:
            print("CatBoost time budget reached; retaining best completed iteration.", flush=True)
        return not self.stopped


class LightBudget:
    order = 25
    before_iteration = False

    def __init__(self, seconds):
        self.start, self.seconds, self.stopped = time.monotonic(), seconds, False
        self.best, self.iteration, self.scores = -np.inf, 0, None

    def __call__(self, env):
        from lightgbm.callback import EarlyStopException

        scores = env.evaluation_result_list
        if len(scores) != 1 or scores[0][1] != "macro_f1":
            raise ValueError("Only Validation Macro F1 may select the LightGBM iteration.")
        if scores[0][2] > self.best:
            self.best, self.iteration, self.scores = scores[0][2], env.iteration, list(scores)
        if time.monotonic() - self.start >= self.seconds:
            self.stopped = True
            print("LightGBM time budget reached; retaining best completed iteration.", flush=True)
            raise EarlyStopException(self.iteration, self.scores)


def fit_model(name, x, y, val_x, val_y, directory, *, iterations=400, threads=4, seconds=720):
    """Validation selects iteration only; gradient fitting uses Train rows only."""
    if name not in MODELS or not 1 <= iterations <= 2000 or not 1 <= threads <= 8 or seconds <= 0:
        raise ValueError("Invalid model/training budget.")
    y, val_y = validate_labels(y) - 1, validate_labels(val_y) - 1
    if set(y) != set(range(6)) or set(val_y) != set(range(6)):
        raise ValueError("Both splits must contain all six classes for Macro F1 selection.")
    started = time.monotonic()
    if name == "catboost":
        from catboost import CatBoostClassifier

        budget = CatBudget(seconds)
        model = CatBoostClassifier(
            iterations=iterations,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            l2_leaf_reg=5,
            random_seed=42,
            thread_count=threads,
            task_type="CPU",
            classes_count=6,
            allow_writing_files=False,
            verbose=25,
        )
        model.fit(
            x,
            y,
            eval_set=(val_x, val_y),
            early_stopping_rounds=50,
            use_best_model=True,
            callbacks=[budget],
        )
        best = int(model.get_best_iteration()) + 1
        path = directory / "catboost.cbm"
        model.save_model(str(path))
        probabilities = model.predict_proba(val_x, thread_count=threads)
        if best < 1 or model.tree_count_ != best or list(model.classes_) != list(range(6)):
            raise ValueError("CatBoost did not retain the best six-class checkpoint.")
    else:
        import lightgbm as lgb

        budget = LightBudget(seconds)
        model = lgb.LGBMClassifier(
            n_estimators=iterations,
            num_leaves=31,
            max_depth=6,
            learning_rate=0.05,
            objective="multiclass",
            num_class=6,
            metric="None",
            min_child_samples=30,
            reg_lambda=5,
            max_bin=127,
            random_state=42,
            n_jobs=threads,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(
            x,
            y,
            eval_set=[(val_x, val_y)],
            eval_metric=macro_eval,
            callbacks=[
                budget,
                lgb.early_stopping(50, first_metric_only=True),
                lgb.log_evaluation(25),
            ],
        )
        best = int(model.best_iteration_)
        if best < 1 or list(model.classes_) != list(range(6)):
            raise ValueError("LightGBM did not select a six-class checkpoint.")
        path = directory / "lightgbm.txt"
        model.booster_.save_model(str(path), num_iteration=best)
        probabilities = model.booster_.predict(val_x, num_iteration=best, num_threads=threads)
    record = {
        "name": name,
        "file": path.name,
        "best_iteration": best,
        "fit_seconds": time.monotonic() - started,
        "budget_reached": budget.stopped,
        "iterations_limit": iterations,
        "threads": threads,
        "seconds_limit": seconds,
    }
    replay = predict_model(name, path, val_x, threads=threads)
    if not np.array_equal(np.asarray(probabilities), replay):
        raise ValueError("Native saved model does not exactly replay Validation probabilities.")
    record["validation_macro_f1"] = float(
        f1_score(val_y, replay.argmax(1), labels=range(6), average="macro", zero_division=0)
    )
    return record, replay


def predict_model(name, path, x, *, threads=4):
    if name == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier().load_model(str(path))
        if list(model.classes_) != list(range(6)):
            raise ValueError("CatBoost class mapping changed.")
        p = model.predict_proba(x, thread_count=threads)
    elif name == "lightgbm":
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(path))
        if model.num_model_per_iteration() != 6:
            raise ValueError("LightGBM class count changed.")
        p = model.predict(x, num_threads=threads)
    else:
        raise ValueError("Unknown tree model.")
    return validate_probabilities(np.asarray(p)[:, None, :])[:, 0]


def candidates(neural, tree, cat, light):
    p = validate_probabilities(np.stack([neural, tree, cat, light], axis=1))
    n, t, c, light = (p[:, i] for i in range(4))
    return {
        BASELINE: 0.3 * n + 0.7 * t,
        "extra_trees": t,
        "catboost": c,
        "lightgbm": light,
        "inner_best_catboost_30_70": 0.3 * n + 0.7 * c,
        "inner_best_lightgbm_30_70": 0.3 * n + 0.7 * light,
        "inner_best_cat_lgbm_30_35_35": 0.3 * n + 0.35 * c + 0.35 * light,
    }
