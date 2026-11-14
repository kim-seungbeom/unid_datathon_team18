"""Uni-D: 33개 baseline 특징을 받는 LightGBM 분류기."""
from __future__ import annotations
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from models import ShiftLabels


def build_lightgbm(seed: int = 42) -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('classifier', ShiftLabels(LGBMClassifier(
            objective='multiclass', num_class=6, n_estimators=300,
            learning_rate=0.05, num_leaves=15, max_depth=-1,
            min_child_samples=30, reg_lambda=2.0,
            verbosity=-1, random_state=seed, n_jobs=4
        )))
    ])
