"""Uni-D: 33개 baseline 특징을 받는 XGBoost 분류기."""
from __future__ import annotations
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from models import ShiftLabels


def build_xgboost(seed: int = 42) -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('classifier', ShiftLabels(XGBClassifier(
            objective='multi:softprob', num_class=6, n_estimators=300,
            max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.9, reg_lambda=2.0,
            tree_method='hist', random_state=seed, n_jobs=4,
            eval_metric='mlogloss'
        )))
    ])
