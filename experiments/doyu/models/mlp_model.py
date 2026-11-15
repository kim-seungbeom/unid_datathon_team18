"""Uni-D: 33개 baseline 특징을 받는 MLP 분류기."""
from __future__ import annotations
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mlp(seed: int = 42) -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu',
            alpha=0.01, batch_size=128, learning_rate_init=0.001,
            max_iter=250, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, random_state=seed
        ))
    ])
