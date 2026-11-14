"""운영진 v7 train.py의 로지스틱 회귀를 그대로 사용한다.

정의는 baselines/organizer_v7/model.py의 build_model(class_weight)
(median 대체 → StandardScaler → LogisticRegression(C=1, lbfgs, max_iter=3000, seed 42)).
v7 train.py처럼 class_weight=None / 'balanced' 두 가지를 별도 모델 이름으로 제공한다.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from organizer_v7_model import build_model as _organizer_build_model


def build_logistic(seed: int = 42, class_weight=None) -> Pipeline:
    # v7 정의는 random_state=SEED(42)로 고정되어 있어 seed 인자는 사용하지 않는다(lbfgs는 결정적).
    return _organizer_build_model(class_weight)
