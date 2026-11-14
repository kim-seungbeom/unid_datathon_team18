"""Uni-D: 동일한 33개 baseline 특징을 받는 3개 분류기 공통 진입점.

모델 정의는 각 파일에 있습니다.
    models/xgboost_model.py  -> build_xgboost
    models/lightgbm_model.py -> build_lightgbm
    models/mlp_model.py      -> build_mlp
    models/logistic_model.py -> build_logistic (운영진 v7 로지스틱, class_weight none/balanced)
    models/cross_attention_model.py -> CrossAttentionNet (train_cross_attention.py 전용)

ShiftLabels는 이미 저장된 joblib(models.ShiftLabels로 피클됨)이 계속 열리도록 여기에 둡니다.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class ShiftLabels(BaseEstimator, ClassifierMixin):
    """XGBoost/LightGBM의 0..5 레이블을 대회 1..6 레이블로 감싼다."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        if not np.isin(y, np.arange(1, 7)).all():
            raise ValueError('Expected labels 1..6')
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y - 1)
        self.classes_ = np.arange(1, 7)
        return self

    def predict(self, X):
        return self.estimator_.predict(X).astype(int) + 1

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def build_model(name: str, seed: int = 42):
    # 모듈을 함수 안에서 불러와 사용하지 않는 라이브러리는 import하지 않고, 순환 import도 피한다.
    if name == 'xgboost':
        from .xgboost_model import build_xgboost
        return build_xgboost(seed)
    if name == 'lightgbm':
        from .lightgbm_model import build_lightgbm
        return build_lightgbm(seed)
    if name == 'mlp':
        from .mlp_model import build_mlp
        return build_mlp(seed)
    if name in ('logistic', 'logistic_balanced'):
        from .logistic_model import build_logistic
        return build_logistic(seed, 'balanced' if name == 'logistic_balanced' else None)
    raise ValueError(f'Unsupported model: {name}')


MODEL_NAMES = ['xgboost', 'lightgbm', 'mlp', 'logistic', 'logistic_balanced']
