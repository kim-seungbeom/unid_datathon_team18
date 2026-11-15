"""예측 후처리. 과거 예측과 현재·과거 피처만 사용하므로 인과적이다."""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = np.arange(1, 7)


def rule_mask(proba: np.ndarray, feats: pd.DataFrame, use_vdrop: bool = True) -> np.ndarray:
    """사건 규칙과 모순되는 단계의 확률을 0으로 만든다. 모든 단계가 막히면 원래 확률을 쓴다."""
    p = proba.copy()
    r45 = feats["reached45"].to_numpy() > 0.5
    r50 = feats["reached50"].to_numpy() > 0.5
    p[~r45, 1:] = 0          # 45℃ 미도달 → 1단계만
    p[r45, 0] = 0            # 45℃ 도달 → 1단계 금지
    p[~r50, 2:] = 0          # t1 누적최대 < 50 → 3~6 금지
    if use_vdrop:
        vd = feats["v_drop"].to_numpy() > 0.5
        p[~vd, 3:] = 0       # 전압 강하 전 → 4~6 금지
        p[vd, :3] = 0        # 전압 강하 후 → 1~3 금지
    empty = p.sum(axis=1) <= 0
    p[empty] = proba[empty]
    return p


def cummax_labels(labels: np.ndarray, groups: np.ndarray, times: np.ndarray) -> np.ndarray:
    """같은 실험에서 지금까지 예측한 최대 단계 아래로 내려가지 않게 한다."""
    out = labels.copy()
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        idx = idx[np.argsort(times[idx], kind="mergesort")]
        out[idx] = np.maximum.accumulate(labels[idx])
    return out


def apply(proba, feats, groups, times, mode: str) -> np.ndarray:
    """mode: none | cummax | mask_cummax | mask_novdrop_cummax"""
    if mode == "none":
        return LABELS[proba.argmax(1)]
    if mode == "cummax":
        return cummax_labels(LABELS[proba.argmax(1)], groups, times)
    if mode == "mask_cummax":
        return cummax_labels(LABELS[rule_mask(proba, feats, True).argmax(1)], groups, times)
    if mode == "mask_novdrop_cummax":
        return cummax_labels(LABELS[rule_mask(proba, feats, False).argmax(1)], groups, times)
    raise ValueError(mode)


MODES = ["none", "cummax", "mask_cummax", "mask_novdrop_cummax"]
