"""전환(2→3, 4→5, 5→6) 시점과 '신호의 첫 임계값 통과'가 일치하는지 탐색. 집계값만 출력.

각 실험에서 이전 단계가 시작된 뒤(예: 2→3이면 2단계 시작 이후) 신호 s가 처음 θ를 넘는(또는 밑도는) 시각을
예측 전환 시각으로 보고, 실제 전환 시각과의 차이를 잰다. θ는 Train에서 고르고 Val에서 검증한다.
점수: 전환 전후 구간 행 단위 정확도(이전단계 vs 다음단계)를 실험별 평균.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import features as F

ROOT = Path(__file__).resolve().parents[1]


def load(split):
    fr = pd.read_csv(ROOT / f"data/{split}.csv")
    fe = F.build_features(fr)
    fe["experiment_id"] = fr["experiment_id"].astype(str).values
    fe["t"] = fr["relative_time_s"].values
    fe["label"] = fr["label"].values
    # 추가 후보: 압력·온도 상승량(2단계 시작 대비)과 t2-t1 차이
    fe["t2_minus_t1"] = fe["meas_t2"] - fe["meas_t1"]
    fe["thermal_minus_t1"] = fe["thermal_t"] - fe["meas_t1"]
    return fe


def segments(fe, a, b):
    """실험별 (a 또는 b 라벨 행 인덱스, 실제 b 시작 위치) — a에서 b로 바로 이어지는 실험만."""
    out = []
    for g, x in fe.groupby("experiment_id", sort=False):
        x = x.sort_values("t")
        m = x["label"].isin([a, b]).to_numpy()
        if m.sum() == 0 or not (x["label"] == a).any() or not (x["label"] == b).any():
            continue
        out.append(x[m])
    return out


def score(segs, col, theta, sign):
    accs, errs = [], []
    for x in segs:
        v = x[col].to_numpy(float)
        y = (x["label"].to_numpy() == x["label"].max()).astype(int)
        hit = (v >= theta) if sign > 0 else (v <= theta)
        hit = np.nan_to_num(np.maximum.accumulate(hit.astype(float)), nan=0)
        accs.append((hit == y).mean())
        tp = x["t"].to_numpy()
        true_t = tp[np.argmax(y)]
        pred_t = tp[np.argmax(hit)] if hit.any() else tp[-1] + 5
        errs.append(abs(pred_t - true_t))
    return float(np.mean(accs)), float(np.median(errs))


def main():
    tr, va = load("train"), load("validation")
    cands = [c for c in tr.columns if c not in ("experiment_id", "t", "label") and tr[c].dtype.kind == "f"]
    for a, b in ((2, 3), (4, 5), (5, 6)):
        st, sv = segments(tr, a, b), segments(va, a, b)
        res = []
        for c in cands:
            vals = pd.concat([s[c] for s in st]).dropna().to_numpy()
            if len(np.unique(vals)) < 5:
                continue
            for sign in (1, -1):
                best = (-1, None, None)
                for q in np.linspace(0.02, 0.98, 49):
                    th = np.quantile(vals, q)
                    acc, _ = score(st, c, th, sign)
                    if acc > best[0]:
                        best = (acc, th, q)
                acc_v, err_v = score(sv, c, best[1], sign)
                _, err_t = score(st, c, best[1], sign)
                res.append((best[0], acc_v, err_t, err_v, c, sign))
        res.sort(key=lambda r: -r[0])
        print(f"\n=== {a}->{b}: train exps={len(st)} val exps={len(sv)}  (acc=전환 구간 행 정확도, err=중앙 시각 오차 초)")
        for r in res[:12]:
            print(f"  {r[4]:28s} {'>=' if r[5] > 0 else '<='}  train acc {r[0]:.3f} err {r[2]:6.0f}s | val acc {r[1]:.3f} err {r[3]:6.0f}s")


if __name__ == "__main__":
    main()
