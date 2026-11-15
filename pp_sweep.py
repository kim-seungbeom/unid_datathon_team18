"""저장된 OOF/Val 확률로 후처리 변형을 비교한다(재학습 없음). 집계 지표만 출력.

변형: 과거 w초 확률 평균(인과적) → 규칙 마스킹 → argmax → cummax
python pp_sweep.py --run base_gkf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import features as F
import postprocess as P

ROOT = Path(__file__).resolve().parents[1]


def smooth(proba, groups, times, w):
    """같은 실험의 (t-w, t] 확률 평균. w=0이면 그대로."""
    if w <= 0:
        return proba
    out = np.empty_like(proba)
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        idx = idx[np.argsort(times[idx], kind="mergesort")]
        t = times[idx]
        c = np.vstack([np.zeros((1, 6)), np.cumsum(proba[idx], axis=0)])
        s = np.searchsorted(t, t - w, side="right")
        e = np.arange(1, len(idx) + 1)
        out[idx] = (c[e] - c[s]) / (e - s)[:, None]
    return out


def decode(proba, feats, groups, times, w, mask=True):
    p = smooth(proba, groups, times, w)
    if mask:
        p = P.rule_mask(p, feats, True)
    return P.cummax_labels(P.LABELS[p.argmax(1)], groups, times)


def f1(y, pred):
    return float(f1_score(y, pred, labels=list(range(1, 7)), average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base_gkf")
    ap.add_argument("--windows", type=float, nargs="+", default=[0, 10, 15, 20, 30, 45, 60])
    a = ap.parse_args()
    run = ROOT / "outputs" / "final_model" / a.run
    tr, va = pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/validation.csv")
    al = pd.concat([tr, va], ignore_index=True)
    sets = {"oof": (al, F.build_features(al)), "val": (va, F.build_features(va))}
    res = {}
    for k, (fr, fe) in sets.items():
        y = fr["label"].to_numpy(int)
        g = fr["experiment_id"].astype(str).to_numpy()
        t = fr["relative_time_s"].to_numpy(float)
        files = sorted(run.glob(f"{k}_proba_s*.npy"))
        for w in a.windows:
            for mask in (True, False):
                v = [f1(y, decode(np.load(f), fe, g, t, w, mask)) for f in files]
                res[f"{k}/w{int(w)}/{'mask' if mask else 'nomask'}"] = (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0)
    for key, (m, s) in res.items():
        print(f"{key:24s} {m:.4f} ± {s:.4f}")
    (run / "pp_sweep.json").write_text(json.dumps({k: list(v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
