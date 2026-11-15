"""Test처럼 10/15초·혼합 간격일 때 성능이 유지되는지 OOF로 시뮬레이션한다. 집계 지표만 출력.

각 fold의 평가 실험을 (a) 원본 5초, (b) 10초, (c) 15초, (d) 5/10/15초 혼합으로 솎아
같은 행에서 F1을 비교한다. 학습은 원본 5초 데이터 그대로.
python interval_sim.py --blend base:0.5,base_img16_delta:0.5 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import model as M
import postprocess as P
import train as T

ROOT = Path(__file__).resolve().parents[1]
L = list(range(1, 7))


def thin(fr, mode, rng):
    keep = []
    for _, idx in fr.groupby("experiment_id", sort=False).indices.items():
        idx = idx[np.argsort(fr["relative_time_s"].to_numpy()[idx], kind="mergesort")]
        if mode == "5s":
            keep.extend(idx)
        elif mode in ("10s", "15s"):
            keep.extend(idx[:: int(mode[:-1]) // 5])
        else:  # mixed: 5/10/15초 간격을 무작위로
            i = 0
            while i < len(idx):
                keep.append(idx[i])
                i += int(rng.choice([1, 2, 3]))
    return np.sort(np.array(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", default="base:0.5,base_img16_delta:0.5")
    ap.add_argument("--post", default="mask_cummax")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv", dtype={"sample_id": str, "experiment_id": str})
    va = pd.read_csv(ROOT / "data/validation.csv", dtype={"sample_id": str, "experiment_id": str})
    emb_dir = ROOT / "outputs/emb/resnet18"
    E = np.concatenate([T.load_embeddings(emb_dir, "train", tr.sample_id), T.load_embeddings(emb_dir, "validation", va.sample_id)])
    al = pd.concat([tr, va], ignore_index=True)
    y, g = al.label.to_numpy(), al.experiment_id.to_numpy()
    modes = ["5s", "10s", "15s", "mixed"]
    preds = {m: np.zeros(len(al), int) for m in modes}
    masks = {m: np.zeros(len(al), bool) for m in modes}
    blend = M.parse_blend(a.blend)
    for fold, (tri, tei) in enumerate(GroupKFold(5, shuffle=True, random_state=a.seed).split(al, y, g)):
        bp = M.BlendPredictor(blend, a.post, a.seed).fit(al.iloc[tri].reset_index(drop=True), y[tri], E[tri])
        te = al.iloc[tei].reset_index(drop=True)
        for m in modes:
            k = thin(te, m, np.random.default_rng(fold))
            preds[m][tei[k]] = bp.predict(te.iloc[k].reset_index(drop=True), E[tei][k])
            masks[m][tei[k]] = True
        print(f"fold {fold} done", flush=True)
    for m in modes:
        mk = masks[m]
        same = mk & masks["5s"]
        print(f"{m:6s} rows={int(mk.sum()):6d} F1={f1_score(y[mk], preds[m][mk], labels=L, average='macro'):.4f} "
              f"| same rows with 5s model input F1={f1_score(y[mk], preds['5s'][mk], labels=L, average='macro'):.4f}")


if __name__ == "__main__":
    main()
