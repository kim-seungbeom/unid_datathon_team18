"""STEP 5: 경계 전문가(이진 HGB). 메인 모델의 저장된 OOF/Val 확률에 덧씌워 비교한다. 집계 지표만 출력.

메인 argmax가 {a,b}일 때 두 클래스의 확률 합을 전문가 확률 q=P(b|a or b)로 다시 나눈다 → 마스킹 → cummax.
fold 분할·PCA는 evaluate.py와 같은 seed·방식으로 재현한다(학습 부분에서만 fit).

python experts.py --config base_img16 --run base_img16_gkf --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import evaluate as E
import features as F
import postprocess as P
from pp_sweep import f1

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [(2, 3), (4, 5), (5, 6)]


def expert_q(Xtr, ytr, Xte, a, b, seed):
    m = np.isin(ytr, [a, b])
    if len(np.unique(ytr[m])) < 2:
        return None
    model = E.make_model(seed, {})
    model.fit(Xtr[m], (ytr[m] == b).astype(int))
    return model.predict_proba(Xte)[:, 1]


def apply_experts(proba, feats, qs, use):
    p = P.rule_mask(proba, feats, True)
    top = P.LABELS[p.argmax(1)]
    p = p.copy()
    for (a, b) in use:
        q = qs.get((a, b))
        if q is None:
            continue
        rows = np.isin(top, [a, b])
        tot = p[rows, a - 1] + p[rows, b - 1]
        p[rows, a - 1], p[rows, b - 1] = tot * (1 - q[rows]), tot * q[rows]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base_img16")
    ap.add_argument("--run", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--emb_dir", type=Path, default=ROOT / "outputs" / "emb" / "resnet18")
    a = ap.parse_args()
    run = ROOT / "outputs" / "final_model" / (a.run or f"{a.config}_gkf")
    cfg = E.CONFIGS[a.config]
    cols = [c for c in F.feature_columns(include_events=cfg["events"], include_gas=cfg.get("gas", False), include_cond=cfg.get("cond", False)) if c not in cfg.get("drop", [])]
    img = cfg.get("img")

    tr, va = pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/validation.csv")
    ftr, fva = F.build_features(tr), F.build_features(va)
    al, fal = pd.concat([tr, va], ignore_index=True), pd.concat([ftr, fva], ignore_index=True)
    if img:
        Etr = E.load_embeddings(a.emb_dir, "train", tr["sample_id"])
        Eva = E.load_embeddings(a.emb_dir, "validation", va["sample_id"])
        Eal = np.concatenate([Etr, Eva])

    def mats(Fa, Fb, fa, fb, Ea, Eb, seed):
        if not img:
            return Fa[cols].reset_index(drop=True), Fb[cols].reset_index(drop=True)
        return E.add_image(Fa[cols], Fb[cols], Ea, Eb, fa, fb, img, cfg["delta"], seed)

    variants = [("main", [])] + [(f"+{x}{y}", [(x, y)]) for x, y in PAIRS] + [("+all", PAIRS)]
    res = {f"{k}/{n}": [] for k in ("oof", "val") for n, _ in variants}
    for seed in a.seeds:
        # Val: Train으로 전문가 학습
        Xa, Xb = mats(ftr, fva, tr, va, Etr if img else None, Eva if img else None, seed)
        ytr = tr["label"].to_numpy(int)
        qs = {pr: expert_q(Xa, ytr, Xb, *pr, seed) for pr in PAIRS}
        pv = np.load(run / f"val_proba_s{seed}.npy")
        g, t, y = va["experiment_id"].astype(str).to_numpy(), va["relative_time_s"].to_numpy(float), va["label"].to_numpy(int)
        for n, use in variants:
            lab = P.cummax_labels(P.LABELS[apply_experts(pv, fva, qs, use).argmax(1)], g, t)
            res[f"val/{n}"].append(f1(y, lab))
        # OOF
        yal = al["label"].to_numpy(int)
        gal = al["experiment_id"].astype(str).to_numpy()
        tal = al["relative_time_s"].to_numpy(float)
        po = np.load(run / f"oof_proba_s{seed}.npy")
        qo = {pr: np.zeros(len(al)) for pr in PAIRS}
        for tri, tei in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(fal, yal, gal):
            Xa, Xb = mats(fal.iloc[tri], fal.iloc[tei], al.iloc[tri], al.iloc[tei],
                          Eal[tri] if img else None, Eal[tei] if img else None, seed)
            for pr in PAIRS:
                q = expert_q(Xa, yal[tri], Xb, *pr, seed)
                qo[pr][tei] = 0.5 if q is None else q
        for n, use in variants:
            lab = P.cummax_labels(P.LABELS[apply_experts(po, fal, qo, use).argmax(1)], gal, tal)
            res[f"oof/{n}"].append(f1(yal, lab))
        print(f"seed {seed} done", flush=True)
    summary = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in res.items()}
    for k, (m, s) in summary.items():
        print(f"{k:14s} {m:.4f} +- {s:.4f}")
    (run / "experts.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
