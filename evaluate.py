"""평가 하네스. stdout에는 집계 지표만 출력한다(행 값·ID 출력 없음).

예)
  python evaluate.py --data_dir ../data --config base     --mode both --seeds 42 43 44
  python evaluate.py --data_dir ../data --config events   --mode both --seeds 42 43 44

--mode oof : Train+Validation 실험을 합친 5-fold GroupKFold(shuffle, group=experiment_id)
--mode val : Train으로 학습 → 공식 Validation
OOF/Val 확률은 --out_dir(기본 ../outputs/final_model, gitignore 대상)에 npy로 저장한다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

import features as F
import postprocess as P

LABELS = list(range(1, 7))


def load(data_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"{split}.csv")


def make_model(seed: int, params: dict) -> HistGradientBoostingClassifier:
    base = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                l2_regularization=1.0, early_stopping=False, random_state=seed)
    base.update(params)
    return HistGradientBoostingClassifier(**base)


def load_embeddings(emb_dir: Path, split: str, ids: pd.Series) -> np.ndarray:
    """image_features 추출기 출력(features.npy + sample_ids.csv)을 sample_id 순서로 맞춘다."""
    d = emb_dir / split
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise SystemExit(f"{d}: 임베딩 상태가 complete가 아닙니다.")
    E = np.load(d / "features.npy", mmap_mode="r", allow_pickle=False)
    order = pd.read_csv(d / "sample_ids.csv")["sample_id"].astype(str)
    pos = pd.Series(np.arange(len(order)), index=order.values)
    if not ids.astype(str).isin(pos.index).all():
        raise SystemExit(f"{d}: 일부 sample_id의 임베딩이 없습니다.")
    return np.asarray(E[pos.loc[ids.astype(str).values].values], dtype=np.float32)


def add_image(Ftr, Fte, Etr, Ete, frtr, frte, k, delta, seed):
    """PCA는 학습 부분에서만 fit하고 평가 부분에는 transform만 한다."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k, random_state=seed).fit(Etr)
    Ztr, Zte = pca.transform(Etr), pca.transform(Ete)
    cols = [f"img_pc{i}" for i in range(k)]
    Xtr = Ftr.reset_index(drop=True).copy()
    Xte = Fte.reset_index(drop=True).copy()
    Xtr[cols], Xte[cols] = Ztr, Zte
    if delta:
        Xtr["img_change30s"] = F.image_change(frtr.reset_index(drop=True), Ztr)
        Xte["img_change30s"] = F.image_change(frte.reset_index(drop=True), Zte)
    return Xtr, Xte


def fit_predict(Xtr, ytr, Xte, seed, params, balanced):
    m = make_model(seed, params)
    sw = compute_sample_weight("balanced", ytr) if balanced else None
    m.fit(Xtr, ytr, sample_weight=sw)
    proba = np.zeros((len(Xte), 6))
    proba[:, np.asarray(m.classes_) - 1] = m.predict_proba(Xte)
    return proba


def metrics(y, pred) -> dict:
    return {
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class_f1": [round(float(v), 4) for v in
                         f1_score(y, pred, labels=LABELS, average=None, zero_division=0)],
        "confusion": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def score_all_modes(proba, y, feats, groups, times) -> dict:
    return {m: metrics(y, P.apply(proba, feats, groups, times, m)) for m in P.MODES}


CONFIGS = {
    "base": dict(events=False),
    "events": dict(events=True),
    # 경과 시간형 사건 피처(배터리 종류별 타이밍을 외울 위험)를 뺀 변형
    "events_nosec": dict(events=True, drop=["sec_since45", "sec_since50", "sec_since_vdrop"]),
    "events_flags": dict(events=True, drop=["sec_since45", "sec_since50", "sec_since_vdrop",
                                            "meas_p_since_vdrop", "meas_t1_since_vdrop",
                                            "meas_t2_since_vdrop", "thermal_t_since_vdrop"]),
    # 아래는 --emb_dir(이미지 임베딩, GPU 서버에서 추출) 필요
    # base_* : 사건 피처 없이(OOF에서 사건 피처가 후처리 이득을 없앰) 이미지 추가
    "base_img16": dict(events=False, img=16, delta=False),
    "base_img32": dict(events=False, img=32, delta=False),
    "base_img16_delta": dict(events=False, img=16, delta=True),
    # 가스·환경 센서 11개 추가 (현재값, 30/120초 창, 누적최대, 첫 관측 대비 상승)
    "base_gas": dict(events=False, gas=True),
    "base_gas_img16_delta": dict(events=False, gas=True, img=16, delta=True),
    # 실험 조건 4개(배터리 종류·용량·SOC·열폭주 유도 방식) 추가
    "base_cond": dict(events=False, cond=True),
    "base_cond_img16_delta": dict(events=False, cond=True, img=16, delta=True),
    "events_img16": dict(events=True, img=16, delta=False),
    "events_img32": dict(events=True, img=32, delta=False),
    "events_img16_delta": dict(events=True, img=16, delta=True),
    "events_img32_delta": dict(events=True, img=32, delta=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "final_model")
    ap.add_argument("--config", choices=CONFIGS, default="events")
    ap.add_argument("--mode", choices=["oof", "val", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--hgb", default="{}", help="HGB 파라미터 JSON 덮어쓰기(튜닝 실험용)")
    ap.add_argument("--emb_dir", type=Path, default=None,
                    help="train/, validation/ 하위에 features.npy·sample_ids.csv·metadata.json")
    args = ap.parse_args()
    HGB = json.loads(args.hgb)

    cfg = CONFIGS[args.config]
    name = args.config + ("_bal" if args.balanced else "") + (f"_{args.tag}" if args.tag else "")
    out = args.out_dir / name
    out.mkdir(parents=True, exist_ok=True)

    tr, va = load(args.data_dir, "train"), load(args.data_dir, "validation")
    ftr, fva = F.build_features(tr), F.build_features(va)
    cols = [c for c in F.feature_columns(include_events=cfg["events"], include_gas=cfg.get("gas", False), include_cond=cfg.get("cond", False)) if c not in cfg.get("drop", [])]
    ytr, yva = tr["label"].to_numpy(int), va["label"].to_numpy(int)
    img = cfg.get("img")
    if img:
        if args.emb_dir is None:
            raise SystemExit("이 config는 --emb_dir가 필요합니다.")
        Etr = load_embeddings(args.emb_dir, "train", tr["sample_id"])
        Eva = load_embeddings(args.emb_dir, "validation", va["sample_id"])
    print(f"[{name}] features={len(cols)}{f'+img{img}' if img else ''} train_rows={len(tr)} val_rows={len(va)}")

    def matrices(Ftr, Fte, frtr, frte, Etr_, Ete_, seed):
        if not img:
            return Ftr[cols], Fte[cols]
        return add_image(Ftr[cols], Fte[cols], Etr_, Ete_, frtr, frte, img, cfg["delta"], seed)

    report = {"config": name, "n_features": len(cols), "features": cols, "image_pcs": img, "runs": {}}
    for seed in args.seeds:
        r = {}
        if args.mode in ("val", "both"):
            t0 = time.time()
            Xtr, Xva = matrices(ftr, fva, tr, va, Etr if img else None, Eva if img else None, seed)
            pv = fit_predict(Xtr, ytr, Xva, seed, HGB, args.balanced)
            np.save(out / f"val_proba_s{seed}.npy", pv)
            r["val"] = score_all_modes(pv, yva, fva, va["experiment_id"].astype(str).to_numpy(),
                                       va["relative_time_s"].to_numpy(float))
            r["val_sec"] = round(time.time() - t0, 1)
        if args.mode in ("oof", "both"):
            t0 = time.time()
            al = pd.concat([tr, va], ignore_index=True)
            fal = pd.concat([ftr, fva], ignore_index=True)
            yal = al["label"].to_numpy(int)
            gal = al["experiment_id"].astype(str).to_numpy()
            Eal = np.concatenate([Etr, Eva]) if img else None
            oof = np.zeros((len(al), 6))
            # StratifiedGroupKFold의 shuffle은 분할을 거의 바꾸지 않아 seed 변동을 못 잰다 → GroupKFold(shuffle)
            cv = GroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            for tri, tei in cv.split(fal, yal, gal):
                Xa, Xb = matrices(fal.iloc[tri], fal.iloc[tei], al.iloc[tri], al.iloc[tei],
                                  Eal[tri] if img else None, Eal[tei] if img else None, seed)
                oof[tei] = fit_predict(Xa, yal[tri], Xb, seed, HGB, args.balanced)
            np.save(out / f"oof_proba_s{seed}.npy", oof)
            r["oof"] = score_all_modes(oof, yal, fal, gal, al["relative_time_s"].to_numpy(float))
            r["oof_sec"] = round(time.time() - t0, 1)
        report["runs"][seed] = r
        line = [f"seed={seed}"]
        for k in ("oof", "val"):
            if k in r:
                line.append(k + " " + " ".join(f"{m}={r[k][m]['macro_f1']:.4f}" for m in P.MODES))
        print(" | ".join(line), flush=True)

    # seed 요약
    summ = {}
    for k in ("oof", "val"):
        for m in P.MODES:
            v = [report["runs"][s][k][m]["macro_f1"] for s in args.seeds if k in report["runs"][s]]
            if v:
                summ[f"{k}/{m}"] = {"mean": round(float(np.mean(v)), 4),
                                    "std": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4)}
    report["summary"] = summ
    for k, v in summ.items():
        print(f"  {k:28s} mean={v['mean']:.4f} std={v['std']:.4f}")
    s0 = args.seeds[0]
    for k in ("oof", "val"):
        if k in report["runs"][s0]:
            for m in ("none", "mask_cummax"):
                print(f"  seed{s0} {k}/{m} per-class F1: {report['runs'][s0][k][m]['per_class_f1']}")
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"saved {out / 'report.json'}")


if __name__ == "__main__":
    main()
