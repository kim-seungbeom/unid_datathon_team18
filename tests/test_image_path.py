"""합성 임베딩으로 이미지 입력 경로를 검사한다(실제 데이터 없음)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as E  # noqa: E402
import features as F  # noqa: E402
from test_features import both  # noqa: E402


def write_emb(d: Path, ids, dim=8, seed=0):
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    np.save(d / "features.npy", rng.normal(size=(len(ids), dim)).astype(np.float32))
    pd.DataFrame({"sample_id": ids}).to_csv(d / "sample_ids.csv", index=False)
    (d / "metadata.json").write_text(json.dumps({"status": "complete"}))


def test_load_reorders_by_sample_id(tmp_path):
    df = both()
    ids = df.sample_id.tolist()
    write_emb(tmp_path / "train", ids[::-1])
    raw = np.load(tmp_path / "train" / "features.npy")
    got = E.load_embeddings(tmp_path, "train", df.sample_id)
    np.testing.assert_allclose(got[0], raw[-1])


def test_image_change_causal_and_independent():
    df = both()
    Z = np.random.default_rng(1).normal(size=(len(df), 4))
    full = F.image_change(df, Z)
    m = (df.experiment_id == "e1").to_numpy()
    sub = df[m].iloc[:20]
    part = F.image_change(sub, Z[np.flatnonzero(m)[:20]])
    np.testing.assert_allclose(full[np.flatnonzero(m)[:20]], part)
    assert full[0] == 0


def test_add_image_pca_fit_on_train_only():
    df = both()
    f = F.build_features(df)
    cols = F.feature_columns()
    Etr = np.random.default_rng(2).normal(size=(len(df), 8))
    Ete = Etr + 100  # 평가 쪽 분포가 달라도 학습 PCA로만 변환
    a, b = E.add_image(f[cols], f[cols], Etr, Ete, df, df, 3, True, 42)
    assert a.shape[1] == len(cols) + 4 and b.shape == a.shape
    assert not np.allclose(a["img_pc0"], b["img_pc0"])
