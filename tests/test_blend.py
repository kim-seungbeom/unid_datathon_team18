"""합성 데이터로 최종 블렌드의 학습·저장·로드·인과 재현성을 검사한다."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as M  # noqa: E402
import train as T  # noqa: E402
from test_features import synth  # noqa: E402


@pytest.fixture(scope="module")
def data():
    frame = pd.concat([synth(f"syn-{i}", n=60, seed=i) for i in range(4)], ignore_index=True)
    labels = np.tile(np.repeat(np.arange(1, 7), 10), 4)
    emb = np.random.default_rng(1).normal(size=(len(frame), 512)).astype("float32")
    emb[:, 0] += labels * 3
    return frame, labels, emb


def test_parse_blend():
    assert M.parse_blend("base:1,base_img16_delta:3") == [("base", 0.25), ("base_img16_delta", 0.75)]
    for bad in ("nope:1", "base:0", "base:1,base:1", "base"):
        with pytest.raises(M.SubmissionError):
            M.parse_blend(bad)
    with pytest.raises(M.SubmissionError):
        M.BlendPredictor(M.parse_blend("base:1"))  # 이미지 없는 최종 모델 거부


def test_blend_contract(data, tmp_path):
    frame, labels, emb = data
    bp = M.BlendPredictor(M.parse_blend("base:0.5,base_img16_delta:0.5")).fit(frame, labels, emb)
    base = pd.Series(bp.predict(frame, emb), index=frame.sample_id)
    # 확률 = 멤버 확률 가중 평균
    p, _ = bp.predict_proba(frame, emb)
    p0, _ = bp.members[0].predict_proba(frame, None)
    p1, _ = bp.members[1].predict_proba(frame, emb)
    np.testing.assert_allclose(p, 0.5 * p0 + 0.5 * p1)
    order = np.random.default_rng(3).permutation(len(frame))
    sh = pd.Series(bp.predict(frame.iloc[order], emb[order]), index=frame.sample_id.iloc[order])
    pd.testing.assert_series_equal(base, sh.reindex(base.index))
    for idx in frame.groupby("experiment_id").indices.values():
        for part in (idx, idx[: len(idx) // 2]):
            np.testing.assert_array_equal(bp.predict(frame.iloc[part], emb[part]), base.iloc[part])
    bp.save(tmp_path / "a")
    loaded = M.Predictor.load(tmp_path / "a")
    assert isinstance(loaded, M.BlendPredictor) and loaded.config["img_pcs"] == 16
    np.testing.assert_array_equal(loaded.predict(frame, emb), base.values)


def test_train_cli_default_is_blend(data, tmp_path):
    frame, labels, emb = data
    d, cache = tmp_path / "data", tmp_path / "emb"
    for split, pos in (("train", np.arange(180)), ("validation", np.arange(180, 240))):
        fr = frame.iloc[pos].copy()
        fr["label"] = labels[pos]
        (d / split).mkdir(parents=True)
        fr.to_csv(d / split / f"{split}.csv", index=False)
        (cache / split).mkdir(parents=True)
        np.save(cache / split / "features.npy", emb[pos])
        fr[["sample_id"]].to_csv(cache / split / "sample_ids.csv", index=False)
        (cache / split / "metadata.json").write_text(json.dumps({"status": "complete", "model": "resnet18"}))
    T.main(["--data_dir", str(d), "--emb_dir", str(cache), "--out", str(tmp_path / "art")])
    assert (tmp_path / "art" / "blend.json").is_file()
    T.main(["--data_dir", str(d), "--emb_dir", str(cache), "--splits", "train", "--out", str(tmp_path / "art_tr")])
    assert isinstance(M.Predictor.load(tmp_path / "art_tr"), M.BlendPredictor)
