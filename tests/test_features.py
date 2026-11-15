"""합성 데이터만 사용하는 피처·후처리 테스트."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features as F  # noqa: E402
import postprocess as P  # noqa: E402


def synth(exp="e1", n=80, step=5.0, seed=0, steps=None):
    rng = np.random.default_rng(seed)
    t = np.cumsum(steps) - steps[0] if steps is not None else np.arange(n) * step
    n = len(t)
    t1 = np.linspace(25, 120, n) + rng.normal(0, 0.2, n)
    v = np.concatenate([np.linspace(4.0, 4.2, n // 2), np.linspace(4.2, 2.0, n - n // 2)])
    df = pd.DataFrame({
        "sample_id": [f"{exp}_{i}" for i in range(n)], "experiment_id": exp,
        "relative_time_s": t, "meas_v": v, "meas_a": rng.normal(1, 0.1, n),
        "meas_t1": t1, "meas_t2": t1 - 2, "meas_p": np.linspace(100, 140, n),
        "thermal_t": t1 + 1,
    })
    df.loc[5, "meas_p"] = np.nan
    return df


def both():
    return pd.concat([synth("e1", seed=1), synth("e2", n=60, seed=2)], ignore_index=True)


def test_order_invariant():
    df = both()
    a = F.build_features(df)
    perm = np.random.default_rng(3).permutation(len(df))
    b = F.build_features(df.iloc[perm]).loc[df.index[perm]]
    pd.testing.assert_frame_equal(a.loc[df.index[perm]], b)


def test_experiments_independent():
    df = both()
    alone = F.build_features(df[df.experiment_id == "e1"])
    mixed = F.build_features(df).loc[alone.index]
    pd.testing.assert_frame_equal(alone, mixed)


@pytest.mark.parametrize("keep", [10, 33, 55])
def test_prefix_causal(keep):
    df = synth("e1", seed=4)
    full = F.build_features(df).iloc[:keep]
    pref = F.build_features(df.iloc[:keep])
    pd.testing.assert_frame_equal(full, pref)


def test_irregular_seconds():
    steps = np.array([5, 5, 10, 15, 5, 10, 15, 15, 5] * 6, dtype=float)
    df = synth("e1", steps=steps)
    f = F.build_features(df)
    t = df.relative_time_s.to_numpy()
    r = f["reached45"].to_numpy() > 0
    first = np.flatnonzero(r)[0]
    np.testing.assert_allclose(f["sec_since45"].to_numpy()[r], t[r] - t[first])
    assert (f["sec_since45"].to_numpy()[~r] == -1).all()


def test_duplicate_time_rejected():
    df = synth("e1")
    df.loc[3, "relative_time_s"] = df.loc[2, "relative_time_s"]
    with pytest.raises(F.FeatureError):
        F.build_features(df)


def test_event_values():
    df = synth("e1")
    f = F.build_features(df)
    t1 = df.meas_t1.to_numpy()
    assert np.allclose(f["t1_cummax"], np.maximum.accumulate(t1))
    vd = f["v_drop"].to_numpy() > 0
    assert vd.any() and not vd[0]
    assert (np.diff(vd.astype(int)) >= 0).all()
    assert (f.loc[~vd, "meas_p_since_vdrop"] == 0).all()


def test_cummax_postprocess():
    labels = np.array([1, 2, 1, 3, 2, 4])
    groups = np.array(["a"] * 6)
    times = np.arange(6.0)
    assert P.cummax_labels(labels, groups, times).tolist() == [1, 2, 2, 3, 3, 4]
    perm = np.array([5, 0, 3, 1, 4, 2])
    out = P.cummax_labels(labels[perm], groups[perm], times[perm])
    assert out.tolist() == [4, 1, 3, 2, 3, 2]


def test_rule_mask():
    feats = pd.DataFrame({"reached45": [0, 1, 1, 1], "reached50": [0, 0, 1, 1], "v_drop": [0, 0, 0, 1]})
    proba = np.full((4, 6), 1 / 6)
    lab = P.LABELS[P.rule_mask(proba, feats).argmax(1)]
    assert lab.tolist() == [1, 2, 2, 4]
