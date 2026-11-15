"""인과적(과거 방향) 피처 생성.

모든 피처는 같은 experiment_id 안에서 relative_time_s <= 현재 시점인 행만 사용한다.
실험 전체 통계, 미래 행, 다른 실험, label, 실험 조건 상수(bat_class 등)는 쓰지 않는다.
입력 행 순서와 무관하게 같은 결과를 내고, 결과는 입력 행 순서로 돌려준다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 현재값·시간창 통계를 만들 센서. thermal_t는 열화상 장치가 준 수치(픽셀 온도 지도 아님).
SENSORS = ["meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p", "thermal_t"]
WINDOWS_S = (30.0, 120.0)
SLOPE_SIGNALS = ["meas_t1", "meas_t2", "meas_p"]

# 사건 임계값: Train 분석(REPORT_FOLLOWUP.md)에서 관측된 라벨 경계값. 데이터로 다시 fit하지 않는 고정 상수.
T1_STAGE2_C = 45.0
T1_STAGE3_MIN_C = 50.0
V_DROP_RATIO = 0.83
DROP_DELTA_SIGNALS = ["meas_p", "meas_t1", "meas_t2", "thermal_t"]

# 가스·환경 센서: include_gas=True인 설정에서만 모델 입력에 쓴다(기존 설정의 피처 목록은 그대로).
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf",
       "meas_n2o", "meas_no", "meas_no2", "meas_so2", "atmp_t"]
GAS_WINDOWS_S = (30.0, 120.0)


# 실험 조건(실험 안에서 상수, 현재 행 값): include_cond=True 설정에서만 입력.
COND = ["bat_class", "bat_cc", "soc_status", "tr_method"]


def gas_columns() -> list[str]:
    cols = []
    for s in GAS:
        cols.append(s)
        for w in GAS_WINDOWS_S:
            cols += [f"{s}_mean{int(w)}s", f"{s}_max{int(w)}s", f"{s}_delta{int(w)}s"]
        cols += [f"{s}_cummax", f"{s}_rise"]
    return cols


EVENT_COLS = [
    "t1_cummax", "reached45", "sec_since45", "reached50", "sec_since50",
    "v_ratio", "v_ratio_cummin", "v_drop", "sec_since_vdrop",
] + [f"{s}_since_vdrop" for s in DROP_DELTA_SIGNALS]


class FeatureError(ValueError):
    """데이터 값을 메시지에 담지 않는 오류."""


def _numeric(frame: pd.DataFrame, cols) -> np.ndarray:
    out = frame[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def _ffill(x: np.ndarray) -> np.ndarray:
    """과거 방향 결측 채움(열 단위). 첫 관측 이전은 NaN 유지."""
    x = x.copy()
    n = x.shape[0]
    idx = np.where(np.isfinite(x), np.arange(n)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = x[idx, np.arange(x.shape[1])[None, :]]
    seen = np.maximum.accumulate(np.isfinite(x), axis=0)
    filled[~seen] = np.nan
    return filled


def _window_stats(t: np.ndarray, x: np.ndarray, w: float):
    """(t-w, t] 평균·최댓값, 그리고 현재값 - (t-w 이하 마지막 관측값)."""
    n = len(t)
    start = np.searchsorted(t, t - w, side="right")
    end = np.arange(1, n + 1)
    valid = np.isfinite(x)
    xs = np.where(valid, x, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(xs)])
    ccnt = np.concatenate([[0], np.cumsum(valid)])
    cnt = ccnt[end] - ccnt[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (csum[end] - csum[start]) / cnt
    mean[cnt == 0] = np.nan
    vmax = np.full(n, np.nan)
    for i in range(n):
        seg = x[start[i]:end[i]]
        if np.isfinite(seg).any():
            vmax[i] = np.nanmax(seg)
    lag = start - 1
    delta = np.full(n, np.nan)
    ok = lag >= 0
    delta[ok] = x[ok] - x[lag[ok]]
    return mean, vmax, delta


def _window_slope(t: np.ndarray, x: np.ndarray, w: float) -> np.ndarray:
    """(t-w, t] 유효 관측의 최소제곱 기울기(단위/초). 관측 2개 미만이면 NaN."""
    n = len(t)
    start = np.searchsorted(t, t - w, side="right")
    out = np.full(n, np.nan)
    for i in range(n):
        tt, xx = t[start[i]:i + 1], x[start[i]:i + 1]
        m = np.isfinite(xx)
        if m.sum() >= 2:
            tt, xx = tt[m], xx[m]
            tc = tt - tt.mean()
            den = (tc * tc).sum()
            if den > 0:
                out[i] = (tc * (xx - xx.mean())).sum() / den
    return out


def _first_time(cond: np.ndarray, t: np.ndarray) -> float:
    hit = np.flatnonzero(cond)
    return float(t[hit[0]]) if len(hit) else np.nan


def _gas_feats(t: np.ndarray, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """가스 센서: 현재값, 30/120초 창 통계, 누적 최대, 실험 첫 관측 대비 상승량(첫 관측은 과거)."""
    feats = {}
    for s in GAS:
        if s not in raw:
            continue
        x = _ffill(raw[s][:, None])[:, 0]
        feats[s] = x
        for w in GAS_WINDOWS_S:
            m, mx, d = _window_stats(t, raw[s], w)
            feats[f"{s}_mean{int(w)}s"], feats[f"{s}_max{int(w)}s"], feats[f"{s}_delta{int(w)}s"] = m, mx, d
        cm = np.fmax.accumulate(np.where(np.isfinite(x), x, -np.inf))
        cm[~np.isfinite(cm)] = np.nan
        feats[f"{s}_cummax"] = cm
        first = x[np.flatnonzero(np.isfinite(x))[0]] if np.isfinite(x).any() else np.nan
        feats[f"{s}_rise"] = x - first
    return feats


def _one_experiment(t: np.ndarray, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(t)
    feats: dict[str, np.ndarray] = {}
    filled = {k: _ffill(v[:, None])[:, 0] for k, v in raw.items() if k in SENSORS}

    for s in SENSORS:
        x = filled[s]
        feats[s] = x
        for w in WINDOWS_S:
            m, mx, d = _window_stats(t, raw[s], w)
            tag = int(w)
            feats[f"{s}_mean{tag}s"] = m
            feats[f"{s}_max{tag}s"] = mx
            feats[f"{s}_delta{tag}s"] = d
    for s in SLOPE_SIGNALS:
        for w in WINDOWS_S:
            feats[f"{s}_slope{int(w)}s"] = _window_slope(t, raw[s], w)

    # --- 사건 피처 ---
    t1 = filled["meas_t1"]
    t1_cummax = np.fmax.accumulate(np.where(np.isfinite(t1), t1, -np.inf))
    t1_cummax[~np.isfinite(t1_cummax)] = np.nan
    feats["t1_cummax"] = t1_cummax
    for thr, name in ((T1_STAGE2_C, "45"), (T1_STAGE3_MIN_C, "50")):
        reached = np.nan_to_num(t1_cummax, nan=-np.inf) >= thr
        t0 = _first_time(reached, t)
        feats[f"reached{name}"] = reached.astype(float)
        feats[f"sec_since{name}"] = np.where(reached, t - t0, -1.0)

    v = filled["meas_v"]
    vmax = np.fmax.accumulate(np.where(np.isfinite(v), v, -np.inf))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where((vmax > 0) & np.isfinite(v), v / vmax, np.nan)
    rmin = np.fmin.accumulate(np.where(np.isfinite(ratio), ratio, np.inf))
    rmin[~np.isfinite(rmin)] = np.nan
    drop = np.nan_to_num(rmin, nan=np.inf) < V_DROP_RATIO
    td = _first_time(drop, t)
    feats["v_ratio"] = ratio
    feats["v_ratio_cummin"] = rmin
    feats["v_drop"] = drop.astype(float)
    feats["sec_since_vdrop"] = np.where(drop, t - td, -1.0)
    first = np.flatnonzero(drop)
    for s in DROP_DELTA_SIGNALS:
        x = filled[s]
        out = np.zeros(n)
        if len(first):
            base = x[first[0]]
            out[first[0]:] = x[first[0]:] - base
        feats[f"{s}_since_vdrop"] = out
    feats.update(_gas_feats(t, raw))
    return feats


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """frame에 들어 있는 행만으로 피처를 만들어 frame과 같은 순서·index로 돌려준다."""
    need = {"experiment_id", "relative_time_s", *SENSORS}
    if not need.issubset(frame.columns):
        raise FeatureError("필수 열이 없습니다.")
    times = pd.to_numeric(frame["relative_time_s"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times).all():
        raise FeatureError("relative_time_s에 유효하지 않은 값이 있습니다.")
    groups = frame["experiment_id"].astype(str).to_numpy()
    gas_present = [s for s in GAS if s in frame.columns]
    raw_all = _numeric(frame, SENSORS + gas_present)
    names = SENSORS + gas_present

    pieces = []
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        idx = idx[np.argsort(times[idx], kind="mergesort")]
        t = times[idx]
        if (np.diff(t) <= 0).any():
            raise FeatureError("같은 실험 안에 중복 시각이 있습니다.")
        raw = {s: raw_all[idx, j] for j, s in enumerate(names)}
        f = pd.DataFrame(_one_experiment(t, raw))
        f.index = idx
        pieces.append(f)
    out = pd.concat(pieces).sort_index()
    out.index = frame.index
    for c in COND:
        if c in frame.columns:
            out[f"cond_{c}"] = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
    return out


def image_change(frame: pd.DataFrame, Z: np.ndarray, window_s: float = 30.0) -> np.ndarray:
    """현재 이미지 벡터 Z[i]와 같은 실험의 (t - window_s) 이하 마지막 프레임 벡터의 L2 거리. 없으면 0."""
    times = pd.to_numeric(frame["relative_time_s"], errors="coerce").to_numpy(dtype=float)
    groups = frame["experiment_id"].astype(str).to_numpy()
    out = np.zeros(len(frame))
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        idx = idx[np.argsort(times[idx], kind="mergesort")]
        t = times[idx]
        lag = np.searchsorted(t, t - window_s, side="right") - 1
        ok = lag >= 0
        d = np.zeros(len(idx))
        d[ok] = np.linalg.norm(Z[idx[ok]] - Z[idx[lag[ok]]], axis=1)
        out[idx] = d
    return out


def feature_columns(include_events: bool = True, include_gas: bool = False,
                    include_cond: bool = False) -> list[str]:
    demo = pd.DataFrame({
        "experiment_id": ["a", "a"], "relative_time_s": [0.0, 5.0],
        **{s: [1.0, 1.0] for s in SENSORS},
    })
    cols = list(build_features(demo).columns)   # 가스 열이 없는 demo → 기존 목록과 동일
    if not include_events:
        cols = [c for c in cols if c not in EVENT_COLS]
    if include_gas:
        cols += gas_columns()
    if include_cond:
        cols += [f"cond_{c}" for c in COND]
    return cols
