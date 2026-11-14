"""센서(승범) + 이미지 인코더(성현) 특징을 행 단위로 결합해 최종 분류기(도유)에 넣는다.

- 센서: sensor_features.build_sensor_features (같은 실험의 현재·과거 행만, fit 없음)
- 이미지: image_features 고정 인코더 벡터. 미리 추출한 폴더를 sample_id로 연결하거나,
  같은 extract() 함수로 그 자리에서 계산한다(학습·추론 전처리 동일).
- 결측 대체·표준화·PCA는 sklearn Pipeline 안에서 Train(또는 CV train fold)으로만 fit한다.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
# 실험 폴더가 먼저 오도록 뒤에 추가한다(루트 model.py 등과 이름 충돌 방지).
for _path in (REPO_ROOT, REPO_ROOT / 'image_features'):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from schema import LABELS, read_frame  # noqa: E402
from sensor_features import (SENSOR_FEATURE_VERSION, build_sensor_features,  # noqa: E402
                             uses_history, validate_feature_names, validate_imputation)
from temporal import TEMPORAL_VERSION  # noqa: E402

from models import build_model  # noqa: E402

ENCODERS = ('none', 'resnet18', 'dinov2', 'convnext_tiny')
EMBED_PREFIX = 'img_'
LABEL_LIST = list(LABELS)


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def sensor_spec(config_path: Path | None, sensor_features=None, imputation=None) -> dict:
    """승범님 configs/*.json의 sensor_features·sensor_imputation을 그대로 가져온다."""
    names, imp = ['meas_v', 'meas_a', 'meas_t1', 'meas_t2', 'meas_p'], 'median'
    if config_path is not None:
        config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        names = config.get('sensor_features', names)
        imp = config.get('sensor_imputation', imp)
    if sensor_features:
        names = list(sensor_features)
    if imputation:
        imp = imputation
    names = list(validate_feature_names(names))
    imp = validate_imputation(imp)
    return {
        'sensor_features': names,
        'sensor_imputation': imp,
        'sensor_feature_version': SENSOR_FEATURE_VERSION,
        'temporal_version': TEMPORAL_VERSION if uses_history(names, imp) else None,
    }


def check_versions(bundle: dict) -> None:
    if bundle['sensor_feature_version'] != SENSOR_FEATURE_VERSION:
        raise ValueError('Saved sensor feature version does not match sensor_features.py.')
    expected = (TEMPORAL_VERSION
                if uses_history(bundle['sensor_features'], bundle['sensor_imputation']) else None)
    if bundle['temporal_version'] != expected:
        raise ValueError('Saved temporal version does not match temporal.py.')


# ---------------------------------------------------------------------------
# 이미지 벡터
# ---------------------------------------------------------------------------
def load_embeddings(directory: Path, sample_ids) -> tuple[np.ndarray, dict]:
    """성현님 추출 결과 폴더(features.npy, sample_ids.csv, metadata.json)를 행 순서에 맞춘다."""
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if meta.get('status') != 'complete':
        raise ValueError(f'Embedding extraction is not complete: {directory}')
    ids = pd.read_csv(directory / 'sample_ids.csv', dtype=str,
                      keep_default_na=False)['sample_id']
    features = np.load(directory / 'features.npy', mmap_mode='r', allow_pickle=False)
    if len(ids) != features.shape[0] or not ids.is_unique:
        raise ValueError('Embedding IDs and feature rows do not match.')
    positions = pd.Index(ids).get_indexer(pd.Index(sample_ids))
    if (positions < 0).any():
        raise ValueError(f'{int((positions < 0).sum())} sample_ids have no embedding in {directory}.')
    values = np.asarray(features[positions], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Embeddings contain non-finite values.')
    return values, meta


def extract_embeddings(csv_path: Path, image_root: Path, encoder: str, output_dir: Path, *,
                       device='cuda', batch_size=32, workers=4, l2_normalize=False,
                       _encoder=None) -> Path:
    """image_features.extract_features.extract를 그대로 호출한다(새 폴더에만 저장)."""
    from extract_features import extract

    extract(model_name=encoder, csv_path=Path(csv_path), image_root=Path(image_root),
            output_dir=Path(output_dir), batch_size=batch_size, workers=workers, device=device,
            l2_normalize=l2_normalize, _encoder=_encoder)
    return Path(output_dir)


def embedding_info(meta: dict) -> dict:
    keys = ('model', 'weights', 'dimension', 'l2_normalize', 'preprocessing')
    return {k: meta.get(k) for k in keys}


def check_embedding_compatible(saved: dict, meta: dict) -> None:
    for key in ('model', 'weights', 'dimension', 'l2_normalize'):
        if saved.get(key) != meta.get(key):
            raise ValueError(f'Embedding {key} differs from training: {meta.get(key)!r}')
    if json.dumps(saved.get('preprocessing'), sort_keys=True) != json.dumps(
            meta.get('preprocessing'), sort_keys=True):
        raise ValueError('Embedding preprocessing differs from training.')


# ---------------------------------------------------------------------------
# 특징 표
# ---------------------------------------------------------------------------
def build_matrix(frame: pd.DataFrame, spec: dict, embeddings: np.ndarray | None) -> pd.DataFrame:
    sensors = build_sensor_features(frame, spec['sensor_features'],
                                    imputation=spec['sensor_imputation'])
    sensors = sensors.reset_index(drop=True)
    sensors.columns = [f'sensor_{c}' for c in sensors.columns]
    if embeddings is None:
        return sensors
    if len(embeddings) != len(frame):
        raise ValueError('Embedding rows do not match the frame.')
    emb = pd.DataFrame(embeddings, columns=[f'{EMBED_PREFIX}{i:04d}'
                                            for i in range(embeddings.shape[1])])
    return pd.concat([sensors, emb], axis=1)


def features_for(frame, csv_path, image_root, bundle_or_spec, *, embeddings_dir=None,
                 device='cuda', work_dir=None, _encoder=None):
    """학습·추론 공통: 센서 특징 + (미리 추출 또는 즉시 추출한) 이미지 벡터."""
    encoder = bundle_or_spec.get('encoder')
    if not encoder:
        return build_matrix(frame, bundle_or_spec, None), None
    l2 = bool((bundle_or_spec.get('embedding') or {}).get('l2_normalize', False))
    if embeddings_dir is None:
        holder = None
        if work_dir is None:
            holder = tempfile.TemporaryDirectory(prefix='doyu_embed_')
            work_dir = Path(holder.name) / encoder
        try:
            embeddings_dir = extract_embeddings(csv_path, image_root, encoder, work_dir,
                                                device=device, l2_normalize=l2,
                                                _encoder=_encoder)
            values, meta = load_embeddings(embeddings_dir, frame['sample_id'])
        finally:
            if holder is not None:
                holder.cleanup()
    else:
        values, meta = load_embeddings(embeddings_dir, frame['sample_id'])
    if meta.get('model') != encoder:
        raise ValueError(f'Embedding folder is {meta.get("model")}, expected {encoder}.')
    if bundle_or_spec.get('embedding'):
        check_embedding_compatible(bundle_or_spec['embedding'], meta)
    return build_matrix(frame, bundle_or_spec, values), meta


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def build_fusion_pipeline(model_name: str, columns, pca_dim: int, seed: int = 42) -> Pipeline:
    sensor_cols = [c for c in columns if not c.startswith(EMBED_PREFIX)]
    emb_cols = [c for c in columns if c.startswith(EMBED_PREFIX)]
    parts = [('sensor', SimpleImputer(strategy='median', keep_empty_features=True), sensor_cols)]
    if emb_cols:
        steps = [('scale', StandardScaler())]
        if pca_dim > 0:
            steps.append(('pca', PCA(n_components=pca_dim, random_state=seed)))
        parts.append(('image', Pipeline(steps), emb_cols))
    fusion = ColumnTransformer(parts, remainder='drop', verbose_feature_names_out=False)
    return Pipeline([('fusion', fusion), ('model', build_model(model_name, seed))])


def effective_pca_dim(pca_dim: int, n_rows: int, n_emb: int) -> int:
    if pca_dim <= 0 or n_emb == 0:
        return 0
    return max(1, min(pca_dim, n_emb, n_rows - 1))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABEL_LIST, average='macro', zero_division=0))


def save_metrics(y_true, y_pred, prefix: Path, *, verbose=True) -> dict:
    """classification_report·혼동행렬 저장. 반환값은 macro/weighted P·R·F1."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, labels=LABEL_LIST, output_dict=True,
                                   zero_division=0)
    pd.DataFrame(report).T.to_csv(f'{prefix}_classification_report.csv')
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABEL_LIST),
                 index=[f'actual_{c}' for c in LABEL_LIST],
                 columns=[f'pred_{c}' for c in LABEL_LIST]).to_csv(f'{prefix}_confusion_matrix.csv')
    summary = {}
    for avg in ('macro', 'weighted'):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_LIST,
                                                     average=avg, zero_division=0)
        summary.update({f'{avg}_precision': float(p), f'{avg}_recall': float(r),
                        f'{avg}_f1': float(f)})
    if verbose:
        print(classification_report(y_true, y_pred, labels=LABEL_LIST, digits=4,
                                    zero_division=0))
        for avg in ('macro', 'weighted'):
            print(f"{avg:>8} Precision: {summary[f'{avg}_precision']:.4f}  "
                  f"Recall: {summary[f'{avg}_recall']:.4f}  F1: {summary[f'{avg}_f1']:.4f}")
    return summary


__all__ = ['ENCODERS', 'LABEL_LIST', 'REPO_ROOT', 'build_fusion_pipeline', 'build_matrix',
           'effective_pca_dim', 'features_for', 'load_embeddings', 'macro_f1', 'read_frame',
           'save_metrics', 'sensor_spec', 'check_versions', 'embedding_info']
