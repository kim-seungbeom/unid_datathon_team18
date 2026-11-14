"""센서(승범) + 이미지 인코더(성현) + 최종 분류기(도유) 결합 학습·비교.

실행(레포 루트, 서버에서 사용자가 직접):
    python experiments/doyu/train_fusion.py --data_dir /home/ubuntu/data/battery_data_light \
        --config configs/sensor_history_voltage.json --encoder resnet18 \
        --train_embeddings <성현 추출 train 폴더> --val_embeddings <성현 추출 validation 폴더> \
        --out_dir runs/doyu/fusion/resnet18_voltage

- 모델 선택은 Train 내부 GroupKFold(experiment_id) CV. Validation은 모델별 최종 확인용.
- 결측 대체·표준화·PCA는 각 fold의 train 부분에서만 fit (Pipeline 내부).
- --train_embeddings/--val_embeddings를 생략하면 같은 extract()로 out_dir/embeddings/에 추출.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import fusion as F
from models import MODEL_NAMES
from schema import validate_split


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True, help='새 실험 폴더(덮어쓰지 않음)')
    p.add_argument('--config', type=Path, help='승범님 configs/*.json (sensor_features 등 사용)')
    p.add_argument('--sensor_features', nargs='+', help='config 대신 직접 지정')
    p.add_argument('--sensor_imputation', choices=['median', 'forward_fill'])
    p.add_argument('--encoder', choices=F.ENCODERS, default='resnet18')
    p.add_argument('--train_embeddings', type=Path)
    p.add_argument('--val_embeddings', type=Path)
    p.add_argument('--pca_dim', type=int, default=32, help='0이면 PCA 없이 표준화만')
    p.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda', help='즉시 추출할 때 인코더 장치')
    p.add_argument('--refit_train_val', action='store_true',
                   help='저장 모델을 Train+Validation으로 재학습(제출용, dataana 분석에는 부적합)')
    a = p.parse_args()
    if (a.train_embeddings is None) != (a.val_embeddings is None):
        p.error('--train_embeddings와 --val_embeddings는 함께 지정하세요.')
    a.out_dir.mkdir(parents=True, exist_ok=False)

    spec = F.sensor_spec(a.config, a.sensor_features, a.sensor_imputation)
    spec['encoder'] = None if a.encoder == 'none' else a.encoder
    train_csv = a.data_dir / 'train' / 'train.csv'
    val_csv = a.data_dir / 'validation' / 'validation.csv'
    train = F.read_frame(train_csv, labeled=True)
    val = F.read_frame(val_csv, labeled=True)
    validate_split(train, val)

    X, meta = F.features_for(train, train_csv, train_csv.parent, spec,
                             embeddings_dir=a.train_embeddings, device=a.device,
                             work_dir=a.out_dir / 'embeddings' / 'train')
    if meta is not None:
        spec['embedding'] = F.embedding_info(meta)
    X_val, _ = F.features_for(val, val_csv, val_csv.parent, spec,
                              embeddings_dir=a.val_embeddings, device=a.device,
                              work_dir=a.out_dir / 'embeddings' / 'validation')
    y, y_val = train['label'].to_numpy(), val['label'].to_numpy()
    columns = list(X.columns)
    n_emb = sum(c.startswith(F.EMBED_PREFIX) for c in columns)
    print(f'features: sensor={len(columns) - n_emb} image={n_emb} (encoder={a.encoder}, '
          f'pca_dim={a.pca_dim})', flush=True)

    groups = train['experiment_id'].to_numpy()
    n_splits = min(a.folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError('Need >= 2 independent experiments for GroupKFold.')
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))

    rows = []
    for name in a.models:
        print(f'\n=== {name} ===', flush=True)
        scores = []
        for fold, (tr, te) in enumerate(splits, 1):
            pca = F.effective_pca_dim(a.pca_dim, len(tr), n_emb)
            model = F.build_fusion_pipeline(name, columns, pca, a.seed)
            model.fit(X.iloc[tr], y[tr])
            scores.append(F.macro_f1(y[te], model.predict(X.iloc[te])))
            print(f'fold {fold}: {scores[-1]:.5f}', flush=True)
        rows.append({'model': name, 'cv_macro_f1_mean': float(np.mean(scores)),
                     'cv_macro_f1_std': float(np.std(scores)),
                     **{f'fold_{i + 1}': s for i, s in enumerate(scores)}})
    summary = pd.DataFrame(rows).sort_values('cv_macro_f1_mean', ascending=False)
    summary.to_csv(a.out_dir / 'cv_scores.csv', index=False)
    print('\nCV ranking:\n', summary[['model', 'cv_macro_f1_mean', 'cv_macro_f1_std']]
          .to_string(index=False))
    print('Selected from TRAIN CV:', summary.iloc[0]['model'])

    val_rows = []
    for name in summary['model']:
        pca = F.effective_pca_dim(a.pca_dim, len(X), n_emb)
        model = F.build_fusion_pipeline(name, columns, pca, a.seed).fit(X, y)
        pred = model.predict(X_val)
        print(f'\n[{name}] Holdout validation')
        metrics = F.save_metrics(y_val, pred, a.out_dir / f'validation_{name}')
        pd.DataFrame({'sample_id': val['sample_id'], 'experiment_id': val['experiment_id'],
                      'relative_time_s': val['relative_time_s'], 'actual': y_val, 'pred': pred}
                     ).to_csv(a.out_dir / f'validation_{name}_predictions.csv', index=False)
        val_rows.append({'model': name, **metrics})
        if a.refit_train_val:
            X_all = pd.concat([X, X_val], ignore_index=True)
            pca = F.effective_pca_dim(a.pca_dim, len(X_all), n_emb)
            model = F.build_fusion_pipeline(name, columns, pca, a.seed).fit(
                X_all, np.r_[y, y_val])
        bundle = {'kind': 'fusion', 'format_version': 1, 'pipeline': model, 'model_name': name,
                  'columns': columns, 'pca_dim': a.pca_dim, 'seed': a.seed,
                  'refit_train_val': bool(a.refit_train_val), 'label_values': F.LABEL_LIST,
                  **spec}
        joblib.dump(bundle, a.out_dir / f'best_{name}.joblib')
        print('Saved:', a.out_dir / f'best_{name}.joblib')
    pd.DataFrame(val_rows).to_csv(a.out_dir / 'validation_scores.csv', index=False)
    (a.out_dir / 'run_config.json').write_text(json.dumps({
        **{k: v for k, v in spec.items()}, 'pca_dim': a.pca_dim, 'models': a.models,
        'folds': n_splits, 'seed': a.seed, 'refit_train_val': bool(a.refit_train_val),
        'train_embeddings': str(a.train_embeddings) if a.train_embeddings else 'extracted',
        'val_embeddings': str(a.val_embeddings) if a.val_embeddings else 'extracted',
    }, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
