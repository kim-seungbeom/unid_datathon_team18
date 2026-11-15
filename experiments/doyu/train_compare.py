"""33차원 특징으로 XGBoost / LightGBM / MLP를 동일한 GroupKFold에서 비교.

실행(레포 루트): python experiments/doyu/train_compare.py --data_dir /home/ubuntu/data/battery_data_light
결과 기본 위치: runs/doyu/tabular/ (best_{xgboost,lightgbm,mlp}.joblib, cv_scores 등, Git 제외)
주의: 모델 선택은 Train 내부 CV로 하며 제공된 validation은 최종 확인에만 사용.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from organizer_features import M, REPO_ROOT
from models import MODEL_NAMES, build_model


def load_split(data_dir, split):
    frame = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id':'string','experiment_id':'string'})
    paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
    img = M.extract_image_features(frame, paths)
    X = M.build_features(frame, img)
    y = frame['label'].astype(int).to_numpy()
    if not np.isin(y, M.LABELS).all():
        raise ValueError('Expected 1..6 labels')
    return frame, X, y


def evaluate(model, X, y, frame, out_dir, name):
    pred = model.predict(X)
    score = f1_score(y, pred, labels=M.LABELS, average='macro', zero_division=0)
    pd.DataFrame(confusion_matrix(y, pred, labels=M.LABELS),
                 index=[f'true_{i}' for i in M.LABELS],
                 columns=[f'pred_{i}' for i in M.LABELS]).to_csv(out_dir / f'{name}_confusion_matrix.csv')
    report = classification_report(y, pred, labels=M.LABELS, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(out_dir / f'{name}_classification_report.csv')
    p = frame[['sample_id','experiment_id','relative_time_s','image_path']].copy()
    p['actual'] = y
    p['pred'] = pred
    p['correct'] = (pred == y)
    p.to_csv(out_dir / f'{name}_predictions.csv', index=False)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, default=REPO_ROOT / 'runs' / 'doyu' / 'tabular')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--refit_train_val', action='store_true')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tr, X, y = load_split(args.data_dir, 'train')
    va, X_val, y_val = load_split(args.data_dir, 'validation')
    if not set(tr.experiment_id).isdisjoint(va.experiment_id):
        raise ValueError('train/validation experiment_id overlap!')
    groups = tr.experiment_id.to_numpy()
    n_splits = min(args.folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError('Need >= 2 independent experiments')
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
    rows = []
    for name in args.models:
        scores=[]
        print(f'\n=== {name} ===', flush=True)
        for fold,(ii,jj) in enumerate(splits, 1):
            m=build_model(name)
            m.fit(X.iloc[ii],y[ii])
            p=m.predict(X.iloc[jj])
            s=f1_score(y[jj],p,labels=M.LABELS,average='macro',zero_division=0)
            scores.append(s)
            print(f'fold {fold}: {s:.5f}',flush=True)
        rows.append({'model':name,'cv_macro_f1_mean':float(np.mean(scores)),
                     'cv_macro_f1_std':float(np.std(scores)),
                     **{f'fold_{i+1}':s for i,s in enumerate(scores)}})
    summary=pd.DataFrame(rows).sort_values('cv_macro_f1_mean',ascending=False)
    summary.to_csv(args.out_dir/'cv_scores.csv', index=False)
    selected=summary.iloc[0]['model']
    print('\nCV ranking:\n',summary[['model','cv_macro_f1_mean','cv_macro_f1_std']].to_string(index=False))
    print('Selected from TRAIN CV:',selected)
    # 모델마다 최종 모델을 따로 저장한다. 선택은 여전히 TRAIN CV 기준이며,
    # validation 점수는 모델별 참고용으로만 기록한다.
    val_rows=[]
    for name in summary['model']:
        model=build_model(name).fit(X,y)
        val_score=evaluate(model,X_val,y_val,va,args.out_dir,f'validation_{name}')
        print(f'[{name}] Holdout validation Macro F1: {val_score:.5f}')
        val_rows.append({'model':name,'holdout_macro_f1':val_score})
        if args.refit_train_val:
            model=build_model(name).fit(pd.concat([X,X_val],ignore_index=True),np.r_[y,y_val])
        bundle={'pipeline':model,'features':list(X.columns),'model_name':name,
                'refit_train_val':bool(args.refit_train_val)}
        model_out=args.out_dir/f'best_{name}.joblib'
        joblib.dump(bundle,model_out)
        print('Saved:',model_out)
    pd.DataFrame(val_rows).to_csv(args.out_dir/'validation_scores.csv',index=False)

if __name__ == '__main__':
    main()
