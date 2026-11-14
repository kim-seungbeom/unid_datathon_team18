"""3개 tabular 모델(best_*.joblib)로 제출 CSV 생성 + 정답이 있으면 precision/recall/f1.

운영진 v7 test.py와 같은 bundle 규격({'pipeline','features',...})을 사용한다.
입력 CSV에 label 열이 있으면 분류 지표와 혼동행렬을 출력·저장하고, 없으면 예측만 한다.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from organizer_features import M
# Unpickling requires models.ShiftLabels; import it here to give clearer dependency.
import models  # noqa: F401


def report_metrics(frame, pred, output):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support)

    y_true = pd.to_numeric(frame['label'], errors='raise').astype(int).to_numpy()
    y_pred = pred.loc[frame.sample_id.values].to_numpy()
    print('\n[분류 지표]')
    print(classification_report(y_true, y_pred, labels=M.LABELS, digits=4, zero_division=0))
    for avg in ('macro', 'weighted'):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=M.LABELS, average=avg, zero_division=0)
        print(f'{avg:>8} Precision: {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}')

    stem = output.with_suffix('')
    report = classification_report(y_true, y_pred, labels=M.LABELS, output_dict=True,
                                   zero_division=0)
    pd.DataFrame(report).T.to_csv(f'{stem}_classification_report.csv')
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=M.LABELS),
                 index=[f'actual_{c}' for c in M.LABELS],
                 columns=[f'pred_{c}' for c in M.LABELS]).to_csv(f'{stem}_confusion_matrix.csv')
    print(f'저장: {stem}_classification_report.csv, {stem}_confusion_matrix.csv')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--image_root',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('submission.csv'))
    p.add_argument('--sample_submission',type=Path)
    a=p.parse_args()
    frame=pd.read_csv(a.input,dtype={'sample_id':'string','experiment_id':'string'})
    if not frame.sample_id.is_unique: raise ValueError('sample_id not unique')
    paths=M.resolve_image_paths(frame,[a.image_root,a.image_root.parent])
    X=M.build_features(frame,M.extract_image_features(frame,paths))
    b=joblib.load(a.model)
    preds=pd.Series(b['pipeline'].predict(X[b['features']]).astype(int),
                    index=frame.sample_id.to_numpy())
    ids=(pd.read_csv(a.sample_submission,dtype={'sample_id':'string'})['sample_id'].tolist()
         if a.sample_submission else frame.sample_id.tolist())
    if set(ids)!=set(preds.index) or len(ids)!=len(preds):
        raise ValueError('sample_submission IDs do not match the input')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    submission=pd.DataFrame({'sample_id':ids,'label':preds.loc[ids].to_numpy()})
    if not submission.label.isin(M.LABELS).all(): raise ValueError('labels must be 1..6')
    submission.to_csv(a.output,index=False)
    print('Saved',a.output)
    if 'label' in frame.columns:
        report_metrics(frame, preds, a.output)
    else:
        print('입력에 label 열이 없어 지표 계산을 건너뜁니다.')

if __name__ == '__main__': main()
