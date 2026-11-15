"""train_fusion.py의 best_*.joblib로 제출 CSV 생성 + 정답(label)이 있으면 precision/recall/f1.

이미지 벡터는 기본적으로 저장된 인코더 설정으로 입력 이미지에서 직접 계산한다(입력 행만 사용).
이미 같은 인코더로 추출한 폴더가 있으면 --embeddings로 지정할 수 있다.

예) python experiments/doyu/predict_fusion.py --input $DATA/test/test.csv --image_root $DATA/test \
        --model runs/doyu/fusion/<실험>/best_lightgbm.joblib \
        --output outputs/doyu/fusion/<실험>/lightgbm/submission.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

import fusion as F


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--image_root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--embeddings', type=Path, help='같은 인코더로 미리 추출한 폴더(선택)')
    p.add_argument('--sample_submission', type=Path)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists. Choose a new path.')

    bundle = joblib.load(a.model)
    if bundle.get('kind') != 'fusion' or bundle.get('label_values') != F.LABEL_LIST:
        raise ValueError('Not a train_fusion.py bundle.')
    F.check_versions(bundle)
    frame = F.read_frame(a.input, labeled=False)
    X, _ = F.features_for(frame, a.input, a.image_root, bundle, embeddings_dir=a.embeddings,
                          device=a.device)
    if list(X.columns) != bundle['columns']:
        raise ValueError('Feature columns differ from training.')
    preds = pd.Series(bundle['pipeline'].predict(X).astype(int), index=frame['sample_id'])
    ids = (pd.read_csv(a.sample_submission, dtype={'sample_id': str},
                       keep_default_na=False)['sample_id'].tolist()
           if a.sample_submission else frame['sample_id'].tolist())
    if set(ids) != set(preds.index) or len(ids) != len(preds):
        raise ValueError('sample_submission IDs do not match the input')
    submission = pd.DataFrame({'sample_id': ids, 'label': preds.loc[ids].to_numpy()})
    if not submission['label'].isin(F.LABEL_LIST).all():
        raise ValueError('labels must be 1..6')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8', newline='') as handle:
        submission.to_csv(handle, index=False)
    print('Saved', a.output)
    if 'label' in frame.columns:
        y_true = F.read_frame(a.input, labeled=True)['label'].to_numpy()
        print('\n[분류 지표]')
        F.save_metrics(y_true, preds.loc[frame['sample_id']].to_numpy(), a.output.with_suffix(''))
        print(f'저장: {a.output.with_suffix("")}_classification_report.csv / _confusion_matrix.csv')
    else:
        print('입력에 label 열이 없어 지표 계산을 건너뜁니다.')


if __name__ == '__main__':
    main()
