"""Validation 예측 분석: 실험별 실제/예측 단계 그래프와 단계 전환 감지 지연 시간.

사용법 (레포 루트에서):
    python experiments/doyu/dataana.py --data_dir /home/ubuntu/data/battery_data_light \
        --model runs/doyu/tabular/best_lightgbm.joblib --out_dir runs/doyu/dataana/lightgbm

결과는 --out_dir(기본 runs/doyu/dataana/)에 저장됩니다. runs/는 Git 제외 폴더입니다.
    val_predictions.csv   : 샘플별 실제(actual) / 예측(pred)
    plots/<experiment_id>.png : 실험별 실제 vs 예측 step 그래프
    transitions.csv       : 실제 단계가 상승한 시점 목록
    delays.csv            : 전환별 감지 지연 시간(초), 감지 실패는 NaN
    classification_report.csv : 클래스별 precision / recall / f1
    confusion_matrix.csv / .png / _norm.png : 혼동 행렬 (개수, 행 기준 비율)
"""
import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use('Agg')  # 화면 없는 서버에서 파일로만 저장
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABELS = [1, 2, 3, 4, 5, 6]

sys.path.insert(0, str(Path(__file__).resolve().parent))  # experiments/doyu
import models  # noqa: E402, F401  (best_*.joblib 언피클에 필요한 ShiftLabels)
import train_compare as T  # noqa: E402  (load_split 재사용)
from organizer_features import REPO_ROOT  # noqa: E402


def build_result(val, X_va, y_va, m, features):
    result = val[['sample_id', 'experiment_id', 'relative_time_s']].copy()
    result['actual'] = pd.Series(y_va).to_numpy()
    result['pred'] = m.predict(X_va[features])
    return result.sort_values(['experiment_id', 'relative_time_s']).reset_index(drop=True)


def plot_experiment(exp, exp_id, out_path):
    plt.figure(figsize=(12, 4))
    plt.step(exp['relative_time_s'], exp['actual'], where='post', label='Actual')
    plt.step(exp['relative_time_s'], exp['pred'], where='post', linestyle='--', label='Prediction')
    plt.xlabel('Time (s)')
    plt.ylabel('Risk stage')
    plt.yticks(range(1, 7))
    plt.title(exp_id)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def report_metrics(result, out_dir):
    y_true, y_pred = result['actual'], result['pred']

    # 클래스별 precision / recall / f1 + macro·weighted 평균
    print('[분류 지표]')
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0))
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / 'classification_report.csv')

    print('Macro Precision: {:.4f}'.format(precision_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)))
    print('Macro Recall   : {:.4f}'.format(recall_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)))
    print('Macro F1       : {:.4f}'.format(f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)))

    # 혼동 행렬 (행 = 실제, 열 = 예측)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f'actual_{c}' for c in LABELS], columns=[f'pred_{c}' for c in LABELS])
    print('\n[혼동 행렬] (행 = 실제, 열 = 예측)')
    print(cm_df)
    cm_df.to_csv(out_dir / 'confusion_matrix.csv')

    # 개수 / 행 기준 비율(= 실제 단계별 recall 분포) 두 가지 그림 저장
    for normalize, name, fmt in [(None, 'confusion_matrix.png', 'd'), ('true', 'confusion_matrix_norm.png', '.2f')]:
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=LABELS, normalize=normalize,
            values_format=fmt, cmap='Blues', ax=ax, colorbar=False,
        )
        ax.set_xlabel('Predicted stage')
        ax.set_ylabel('Actual stage')
        ax.set_title('Confusion matrix' + (' (row-normalized)' if normalize else ''))
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=120)
        plt.close(fig)
    print()


def find_transitions(exp):
    exp = exp.sort_values('relative_time_s').copy()
    # 실제 위험 단계가 이전보다 상승했는지 확인
    exp['actual_change'] = exp['actual'].diff()
    # 위험 단계가 상승한 행만 추출
    return exp[exp['actual_change'] > 0]


def detection_delays(exp, transitions):
    delays = []
    for _, row in transitions.iterrows():
        change_time = row['relative_time_s']
        target_stage = row['actual']

        # 전환 이후, 다음 실제 단계 전환 전까지 검사
        future = exp[exp['relative_time_s'] >= change_time].copy()
        next_change = future[future['actual'] != target_stage]
        if not next_change.empty:
            future = future[future['relative_time_s'] < next_change.iloc[0]['relative_time_s']]

        # 목표 위험 단계 이상을 예측한 첫 시점
        detected = future[future['pred'] >= target_stage]
        if not detected.empty:
            delay = detected.iloc[0]['relative_time_s'] - change_time
        else:
            delay = float('nan')  # 해당 구간에서 감지 실패

        delays.append({
            'experiment_id': exp['experiment_id'].iloc[0],
            'stage': int(target_stage),
            'change_time_s': change_time,
            'delay_s': delay,
        })
    return delays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='best_{xgboost,lightgbm,mlp}.joblib')
    parser.add_argument('--out_dir', type=Path, default=REPO_ROOT / 'runs' / 'doyu' / 'dataana')
    parser.add_argument('--embeddings', type=Path,
                        help='fusion 모델: validation 인코더 추출 폴더(생략 시 즉시 추출)')
    parser.add_argument('--device', default='cuda', help='fusion 모델 즉시 추출 장치')
    args = parser.parse_args()

    out_dir = args.out_dir
    plot_dir = out_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Train만으로 학습된 best_*.joblib 사용 (--refit_train_val 모델은 Validation을 학습에 써서 부적합)
    bundle = joblib.load(args.model)
    m = bundle['pipeline']
    if bundle.get('kind') == 'fusion':  # train_fusion.py: 센서(승범) + 인코더(성현)
        import fusion as F

        F.check_versions(bundle)
        val_csv = args.data_dir / 'validation' / 'validation.csv'
        val = F.read_frame(val_csv, labeled=True)
        val['relative_time_s'] = pd.to_numeric(val['relative_time_s'], errors='coerce')
        X_va, _ = F.features_for(val, val_csv, val_csv.parent, bundle,
                                 embeddings_dir=args.embeddings, device=args.device)
        y_va, features = val['label'], bundle['columns']
    else:  # train_compare.py: 운영진 v7 33개 특징
        val, X_va, y_va = T.load_split(args.data_dir, 'validation')
        features = bundle['features']

    result = build_result(val, X_va, y_va, m, features)
    result.to_csv(out_dir / 'val_predictions.csv', index=False)
    report_metrics(result, out_dir)

    all_transitions, all_delays = [], []
    for exp_id, exp in result.groupby('experiment_id', sort=False):
        exp = exp.sort_values('relative_time_s')
        plot_experiment(exp, exp_id, plot_dir / f'{exp_id}.png')

        transitions = find_transitions(exp)
        all_transitions.append(transitions[['experiment_id', 'relative_time_s', 'actual']])
        all_delays.extend(detection_delays(exp, transitions))

    pd.concat(all_transitions).to_csv(out_dir / 'transitions.csv', index=False)
    delay_df = pd.DataFrame(all_delays)
    delay_df.to_csv(out_dir / 'delays.csv', index=False)

    print(delay_df)
    print('\n[단계별 감지 지연 요약]')
    summary = delay_df.groupby('stage')['delay_s'].agg(
        n='size',
        detected='count',
        mean_s='mean',
        median_s='median',
        max_s='max',
    )
    summary['missed'] = summary['n'] - summary['detected']
    print(summary)
    print(f'\n저장 위치: {out_dir}')


if __name__ == '__main__':
    main()
