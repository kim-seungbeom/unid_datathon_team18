# 실험별 균형 학습 비교

기존 frozen 신경망 + 원본 ExtraTrees 50:50을 보존하고, frozen 신경망의 **행별 loss 가중치만** 변경한다. 행이 많은 실험이 학습에 과도하게 반영되는지 확인하는 한 가지 가설이다. 여러 실험에서 3→2, 5→4 오류가 반복된다는 사실만으로 이 가설이 증명되지는 않는다.

학습 단계의 행 수를 N, 실험 수를 G, 해당 실험의 행 수를 n_g라 하면 각 행의 가중치는 `N / (G * n_g)`이다. 전체 가중치 평균은 1이고, 각 실험의 가중치 합은 N/G로 같다. loss는 `mean(weight * per_row_cross_entropy)`이며 미니배치 안에서 가중치 합으로 다시 나누지 않는다. 클래스 가중치·resampling·특정 실험 ID 예외처리·예측 라벨 보정은 추가하지 않는다.

- 초기 가중치, frozen ResNet18, raw 센서 5개, 중앙값·표준화, shuffle, AdamW, learning rate, dropout은 저장된 frozen 설정과 같다. 초기 전체 가중치와 frozen backbone 상태 해시도 검사한다.
- 기존 모델은 재학습하지 않고 저장된 모델의 Validation 지표를 먼저 재현한다. 대조군 코드와 가중치는 변경하지 않는다.
- 기존과 동일한 Train 내부 실험 분할에서 epoch 수를 선택한다. 이때 가중치와 전처리는 내부 학습 부분에서만 계산한다. 공식 Validation은 epoch 선택에 사용하지 않는다.
- 선택된 epoch 수만큼 전체 Train에서 새로 초기화해 학습한다. 가중치와 전처리도 전체 Train에서 새로 계산한다. 따라서 두 조건의 epoch 수는 달라질 수 있지만 선택 절차는 같다.
- 전처리는 기존처럼 행별 동일 비중으로 계산하고, loss만 실험별 균형을 적용한다. Validation 평가에는 가중치를 적용하지 않는다.
- 원본 ExtraTrees와 50:50 확률 평균은 고정한다. Test는 `train` 명령에서 읽지 않는다.

## 서버 실행

실행 중인 학습이 없는 개인 clone에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short --branch
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-experiment-balance
```

tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail

LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/experiment_balance.py train \
  --recipe runs/seungbeom/frozen_tree_blend_validation/recipe.json \
  --data_dir /home/ubuntu/data/battery_data_light \
  --output_dir runs/seungbeom/experiment_balance_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/experiment_balance_seed42.log
```

진행 순서는 원본 지표 재현 → `Stage 1/2` 내부 Train epoch 선택 → `Stage 2/2` 전체 Train 재학습 → 공식 Validation 비교·저장 모델 재현이다. 두 단계 모두 신경망 학습이므로 앞선 진단보다 시간이 걸린다. `selection_history.json`, `refit_history.json`은 매 epoch 갱신된다. 기존 폴더를 덮어쓰거나 중단 지점부터 재개하지 않으므로 재실행 시 새 output_dir와 로그 이름을 사용한다.

```bash
python experiments/seungbeom/experiment_balance.py report \
  --output_dir runs/seungbeom/experiment_balance_seed42 \
  --class_details \
  > runs/seungbeom/experiment_balance_seed42/comparison.txt

less -S runs/seungbeom/experiment_balance_seed42/comparison.txt
```

보고서는 추론을 다시 하지 않는다. `balanced_neural`은 기존 `original_frozen`과, `balanced_50_50`은 기존 `original_50_50`과 비교한다. 첫 표의 delta는 모두 기존 50:50 기준이다.

전체 Macro F1과 6개 클래스 F1을 먼저 보고, 3→2·5→4·5→3 오류율을 확인한다. 반대 방향인 2→3·4→5도 함께 표시해 오류가 이동했는지 확인한다. `improved/tied/worse`는 실제 해당 클래스가 존재하는 Validation 실험 중 오류가 감소/동일/증가한 실험 수다. 단순 오류 개수는 Macro F1 기여도가 아니고, 실험별 작은 표본의 변화도 함께 고려해야 한다. 실험별 상세 집계는 `group_pair_errors.csv`에 저장한다.

이 비교는 하나의 Train 내부 그룹 holdout/seed다. 3-fold 결과가 아니며, 이미 모델 선택에 사용한 Validation의 독립 최종 성능으로 해석하지 않는다. 균형 학습은 유효 클래스 비중도 바꿀 수 있으므로 약한 클래스 개선과 다른 클래스 하락을 함께 확인한다. 원본·개별 정답/예측 및 생성 결과는 Git에 올리지 않는다.

## 비교 후 선택한 후보 CSV 생성

학습 명령은 제출 CSV를 만들지 않는다. 결과를 확인한 뒤 후보를 내보내려면:

```bash
python experiments/seungbeom/experiment_balance.py predict \
  --run_dir runs/seungbeom/experiment_balance_seed42 \
  --method balanced_50_50 \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output runs/seungbeom/submissions_experiment_balance_seed42/submission_balanced_50_50.csv \
  --device cuda
```

신경망 단독은 `--method balanced_neural`과 새 출력 파일명을 지정한다. 추론은 현재 행의 이미지·센서와 저장된 모델·전처리만 사용하며 실험 ID·행 수·정답을 사용하지 않는다. Train/Validation CSV를 다시 열지 않고, 제출 양식의 ID/순서와 1~6 정수 라벨을 확인한다. 기존 제출 파일은 덮어쓰지 않는다.
