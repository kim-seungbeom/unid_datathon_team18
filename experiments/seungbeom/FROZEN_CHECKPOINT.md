# Frozen 내부 최고 가중치와 전체 Train 재학습 비교

기존 frozen + 원본 ExtraTrees 50:50은 보존한다. 원래 frozen 설정을 그대로 읽어 같은 초기 가중치, raw 센서 5개, 중앙값·표준화, ResNet18 고정, AdamW, 일반 CE, shuffle, seed, Train 내부 실험 분할을 사용한다. 실험별 가중치·클래스 가중치·새 피처를 추가하지 않는다. 과거 학습 로그만으로 당시 가중치를 복구할 수 없으므로 새 학습이 필요하다.

- 내부 Train holdout의 **Macro F1이 엄격히 증가할 때 즉시** `inner_best.pt`를 교체한다. 동점은 앞선 epoch를 유지한다. 가중치는 CPU에 복사해 저장하므로 다음 optimizer 업데이트에 따라 바뀌지 않는다.
- 체크포인트에는 전체 신경망 가중치(고정 backbone 포함), 그 모델의 전처리, 모델 설정, 라벨 순서, 선택 epoch·점수·분할과 재현 메타데이터가 들어 있다. 추론용이며 optimizer 상태를 이용한 학습 재개 기능은 아니다.
- 내부 학습 부분으로만 전처리·가중치를 학습한다. 내부 holdout으로 epoch를 선택하고, 저장 모델을 불러와 내부 검증 확률과 점수가 재현되는지 확인한다.
- 선택한 epoch 수만큼 전체 Train에서 **새로 초기화하고 전처리도 다시 학습**한다. 결과를 별도 `refit.pt`에 저장한다. `inner_best.pt`를 이어서 학습하거나 덮어쓰지 않는다.
- 두 모델과 각각의 원본 ExtraTrees 50:50 조합을 같은 공식 Validation에서 비교한다. 공식 Validation은 이 실행의 epoch 선택에 사용하지 않는다. 내부 점수와 공식 점수는 서로 다른 행의 성능이다.
- 기존 모델 재현, 새 체크포인트 재현, 파일/소스/환경 일치 검사를 수행한다. `train` 명령은 Test를 읽거나 제출 CSV를 만들지 않는다.

## 서버 실행

개인 clone에서 실행 중인 학습이 끝난 뒤:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short --branch
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-frozen-checkpoint
```

tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail

LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/frozen_checkpoint.py train \
  --recipe runs/seungbeom/frozen_tree_blend_validation/recipe.json \
  --data_dir /home/ubuntu/data/battery_data_light \
  --output_dir runs/seungbeom/frozen_checkpoint_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/frozen_checkpoint_seed42.log
```

기존 모델 재현 → `Stage 1/2` 내부 선택·최고 체크포인트 저장 → `Stage 2/2` 전체 Train 재학습 → 공식 Validation 비교 순서다. 최고 갱신 시 `Saved inner_best.pt: epoch=... inner_macro_f1=...`가 출력된다. 중간에 내려간 점수로 최고 파일을 교체하지 않는다. 종료할 때까지 같은 clone에서 pull하지 않는다. 재실행은 새 output_dir/로그 이름을 사용한다.

```bash
python experiments/seungbeom/frozen_checkpoint.py report \
  --output_dir runs/seungbeom/frozen_checkpoint_seed42 \
  --class_details \
  > runs/seungbeom/frozen_checkpoint_seed42/comparison.txt

less -S runs/seungbeom/frozen_checkpoint_seed42/comparison.txt
```

| 표의 이름 | 의미 |
|---|---|
| original_frozen / original_50_50 | 보존한 기존 신경망 / 기존 결합 |
| inner_best_neural / inner_best_50_50 | 내부 최고 체크포인트 / 그 모델과 트리의 50:50 |
| refit_neural / refit_50_50 | 새 전체 Train 재학습 / 그 모델과 트리의 50:50 |

처음의 `Best inner Train Macro F1`은 선택용 점수다. 이어지는 표는 모두 **공식 Validation** 점수다. 3→2, 5→4 및 반대 방향 오류도 함께 비교한다. 내부 최고 모델과 refit은 학습에 쓴 행 수와 전처리가 다르다. 원래 refit과 같은 epoch가 선택되면 새 refit 결과는 기존 frozen 재현 확인 역할도 한다. 최고 체크포인트를 보존한다고 Public 상승이 보장되지는 않는다. 기존 선택에 사용했던 Validation이므로 독립 최종 성능이나 3-fold CV로 해석하지 않는다.

## 비교한 후보 내보내기

완료된 실행에서 후보를 고른 다음 별도 파일로 추론한다. 내부 최고 모델과 트리의 50:50 예시:

```bash
python -u experiments/seungbeom/frozen_checkpoint.py predict \
  --run_dir runs/seungbeom/frozen_checkpoint_seed42 \
  --method inner_best_50_50 \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output runs/seungbeom/submissions_frozen_checkpoint_seed42/submission_inner_best_50_50.csv \
  --device cuda
```

`--method`에는 `inner_best_neural`, `inner_best_50_50`, `refit_neural`, `refit_50_50` 중 하나를 지정하고 출력 파일명도 다르게 한다. 자동으로 후보를 선택하거나 리더보드에 업로드하지 않는다. 추론에는 현재 행 이미지·센서와 저장 모델·전처리만 사용하며 Train/Validation CSV나 실험별 행 수를 다시 읽지 않는다. 양식의 ID·순서·1~6 정수 라벨을 검증한다. 생성 가중치·데이터·예측 결과를 Git에 올리지 않는다.
