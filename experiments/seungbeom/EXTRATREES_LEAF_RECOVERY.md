# 진단한 leaf 실험 이어서 완료하기

원래 ExtraTrees와 재학습 leaf 2는 같은 하이퍼파라미터라도 동일한 가중치가 아니다.
PCA·트리 상태의 차이를 `allclose` 허용오차로 덮거나 기존 모델을 교체하지 않는다.
원래 모델을 별도로 보존하고, 이미 학습한 leaf 2·5·10을 같은 재학습 집단으로 비교한다.
새 집단 안에서는 leaf 값 외의 파라미터와 fitted 전처리/PCA가 같아야 한다.

`extratrees_leaf_recovery.py`는 기존 sweep/audit 소스를 수정하지 않는 별도 복구 경로다.
진단 파일·원래 모델·Train/Validation CSV·캐시의 해시를 확인하고, 저장 후보의 파라미터,
클래스, 피처 순서, 학습 행 수, 공유 전처리/PCA 상태를 검사한다. 기존 실패 실행에는
학습 전 manifest가 없었으므로 새 기록은 가져오는 시점의 검증 기록이며 과거 코드 지문을
소급해 증명하는 기록은 아니다. 모델은 신뢰하는 본인 서버 실행에서 만든 파일만 사용한다.

## 1. 코드 받기와 tmux

실행 중인 작업이 끝난 clone에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-et-leaf
```

이미 tmux 안이면 마지막 줄은 생략한다. tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
set -o pipefail
```

## 2. 저장 모델 재사용·Validation 재확인·남은 트리 CV

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/extratrees_leaf_recovery.py recover \
  --recipe runs/seungbeom/frozen_tree_blend_validation/recipe.json \
  --failed_run runs/seungbeom/extratrees_leaf_sweep_seed42 \
  --audit runs/seungbeom/extratrees_leaf_audit_seed42/audit.json \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --validation_csv /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --output_dir runs/seungbeom/extratrees_leaf_recovered_seed42 \
  --group_cv \
  --device cuda \
  2>&1 | tee runs/seungbeom/extratrees_leaf_recovered_seed42.log
```

Train 전체용 트리와 신경망은 다시 학습하지 않는다. 원래 실패 폴더를 읽고 새 폴더에
복사한다. Validation 이미지 추론은 다시 수행한다. 기존 모델과 세 후보 각각의 저장된
집계 지표가 모두 재현돼야 완료된다. 원래 leaf 2와 재학습 leaf 2 사이의 개별 확률·라벨
차이는 별도 진단으로 기록하며, 둘이 같은 모델이라고 표시하지 않는다.

`--group_cv`는 이전에 중단돼 실행하지 못한 Train 실험 단위 3-fold를 추가한다.
각 fold의 학습 행으로 전처리/PCA와 트리를 다시 학습한다. 기존 frozen은 Train 전체를
봤으므로 CV에는 합치지 않는다. 이를 생략하면 공식 Validation 비교만 완료한다.

## 3. 결과 확인

```bash
python experiments/seungbeom/extratrees_leaf_recovery.py report \
  --recipe runs/seungbeom/extratrees_leaf_recovered_seed42/recipe.json \
  --class_details \
  | tee runs/seungbeom/extratrees_leaf_recovered_seed42/comparison.txt
```

- `original_frozen_tree_50_50`: 원래 약 0.89 조합
- `refit_blend_leaf2_50_50`: 같은 설정으로 재학습한 leaf 2 + frozen
- `refit_blend_leaf5_50_50`, `refit_blend_leaf10_50_50`: 새 후보
- `delta_vs_original_blend`: 보존한 제출 후보 대비 Validation 차이
- `delta_vs_refit_leaf2_blend`: 같은 fitted 전처리/PCA를 사용한 재학습 집단 내 비교

Original과 refit은 F1이 같아도 서로 다른 모델이다. 후보의 전체 비교는 공식 Validation,
트리 단독의 분할 변동 확인은 Group CV로 한다. 둘의 점수를 직접 빼서 비교하지 않으며,
Public/Private 점수를 환산하지 않는다. 승자를 자동 선택하거나 기존 모델을 바꾸지 않는다.

## 4. 후보 CSV 생성

복구가 성공한 다음 실행한다. Test는 이 단계에서 추론에만 사용한다.

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/extratrees_leaf_recovery.py predict \
  --recipe runs/seungbeom/extratrees_leaf_recovered_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --leaves 5 10 \
  --output_dir runs/seungbeom/submissions_extratrees_leaf_recovered_seed42 \
  --device cuda
```

출력: `submission_frozen_refit_extratrees_leaf5_50_50.csv`,
`submission_frozen_refit_extratrees_leaf10_50_50.csv`.
`--leaves 2`를 명시하면 재학습 leaf 2 후보가 나오며 원래 0.89 모델의 CSV가 아니다.

추론은 복사한 모델과 현재 입력 행만 사용한다. Train·Validation CSV/캐시, 실패 폴더,
진단 파일을 다시 열지 않는다. 모델·코드·환경 변경과 제출 양식 ID/순서를 검사한다.
생성한 CSV만 제출하며 recipe/manifest와 모델은 서버에 보존한다. 기존 파일은 덮어쓰지 않는다.
