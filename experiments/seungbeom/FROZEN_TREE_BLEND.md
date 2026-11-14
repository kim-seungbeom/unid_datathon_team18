# 새 frozen + 기존 ExtraTrees 50:50

`resnet_finetune.py`의 새 `frozen` 체크포인트와 원래 midpoint recipe의 ExtraTrees를
그대로 불러와 6개 클래스 확률을 각각 0.5씩 평균한다. 정수 라벨을 평균하지 않는다.
`layer4`, Train+Validation 재학습 ExtraTrees, CV3 트리 평균은 사용하지 않는다.
모델·전처리·앙상블 가중치를 다시 학습하거나 탐색하지 않는다.

새 신경망은 저장된 전체 ResNet18을 FP32로 복원하고, 기존 트리는 원래 recipe에
기록된 이미지 연산 정밀도와 전처리를 유지한다. 이미지 경로를 두 번 처리할 수 있으므로
캐시를 서로 바꿔 끼우지 않는다. 비교와 최종 추론은 동일한 경로를 사용한다.
기존 신경망 head도 트리 이미지 특징에서 함께 계산하지만, 새 제출값은 새 frozen과
기존 ExtraTrees 두 모델의 50:50 확률만으로 결정한다.

## 1. 코드 받기와 tmux

같은 clone의 학습·추론이 끝난 뒤 실행한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-frozen-blend
```

이미 tmux 안이면 현재 창을 사용하면 된다. tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail
```

## 2. Validation 비교

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/frozen_tree_blend.py compare \
  --baseline_recipe runs/seungbeom/midpoint_blend_replay_seed42/recipe.json \
  --frozen_run runs/seungbeom/resnet_layer4_seed42/frozen/attempt_f3ab196c7b92 \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --output_dir runs/seungbeom/frozen_tree_blend_validation \
  --device cuda \
  2>&1 | tee runs/seungbeom/frozen_tree_blend_validation.log
```

사용자가 완료한 frozen attempt 경로를 사용한다. 다른 실행을 비교할 때는 그 실행의
report에 나온 경로로 지정한다. 출력 폴더는 프로그램이 생성하고 기존 폴더를 덮어쓰지
않는다. 실패한 비교를 다시 진행할 때는 원인을 해결하고 새 output_dir를 사용한다.

원래 Train/Validation CSV 해시가 일치해야 한다. 새 frozen, 기존 ExtraTrees,
기존 신경망, 기존 50:50의 전체 집계 지표(클래스별 값·혼동행렬 포함)가 재현되어야
`recipe.json`이 생성된다. 이는 저장된 집계 성능의 재현 검사이며 과거의 모든 개별
확률값을 직접 대조했다는 뜻은 아니다. Test는 이 단계에서 읽지 않는다.

완료 후 다시 표를 보려면:

```bash
python experiments/seungbeom/frozen_tree_blend.py report \
  --recipe runs/seungbeom/frozen_tree_blend_validation/recipe.json \
  --class_details \
  | tee runs/seungbeom/frozen_tree_blend_validation/comparison.txt
```

`frozen_tree_50_50`이 이번 후보다. `frozen`은 새 신경망 단독, `extra_trees`는 기존
트리 단독, `baseline_50_50`은 과거 조합, `historical_neural`은 과거 신경망이다.
`delta_vs_frozen`은 새 frozen 단독 대비 차이다. 공식 Validation은 이미 실험 선택에
사용되어 왔으므로 독립적인 최종 평가 점수로 해석하지 않는다.

## 3. 제출 CSV 만들기

비교가 성공한 뒤 같은 환경에서 실행한다.

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/frozen_tree_blend.py predict \
  --recipe runs/seungbeom/frozen_tree_blend_validation/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output runs/seungbeom/submissions_frozen_tree_seed42/submission_frozen_extratrees_50_50.csv \
  --device cuda
```

제출 파일:

```text
/home/ubuntu/seungbeom/unid_datathon_team18/runs/seungbeom/submissions_frozen_tree_seed42/submission_frozen_extratrees_50_50.csv
```

동명의 `.manifest.json`은 코드·모델·입력·출력 지문과 recipe를 보존하는 서버 내부
재현 기록이다. 대회 제출에는 CSV만 사용한다. 기존 CSV나 manifest가 있으면 덮어쓰지
않고 중단한다. 부분 입력에서는 `--sample_submission`을 생략하면 입력 순서를 따른다.
양식을 지정하면 입력과 정확히 같은 ID 집합인지 확인하고 양식 순서로 저장한다.

추론은 현재 입력의 이미지·센서와 저장된 전처리만 사용한다. Train/Validation CSV나
이미지 캐시는 열지 않는다. 기존 모델 파일, 새 frozen 파일, recipe, 소스와 환경이
비교 때와 같은지 확인한다. 기존 모델·기존 제출 CSV를 변경하지 않는다.

tmux에서 분리하려면 `Ctrl+b` 다음 `d`, 돌아오려면:

```bash
tmux attach -t seungbeom-frozen-blend
```

실제 데이터 학습·추론은 사용자가 주최 측 서버에서 실행한다. 로컬 테스트는 독립 합성
입력으로 코드 동작을 검증할 뿐 실제 대회 성능을 측정하지 않는다.
