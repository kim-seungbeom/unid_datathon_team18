# ExtraTrees Train+Validation 재학습 후보

기존 50:50 recipe에 들어 있는 ExtraTrees 파이프라인을 `clone`한다.
원본 센서 5개, 고정 ResNet18 이미지 벡터, 하이퍼파라미터는 유지하고,
센서 중앙값 대체·이미지 벡터 표준화·PCA·ExtraTrees를 Train+Validation으로
다시 학습한다. 신경망과 기존 50:50 가중치·CSV는 변경하지 않는다.

사용자가 제공한 규정은 Train과 Validation을 학습에 사용하는 것을 허용한다.
이번 후보는 Validation을 학습에 포함하므로 독립적인 Validation 점수를
출력하지 않는다. 앞선 Train 그룹 CV는 당시 학습 절차의 검증 기록이며,
이 재학습 모델을 독립적으로 평가한 점수가 아니다. Test는 추론에만 사용한다.

## 1. 서버에서 코드 받기

같은 clone에서 학습·추론이 실행 중이지 않을 때 진행한다.
`git status --short`에 본인 수정 파일이 있으면 먼저 변경을 정리한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -s seungbeom-et-refit
```

이미 tmux 안이면 새로 열 필요 없다. tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail
```

## 2. Train+Validation으로 재학습

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/extratrees_refit.py train \
  --recipe runs/seungbeom/midpoint_blend_replay_seed42/recipe.json \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --validation_csv /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --validation_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/validation \
  --output_dir runs/seungbeom/extratrees_trainval_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/extratrees_trainval_seed42.log
```

출력 폴더는 프로그램이 만든다. 기존 폴더를 덮어쓰지 않는다.
두 CSV와 캐시를 원래 학습 기록의 해시와 대조하며, Train/Validation의 sample_id와
experiment_id 중복을 검사한다. 각 캐시는 sample_id로 정렬한다.

이미지 추출은 재실행하지 않고 캐시를 사용한다. ExtraTrees 학습 자체는 CPU에서
진행하므로 GPU 사용량이 낮아도 정상이다. `Fitting fresh ExtraTrees...` 다음에는
신경망처럼 epoch 로그가 나오지 않는다. `Refit complete`가 나올 때까지 기다린다.

생성되는 파일은 `model.joblib`, `metadata.json`, `status.json`이다.
metadata에는 `refit_train_val: true`, 두 학습 split과 해시, 입력 피처 계약,
모델 설정 지문, 소스·환경, 독립적인 Validation 점수가 없다는 사실을 기록한다.

tmux에서 분리: `Ctrl+b`를 누르고 뗀 다음 `d`. 재접속:

```bash
tmux attach -t seungbeom-et-refit
```

## 3. Test 추론 및 CSV 생성

학습 완료 후 같은 가상환경에서:

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/extratrees_refit.py predict \
  --model_dir runs/seungbeom/extratrees_trainval_seed42 \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output_dir runs/seungbeom/submissions_extratrees_trainval_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/submissions_extratrees_trainval_seed42.log
```

이 단계는 입력 행의 실제 이미지에서 특징을 추출한다. Train/Validation CSV·캐시나
기존 신경망/트리 가중치를 열지 않으며, 모델을 다시 학습하지 않는다.
저장된 이미지 수치 연산 설정과 전처리를 그대로 적용한다.
부분 입력에서는 `--sample_submission`을 생략하면 입력 순서대로 저장한다.
템플릿을 지정하면 입력과 정확히 같은 ID 집합이어야 한다.

제출 파일:

```text
/home/ubuntu/seungbeom/unid_datathon_team18/runs/seungbeom/submissions_extratrees_trainval_seed42/submission_extratrees_trainval.csv
```

VS Code Remote 탐색기에서 파일을 우클릭해 다운로드한 뒤 DAKER에 제출한다.
메모: `고정 ResNet18 이미지 벡터 + 원본 센서 5개, 기존 ExtraTrees 설정 유지. Train+Validation으로 전처리·PCA·모델 재학습.`

기존 50:50 제출 파일은 유지하고 새 후보로 비교한다. 모델 가중치·입력·예측 파일은
Git에 올리지 않는다. 실패한 실행을 다시 시작할 때는 새 출력 폴더를 지정한다.
학습 도중 재개 기능은 없으며 기존 출력은 덮어쓰지 않는다.
