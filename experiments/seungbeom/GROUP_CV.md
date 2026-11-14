# 기존 50:50 구조의 Train 실험 단위 3-fold 검증

기존 midpoint recipe의 설정을 읽어서 고정 ResNet18 벡터 + 원본 센서 5개 +
concat 신경망 / ExtraTrees를 새로 학습한다. 두 예측 확률은 항상 50:50이다.
기존 모델·recipe·제출 CSV는 수정하지 않으며, 이 명령은 제출 CSV를 만들지 않는다.

- 외부 3-fold: 원본 Train을 `experiment_id` 기준 GroupKFold로 분리한다.
- 각 fold의 학습 부분 안에서만 첫 번째 StratifiedGroupKFold 분할을 사용해
  신경망 epoch를 선택한다(최대 5개 내부 분할, seed 42). 내부 분할을 여러 번
  점수로 탐색하지 않는다. 선택한 epoch 수로 외부 학습 부분 전체를 처음부터 재학습한다.
- 설정·학습률·dropout·loss·seed·최대 epoch 등은 원본 신경망 설정 그대로다.
  ExtraTrees도 원본 파이프라인을 `clone`해 하이퍼파라미터만 복사한다.
- 중앙값·표준화·PCA는 각 학습 부분에서 새로 fit한다. 기존에 전체 Train으로
  학습된 전처리나 모델 가중치는 fold 모델에 재사용하지 않는다.
- 이미지 벡터는 학습하지 않은 사전학습 인코더의 원본 Train 캐시를 재사용한다.
  sample_id로 정렬하고 원본 학습 기록의 캐시 해시와 대조한다.
- 공식 Validation/Test CSV와 이미지는 읽지 않는다. 기존 Validation 집계 점수는
  recipe에 기록된 값을 별도 참고란에 표시한다. 이 점수와 CV 점수는 직접 순위 비교하지 않는다.
- 외부 holdout은 epoch 선택·전처리 학습·혼합 가중치 결정에 쓰지 않는다.
  단, 구조 자체는 이전 Validation/Public 실험에서 선택했으므로 완전히 새로운 blind test는 아니다.

## 서버에서 실행

다른 학습이 실행 중인 clone에서는 먼저 작업이 끝난 뒤 pull한다.
`git status --short`에 코드 변경이 있으면 덮어쓰지 말고 먼저 본인 변경을 정리한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -s seungbeom-cv3
```

새 tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail

LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/group_cv.py run \
  --recipe runs/seungbeom/midpoint_blend_replay_seed42/recipe.json \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --output_dir runs/seungbeom/group_cv_raw_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/group_cv_raw_seed42.log
```

출력 디렉터리는 프로그램이 만든다. 미리 `mkdir`로 만들지 않는다.
한 tmux에서 3개 fold를 순서대로 학습한다. 각 fold는 `inner_selection` →
`outer_refit` → ExtraTrees 학습 → outer holdout 평가 순서다.
이미지는 다시 추출하지 않고 Test도 추론하지 않는다.

tmux에서 나가기: `Ctrl+b`를 누르고 뗀 다음 `d`. 돌아오기:

```bash
tmux attach -t seungbeom-cv3
```

## 완료 후 결과 비교

```bash
python experiments/seungbeom/group_cv.py report \
  --output_dir runs/seungbeom/group_cv_raw_seed42 \
  | tee runs/seungbeom/group_cv_raw_seed42/comparison.txt
```

잘려 보이면 다음 명령으로 가로 스크롤한다(`q`로 종료).

```bash
less -S runs/seungbeom/group_cv_raw_seed42/comparison.txt
```

`neural`, `extra_trees`, `blend_50_50`을 같은 fold끼리 비교한다.

- `selected_epoch`: 내부 검증에서 선택한 신경망 학습 횟수. ExtraTrees와 blend 행에는
  같은 fold의 신경망 선택값을 표시한다.
- `fold_mean`: 세 fold Macro F1의 단순 평균.
- `fold_sample_std`: fold 간 표본 표준편차(ddof=1). 신뢰구간이 아니다.
- `pooled_oof_f1`: 모든 Train 행의 학습 밖 예측을 모아 다시 계산한 6-class Macro F1.
  fold 평균과는 다를 수 있다.
- `f1_1`~`f1_6`: 위쪽 표는 fold별, 아래쪽 표는 pooled OOF의 단계별 F1.
- 특정 분할에 클래스가 없으면 경고하고, 지표는 계속 6개 클래스를 평균한다.
- `summary.json`: 위 지표와 fold별 precision/recall/support/confusion matrix.
- `experiment.json`: 설정, fold 행 위치, 원본 입력·소스 해시, 환경 기록.
- fold별 가중치·OOF 예측은 새 실행 폴더에만 저장된다. 데이터와 마찬가지로
  Git이나 외부 AI에 올리지 않는다. 이 CV 체크포인트는 기존 제출 모델 형식과 구분된다.

## 중단 후 이어서 실행

코드·환경·입력을 그대로 유지한 채 동일한 실행 명령에 `--resume`을 추가한다.

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/group_cv.py run \
  --recipe runs/seungbeom/midpoint_blend_replay_seed42/recipe.json \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --output_dir runs/seungbeom/group_cv_raw_seed42 \
  --device cuda --resume \
  2>&1 | tee -a runs/seungbeom/group_cv_raw_seed42.log
```

완료된 fold는 파일 해시와 OOF 지표 재현을 확인하고 재사용한다. 중간에 끊긴 fold는
새 attempt 폴더에서 처음부터 재학습한다. epoch 중간 복원은 지원하지 않는다.
같은 출력 폴더로 두 작업을 동시에 실행하지 않는다.
