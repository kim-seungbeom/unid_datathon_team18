# Frozen + ExtraTrees: leaf 2·5·10 비교

기존 frozen 신경망과 50:50 확률 평균을 고정하고 ExtraTrees의 `min_samples_leaf`만
2·5·10으로 비교한다. 원래 Train CSV와 이미지 캐시를 해시로 확인하고 원래 파이프라인을
복제하므로 나무 개수, seed, 센서 5개, 전처리/PCA 설정은 유지한다. 전처리와 트리는
Train에서만 다시 학습한다. 신경망은 학습하지 않는다. 이 값은 한 잎에 필요한 최소
샘플 수이며, 크게 하면 작은 집단에 대한 분할이 제한된다.
[scikit-learn 공식 정의](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

leaf 2는 재학습 재현 확인용이기도 하다. 기존 트리와 새 leaf 2의 Validation 확률이
동일하고, frozen·트리·50:50의 저장된 전체 집계 지표가 재현돼야 성공 처리한다.
기존 모델과 약 0.89 제출 CSV는 수정하지 않는다. 오류가 나면 실패 폴더를 삭제하지 말고
원인을 확인한 뒤 새 `--output_dir`로 실행한다.

## 1. 서버에서 코드 받기

같은 clone에서 실행 중인 학습/추론이 끝난 후:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-et-leaf
```

이미 tmux 안이면 현재 창을 사용한다. tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail
```

## 2. 학습·Validation·트리 3-fold 비교

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/extratrees_leaf_sweep.py train \
  --recipe runs/seungbeom/frozen_tree_blend_validation/recipe.json \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --validation_csv /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --output_dir runs/seungbeom/extratrees_leaf_sweep_seed42 \
  --group_cv \
  --device cuda \
  2>&1 | tee runs/seungbeom/extratrees_leaf_sweep_seed42.log
```

프로그램이 출력 폴더를 만든다. 순서는 Train 트리 3개 학습 → Validation 이미지 두 경로
처리(FP32 frozen, 원래 정밀도의 트리 임베딩) → 기준 재현 확인 → 선택적인 트리 CV다.
Validation 이미지는 후보마다 반복하지 않고 각 경로에서 한 번씩 처리한다. 과거 결과의
이미지 수치 정밀도를 보존하기 위해 두 경로를 섞지 않는다.

`--group_cv`는 이전 `group_cv.py`와 같은 `GroupKFold(n_splits=3)` 외부 분할을 쓴다.
원래 Train 행 순서·experiment_id가 고정되고 각 fold의 학습 행에서 전처리/PCA까지
다시 학습한다. 3 leaf × 3 fold의 트리만 추가 학습한다. **이 CV는 트리 단독 점수다.**
frozen 신경망은 Train 전체를 이미 학습했으므로 OOF에 결합하면 누수다. 50:50 비교는
공식 Validation에서만 한다. CV가 필요 없으면 `--group_cv` 한 줄을 빼도 된다.

## 3. 결과 보기

```bash
python experiments/seungbeom/extratrees_leaf_sweep.py report \
  --recipe runs/seungbeom/extratrees_leaf_sweep_seed42/recipe.json \
  --class_details \
  | tee runs/seungbeom/extratrees_leaf_sweep_seed42/comparison.txt
```

- `blend_leaf2_50_50`: 보존한 약 0.89 조합과 재현 확인된 기준
- `blend_leaf5_50_50`, `blend_leaf10_50_50`: 새 제출 후보
- `extra_trees_leaf*`: 각 트리 단독, `frozen`: 고정 신경망 단독
- `delta_vs_leaf2_blend`: 공식 Validation의 기준 50:50 대비 차이
- 마지막 CV 표: fold별 점수, 평균·표본 표준편차, 합쳐진 OOF 점수

공식 Validation과 CV는 다른 행을 평가한다. 표준편차는 신뢰구간이 아니다. 자동으로
최종 모델을 교체하거나 승자를 선택하지 않는다. 클래스별 저하와 fold별 변동을 함께
확인하며 Public/Private 점수를 Validation으로 환산하지 않는다.

## 4. 비교 후 후보 CSV 생성 (선택)

Test를 읽는 것은 이 별도 명령뿐이다. 기본값은 leaf 5·10이며 제출 사이트 업로드는 하지 않는다.

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/extratrees_leaf_sweep.py predict \
  --recipe runs/seungbeom/extratrees_leaf_sweep_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --leaves 5 10 \
  --output_dir runs/seungbeom/submissions_extratrees_leaf_seed42 \
  --device cuda
```

출력 파일은 위 폴더의 `submission_frozen_extratrees_leaf5_50_50.csv`,
`submission_frozen_extratrees_leaf10_50_50.csv`다. `status.json`이 `complete`인지 확인하고
CSV만 제출한다. recipe/manifest/가중치/캐시/CSV는 서버 내부 자료로 Git에 올리지 않는다.

predict는 Train·Validation CSV와 캐시를 열지 않고 현재 입력 행의 이미지와 센서만
처리한다. 저장된 클래스 확률을 50:50 평균한 후 정수 1~6으로 바꾸며, sample_submission의
ID·순서를 유지한다. 원래 모델·소스·패키지 환경과 leaf 후보 가중치의 변경도 검사한다.
