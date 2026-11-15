# CatBoost + LightGBM: bounded training

기존 최고 모델/CSV를 보존하는 개인 실험. 학습은 사용자가 제공 GPU 서버에서만 실행한다.
기존 ResNet18 512차원 벡터와 원본 센서 5개를 사용한다. 센서 결측은 Train 중앙값,
Train 전체 결측 칼럼은 0으로 채운다. 이번 후보에는 PCA·표준화·prior·Temperature를
쓰지 않는다. 따라서 기존 PCA ExtraTrees와의 차이가 분류기 하나뿐인 실험은 아니다.

두 모델은 CPU 4 threads로 순차 학습한다. 기본 400회, 각 12분의 soft budget,
Validation Macro F1이 50회 개선되지 않으면 종료한다. 예산은 반복 사이에서 확인하므로
전처리·검증·저장·한 번의 긴 반복까지 포함한 총 실행 시간의 상한은 아니다.
CatBoost depth=6, lr=.05, L2=5; LightGBM leaves=31, depth=6, lr=.05,
min_child_samples=30, L2=5. seed=42, 클래스 가중치 없음.

Train만으로 gradient/센서 중앙값을 학습한다. 공식 Validation으로 최고 반복을 선택하고
그 시점 모델을 native 파일로 저장한다. 이 점수는 재사용한 모델 선택 점수이며 새로운
독립 평가가 아니다. CatBoost 로그의 `test:`는 전달한 공식 Validation을 뜻하며 대회 Test가 아니다.
새 모델에 대한 OOF는 생성하지 않는다. 기존 OOF를 새 트리의 OOF로
대체하여 사용하지 않는다.

비교 후보:
- `catboost`, `lightgbm`: 이미지+센서 트리 단독
- `inner_best_catboost_30_70`, `inner_best_lightgbm_30_70`: 기존 신경망 30% + 새 트리 70%
- `inner_best_cat_lgbm_30_35_35`: 기존 신경망 30% + CatBoost 35% + LightGBM 35%

비율은 고정하고 새로 탐색하지 않는다. 기존 `inner_best_30_70`, ExtraTrees도 대조군으로 표시한다.

## 서버

다른 작업이 실행 중이지 않은 개인 clone에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-cat-lgbm
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
python -m pip install 'catboost==1.2.8' 'lightgbm>=4,<5'
set -o pipefail
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/boosted_trees.py train \
  --baseline_recipe runs/seungbeom/prior30_validation_seed42/recipe.json \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --validation_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/validation \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/cat_lgbm_seed42 \
  --iterations 400 --threads 4 --minutes_per_model 12 --device cuda \
  2>&1 | tee runs/seungbeom/cat_lgbm_seed42.log
```

`--device cuda`는 기존 캐시의 실행 환경 확인과 Test 이미지 추론용이다.
새 트리 학습 자체는 CPU만 사용한다. 기존 LightGBM 4.x가 있으면 pip는 업그레이드하지 않는다.
실행 중인 작업의 환경에서는 설치/pull하지 않는다.

```bash
python experiments/seungbeom/boosted_trees.py report \
  --recipe runs/seungbeom/cat_lgbm_seed42/recipe.json
```

비교 후 선택한 후보만 `--methods`로 전달한다. 아래 예시는 세 모델 결합이다.

```bash
python -u experiments/seungbeom/boosted_trees.py predict \
  --recipe runs/seungbeom/cat_lgbm_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --cache_dir runs/seungbeom/inner_best_probability_cache/test \
  --output_dir runs/seungbeom/submissions_cat_lgbm_seed42 \
  --methods inner_best_cat_lgbm_30_35_35 --device cuda
```

`--methods`에 후보 여러 개를 공백으로 구분할 수 있다. 이미지 추출은 후보 수와 무관하게
한 번만 수행한다. 신경망 확률은 기존 검증된 캐시를 재사용한다. Test 이미지의 512차원
벡터는 원본 이미지에서 한 번 추출하므로, CSV 생성은 확률만 섞던 실험보다 오래 걸린다.
예측에서는 Train/Validation/OOF 입력을 읽거나 재학습하지 않는다.
CSV는 `submission_<method>.csv`, 저장 모델은 `catboost.cbm`, `lightgbm.txt`다.
기준 raw 30:70/ExtraTrees 또는 같은 호출 내 다른 후보와 라벨이 완전히 같으면 건너뛴다.

현재 CLI의 신경망 확률 캐시는 전체 입력에 연결되어 있어 부분 입력 캐시를 거부한다.
최종 선택 시 `test.py`의 임의 입력/이미지 변경 대응 및 모델 패키징을 별도 검증해야 한다.

참고: [CatBoost best model](https://catboost.ai/docs/en/references/training-parameters/common),
[Macro TotalF1](https://catboost.ai/docs/en/concepts/loss-functions-multiclassification),
[LightGBM early stopping](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html).
