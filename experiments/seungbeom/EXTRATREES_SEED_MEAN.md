# ExtraTrees seed 42·43·44 평균 비교

기존 최고 모델과 CSV를 보존한다. 기존 seed 42 ExtraTrees의 **학습된 결측 대체,
표준화, PCA 및 트리**를 읽어 재사용한다. seed 43·44는 같은 전처리를 통과한
Train으로 트리만 학습한다. 나무 400개, leaf=2 등 다른 파라미터는 그대로다.
이미지·신경망 재학습, 전처리 재학습, Validation/Test 학습은 없다.

후보는 세 트리 확률의 동일 비중 평균 및 `inner_best 30% + 평균 트리 70%`다.
기준은 기존 `inner_best 30% + 원본 seed42 트리 70%`이며 tau/Temperature는
적용하지 않는다. 기존 tau는 다른 확률 분포에서 선택되었기 때문이다.
Validation은 재사용된 비교 점수이며 이번 실험의 독립 OOF 검증은 아니다.

## 서버 실행

서버 개인 clone에서 다음을 실행한다. 기존 작업 중인 변경이 있으면 pull 전 보존한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -s seungbeom-et-seeds
```

이미 같은 tmux 세션이 있다면 `tmux attach -t seungbeom-et-seeds`로 접속한다.
tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/extratrees_seed_mean.py train \
  --baseline_recipe runs/seungbeom/prior30_validation_seed42/recipe.json \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --validation_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/validation \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/extratrees_seed_mean_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/extratrees_seed_mean_seed42.log
```

기존 가상환경을 사용하며 새 라이브러리 설치는 필요 없다. 트리 CPU 병렬 수는
원본 설정을 유지한다. 완료된 기존 확률 캐시가 필요하며 이미지 추론은 수행하지 않는다. 원본 PCA의 수치
재현을 위해 행별 변환을 수행하므로 변환 구간에 시간이 걸릴 수 있다.
변환은 4096행마다 진행 상황을 출력한다.

```bash
python experiments/seungbeom/extratrees_seed_mean.py report \
  --recipe runs/seungbeom/extratrees_seed_mean_seed42/recipe.json \
  | tee runs/seungbeom/extratrees_seed_mean_seed42/comparison.txt
```

`extra_trees_seed42/43/44`, `extra_trees_seed_mean`을 비교하고,
`inner_best_30_70`과 `inner_best_et_seed_mean_30_70`의 Macro F1 및 클래스별 F1을
확인한다. 평균이 모든 단일 모델보다 높다는 보장은 없다.

학습 결과는 `members.joblib`, `recipe.json`, `status.json`으로 저장된다.
이 명령은 CSV를 생성하거나 기존 제출 파일을 덮어쓰지 않는다.
재실행 시 기존 출력 폴더는 덮어쓰지 않으므로 새로운 output_dir을 지정한다.
tmux에서 빠져나갈 때는 Ctrl+B 후 D를 누른다.

## 재현 검사

- 원본 Train/Validation CSV와 이미지 임베딩 캐시의 해시·ID 대응을 검사한다.
- 새 트리 학습 전 원본 seed42의 Validation 확률을 기존 캐시와 정확히 대조한다.
- 전처리 상태, 원본 트리 상태, seed 외 파라미터가 유지되는지 검사한다.
- 저장 후 다시 로드한 각 트리의 Validation 확률을 정확히 대조한다.
- sklearn 트리 노드의 메모리 패딩은 복사 시 바뀔 수 있으므로 실제 노드 필드별로
  상태를 해시한다. 파일 자체에는 별도 SHA256 검사를 적용한다.

새 트리는 저장된 PCA의 transform 결과로 학습한다. 원본 트리는 그대로 재사용하며,
원본 PCA fit_transform으로 학습했던 seed42를 새로 학습해 동일하게 만들었다고
주장하지 않는다. 향후 추론에서도 저장된 전처리와 같은 행별 변환을 사용해야 한다.
