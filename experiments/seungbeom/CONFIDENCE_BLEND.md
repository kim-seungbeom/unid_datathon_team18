# 기존 prior30 모델의 샘플별 결합 비율 실험

기존 최고 제출물, inner_best 신경망, 원본 seed42 ExtraTrees, PCA와 전처리는 보존한다.
추가 신경망/트리 학습 없이 기존 Train OOF와 Validation/Test 확률 캐시를 사용한다.

신경망 비중은 각 행의 확률로만 계산한다.

```text
H(p) = -sum(p * log(p)) / log(6)
alpha = clip(0.30 + 0.10 * strength * (H_tree - H_neural), 0.20, 0.40)
p_mix = alpha * p_neural + (1 - alpha) * p_tree
prediction = argmax(p_mix / train_prior ** tau) + 1
```

`strength` 후보는 0, -0.5, +0.5, -1, +1의 다섯 개다. 양수는 낮은 엔트로피를
보이는 모델의 비중을 늘리고, 음수는 그 방향을 반대로 한다. 0은 기존 30:70과 정확히 같다.
Train OOF Macro F1에서 `0.001 * strength²`를 뺀 값으로 하나만 선택한다.
tau는 각 학습 부분에서 원래 raw30 방식으로 선택한 값을 유지하며 새 비율에 맞춰 재탐색하지 않는다.
전체 OOF의 prior/tau가 기존 prior30 recipe와 정확히 일치하는지도 확인한다.

엔트로피는 확률의 집중도를 나타내지만 정답 확률을 보장하지 않는다.
[SciPy 엔트로피 정의](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html),
[scikit-learn의 잘 보정된 확률에 관한 soft voting 안내](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)를
참고했다. 위 제한된 가중치 공식과 후보 범위는 자체 실험 설계이며 성능 향상이 증명된 방법이 아니다.

## 비교와 내보내기

각 OOF fold를 제외하고 prior/tau와 strength를 선택한 뒤 제외된 fold에서 비교한다.
기존 OOF proxy 모델과 구조·epoch 선택을 재사용하므로 독립 nested CV는 아니다.
Validation은 후보 비교에만 사용하며 Test·Public 점수로 strength를 선택하지 않는다.

`recommended=True` 조건은 제외-fold 전체 Macro F1 +0.001 이상, 과반 fold 개선,
최악 fold 하락 0.01 이하, 최종 strength가 0이 아닌 경우다. False여도 명시적
`predict` 명령으로 탐색용 파일을 생성할 수 있다. 그 사실은 recipe와 출력 manifest에 기록된다.
strength가 0이거나 기존 prior30과 모든 예측이 같으면 중복 파일을 생성하지 않는다.
추천 여부나 CSV 생성 자체가 Public/Private 개선을 의미하지 않는다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
set -o pipefail

LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/confidence_blend.py compare \
  --baseline_recipe runs/seungbeom/prior30_validation_seed42/recipe.json \
  --oof_dir runs/seungbeom/prior30_oof_seed42 \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/confidence_blend_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/confidence_blend_seed42.log

python experiments/seungbeom/confidence_blend.py report \
  --recipe runs/seungbeom/confidence_blend_seed42/recipe.json --folds
```

비교 후 탐색용 CSV 생성:

```bash
python -u experiments/seungbeom/confidence_blend.py predict \
  --recipe runs/seungbeom/confidence_blend_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --cache_dir runs/seungbeom/inner_best_probability_cache/test \
  --output_dir runs/seungbeom/submissions_confidence_blend_seed42 \
  --device cuda
```

출력 파일: `runs/seungbeom/submissions_confidence_blend_seed42/submission_inner_best_confidence_prior_tau.csv`.
`predict`는 완료된 확률 캐시와 고정 state만 읽으며 Train/Validation/OOF 데이터는 필요 없다.
최종 제출 코드로 채택하려면 원래 두 모델의 실시간 확률에도 같은 고정 함수를 연결해야 한다.
부분 입력·행 순서 변경·배치 크기 변경은 같은 샘플의 계산에 영향을 주지 않는다.
