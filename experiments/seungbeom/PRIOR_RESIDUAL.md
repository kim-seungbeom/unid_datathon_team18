# 기존 30:70 + prior tau에 제한된 클래스별 보정 추가

기존 inner_best 신경망, 원본 seed42 ExtraTrees, 30:70 비율, 전처리/PCA를 보존한다.
기준은 `prior30_validation_seed42/recipe.json`이며 다른 비율의 prior 파일에 적용하지 않는다.
사용자가 알려준 최고 Public 점수는 0.9093891677이다. 코드가 리더보드 점수를 읽거나
최적화하지는 않으며, 최고 CSV와 이 recipe의 대응은 사용자가 확인해야 한다.

기존 점수 `log(p_c) - tau * log(train_prior_c)`에 클래스별 작은 상수 `b_c`를 더한다.
`b_c`는 Train OOF만으로 선택한다. 범위는 ±0.15, 간격 0.05, 좌표 탐색은 2회다.
Macro F1에서 보정 크기 패널티 `0.01 * mean((b/0.15)^2)`를 뺀 목적함수를 쓴다.
보정 없음이 초기 후보이며 개선이 없으면 그대로다. 이 방식은 특정 오류의 정답을
수작업으로 바꾸거나 Test 클래스 분포에 맞추는 처리가 아니다.

빈도 기반 logit adjustment는 [Menon et al., ICLR 2021](https://research.google/pubs/long-tail-learning-via-logit-adjustment/)
의 접근과 관련 있다. 이번 **추가 상수 탐색과 패널티·채택 조건은 별도로 정한 실험 설계**이며,
그 논문의 방법을 그대로 구현했다거나 그 논문이 이 실험의 성능 향상을 보장한다는 뜻은 아니다.

## 선택과 검증

OOF fold 하나씩을 제외하고 남은 OOF에서 prior·tau·추가 상수를 다시 학습한다.
제외된 fold에는 고정된 보정만 적용한다. 전체 OOF에서 학습한 최종 prior/tau는 기존
prior30과 정확히 일치해야 한다. 추가 상수의 채택 조건은 사전에 고정했다.

- 제외-fold 예측을 모은 Macro F1이 같은 방식의 prior 기준보다 0.001 이상 상승
- 과반 fold에서 상승
- 어느 fold도 0.01 넘게 하락하지 않음
- 최종 추가 상수가 모두 0인 경우 제외

기존 OOF proxy head와 구조·epoch 선택을 재사용하므로 독립적인 nested CV는 아니다.
Validation도 재사용된 비교 데이터다. 채택 여부는 Validation이나 Public 점수로 정하지 않는다.
선택 기준을 통과하지 못하면 `predict`는 Test를 읽기 전에 종료하고 CSV를 만들지 않는다.

## 서버 실행

기존 가상환경·캐시를 사용한다. 이미지 추출, 신경망/트리 재학습, 추가 패키지 설치는 없다.
실행 중인 작업이 없는 개인 clone에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail

LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/prior_residual.py compare \
  --baseline_recipe runs/seungbeom/prior30_validation_seed42/recipe.json \
  --oof_dir runs/seungbeom/prior30_oof_seed42 \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/prior_residual_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/prior_residual_seed42.log
```

결과 다시 보기:

```bash
python experiments/seungbeom/prior_residual.py report \
  --recipe runs/seungbeom/prior_residual_seed42/recipe.json --folds
```

`inner_best_30_70_prior_tau`가 기존 기준, `inner_best_30_70_prior_residual`이 새 후보다.
OOF `adopt=True`인 경우에만 다음 명령이 CSV를 만든다. 기존 최고 파일은 덮어쓰지 않는다.

```bash
python -u experiments/seungbeom/prior_residual.py predict \
  --recipe runs/seungbeom/prior_residual_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --cache_dir runs/seungbeom/inner_best_probability_cache/test \
  --output_dir runs/seungbeom/submissions_prior_residual_seed42 \
  --device cuda
```

출력: `runs/seungbeom/submissions_prior_residual_seed42/submission_inner_best_30_70_prior_residual.csv`.
기존 prior30과 모든 예측이 같으면 중복 파일은 만들지 않는다. 실제 제출은 사용자가 한다.
고정 보정은 각 행에 독립적으로 적용된다. 행 순서 변경·부분 입력에 같은 샘플의 결과가 유지된다.
이 도구의 predict는 검증된 기존 확률 캐시를 사용하므로, 최종 제출 코드에 채택할 때에는
기존 모델의 실시간 추론 결과에도 동일한 고정 보정을 연결해야 한다.
