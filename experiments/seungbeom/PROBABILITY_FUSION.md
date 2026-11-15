# Temperature scaling / L2 logistic stacking

기존 최고 모델과 제출물은 보존한다. 이전 코드·recipe·OOF·확률 캐시는 변경하지 않는다.
실제 데이터 실행은 제공 서버에서 사람이 한다. 로컬 테스트는 독립 합성 데이터와 CPU만 사용한다.

## 고정된 두 실험

1. `temperature_30_70`: 각 멤버의 Train OOF 로그 손실을 최소화하는 온도를 각각 선택한다.
   캐시에 logits가 없어 `softmax(log(clip(p, 1e-12, 1)) / T)`를 사용한다.
   T 탐색 범위는 [0.25, 4], 항등값 T=1과 양쪽 경계를 포함해 확인한다.
   보정 후 신경망:ExtraTrees = 30:70 결합을 유지하며 prior tau는 추가하지 않는다.
   단일 멤버의 argmax를 바꾸는 것이 아니라 앙상블에서의 확신 정도를 조정한다.
   NLL 개선이 Macro F1 개선을 보장하지 않는다. T가 경계에 걸리는지도 출력한다.
2. `logistic_stacking`: 신경망 6개 확률 + ExtraTrees 6개 확률을 순서대로 연결한다.
   입력은 원본 확률 12개, 출력은 6개 클래스다. 추가 scaler, ID, 시간, 실제 라벨 입력 피처는 없다.
   L2 logistic regression, C=0.1, lbfgs, max_iter=2000, tol=1e-7, class_weight=None을 미리 고정한다.
   C/확률 변환/클래스 가중치를 Validation이나 Test로 탐색하지 않는다.

두 방법 모두 기존 모델의 학습을 다시 하지 않는다. 기존 OOF 위에서 작은 보정기/분류기를 학습한다.
각 OOF fold를 제외하고 보정기를 학습한 진단과 전체 OOF에 학습한 진단을 구분해서 출력한다.
단, 기본 OOF 예측을 만드는 신경망은 대체 모델이고 기존 epoch를 재사용했다. 바깥 fold마다
기본 모델 OOF까지 다시 만든 것이 아니므로 **독립적인 nested CV 점수가 아니다**.
Validation은 고정된 모델 비교에만 사용하지만 이전 실험에서도 반복 사용됐다.

최종 온도·스태킹 계수·절편·클래스 순서는 `recipe.json`에 저장한다. pickle 없이 고정 계수로
추론하며, sklearn 예측과 수치적으로 일치하는지도 학습 직후 확인한다. 이후 source/runtime/입력
캐시/원본 모델을 검증한다. Test는 고정된 보정기 적용과 중복 확인에만 사용한다.

이번 도구는 요청한 두 실험을 평가하기 위한 것으로, OOF 진단이 하락한 후보도 CSV 생성은 가능하다.
`diagnostic_improved=False`가 나오면 개선 근거가 약한 후보다. CSV 생성이 제출 추천을 뜻하지 않는다.
기존 30:70 무보정/30:70+tau 및 이번 다른 후보와 라벨이 같으면 중복 CSV는 건너뛴다.
모든 과거 제출 CSV와 비교하는 것은 아니다. 기존 최고 제출물은 별도로 보존한다.

참고: [Temperature scaling 논문](https://proceedings.mlr.press/v70/guo17a.html),
[LogisticRegression 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

## 서버 실행

같은 checkout의 다른 작업이 끝난 뒤 받는다. 미커밋 파일을 지우거나 강제로 덮어쓰지 않는다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short --branch
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-probability-fusion
```

tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
set -o pipefail
```

OOF 생성이나 이미지 추론은 다시 실행하지 않는다. 기존 `prior30_oof_seed42` 및
`inner_best_probability_cache`를 사용한다. 다음 명령은 Train OOF 보정기 학습 후 Validation을 비교한다.
`--device cuda`는 원본 캐시 환경 검증용이며 새 보정기 학습은 CPU로 한다.

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/probability_fusion.py compare \
  --baseline_recipe runs/seungbeom/prior30_validation_seed42/recipe.json \
  --oof_dir runs/seungbeom/prior30_oof_seed42 \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/probability_fusion_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/probability_fusion_seed42.log
```

저장된 결과만 다시 보기:

```bash
python experiments/seungbeom/probability_fusion.py report \
  --recipe runs/seungbeom/probability_fusion_seed42/recipe.json
```

fold별 진단도 보려면 `--folds`를 추가한다. 기존 30:70/30:70+tau 재현 확인,
두 후보의 Macro F1·클래스별 F1·3→2/5→4 및 역방향 오류·로그 손실을 비교한다.
prior tau 점수는 확률이 아니므로 해당 기준에는 NLL을 계산하지 않는다.

## CSV 생성

compare가 정상 완료된 뒤:

```bash
python -u experiments/seungbeom/probability_fusion.py predict \
  --recipe runs/seungbeom/probability_fusion_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --cache_dir runs/seungbeom/inner_best_probability_cache/test \
  --output_dir runs/seungbeom/submissions_probability_fusion_seed42 \
  --device cuda
```

중복이 아니라면 다음 파일을 만든다.

```text
runs/seungbeom/submissions_probability_fusion_seed42/submission_temperature_30_70.csv
runs/seungbeom/submissions_probability_fusion_seed42/submission_logistic_stacking.csv
```

같은 출력 폴더에는 재실행하지 않는다. 확률 캐시는 정확히 같은 전체 CSV/이미지/모델에 묶여 있으며,
일부 입력이 들어오면 전체 캐시를 잘라 쓰는 대신 거부한다. 보정 연산의 행별 불변성은 합성 테스트로
확인했지만, 최종 제출용 `test.py`는 입력 행에서 원본 모델 확률을 계산한 뒤 저장된 보정을 적용하고
부분 입력·이미지 교체·원본 제출 재현을 별도로 검증해야 한다. 이 도구만으로 최종 제출 패키지가
완성된 것은 아니다. 데이터/실제 OOF/가중치/CSV/데이터 포함 로그는 Git에 올리지 않는다.
