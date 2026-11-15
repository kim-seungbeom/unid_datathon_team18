# 35:65 / 25:75 + Train OOF tau

`inner_best_prior_neighbors.py`는 기존 30:70 보정 실험 이후에 사용하는 별도 코드다.
기존 모델·코드·recipe·제출물은 그대로 둔다. 실제 데이터는 사람이 제공 서버에서만 실행한다.

- 두 비율은 신경망:ExtraTrees = 35:65, 25:75로 미리 고정한다.
- 기존 `prior30_oof_seed42`의 동일한 Train OOF로 각각 tau를 다시 선택한다.
- Train에서 구한 클래스 빈도로 `argmax(p / prior**tau) + 1`을 적용한다.
- tau는 0..2 / 0.05 간격, 동률은 작은 tau. 기존 30:70 tau 선택도 재현 검사한다.
- 모델 학습, 새 이미지 추론, 임베딩 추출은 전혀 하지 않는다.
- 먼저 OOF로 선택하고 나서 Validation을 비교한다. Validation/Test로 tau를 선택하지 않는다.
- OOF head는 inner_best의 대체 모델이며 epoch도 기존 선택값을 쓴다. 다른 fold로 tau를
  선택한 진단도 완전한 nested CV가 아니다. 독립 최종 점수/Private 점수로 해석하지 않는다.
- 비율마다 무보정 대비 OOF 진단이 개선되고 tau>0일 때만 `adopt=True`가 된다.
- Test 단계에서는 고정된 값을 적용하고 중복 여부만 확인한다. 예측 분포로 재학습하지 않는다.
- 기존 30:70 무보정, 30:70+tau, 앞서 내보낸 이번 후보와 라벨이 완전히 같으면 CSV를 건너뛴다.
  모든 과거 제출 파일과 비교하는 도구는 아니다. 기록된 두 기준/이번 후보만 비교한다.
- 결과는 0~2개 CSV다. OOF 미채택/중복이면 `skip` 이유를 출력한다. 억지로 내보내는 옵션은 없다.
- compare/predict는 기존 입력·모델·코드·환경과 일치하는 확률 캐시를 요구한다.
  부분 입력에 대한 최종 제출 CLI 검증과 공식 `test.py` 통합은 별도 마무리 작업이다.

## 1. 서버 코드 받기

같은 checkout에서 실행 중인 작업이 끝난 뒤 받는다. 미커밋 변경을 덮어쓰지 않는다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short --branch
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-prior-neighbors
```

tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
set -o pipefail
```

## 2. 기존 OOF로 tau 선택 + Validation 비교

OOF 생성 명령을 다시 돌리지 않는다.

```bash
python -u experiments/seungbeom/inner_best_prior_neighbors.py compare \
  --baseline_recipe runs/seungbeom/prior30_validation_seed42/recipe.json \
  --oof_dir runs/seungbeom/prior30_oof_seed42 \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/prior_neighbors_validation_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/prior_neighbors_validation_seed42.log
```

선택된 tau/adopt와 OOF 진단, 기존 30:70 및 30:70+tau,
35:65 및 25:75 각각의 무보정/보정 Validation 성능을 출력한다.
클래스별 F1과 3→2, 5→4 및 역방향 오류도 비교한다.

다시 보기:

```bash
python experiments/seungbeom/inner_best_prior_neighbors.py report \
  --recipe runs/seungbeom/prior_neighbors_validation_seed42/recipe.json
```

## 3. 고정된 후보 내보내기

위 단계가 오류 없이 끝난 뒤 실행한다. Test에서는 Train/Validation/OOF 데이터 파일을 읽지 않는다.

```bash
python -u experiments/seungbeom/inner_best_prior_neighbors.py predict \
  --recipe runs/seungbeom/prior_neighbors_validation_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --cache_dir runs/seungbeom/inner_best_probability_cache/test \
  --output_dir runs/seungbeom/submissions_prior_neighbors_seed42 \
  --device cuda
```

조건을 통과한 후보만 아래 이름으로 생성한다.

```text
runs/seungbeom/submissions_prior_neighbors_seed42/submission_inner_best_35_65_prior_tau.csv
runs/seungbeom/submissions_prior_neighbors_seed42/submission_inner_best_25_75_prior_tau.csv
```

이미 생성한 디렉터리는 덮어쓰지 않는다. `Output directory exists`가 나오면 이전 실행의
완료 여부와 생성 파일을 먼저 확인한다. Public 상승은 보장하지 않으며 기존 최고 제출물을 보존한다.
데이터/가중치/확률/CSV/데이터 포함 로그는 Git에 올리지 않는다.
