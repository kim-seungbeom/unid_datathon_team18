# 40:60 유지, 35:65·30:70 비교

비율은 **inner_best 신경망 : 기존 ExtraTrees**다. 추가 학습이나 피처 변경 없이
35:65·30:70 두 후보만 비교한다. 기존 `inner_best_ratios.py`, 가중치, 40:60 CSV를
수정하지 않는다. Temperature 조절은 이번 실행에 포함하지 않는다.

사람이 운영진 GPU 서버에서 아래 순서로 실행한다. 처음에는 Validation과 Test 각각
멤버당 한 번의 추론이 필요하다. 이전 제출 CSV에는 멤버별 확률이 없기 때문이다.
새 코드는 확률을 서버 전용 캐시에 저장하며, 같은 입력·이미지·모델·환경이면 다음
실행부터 모델 추론을 건너뛴다. 캐시에도 대회 정보가 있으므로 Git에 올리거나 AI에 보내지 않는다.

## 1. 코드 받기와 tmux

실행 중인 학습·추론이 끝난 뒤:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-inner-refine
```

이미 tmux 안이면 마지막 명령은 생략한다. tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail
```

## 2. Validation 비교

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/inner_best_refine.py compare \
  --baseline_recipe runs/seungbeom/inner_best_ratios_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/inner_best_refine_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/inner_best_refine_seed42.log
```

완료 후 결과만 다시 읽기:

```bash
python experiments/seungbeom/inner_best_refine.py report \
  --recipe runs/seungbeom/inner_best_refine_seed42/recipe.json
```

`delta_vs_40_60`은 Validation Macro F1 차이이며 Public 점수 예상치가 아니다.
`changed_rows_vs_40_60`은 예측이 달라진 행 수이며 정답으로 개선된 행 수가 아니다.
단계별 정밀도·재현율은 `report`에 `--class_details`를 붙인다.
기존 멤버 및 50:50·40:60·60:40 지표가 모두 재현되어야 새 recipe가 완성된다.
오류가 나면 Test 추론을 진행하지 않고 마지막 코드 오류를 확인한다.

## 3. CSV 두 개 생성

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/inner_best_refine.py predict \
  --recipe runs/seungbeom/inner_best_refine_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --cache_dir runs/seungbeom/inner_best_probability_cache/test \
  --output_dir runs/seungbeom/submissions_inner_best_refine_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/inner_best_refine_predict_seed42.log
```

서버 생성 파일:

```text
runs/seungbeom/submissions_inner_best_refine_seed42/
  submission_inner_best_35_65.csv
  submission_inner_best_30_70.csv
  submission_inner_best_35_65.manifest.json
  submission_inner_best_30_70.manifest.json
  status.json
```

완료 메시지와 `status.json`의 `complete`를 확인하고 CSV만 제출한다.
후보끼리 `Candidate disagreement rows: 0`이면 두 CSV의 라벨은 같으므로 중복 제출할
필요가 없다. 40:60 대비 변경 행 수가 0인 후보도 동일한 입력의 40:60 예측과 같다.
Test의 정답은 읽거나 추정하지 않으며, 차이 집계로 비율을 재학습하지 않는다.

재실행할 때 출력 폴더는 `_v2` 같은 새 이름을 쓴다. 같은 데이터·가중치라면
`--cache_dir`은 그대로 사용하면 된다. 캐시가 미완료이거나 입력이 바뀌었으면 새
캐시 폴더를 지정하거나 옵션을 생략해 직접 추론한다. 일부 Test 입력에는 전체 Test
캐시를 재사용하지 않는다. Test 추론은 Train/Validation CSV·Validation 캐시를 읽지 않는다.
기존 최고 파일과 가중치·recipe는 보존한다. tmux 분리는 `Ctrl+b`, 손을 뗀 뒤 `d`다.
