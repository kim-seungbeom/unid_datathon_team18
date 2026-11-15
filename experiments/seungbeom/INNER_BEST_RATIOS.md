# inner_best + ExtraTrees 비율 비교 (추가 학습 없음)

비율 순서는 **inner_best 신경망 : 기존 ExtraTrees**다. 저장된 `inner_best.pt`와
원래 트리 모델·전처리를 그대로 불러온다. 비교는 50:50·40:60·60:40이며,
CSV는 새 후보 40:60·60:40 두 개만 만든다. 기존 50:50 CSV와 가중치는 수정하지 않는다.
학습하거나 라벨 CSV를 평균하지 않고, 두 모델의 클래스 확률을 가중 평균한 뒤
`argmax + 1`로 라벨을 정한다. Validation 순위가 Public/Private 순위를 보장하지 않는다.

실제 실행은 사람이 운영진 서버에서 한다. 아래 경로는 승범 개인 clone 기준이다.
기존 학습/추론이 끝난 뒤 pull하고, 별도 패키지 설치 없이 같은 가상환경을 사용한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-inner-ratios
```

tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail

LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/inner_best_ratios.py compare \
  --run_dir runs/seungbeom/frozen_checkpoint_seed42 \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --output_dir runs/seungbeom/inner_best_ratios_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/inner_best_ratios_seed42.log
```

신경망·트리·50:50의 전체 집계 지표가 기존 checkpoint 결과와 같아야 `recipe.json`이 완성된다.
비교가 실패하면 Test 추론으로 진행하지 말고 로그의 마지막 코드 오류를 확인한다.
`report`는 저장된 결과만 읽으므로 추론을 다시 하지 않는다.

```bash
python experiments/seungbeom/inner_best_ratios.py report \
  --recipe runs/seungbeom/inner_best_ratios_seed42/recipe.json \
  --class_details \
  | tee runs/seungbeom/inner_best_ratios_seed42/comparison.txt
```

비교 완료 후 두 CSV를 한 번에 생성:

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/inner_best_ratios.py predict \
  --recipe runs/seungbeom/inner_best_ratios_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output_dir runs/seungbeom/submissions_inner_best_ratios_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/inner_best_ratios_predict_seed42.log
```

생성 위치:

```text
runs/seungbeom/submissions_inner_best_ratios_seed42/
  submission_inner_best_40_60.csv
  submission_inner_best_60_40.csv
  submission_inner_best_40_60.manifest.json
  submission_inner_best_60_40.manifest.json
  status.json
```

완료 메시지와 `status.json`의 `complete`를 확인한 뒤 CSV 두 개만 제출한다.
원본 데이터·가중치·예측 CSV·실행 산출물은 Git에 올리지 않는다.
출력 디렉터리가 이미 있으면 덮어쓰지 않으므로 `_v2` 등 새 이름을 지정하고 이후
명령의 recipe 경로도 맞춘다. tmux 분리는 `Ctrl+b`, 손을 뗀 뒤 `d`다.

재현 검사는 checkpoint 파일 해시, 전처리·모델 identity, 추론 코드, 런타임,
Validation CSV·이미지 해시를 확인한다. f639869 → d0df9bf의 학습 예산 옵션 추가는
추론 함수가 동일함을 검토한 두 파일 해시에 한해 허용한다. 그 외 의존 파일은 정확히
일치해야 한다. 새 비율 비교 이후의 코드 변경은 다시 비교해야 한다.
Test 추론은 Train/Validation CSV나 다른 Test 행을 조회하지 않으며, 두 비율은
같은 한 번의 멤버 추론 결과를 사용한다. 로컬 테스트는 독립적으로 만든 가상 데이터의
코드 동작 검사이며, 대회 Validation 성능 측정이 아니다.
