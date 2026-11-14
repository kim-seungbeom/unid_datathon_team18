# 실험별 행 수와 오류 확인

기존 `frozen_tree_blend_validation/recipe.json`에 저장된 frozen 신경망·원본 ExtraTrees·50:50을 분석한다. 모델 학습, 전처리 재학습, 비율 변경, Test 접근, 제출 CSV 생성은 하지 않는다. 기존 모델과 제출 파일은 그대로 보존한다.

Train은 `sample_id`, `experiment_id`, `label`만 읽어 실험·클래스별 행 수를 집계한다. Validation은 기존 추론 경로로 예측하고, 저장된 전체 지표(혼동행렬 포함)가 일치해야 오류 보고서를 저장한다. 모델마다 기존 이미지 연산 정밀도를 유지하며, 다른 Train/Validation CSV는 체크섬 검증으로 거절한다. 실험 ID와 전체 실험 행 수는 진단용이며 모델 입력으로 추가하지 않는다.

실제 데이터 분석은 사용자가 주최 측 GPU 서버에서 실행한다. 로컬 테스트는 독립적인 합성 입력으로 코드 동작만 검증한다.

## 1. 서버에서 코드 받기

실행 중인 작업이 없는 개인 clone에서 실행한다. `git status`에 미커밋 변경이 있으면 덮어쓰거나 삭제하지 말고 별도로 보존한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short --branch
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
```

## 2. tmux에서 분석 한 번 실행

```bash
tmux new -A -s seungbeom-diagnostics
```

같은 이름의 세션이 있으면 연결된다. 기존 명령이 실행 중이면 기다리거나 다른 세션 이름을 쓴다. tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail

LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/experiment_diagnostics.py analyze \
  --recipe runs/seungbeom/frozen_tree_blend_validation/recipe.json \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --validation_csv /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --output_dir runs/seungbeom/experiment_diagnostics_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/experiment_diagnostics_seed42.log
```

학습은 없지만 Validation 이미지 추론은 수행한다. 실행 중 `Ctrl+B`를 누른 뒤 `D`를 누르면 세션을 나갈 수 있다. 재접속은 `tmux attach -t seungbeom-diagnostics`.

기존 출력 폴더를 덮어쓰지 않는다. 실패한 실행을 다시 수행할 경우 새 `--output_dir`와 로그 이름을 지정한다. `replay.json`에 False가 있으면 기존 모델·환경·입력 재현부터 확인한다.

## 3. 결과 다시 보기 — 추론 없음

```bash
python experiments/seungbeom/experiment_diagnostics.py report \
  --output_dir runs/seungbeom/experiment_diagnostics_seed42 \
  --top 10 \
  > runs/seungbeom/experiment_diagnostics_seed42/report.txt

less -S runs/seungbeom/experiment_diagnostics_seed42/report.txt
```

`less`에서 위아래·좌우 방향키로 이동하고 `q`로 종료한다. VS Code에서 `report.txt`를 열어도 된다. `--method frozen` 또는 `--method extra_trees`를 추가하면 해당 모델의 실험별 오류 순위를 본다. 기본은 `frozen_tree_50_50`. `--top 1000`이면 실험별 표 전체를 볼 수 있다.

## 읽을 순서

1. Train의 `max/min`, 큰 실험 상위 10%의 행 점유율로 학습 행 편중을 확인한다. `top10%`는 실험 개수의 10%를 올림한 값이다.
2. 클래스별 행 수와 `experiments_present`를 함께 본다. 행이 많아도 몇 개 실험에만 등장하는 클래스일 수 있다.
3. Validation에서 오류 **개수** 순위와 **오류율** 순위를 모두 본다. 행이 많은 실험은 오류 개수도 많아질 수 있고, 작은 실험의 오류율은 불안정할 수 있다.
4. `top3_error_share`와 해당 세 실험의 `top3_groups_row_share`를 비교한다. 이는 틀린 행의 집중도이며 Macro F1 하락 기여도는 아니다. 오류가 0이면 오류 점유율도 0으로 표시한다.
5. 단계별 precision/recall과 혼동 쌍을 보고, 아래 CSV에서 그 오류가 어느 실험에 집중되는지 확인한다.

출력은 `runs/seungbeom/experiment_diagnostics_seed42/` 아래에 있다.

| 파일 | 내용 |
| --- | --- |
| `experiment_counts.csv` | Train/Validation 실험별 행 수·점유율·클래스별 행 수 |
| `class_coverage.csv` | 클래스별 행 수·등장 실험 수·최대 실험의 해당 클래스 점유율 |
| `validation_group_metrics.csv` | 모델·실험별 오류 수·오류율·오류 점유율·F1 |
| `validation_group_classes.csv` | 모델·실험·클래스별 support·precision·recall·F1 |
| `validation_confusions.csv` | 모델·실험별 실제 클래스→예측 클래스의 집계 건수(정분류 포함) |
| `summary.json`, `replay.json`, `status.json` | 요약·재현 검사·완료 및 파일 무결성 기록 |

비율은 0~1이다. `macro_f1_6`은 그룹에 없는 클래스도 0점으로 포함하고, `macro_f1_present`는 그 그룹의 실제 정답에 등장한 클래스만 평균한다. 그룹마다 클래스 구성이 다르므로 이 값으로 그룹을 단순 우열 비교하거나 Public 점수로 환산하지 않는다. 정답 표본이 없는 클래스의 recall은 CSV의 빈칸(NA)이다. 전체 Macro F1은 기존처럼 여섯 클래스를 모두 사용한다.

Train과 Validation은 서로 다른 실험이다. Train 실험의 행 수와 같은 ID의 Validation 오류를 연결하는 분석이 아니다. 전체 실험 편중과 미학습 실험에서의 오류 집중을 각각 확인한다. Validation은 이미 여러 모델 비교에 사용됐으므로 이번 분석은 독립적인 최종 평가가 아니다. 행 편중이 관찰되어도 균형 학습이 성능을 높인다는 증거는 아니며, 후속 실험에서 따로 비교해야 한다.

개별 샘플 ID·이미지 경로·행별 정답/예측/확률은 출력하지 않는다. 생성되는 실험별 집계 파일도 서버의 개인 분석 결과로 보관하며 Git에 커밋하지 않는다.
