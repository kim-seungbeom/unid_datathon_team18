# 승범 제출 후보 CSV 생성

완료된 학습 결과에서 신경망 후보 하나와 트리 기반 후보 네 개를 추론합니다.
새 학습이나 Test를 이용한 전처리 학습은 하지 않습니다. 후보는 같은 Validation의
결과를 비교해서 선택하며, Validation 점수와 실제 리더보드 점수는 같다고 가정하지 않습니다.

| 출력 CSV | 저장된 모델과 예측 방식 |
| --- | --- |
| `submission_neural_concat.csv` | 이미지 벡터와 센서 특징을 연결하는 신경망의 최고 체크포인트 |
| `submission_classwise_ensemble.csv` | Train OOF에서 학습한 클래스별 고정 가중치 |
| `submission_extra_trees.csv` | 같은 번들에 저장된 Extra Trees 단독 예측 |
| `submission_equal_average.csv` | 같은 번들의 모델 확률을 동일 비율로 평균 |
| `submission_xgboost.csv` | 같은 번들에 저장된 XGBoost 단독 예측 |

모든 후보는 이미지와 센서를 입력으로 사용합니다. 트리 후보 네 개는 같은 모델 번들에서
나오므로 새로 네 번 학습하는 것이 아닙니다. 같은 추론 호출에서 계산한 이미지 특징과
모델 확률을 재사용합니다. 신경망 후보는 자신의 저장된 입력 처리 방식으로 별도 추론합니다.

## 1. 서버에서 코드 받기

실행 중인 학습이 끝난 뒤, 개인 서버 레포에서 실행합니다. 미커밋 변경이 있다면
먼저 보존하고 진행합니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new-session -A -s seungbeom-submit
```

tmux 안에서 폴더와 환경을 설정합니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
```

## 2. 후보 다섯 개 추론

아래 두 실행 폴더는 이전 학습을 완료한 폴더여야 합니다. 신경망은 `best.pt`, 트리
앙상블은 `ensemble.joblib`를 불러옵니다. 실제 데이터 경로는 서버에서만 사용합니다.

```bash
python -u experiments/seungbeom/export_submissions.py \
  --neural_run runs/seungbeom/neural_fusion_seed42/embedding_concat_s42_3daac9bb72d0ba69 \
  --ensemble_run runs/seungbeom/classwise_resnet18_seed42 \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output_dir runs/seungbeom/submissions_round1 \
  --device cuda
```

입력 CSV의 행에 해당하는 이미지를 다시 읽어 추론하므로 이미지 추출 진행 로그가
나오는 것이 정상입니다. 과거 Test 캐시를 읽어서 바뀐 이미지를 무시하지 않습니다.

기존 출력은 덮어쓰지 않습니다. 실패 후 다시 실행하거나 별도 후보 묶음을 만들 때는
`--output_dir runs/seungbeom/submissions_round2`처럼 새 폴더를 사용합니다.
tmux에서 나올 때는 `Ctrl+B` 후 `D`, 다시 들어갈 때는 `tmux attach -t seungbeom-submit`입니다.

## 3. 생성 결과 확인과 업로드

완료 메시지를 확인한 뒤 서버에서 파일 목록을 확인합니다.

```bash
ls -lh runs/seungbeom/submissions_round1/submission_*.csv
```

생성 코드는 제출 양식의 ID·행 순서, `sample_id,label` 열 순서, 정수 라벨 1~6을 검사합니다.
누락·중복·추가 행이 있는 파일은 정상 완료 결과로 처리하지 않습니다. 출력 폴더에는
사용한 모델·체크포인트와 CSV의 해시 등 재현 정보를 담은 manifest도 함께 보관합니다.
정상 완료 여부는 `export_status.json`, 후보별 재현 정보는 `manifest.json`에 기록됩니다.
완료 표의 `identical_to`에 다른 파일명이 나오면 두 CSV가 완전히 같으므로 같은 예측을
중복 제출할 필요가 없습니다.

리더보드에는 위 CSV를 사용자가 하나씩 직접 업로드합니다. manifest와 가중치는
예측 CSV 대신 업로드하지 않습니다. 제출 횟수·마감·최종 파일 선택은 대회 화면에서
확인하고, 선택한 파일명과 해당 모델을 함께 기록합니다. 이 명령 자체는 업로드하지 않습니다.

실제 제출 CSV·입력 데이터·가중치와 추론 보고서는 서버 실행 폴더에 보관하며 Git에 올리지
않습니다. CSV 형식 검사는 점수나 이미지 활용의 적절성을 보증하는 검사가 아닙니다.

## 4. 선택한 후보 하나만 다시 생성

신경망 후보를 재현할 때는 동일한 최고 체크포인트를 지정합니다.

```bash
python experiments/seungbeom/neural_experiment.py predict \
  --checkpoint runs/seungbeom/neural_fusion_seed42/embedding_concat_s42_3daac9bb72d0ba69/best.pt \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output runs/seungbeom/submissions_reproduced/submission_neural_concat.csv \
  --device cuda
```

트리 후보는 같은 번들과 정확한 예측 방식을 지정합니다. 아래는 클래스별 앙상블 예시입니다.
다른 후보는 `--methods` 값을 `extra_trees`, `equal_average`, `xgboost`로 바꿉니다.

```bash
python experiments/seungbeom/tree_candidates.py \
  --checkpoint runs/seungbeom/classwise_resnet18_seed42/ensemble.joblib \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output_dir runs/seungbeom/tree_reproduced \
  --methods classwise_ensemble \
  --device cuda
```

신경망과 트리 번들의 저장 형식이 다릅니다. 최종 코드 제출 전에는 선택한 방식과 가중치를
대회가 요구하는 `test.py` 진입점에 연결하고, 반복·순서 변경·부분 입력 추론의 재현성을
해당 진입점으로 확인합니다.
