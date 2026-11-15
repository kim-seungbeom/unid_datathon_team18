# 승범 개인 멀티모달 앙상블

개인 작업 브랜치는 `feat/seungbeom-classwise-ensemble`입니다. 코드 수정은 로컬에서 하고,
Git으로 받은 코드를 사용자가 대회 서버에서 실행합니다. 실행 진입점은 이 폴더의
`classwise_ensemble.py` 하나이며, 결합 계산과 모델 정의는 같은 폴더의 보조 모듈로 분리합니다.
다른 사람의 실험 코드와 루트 `train.py`, `test.py`는 변경하지 않습니다.

## 무엇을 비교하나

현재 행의 센서 5개와 고정 이미지 인코더 벡터를 함께 사용하는 XGBoost, LightGBM,
ExtraTrees를 기본으로 비교합니다. 이미지 인코더 기본값은 ResNet18입니다.
더 나은 성능을 보장하는 조합은 아니며, 아래 세 방식을 동일한 Validation에서 비교합니다.

- 개별 모델
- 모델 확률의 단순 평균
- 클래스별로 모델 비중을 다르게 적용한 결합

클래스별 비중은 `[모델 수, 6]`이고 각 클래스의 모델 비중 합은 1입니다.
각 단계의 결합 점수를 계산한 뒤 가장 높은 단계 하나를 1~6으로 출력합니다.
단계별 비중이 다르므로 결합 점수 자체를 보정된 확률로 해석하지 않습니다.
기본 가중치 탐색은 과도한 편중을 줄이는 정규화 계수 `0.001`, 최대 `3`라운드를 사용합니다.

## 학습·검증 구분

1. Train을 `experiment_id` 단위 GroupKFold로 나눕니다. 한 실험의 행은 같은 fold에 있습니다.
2. 각 fold의 학습 부분에서만 결측 대체·표준화·PCA·분류기를 학습합니다.
3. 해당 fold에서 학습하지 않은 행의 예측을 모아 Train 전체의 OOF 예측을 만듭니다.
4. Train OOF로만 결합 비중과 적용 전략을 결정하고 고정합니다.
5. 기본 모델을 전체 Train으로 다시 학습하고 Validation에서 비교합니다.

가중치 최적화에 사용한 OOF 점수는 학습 진단값입니다. 그 점수를 독립 검증 성능으로
보고하지 않습니다. Validation 점수는 가중치 자동 조정에 사용하지 않습니다.
Test는 추론에만 사용합니다. 이 명령에는 Train+Validation 재학습 기능이 없습니다.
Validation을 보고 설정을 여러 번 바꾸면 해당 Validation도 모델 선택에 사용한 데이터가 됩니다.

## 서버에서 실행

아래 명령은 이 브랜치가 push된 뒤 서버에서 실행합니다. 실행 중인 학습이 없는 상태에서
코드를 받고, `git status`에 수정 파일이 있으면 먼저 보존합니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git fetch origin
git switch feat/seungbeom-classwise-ensemble
git pull --ff-only origin feat/seungbeom-classwise-ensemble

bash experiments/seungbeom/setup_env.sh
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
tmux new-session -A -s seungbeom-ensemble
```

새 tmux 안에서도 프로젝트 폴더로 이동하고 개인 환경을 활성화합니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate

python -u experiments/seungbeom/classwise_ensemble.py train \
  --data_dir /home/ubuntu/data/battery_data_light \
  --run_dir runs/seungbeom/classwise_resnet18_seed42 \
  --encoder resnet18 \
  --models xgboost lightgbm extra_trees \
  --folds 5 \
  --pca_dim 32 \
  --seed 42 \
  --device cuda
```

인코더는 대회 서버 GPU에서 이미지 벡터를 추출합니다. 기본 분류기들은 같은 서버의 CPU에서
학습합니다. 5개 fold와 전체 Train 재학습이 있어 단일 모델 학습보다 시간이 더 걸립니다.
이미지 추출은 입력 구성에 따른 수치 차이를 줄이기 위해 기본 `--batch_size 1`, `--workers 0`입니다.
큰 배치를 선택하면 처리 속도와 메모리 사용이 달라지고, 최종 추론과의 수치 일치도 다시 확인해야 합니다.
tmux에서 빠져나오려면 `Ctrl+B`를 누른 뒤 `D`, 다시 들어가려면
`tmux attach -t seungbeom-ensemble`을 사용합니다. 새 실험은 새 `--run_dir`를 지정합니다.

학습이 끝나면 다음 명령으로 비교합니다.

```bash
python experiments/seungbeom/classwise_ensemble.py report \
  --run_dir runs/seungbeom/classwise_resnet18_seed42 \
  --class_details
```

단계별 F1·Precision·Recall과 전체 Macro F1을 함께 확인합니다. 클래스별 결합이
단순 평균이나 단일 모델보다 낮으면 개선됐다고 보고하지 않습니다.

이미 추출한 동일 인코더의 벡터가 있으면 학습 명령에 두 인자를 함께 추가할 수 있습니다.
`<...>`는 서버에 실제 존재하는 본인의 폴더 경로로 바꿉니다.

```text
--train_embeddings <Train 벡터 폴더> --val_embeddings <Validation 벡터 폴더>
```

각 폴더에는 `features.npy`, `sample_ids.csv`, `metadata.json`이 있어야 합니다.
행은 `sample_id`로 연결하고 인코더·가중치·차원·전처리 설정을 확인합니다.
학습 입력 이미지를 바꿨다면 예전 캐시를 재사용하지 않고 다시 추출합니다.
캐시의 추출 배치 크기가 다르면 같은 이미지에서도 미세한 수치 차이가 생길 수 있습니다.
캐시로 계산한 Validation 점수가 이미지 재추출 결과와 정확히 같다고 가정하지 않습니다.

`--run_dir` 아래에 다음 결과가 저장됩니다.

| 파일 | 내용 |
| --- | --- |
| `ensemble.joblib` | 전체 Train으로 학습한 기본 모델·전처리·고정 결합 비중 |
| `class_weights.json` | 모델별·단계별 비중과 Train OOF에서 선택한 전략 |
| `metrics.json`, `comparison.tsv` | Train OOF 진단값과 Validation 비교 지표 |
| `train_oof.npz` | Train 행별 OOF 확률·정답·fold·식별자, 서버 내부 분석용 |
| `validation_predictions.npz` | Validation 확률·정답·식별자, 서버 내부 분석용 |
| `weight_search.json` | 비중 탐색 설정과 진단 결과 |
| `config.json`, `environment.json`, `split.json` | 실행 설정·패키지/소스 버전·분할 정보 |
| `status.json` | 진행·완료·실패 상태 |

가중치 선택 결과가 `best_single`이나 `equal_average`일 수도 있습니다. 이 경우 해당 비중을
그대로 저장하며, 클래스별 조합이 더 좋아졌다고 해석하지 않습니다.

## 예측

```bash
python experiments/seungbeom/classwise_ensemble.py predict \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --checkpoint runs/seungbeom/classwise_resnet18_seed42/ensemble.joblib \
  --output outputs/seungbeom/classwise_resnet18_seed42/submission.csv \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --device cuda
```

추론은 전달된 CSV 행과 이미지에서 새로 벡터를 계산하며 저장된 전처리와 모델만 적용합니다.
예측 명령은 이미지 캐시를 받지 않습니다. 출력은 `sample_id,label`이고,
`--sample_submission`을 주면 ID가 정확히 같은지 검사한 뒤 그 양식의 행 순서를 따릅니다.
일부 입력만 별도 추론할 때는 이 인자를 생략하면 입력 CSV 순서대로 출력합니다.

최종 제출용 루트 `test.py` 연결은 별도 작업입니다. 이 개인 실험의 joblib를 기존 루트
PyTorch 추론기에 그대로 넘기지 않습니다. 실제 최종 모델을 정하면 필수 제출 CLI에 연결하고,
같은 입력의 재현성·행 순서·부분 입력·빈 이미지 교체에 따른 실제 예측 변화를 확인해야 합니다.
이미지 피처를 연결했다는 사실만으로 이미지 활용 요건이 검증된 것은 아닙니다.

## 센서·인코더 설정

- 기본 센서는 `meas_v`, `meas_a`, `meas_t1`, `meas_t2`, `meas_p`이며 결측은 학습 부분의 중앙값으로 대체합니다.
- `--sensor_config configs/sensor_raw.json`처럼 센서 설정을 지정할 수 있습니다.
  기존 신경망 설정의 Focal Loss나 학습률이 트리 모델에 그대로 적용되는 것은 아닙니다.
- 과거 피처나 직전값 대체를 선택하면 현재·과거의 같은 실험 행만 사용합니다.
  필요한 과거 행이 입력에서 빠지면 피처와 예측이 달라질 수 있으므로 부분 입력 검증을 별도로 해야 합니다.
- ResNet18·ConvNeXt·DINOv2의 가중치 출처, 고정 revision과 전처리는
  [이미지 인코더 안내](../../image_features/README.md)에 있습니다. 최종 README에는 실제 사용한
  출처·버전·라이선스를 기록합니다. 외부 학습 데이터나 외부 추론 API는 사용하지 않습니다.
- 기본 가중치는 [공식 ResNet18 `IMAGENET1K_V1`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)입니다.
  Torchvision 코드 라이선스는 [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE)이고,
  사전학습 모델·데이터의 이용 조건은 [공식 가중치 안내](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights)를 별도로 따릅니다.
  실제 Torch/Torchvision 패키지 버전은 `environment.json`, 가중치 식별자와 이미지 전처리는 저장 모델에 기록합니다.
- DINOv2를 직접 추출하려면 개인 환경에 `transformers>=4.36,<5`, `safetensors`가 추가로 필요합니다.

## 환경과 파일 보관

`setup_env.sh`는 기본적으로 `/home/ubuntu/venv/bin/python`을 기준으로
`$HOME/seungbeom/venv-ensemble`을 만듭니다. 공용 site-packages는 `.pth`로 참조하고,
추가 패키지는 개인 환경에 설치합니다. 공용 torch/torchvision은 설치·업그레이드하지 않습니다.
필요하면 실행 전에 `ENSEMBLE_BASE_PYTHON`, `ENSEMBLE_VENV`로 경로를 지정할 수 있습니다.
설치 후 공용 CUDA 패키지가 그대로 사용되는지 확인합니다.
기존 개인 환경의 Python 주·부 버전이 공용 환경과 다르면 새 환경 경로를 지정하도록 중단합니다.

코드·설정·문서만 Git으로 전달합니다. 모델, OOF 확률, 전처리 통계, 개별 예측 및 보고서는
서버의 `runs/seungbeom/`, `outputs/seungbeom/` 아래에 보관합니다.
로컬 검증은 가상 데이터와 임의 이미지 벡터만 사용하며 실제 성능은 서버 실행 후 확인합니다.
