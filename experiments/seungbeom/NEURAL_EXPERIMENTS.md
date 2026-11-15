# 승범 개인 신경망 비교

기존 `classwise_ensemble.py`의 트리 앙상블과 나란히 비교하는 별도 실험입니다.
실제 성능 향상을 확인한 설정은 아직 없습니다. 코드 수정·가상 CPU 검증은 로컬에서,
실제 학습·추론은 사용자가 주최 측 서버에서 수행합니다.

## 첫 비교: 같은 ResNet18 벡터, 결합 방식만 변경

이미 완료한 이미지 추출 결과를 재사용합니다. 이미지 인코더는 기존 공개 사전학습
ResNet18로 고정하고 센서 5개, seed, 학습 조건도 유지합니다.

| 후보 | 이미지·센서 결합 |
| --- | --- |
| `concat` | 원래 특징을 연결한 뒤 분류 |
| `projected_concat` | 두 특징을 같은 차원으로 투영한 뒤 연결 |
| `gated` | 같은 투영층을 사용하고 입력별 게이트로 두 특징을 결합 |

`projected_concat`과 `gated`가 게이트 효과를 비교하는 대조군입니다. 센서 특징은 현재 행만
사용하며 정답은 게이트 입력에 넣지 않습니다. 게이트 값은 물리적 중요도나 인과관계가 아닙니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
tmux new-session -A -s seungbeom-neural
```

새 tmux 안에서도 같은 폴더로 이동하고 개인 환경을 활성화합니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate

python experiments/seungbeom/neural_sweep.py plan \
  --stage fusion --max_trials 3 \
  --output runs/seungbeom/neural_plans/fusion_seed42.json
```

이 명령은 전체 설정과 변경 항목을 출력하고 계획만 저장합니다. 데이터 읽기나 학습은
시작하지 않습니다. 세 후보 모두 최대 50 epoch, 조기 종료 patience 8, 학습률 0.001,
일반 CE·shuffle을 사용합니다. 이제 계획을 순차 실행합니다.

```bash
python -u experiments/seungbeom/neural_sweep.py run \
  --plan runs/seungbeom/neural_plans/fusion_seed42.json \
  --data_dir /home/ubuntu/data/battery_data_light \
  --run_root runs/seungbeom/neural_fusion_seed42 \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --val_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/validation \
  --device cuda
```

두 벡터 폴더가 기존 실행 결과에 실제로 있는 경우에 사용하는 명령입니다. 없으면
`--train_embeddings`, `--val_embeddings` 두 인자를 모두 생략하여 직접 추출합니다.
추출 시에도 대회 데이터는 외부 서비스로 보내지 않습니다. 기존 캐시의 인코더·가중치·전처리와
ID를 확인하고, 이미지가 달라졌다면 이전 캐시를 재사용하지 않습니다.

tmux에서 나올 때는 `Ctrl+B` 후 `D`, 돌아올 때는 `tmux attach -t seungbeom-neural`입니다.

## 결과 확인

계획에 들어 있는 정확한 폴더를 자동으로 대응시키므로 폴더 이름을 추측할 필요가 없습니다.

```bash
python experiments/seungbeom/neural_sweep.py compare \
  --plan runs/seungbeom/neural_plans/fusion_seed42.json \
  --run_root runs/seungbeom/neural_fusion_seed42
```

완료된 실행의 최고 체크포인트 Validation Macro F1을 비교합니다. 최고 epoch, 클래스별 지표,
학습 이력과 비용 정보는 각 실행 폴더의 아래 파일에 기록됩니다.

| 파일 | 의미 |
| --- | --- |
| `best.pt` | 최고 Validation Macro F1 체크포인트, 추론 설정·전처리 포함 |
| `best_metrics.json` | 최고 체크포인트의 클래스별 F1·precision·recall·혼동행렬 |
| `history.json` | epoch별 학습 손실·일반 CE 검증 손실·지표·학습률·시간 |
| `summary.json` | 실제 epoch·최고 epoch·종료 이유·파라미터 수·학습·추론 시간 |
| `config.json`, `identity.json` | 전체 정규화 설정과 코드·입력의 재현 정보 |
| `status.json` | 실행 중·완료·실패 구분 |

시간은 같은 서버·동일한 다른 작업 부하에서 비교합니다. 검증 추론은 모든 후보에서
한 행씩 실행하지만, 벡터 재사용 모델의 시간에는 이미지 벡터 추출이 포함되지 않습니다.
CNN·ResNet18 직접 학습 모델은 이미지 인코더까지 포함합니다. `summary.json`의
`inference_scope`를 확인하고 서로 다른 범위의 시간을 전체 추론 속도로 비교하지 않습니다.
optimizer는 AdamW이고 학습률 스케줄러는 사용하지 않습니다.

`--run_root` 아래 `sweep_status.json`은 계획 전체의 pending/running/complete/failed 상태를
보관합니다. 실패한 실행은 정상 완료 결과로 집계하지 않습니다. 같은 명령을 다시 실행하면
동일 설정·seed·입력·코드의 완료 결과만 재사용합니다. 실패하거나 다른 설정인 폴더를
덮어쓰지 않으므로, 수정 후 다시 실행할 때는 새로운 `--run_root`를 사용합니다.
최고 가중치로 중단 학습을 이어서 재개하는 기능은 제공하지 않습니다.

### 학습 곡선 보기

세 후보가 모두 완료된 뒤, 활성화된 개인 환경에 선택 패키지를 설치하고 아래를 실행합니다.
계획에 기록된 폴더마다 학습 손실, 검증 CE, Macro F1, 단계별 F1 곡선을 저장합니다.
점선은 최고 모델의 epoch입니다. 가중 CE·Focal 학습 손실과 일반 CE 검증 손실은 정의가
다르므로 값의 높낮이를 직접 비교하지 않습니다. 이미지 파일은 서버에서 직접 확인합니다.

```bash
python -m pip install -r requirements-eda.txt
python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

plan = json.loads(Path("runs/seungbeom/neural_plans/fusion_seed42.json").read_text())
root = Path("runs/seungbeom/neural_fusion_seed42")
for trial in plan["trials"]:
    run = root / trial["id"]
    subprocess.run([
        sys.executable, "experiments/seungbeom/neural_plot.py",
        "--run_dir", str(run), "--output", str(run / "learning_curves.png"),
    ], check=True)
PY
```

기존 그림은 덮어쓰지 않습니다. 다시 생성하려면 출력 파일명을 바꿉니다.

## 다음 비교는 작은 계획으로

선택한 후보의 `config.json`을 `--base_config`로 주면 그 조건에서 지정한 요소만 바꿉니다.
계획·결과 경로는 매번 새 이름을 사용합니다.

| `--stage` | 변경하는 내용 | 후보 수 |
| --- | --- | --- |
| `lr` | 학습률 0.0001, 0.0003, 0.001 | 3 |
| `regularization` | dropout 0.1/0.3 × weight decay 0.0001/0.001 | 4 |
| `fusion` | concat / projected_concat / gated | 3 |
| `encoders` | 작은 CNN / ResNet18을 사전학습 없이 학습 | 2 |
| `imbalance` | 일반 CE / 가중 CE, 선택적으로 `--include_focal` | 2 또는 3 |
| `seeds` | 동일 후보를 seed 42, 43, 44로 반복 | 기본 3 |
| `temporal` | 원본 / 전압 30초 / 압력 30초 / 온도 변화율 / 결합 | 5 |
| `pretrained` | 고정 사전학습 ResNet18 / ConvNeXt-Tiny / DINOv2-Small | 3 |
| `ambient` | 원본 5개 / 주변 온도 추가 / 주변 대비 온도차 2개까지 추가 | 3 |

### 주변 온도 비교: 원본 정보와 차이 표현을 분리

현재 서버 clone에서 진행 중인 학습·추론이 모두 끝난 뒤 같은 브랜치를 pull합니다.
이번 비교는 기존 ResNet18 벡터·concat·seed 42·학습 설정을 그대로 유지하고 센서 입력만
바꿉니다. 과거 피처는 포함하지 않습니다. 원본 대조군도 새 결과 폴더에서 다시 학습합니다.

| `sensors` 표시 | 센서 입력 수 | 입력 |
| --- | --- | --- |
| `raw` | 5 | `meas_v`, `meas_a`, `meas_t1`, `meas_t2`, `meas_p` |
| `ambient` | 6 | 원본 5개 + `atmp_t` |
| `ambient_delta` | 8 | 위 6개 + `temp1_minus_ambient`, `temp2_minus_ambient` |

두 파생값은 같은 행의 `meas_t1 - atmp_t`, `meas_t2 - atmp_t`입니다. 부호를 유지하며
절댓값이나 고정 임계값을 사용하지 않습니다. 새 원본 센서는 `atmp_t` 하나이고,
차이 두 개는 이미 있는 정보의 다른 표현입니다. 따라서 `ambient - raw`는 주변 온도
추가 효과, `ambient_delta - ambient`는 차이 표현을 명시적으로 제공한 효과를 봅니다.
온도차 표현만으로 성능 향상을 보장하지 않습니다.

유효하지 않은 숫자·무한대는 결측으로 처리합니다. 차이에 필요한 값이 결측이면 차이도
결측으로 두고 **Train에서 구한 각 피처의 중앙값**으로 채운 다음 Train 평균·표준편차로
표준화합니다. 원본 값을 먼저 중앙값으로 채운 뒤 차이를 계산하는 방식은 아닙니다.
Validation·추론에서는 저장된 전처리를 그대로 적용합니다. `atmp_t` 열이 없으면 오류로
중단하며, 몰래 0으로 열을 만들지 않습니다. 열은 있지만 Train에서 모두 결측이면 기존
공통 전처리의 0 대체 규칙을 적용합니다. 이는 유용한 환경 정보가 추가됐다는 뜻이 아닙니다.

서버에서 아래 순서로 실행합니다. 추가 패키지는 필요하지 않습니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new-session -A -s seungbeom-ambient
```

새 tmux 터미널에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate

python experiments/seungbeom/neural_sweep.py plan \
  --stage ambient --max_trials 3 \
  --base_config runs/seungbeom/neural_fusion_seed42/embedding_concat_s42_3daac9bb72d0ba69/config.json \
  --output runs/seungbeom/neural_plans/ambient_seed42.json

python -u experiments/seungbeom/neural_sweep.py run \
  --plan runs/seungbeom/neural_plans/ambient_seed42.json \
  --data_dir /home/ubuntu/data/battery_data_light \
  --run_root runs/seungbeom/neural_ambient_seed42 \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --val_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/validation \
  --device cuda

python experiments/seungbeom/neural_sweep.py compare \
  --plan runs/seungbeom/neural_plans/ambient_seed42.json \
  --run_root runs/seungbeom/neural_ambient_seed42 \
  --compact --class_details \
  | tee runs/seungbeom/neural_ambient_seed42/comparison.txt
```

계획은 기존 파일을 덮어쓰지 않습니다. 같은 계획으로 실행만 이어갈 때는 `plan`을 생략하고
`run`부터 실행합니다. 완료·동일 설정 결과만 재사용하며 실패한 실행은 새로운 run_root가
필요합니다. 기존 이미지 벡터만 재사용하고 세 분류 신경망은 자동으로 순차 학습합니다.
`compare`의 `Completed runs: 3/3`을 확인하고 전체 Macro F1과 단계별 F1을 함께 봅니다.
화면이 잘리면 `less -S runs/seungbeom/neural_ambient_seed42/comparison.txt`로 열고
좌우 화살표로 이동하며 `q`로 종료합니다. seed 하나의 차이는 재현되는 개선으로 확정하지
않고, 유망한 조건만 후속 seed·실험 그룹 검증으로 확인합니다.

### 다음 실행: 사전학습 이미지 인코더 비교

**현재 서버의 temporal sweep이 전부 끝난 뒤** 같은 브랜치를 pull합니다. 실행 중인
서버 clone에서는 pull하지 않습니다. 로컬 수정·push는 서버 파일을 자동으로 바꾸지 않습니다.
이 비교는 센서 원본 5개·concat·seed·학습 조건을 유지하고 이미지 인코더와 그 출력 차원만
바꿉니다. 이미지 인코더는 고정하고 결합 신경망만 학습합니다. `encoders` 단계의 무작위
초기화 ResNet과 구별됩니다. 출력 차원에 따라 결합층 파라미터 수도 달라지는 비교입니다.

```bash
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
python -m pip install 'transformers>=4.36,<5' safetensors

python experiments/seungbeom/neural_sweep.py plan \
  --stage pretrained --max_trials 3 \
  --base_config runs/seungbeom/neural_fusion_seed42/embedding_concat_s42_3daac9bb72d0ba69/config.json \
  --output runs/seungbeom/neural_plans/pretrained_seed42.json

python -u experiments/seungbeom/neural_sweep.py run \
  --plan runs/seungbeom/neural_plans/pretrained_seed42.json \
  --data_dir /home/ubuntu/data/battery_data_light \
  --run_root runs/seungbeom/neural_pretrained_seed42 \
  --encoder_embeddings resnet18 \
    runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
    runs/seungbeom/classwise_resnet18_seed42/embeddings/validation \
  --device cuda

python experiments/seungbeom/neural_sweep.py compare \
  --plan runs/seungbeom/neural_plans/pretrained_seed42.json \
  --run_root runs/seungbeom/neural_pretrained_seed42 \
  --compact --class_details
```

ResNet18은 기존 이미지 벡터만 재사용하고 분류 신경망은 다시 학습합니다. ConvNeXt·DINOv2는
각 실행 폴더에서 Train/Validation 이미지를 순차 추출합니다. 입력 행·배치 변화에 대한 기존
재현 경로를 유지하기 위해 추출 batch size 1을 유지하므로 추출에 시간이 걸릴 수 있습니다.
첫 실행 시 공개 가중치를 내려받습니다. 데이터는 외부로 보내지 않습니다.

다른 인코더의 완료 캐시가 이미 있으면 `--encoder_embeddings 모델명 Train폴더 Validation폴더`를
추가합니다. 인코더 이름·차원·가중치·전처리·완료 상태를 검사하며, 서로 다른 인코더에 같은
`--train_embeddings`/`--val_embeddings` 한 쌍을 공통 적용하는 명령은 거절합니다.
동일 입력·코드·캐시·설정의 완료 실험만 재사용하며 실패 폴더는 새 run_root로 재시도합니다.
공통 비교표의 `image_encoder`, `best_epoch`, 완료 수를 확인합니다. 서로 다른 인코더의 캐시는
동일 provenance가 아니므로 이 단계에서는 `--paired`를 사용하지 않습니다.

출처·버전·라이선스는 [이미지 인코더 안내](../../image_features/README.md)의 명시적 ImageNet V1
가중치와 고정 DINOv2 revision을 유지합니다. 공식 설명:
[ConvNeXt-Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html),
[DINOv2-Small](https://huggingface.co/facebook/dinov2-small).

### 현재 진행: 최고 concat 모델에 시간 피처 추가

이미 완료한 concat 실행의 `config.json`을 기준으로 **센서 피처만** 바꿉니다.
같은 코드·seed의 원본 대조군도 새 폴더에서 다시 학습합니다. 이미지 벡터는 기존 캐시를
재사용하므로 인코더 추출을 반복하지 않습니다. 다섯 실험은 자동으로 순차 실행됩니다.

```bash
python experiments/seungbeom/neural_sweep.py plan \
  --stage temporal --max_trials 5 \
  --base_config runs/seungbeom/neural_fusion_seed42/embedding_concat_s42_3daac9bb72d0ba69/config.json \
  --output runs/seungbeom/neural_plans/temporal_seed42.json

python -u experiments/seungbeom/neural_sweep.py run \
  --plan runs/seungbeom/neural_plans/temporal_seed42.json \
  --data_dir /home/ubuntu/data/battery_data_light \
  --run_root runs/seungbeom/neural_temporal_seed42 \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --val_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/validation \
  --device cuda

python experiments/seungbeom/neural_sweep.py compare \
  --plan runs/seungbeom/neural_plans/temporal_seed42.json \
  --run_root runs/seungbeom/neural_temporal_seed42 \
  --compact --class_details
```

| `sensors` 표시 | 센서 입력 수 | 원본 5개에 추가하는 값 |
| --- | --- | --- |
| `raw` | 5 | 없음 |
| `voltage30` | 7 | `meas_v_mean30s`, `meas_v_delta30s` |
| `pressure30` | 7 | `meas_p_mean30s`, `meas_p_delta30s` |
| `temperature30` | 7 | `meas_t1_slope30s`, `meas_t2_slope30s` |
| `combined30` | 11 | 위 여섯 피처 모두 |

전압·압력 평균은 같은 실험의 `(t-30초, t]` 실제 유효 관측 평균입니다.
변화량은 현재값에서 `t-30초` 이하의 가장 최근 유효값을 뺍니다. 기준값이 30초보다 오래된
관측일 수 있으므로 초당 변화율로 부르지 않습니다. 온도 기울기는 `(t-30초, t]`의
유효 온도와 실제 초 단위 시간으로 최소제곱 직선을 맞춘 기울기(℃/초)입니다.
관측이 두 개 미만이면 결측으로 둡니다. 같은 실험의 중복 시각은 오류로 처리합니다.

부족한 이력으로 생긴 결측은 Train에서 계산한 피처별 중앙값으로 대체합니다
(Train에서 전부 결측인 피처만 0). Train 평균·표준편차로 표준화하고 Validation·Test에는
저장된 값을 적용합니다. 이력 계산은 배치·shuffle 전에 입력 CSV 전체에서 수행합니다.
미래값, 다른 실험, 실험 전체 요약, 식별자·정답 파생값은 모델 입력에 쓰지 않습니다.
호출 사이에 이력을 저장하거나 다른 CSV에서 가져오지 않습니다.

이 단계는 **Validation 비교용**입니다. 행 순서 변경·실험 분리·미래 제거에는 동일하지만,
필요한 과거 행을 임의로 삭제하면 피처와 예측이 달라질 수 있습니다. OT의 과거 이력 허용과
제출 안내의 임의 부분 입력 조건의 적용 방식은 별도 확인이 필요하며, 이 실험의 완료를
최종 제출 적합 판정으로 기록하지 않습니다. 결과를 보고 개선 후보만 추가 seed로 비교합니다.
기존 원본 concat 가중치와 제출 CSV는 유지합니다.

`encoders`는 동일 해상도·현재 행 센서·concat 조건의 구조 비교입니다. 공개 사전학습 벡터를
쓰는 첫 실험과는 사전학습 여부 및 입력 전처리가 다르므로 인코더 하나만의 효과로 해석하지
않습니다. 이 단계에는 이미지 벡터 경로 인자를 주지 않습니다.

여러 최종 후보의 반복 실험은 `--base_config`를 반복 지정해 한 계획으로 만들 수 있습니다.
후보 2개 × seed 3개라면 `--max_trials 6`을 명시합니다. 완료 후 같은 계획을 비교하면서
`--require_seeds 42 43 44 --paired`를 추가하면 동일 seed별 차이와 평균·표본 표준편차를 봅니다.
전체 설정에서 seed를 제외한 값과 코드·입력이 동일한 실행끼리만 묶습니다. 하나의 최고 seed로
우위를 주장하지 않으며 표본 표준편차는 신뢰구간이 아닙니다.

## 선택 기준과 제출

최고 체크포인트는 엄격히 더 높은 Validation Macro F1일 때만 바꾸고 동점은 먼저 나온
체크포인트를 유지합니다. 조기 종료도 정상 완료입니다. `min_delta`가 양수일 때도 작은
개선은 최고 체크포인트에 반영하며, 유의미한 개선 기준은 조기 종료 카운터에만 적용합니다.
`early_stopping_patience=0`이면 조기 종료를 끌 수 있습니다.

기존 트리 앙상블의 같은 Validation 점수와 비교하되, 새로운 모델의 성능 향상을 미리
가정하지 않습니다. Validation으로 선택한 최고 점수는 독립적인 최종 일반화 점수와 다릅니다.
Test는 가중치·피처·학습 조건 선택에 사용하지 않습니다. 실제로 이미지와 센서가 모두
예측에 기여하는지, 저장·복원과 부분 입력에 대한 추론이 재현되는지도 최종 제출 전에 확인합니다.

원본 이미지·학습 데이터·확률·체크포인트·실제 보고서는 서버의 비공개 실행 폴더에만 둡니다.
최종 후보를 정한 뒤 개인 `neural_experiment.py predict`로 추론하며, 기존 루트 `test.py`와
기존 앙상블의 joblib는 서로 다른 형식이므로 제출 진입점 연결은 별도 작업입니다.

구조 참고: [Gated Multimodal Units](https://arxiv.org/abs/1702.01992),
[ResNet](https://arxiv.org/abs/1512.03385). 여기의 게이트는 Linear+ReLU 투영을 사용한
실험용 변형이며 논문 결과를 이 데이터의 성능 근거로 사용하지 않습니다. 공개 이미지 벡터의
가중치·전처리·라이선스는 [이미지 인코더 안내](../../image_features/README.md)를 참고합니다.
