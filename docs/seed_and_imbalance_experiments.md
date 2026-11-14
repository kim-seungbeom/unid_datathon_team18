# 반복 seed와 클래스 불균형 비교

현재 단계는 이미 실행한 원본 센서 기준과 전압 이력 후보의 seed 42 결과에 seed 43·44를 각각 더하는 것입니다. 먼저 추가 4회를 완료해 같은 seed끼리 비교하고, 그다음 클래스 불균형 대응을 별도 실험으로 평가합니다. 이 문서는 실제 성능·분포 수치를 포함하지 않으며, 낮은 클래스 F1의 원인을 불균형으로 확정하지 않습니다.

코드 수정·독립적인 가상 입력 CPU 검증은 로컬에서 수행합니다. 아래 실제 데이터 명령은 사용자가 서버에서 직접 실행합니다. 코드는 `feat/seungbeom-sensor`에 유지하며 `main`에 merge하지 않습니다. [센서 피처 비교 안내](sensor_ablation_experiments.md)의 Train-only 전처리와 [인과적 이력 규칙](temporal_sensor_experiments.md)을 그대로 적용합니다.

## 1. 같은 두 설정을 seed 43·44로 반복하기

| 기준 설정 | 추가 설정 | 기존 대비 변경 |
| --- | --- | --- |
| `configs/sensor_raw.json` | `configs/sensor_raw_seed43.json`, `configs/sensor_raw_seed44.json` | seed만 43·44로 변경 |
| `configs/sensor_history_voltage.json` | `configs/sensor_history_voltage_seed43.json`, `configs/sensor_history_voltage_seed44.json` | seed만 43·44로 변경 |

원본은 센서 5개, 전압 이력은 같은 5개와 `meas_v_mean30s`, `meas_v_delta30s`를 사용합니다. 두 계열 모두 median 대체, standard scaling, 10 epochs, batch size 32, sensor hidden 32, learning rate 0.001, weight decay 0.0001, dropout 0.1을 유지합니다. 기존 설정에 생략된 전처리·학습 목적 항목은 동일한 기본값으로 읽습니다. 손실은 일반 cross entropy, Train 로더는 shuffle입니다.

변경 코드가 commit·push된 뒤, 학습 중인 작업이 없는 본인 서버 clone에서 갱신합니다. `git status`에 본인이 수정한 파일이 있으면 먼저 보존하고 해결한 뒤 진행합니다. `seungbeom`은 승범의 예시이며 다른 팀원은 본인 경로를 사용합니다. 공용 데이터·venv를 복제하거나 공용 패키지를 다시 설치하지 않습니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short --branch
git fetch origin
git switch feat/seungbeom-sensor
git pull --ff-only origin feat/seungbeom-sensor
```

기존 `seungbeom-sensor` tmux 창 안에 있다면 그대로 진행합니다. tmux 밖이라면 아래 명령으로 기존 세션에 접속하거나 새로 만듭니다.

```bash
tmux new-session -A -s seungbeom-sensor
```

tmux 안에서 다음 명령으로 위치와 환경을 설정하고, 팀원과 GPU 사용을 조율한 뒤 아래 4회를 하나씩 완료합니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/venv/bin/activate
```

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_raw_seed43.json \
  --run_dir runs/seungbeom/sensor_raw_seed43
```

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_history_voltage_seed43.json \
  --run_dir runs/seungbeom/sensor_history_voltage_seed43
```

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_raw_seed44.json \
  --run_dir runs/seungbeom/sensor_raw_seed44
```

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_history_voltage_seed44.json \
  --run_dir runs/seungbeom/sensor_history_voltage_seed44
```

seed가 이미 설정 파일 이름에 포함되므로 run 이름에 `_seed43_seed43`처럼 중복해서 붙이지 않습니다. 모든 run 디렉터리는 새 경로여야 합니다. 기존 폴더가 있으면 덮어쓰지 말고 새 이름을 정한 뒤 비교 명령에도 반영합니다. 학습 중에는 해당 clone에서 pull하거나 브랜치를 전환하지 않습니다. tmux에서 `Ctrl-b`, `d`로 분리하고, 이후 `tmux attach -t seungbeom-sensor`로 돌아올 수 있습니다.

## 2. 여섯 실행을 비교하기

기존 seed 42 두 실행이 동일한 학습 설정·Train/Validation 분할을 사용했다면 함께 비교합니다. 이번 확장은 기존 CE·shuffle·전처리 기본 동작을 유지합니다. 실제 사용한 폴더명이 다르면 아래 경로만 바꿉니다. 설정 비교가 코드 동등성을 자동으로 입증하는 것은 아니므로, 별도로 학습 구현을 바꾼 결과를 같은 seed 반복으로 묶지 않습니다.

```bash
python scripts/compare_runs.py \
  runs/seungbeom/sensor_raw_seed42 \
  runs/seungbeom/sensor_raw_seed43 \
  runs/seungbeom/sensor_raw_seed44 \
  runs/seungbeom/sensor_history_voltage_seed42 \
  runs/seungbeom/sensor_history_voltage_seed43 \
  runs/seungbeom/sensor_history_voltage_seed44 \
  --class_details
```

기본 표는 loss, sampling, seed, best epoch, 6클래스 Macro F1과 클래스별 F1을 포함합니다. `--class_details`는 저장된 confusion matrix를 바탕으로 클래스별 precision·recall·F1·support를 함께 표시합니다. confusion matrix가 저장된 이전 형식의 결과도 읽을 수 있습니다. 결과 파일은 서버에서 관리하고 공개 Git에 커밋하지 않습니다.

```bash
python scripts/summarize_seeds.py \
  runs/seungbeom/sensor_raw_seed42 \
  runs/seungbeom/sensor_raw_seed43 \
  runs/seungbeom/sensor_raw_seed44 \
  runs/seungbeom/sensor_history_voltage_seed42 \
  runs/seungbeom/sensor_history_voltage_seed43 \
  runs/seungbeom/sensor_history_voltage_seed44 \
  --require_seeds 42 43 44 \
  --paired
```

요약 도구는 저장 설정의 기본값을 채운 뒤 **seed를 제외한 전체 설정**이 같은 실행을 묶습니다. 피처 순서, 전처리, 손실·sampling, 나머지 학습 조건이 다르면 별도 그룹입니다. Train/Validation 분할도 같아야 합니다. `--require_seeds 42 43 44`로 각 그룹의 필요한 seed를 확인합니다.

그룹별 평균과 표본 표준편차(`ddof=1`)를 확인합니다. `--paired`는 첫 번째로 나온 그룹인 원본 센서를 기준으로, 같은 seed의 `전압 이력 Macro F1 − 원본 Macro F1`을 계산하고 그 차이의 평균·표본 표준편차를 요약합니다. 여러 seed에서 개선 방향이 유지되는지 살펴보는 도구이며, 세 번의 반복에 대한 표준편차는 신뢰구간이나 통계적 유의성 판정이 아닙니다. 같은 seed라도 입력 차원이 다르면 모델 초기화와 학습 과정이 완전히 같아지는 것은 아닙니다.

## 3. 낮은 클래스 F1의 원인 구분하기

F1이 낮다는 사실만으로 해당 클래스의 Train 표본이 적다고 결론내릴 수 없습니다. recall이 낮으면 실제 해당 클래스를 놓치는 경우가 많고, precision이 낮으면 다른 클래스를 해당 클래스로 잘못 예측하는 경우가 많습니다. 두 경우를 클래스별 지표와 confusion matrix로 구분합니다. 피처의 구분력, 클래스 간 유사성, 전이 구간의 모호함, 학습의 변동도 원인 후보입니다.

가중치는 **Train 클래스 빈도**로 계산합니다. Validation의 낮은 F1을 뒤집은 값이나 특정 클래스 번호를 하드코딩하지 않습니다. Validation은 기존 분포와 전체 행을 그대로 평가하며, 가중 sampling이나 Train 통계 재학습을 적용하지 않습니다. Test는 추론에만 사용합니다. Train에 없는 클래스는 가중치나 재추출만으로 학습할 수 없으며, 그 클래스에 임의의 행을 만들지 않습니다.

## 4. 불균형 대응을 하나씩 비교하기

아래 6개 설정은 각 원본 계열의 seed 42·피처·10-epoch MLP 조건을 유지합니다. 모든 파일에 `loss`, `sampling`, `class_weight_power=0.5`, `focal_gamma=2.0`을 명시합니다. 해당 손실에서 사용하지 않는 파라미터는 효과를 주지 않습니다.

| 원본 5개 설정 (`configs/`) | 전압 이력 설정 (`configs/`) | 한 번에 바꾸는 항목 |
| --- | --- | --- |
| `sensor_raw_weighted_ce.json` | `sensor_history_voltage_weighted_ce.json` | `loss=weighted_cross_entropy`, `sampling=shuffle` |
| `sensor_raw_focal.json` | `sensor_history_voltage_focal.json` | 가중치 없는 `loss=focal`, `sampling=shuffle` |
| `sensor_raw_balanced_sampling.json` | `sensor_history_voltage_balanced_sampling.json` | `loss=cross_entropy`, `sampling=balanced` |

**Weighted cross entropy:** Train의 클래스별 행 수를 `n_c`라 할 때, 존재하는 클래스에 `w_c ∝ n_c^(-0.5)`를 부여하고 정규화합니다. 없는 클래스의 가중치는 0입니다. PyTorch는 `CrossEntropyLoss(weight=...)`로 클래스별 손실 가중치를 지원합니다. 역빈도 `1/n_c`보다 완만한 역제곱근을 먼저 비교하는 것은 이번 실험의 선택이며 개선이 검증된 법칙은 아닙니다. 가상의 Train 행 수가 1,000 대 100이면 소수/다수 클래스의 가중치 비율은 역빈도에서 10배, 역제곱근에서 약 3.16배입니다. 가중치를 올리면 recall이 좋아지면서 precision이 나빠질 수도 있습니다. [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

**Focal loss:** 정답 클래스 softmax 확률을 `p_y`라 할 때 `−(1−p_y)^2 log(p_y)`를 사용합니다. 클래스 빈도 가중치는 결합하지 않습니다. 예를 들어 `p_y=0.9`인 쉬운 예제의 CE에는 0.01, `p_y=0.2`인 어려운 예제에는 0.64가 곱해집니다. 원 논문은 dense object detection의 전경·배경 불균형을 다뤘습니다. 여기서는 그 원리를 6클래스 softmax 분류에 적용하며, 논문의 탐지 성능 개선이 이 대회의 개선을 보장하지 않습니다. [Lin et al., Focal Loss, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

**Balanced sampling:** Train 행마다 `1/n_y`의 추출 가중치를 주고, `WeightedRandomSampler`로 복원 추출합니다. 한 epoch의 추출 수는 기존 Train 행 수와 같아 업데이트 횟수의 비교 기준을 유지합니다. 각 클래스의 추출 확률이 기대값에서 같아질 뿐, 각 배치가 반드시 같은 클래스 수를 담지는 않습니다. 일부 행이 반복되고 일부는 해당 epoch에 나오지 않을 수 있습니다. 시간상 인접한 센서 행의 반복은 독립적인 새 정보를 늘리는 일이 아니므로 과도한 반복이 도움될지는 검증해야 합니다. 별도 seed generator로 sampling을 재현하고, 원래 Train 전체에서 인과적 전처리·이력 피처를 계산한 뒤 완성된 데이터셋의 인덱스를 추출합니다. [PyTorch WeightedRandomSampler](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)

처음에는 원본 5개 기준에서 세 대응을 따로 비교합니다. weighted CE와 balanced sampling을 한 실행에 동시에 적용하지 않습니다. 아래 반복은 선택 사항이며, **사용자가 서버에서** 현재 작업 완료 후 실행합니다. 오류나 기존 경로가 있으면 중단하고 원인을 확인합니다.

```bash
imbalance_candidates=(
  sensor_raw_weighted_ce
  sensor_raw_focal
  sensor_raw_balanced_sampling
)
for imbalance_config in "${imbalance_candidates[@]}"; do
  imbalance_run_dir="runs/seungbeom/${imbalance_config}_seed42"
  if [ -e "$imbalance_run_dir" ]; then
    printf 'Existing run directory: %s\n' "$imbalance_run_dir"
    break
  fi
  python -u train.py \
    --data_dir /home/ubuntu/data/battery_data_light \
    --config "configs/${imbalance_config}.json" \
    --run_dir "$imbalance_run_dir" || break
done
```

```bash
python scripts/compare_runs.py \
  runs/seungbeom/sensor_raw_seed42 \
  runs/seungbeom/sensor_raw_weighted_ce_seed42 \
  runs/seungbeom/sensor_raw_focal_seed42 \
  runs/seungbeom/sensor_raw_balanced_sampling_seed42 \
  --class_details
```

전압 이력에서도 확인하려면 위 반복의 후보 목록을 아래 세 이름으로 바꾸고 동일하게 순차 실행합니다. 비교 기준은 `sensor_history_voltage_seed42`입니다. 원본과 전압 이력의 결과 차이에 손실 변경 효과를 섞지 않습니다.

```bash
imbalance_candidates=(
  sensor_history_voltage_weighted_ce
  sensor_history_voltage_focal
  sensor_history_voltage_balanced_sampling
)
```

```bash
python scripts/compare_runs.py \
  runs/seungbeom/sensor_history_voltage_seed42 \
  runs/seungbeom/sensor_history_voltage_weighted_ce_seed42 \
  runs/seungbeom/sensor_history_voltage_focal_seed42 \
  runs/seungbeom/sensor_history_voltage_balanced_sampling_seed42 \
  --class_details
```

서로 다른 loss의 Train loss 수치는 직접 우열 비교하지 않습니다. 일차 선택은 동일한 Validation 전체에 대한 6클래스 Macro F1로 하되, 약한 클래스의 precision·recall과 confusion matrix, 다른 클래스의 손실을 함께 살핍니다. seed 42에서 고른 후보는 임시 후보이며, 추가 seed에서도 같은 기준 설정과 비교합니다. Validation 반복 선택은 최종 Test 성능을 입증하지 않습니다.

후속 후보로는 Cui 등의 effective-number 가중치 `w_c ∝ (1−β)/(1−β^n_c)`가 있습니다. 클래스 표본이 늘수록 새 표본의 추가 정보가 감소한다는 관점의 방법이며, β 선택이 추가로 필요합니다. 이번에는 구현하지 않습니다. 논문의 long-tailed 이미지 분류 결과를 센서 데이터 성능으로 일반화하지 않습니다. [Cui et al., Class-Balanced Loss, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)

## 5. 저장 정보와 팀원 전달

새 학습은 run의 `training_objective.json`에 Train 클래스 수와 적용 가중치·손실·sampling 설정을 기록합니다. 이 파일은 실제 데이터에서 얻은 집계를 포함하므로 다른 run 결과·가중치와 함께 팀 내부 서버에서 관리하고 공개 Git에 올리지 않습니다. `config.json`, `preprocessing.json`, `split.json`, `environment.json`, `best_metrics.json`, `best.pt`도 기존 경계에 따라 관리합니다. 이전 run에 `training_objective.json`이 없어도 비교 도구는 기본 학습 설정을 해석할 수 있습니다.

학습 코드 전달에는 새 `training_objectives.py`도 포함합니다. `model.py`, `train.py`, `test.py` 진입점과 1..6 출력 라벨을 유지하며, 체크포인트의 외부 버전은 v4를 유지합니다. 추론은 저장된 모델·피처·Train 전처리를 복원하고 Train용 loss나 sampler로 입력 분포를 바꾸지 않습니다.

MLP 센서 단독과 이미지+센서 모델은 같은 분류 loss·sampling 구성을 사용할 수 있습니다. 센서 단독에서 고른 후보도 결합 모델의 Validation에서 다시 비교해야 합니다. 팀원에게는 선택한 피처 순서·전처리·학습 설정과 필요한 파일을 내부 서버에서 전달하며, 최종 대회 모델은 이미지와 센서를 모두 사용합니다. 이 문서는 실제 서버 학습을 실행하거나 완료했다고 주장하지 않습니다.

2026-11-14 로컬 검증에서 기존 파이프라인과 새 손실·sampling·seed 요약·클래스별 지표의 가상 CPU 테스트 165개 및 명시한 코드의 Ruff 검사를 통과했습니다. 실제 대회 데이터 학습·평가는 사용자가 서버에서 수행합니다.
