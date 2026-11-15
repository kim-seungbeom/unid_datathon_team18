# 센서별 이력·추가 센서·스케일링 비교

이번 비교의 기준은 `configs/sensor_raw.json`의 원본 센서 5개(`meas_v`, `meas_a`, `meas_t1`, `meas_t2`, `meas_p`)입니다. 온도 파생 3개를 추가한 8개 입력 설정은 이전 후보로 유지하지만 이번 비교의 기준에는 넣지 않습니다. 우선 전압·전류·압력의 30초 피처를 각각 추가해 어떤 센서의 과거 정보가 기여하는지 분리합니다. 이후 가스, 주변 온도, 열화상 온도 수치, 결측 플래그와 Robust scaling을 선택적으로 비교합니다. 아직 실제 성능이나 최적 조합을 확인한 문서는 아닙니다.

코드 수정과 독립적인 가상 입력 CPU 검증은 로컬에서 수행합니다. 실제 데이터 확인·학습·평가는 사용자가 GPU 서버에서 직접 실행하며, 아래 명령은 에이전트 실행용이 아닙니다. 코드는 `feat/seungbeom-sensor`에 유지하고 별도 요청 없이 `main`에 merge하지 않습니다.

## 공식 정의와 후보의 근거

[운영진 자료집](https://hkksi.notion.site/Uni-DTHON-416b516e192380cb9459d0149ba78fd5)을 2026-11-14에 확인했습니다. 데이터 파일이나 자료집의 데이터 미리보기 이미지를 읽어 얻은 판단은 아닙니다. 자료집과 DAKER 대회 페이지가 충돌하면 대회 페이지를 따릅니다.

| 원본 컬럼 | 공식 정의 | 이번 비교에서의 취급 |
| --- | --- | --- |
| `meas_v` | 전압, V | 현재값과 30초 평균·변화량 비교 |
| `meas_a` | 전류, A; 모든 행에서 0 | 전류 이력 추가와 전류 제외를 낮은 우선순위의 대조 실험으로 유지 |
| `meas_p` | 압력; 단위 미확인 | 단위를 임의로 정하지 않고 현재값과 30초 평균·변화량 비교 |
| `meas_co`, `meas_co2`, `meas_ch4` | CO·CO₂·CH₄ 농도, ppm | 가스 3개를 함께 추가 |
| `atmp_t` | 시험실의 주변 온도, ℃ | 표면·양극부 온도에서 각각 빼는 후보 |
| `thermal_t` | 제공된 열화상 온도 수치, ℃ | 수치 하나를 센서 입력으로 추가; 픽셀별 온도 지도는 아님 |

`thermal_t`의 최대·평균 여부, 산출 영역·방식, 라벨 생성과의 관계는 자료집에 정의되어 있지 않습니다. 이를 최고 온도, 평균 온도 또는 정답에서 만든 값이라고 단정하지 않습니다. 의사색 픽셀을 실제 온도로 바꾸는 기준도 제공되지 않습니다. 이번 센서 단독 실험은 이미지를 읽지 않으며, `thermal_t` 수치 추가가 최종 모델의 이미지 사용을 대신하지 않습니다.

자료집은 결측이 가열 센서 `indu_t`, `indu_t.1`에만 있다고 설명합니다. 따라서 원본 5개와 가스의 결측 플래그는 공식 설명대로라면 상수 0이며, 정보가 추가되지 않을 것으로 예상되는 대조 실험입니다. 전류 평균·변화량도 상수 입력에서 새 측정 정보를 만들지 않습니다. 요청한 설정은 남기되 전압·압력·가스보다 낮은 우선순위로 판단합니다. 이는 실제 분포를 에이전트가 확인한 결과가 아닙니다.

상수 피처를 추가·제거해도 입력층 크기와 난수 초기화 과정이 달라져 점수가 움직일 수 있습니다. 이런 변화만으로 새로운 측정 정보의 효과라고 판단하지 않고 반복 seed와 함께 해석합니다.

이번 후보에는 가열 센서, 중복 온도 합, 실험 조건 메타데이터를 추가하지 않습니다. `indu_t`는 과충전 방식에서 결측이고 `indu_t.1`은 Test에 컬럼은 있지만 전부 결측입니다. 두 항목은 서로 다른 센서입니다. `meas_t1_plus_t2`는 두 온도의 합과 중복됩니다. 실험 조건은 공식 허용 범위지만 이번 비교에서는 센서와 전처리 효과에 범위를 한정합니다. 식별자·파일명·정답은 입력 피처로 사용하지 않습니다.

## 비교 설정

모든 새 설정은 원본 기준과 같은 seed 42, 10 epochs, batch size 32, 학습률 0.001, weight decay 0.0001, sensor hidden 32, dropout 0.1을 사용합니다. `sensor_imputation`은 모두 `median`이며 `sensor_scaling`은 robust 두 설정을 제외하고 `standard`로 명시합니다. 원본 기준 설정에서 생략된 두 항목도 같은 기본값을 사용합니다. 피처 수에 따라 입력층 차원만 달라지고 나머지 MLP 설정은 같습니다.

| 설정 파일 이름 (`configs/` 아래) | 입력 수 | 기준 대비 변경 | 비교 상대 |
| --- | --- | --- | --- |
| `sensor_raw.json` | 5 | 원본 기준 | — |
| `sensor_history_voltage.json` | 7 | 전압 30초 평균·변화량 | raw |
| `sensor_history_current.json` | 7 | 전류 30초 평균·변화량 | raw; 낮은 우선순위 |
| `sensor_history_pressure.json` | 7 | 압력 30초 평균·변화량 | raw |
| `sensor_gas.json` | 8 | CO·CO₂·CH₄ | raw |
| `sensor_ambient.json` | 7 | `temp1_minus_ambient`, `temp2_minus_ambient` | raw |
| `sensor_thermal.json` | 6 | `thermal_t` | raw |
| `sensor_thermal_ambient.json` | 7 | `thermal_t`, `thermal_minus_ambient` | thermal |
| `sensor_missing.json` | 10 | 원본 5개 각각의 `_missing` | raw; 낮은 우선순위 |
| `sensor_raw_robust.json` | 5 | robust scaling만 변경 | raw |
| `sensor_gas_robust.json` | 8 | 가스 입력의 scaling만 변경 | gas |
| `sensor_gas_missing.json` | 11 | 가스 3개 각각의 `_missing` | gas; 낮은 우선순위 |
| `sensor_no_current.json` | 4 | 상수 전류 제거 | raw |

한 후보의 효과를 다른 변경과 섞지 않도록 표의 비교 상대를 사용합니다. 특히 gas robust와 gas missing은 원본 5개 기준뿐 아니라 gas 8개 기준과 직접 비교해야 합니다. 선택적 후보는 전체를 한꺼번에 결합하기 전에 개별 효과부터 확인합니다.

## 피처와 전처리 계약

- `*_mean30s`: 같은 실험의 현재 시각 `t`에서 `(t-30, t]` 구간에 존재하는 유효 원본 관측의 평균입니다. 행 개수가 아닌 실제 초 단위를 사용하며, 대체한 값을 새 관측처럼 평균에 더하지 않습니다.
- `*_delta30s`: 현재값에서 `t-30` 이하의 가장 최근 유효 관측값을 뺍니다. 정확히 30초 전의 행이 없어도 이전 관측을 사용할 수 있으므로 초당 변화율은 아닙니다. 필요한 과거가 없으면 결측으로 남긴 후 Train 중앙값으로 대체합니다.
- `temp1_minus_ambient = meas_t1 - atmp_t`, `temp2_minus_ambient = meas_t2 - atmp_t`, `thermal_minus_ambient = thermal_t - atmp_t`입니다.
- `*_missing`: 원본 수치 변환 직후의 결측 여부입니다. 변환 실패·무한대도 결측으로 처리하며, forward fill을 선택하는 후속 실험에서도 대체하기 전 상태를 기록합니다. 플래그는 정규화하지 않아 0·1로 유지합니다.
- 선택된 피처에 필요한 추가 원본 컬럼이 없으면 명확한 오류를 냅니다. 컬럼 부재를 전부 결측인 컬럼으로 조용히 바꾸지 않습니다. 같은 피처 순서를 학습·검증·추론에 유지합니다.

표준 방식은 결측 대체 후 Train의 특징별 평균·표준편차를 저장합니다. Robust 방식은 결측 대체 후 Train의 특징별 중앙값과 `Q75 - Q25`를 저장하고 `(x - 중앙값) / IQR`로 변환합니다. 매우 작은 IQR에는 스케일 1을 사용합니다. 두 방식 모두 Validation·Test에서 통계를 다시 학습하지 않습니다. Robust scaling은 이상값을 삭제하거나 클리핑하는 처리가 아니며, 큰 수치가 자동으로 사라지거나 성능이 개선된다고 보장하지 않습니다. [scikit-learn RobustScaler 설명](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)

운영진 자료집의 허용 범위와 별개로 이 저장소의 비교 실험은 전처리 학습을 Train에만 한정합니다. 미래 관측·다른 실험·전체 실험 요약을 피처나 대체 기준에 넣지 않습니다. shuffle, 실험 분리, 미래 제거에 대한 결과 일관성을 유지합니다. 필요한 과거 행을 제거하면 이력 피처가 바뀔 수 있으므로, 이는 미래 사용 여부와 구분해 진단해야 합니다. 자세한 시간 규칙과 forward fill 비교는 [기존 시계열 안내](temporal_sensor_experiments.md)를 참고하세요.

새 전처리 상태는 v4로 저장하며 scaling 방식, 피처 순서, Train에서 학습한 대체값·중심·스케일을 포함합니다. 기존 전처리 상태도 읽을 수 있게 유지합니다. 추론에서는 설정 파일을 임의로 바꾸는 대신 체크포인트에 저장된 전처리를 복원합니다. 피처나 scaling을 바꾼 후보는 별도로 학습해야 합니다.

코드에서 `prep.centers`는 선택한 방식의 중심값입니다. 저장 상태의 `means` 키는 기존 형식과의 호환을 위한 이름으로, robust에서는 평균 대신 중앙값을 담습니다. 직접 공식을 재구현하기보다 `SensorPreprocessor.from_dict(state).transform(frame)`을 사용합니다.

## 서버에서 실행하기

변경이 commit·push된 뒤, 사용자가 본인 서버 clone의 `feat/seungbeom-sensor` 브랜치에 pull한 코드로 실행합니다. 실행 중인 clone은 갱신하지 않습니다. 아래 `seungbeom` 경로는 승범의 예시이며 다른 팀원은 본인 폴더를 사용합니다. 공용 데이터나 환경을 복제하지 않습니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/venv/bin/activate
```

GPU 사용을 조율한 뒤 tmux에서 한 실험씩 완료합니다. 모든 `--run_dir`는 새 디렉터리를 사용합니다. 이미 존재하면 덮어쓰지 않고 새 이름을 정한 뒤 비교 명령에도 같은 이름을 반영합니다. 같은 코드·학습 조건의 기준 실행이 없다면 먼저 실행합니다.

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_raw.json \
  --run_dir runs/seungbeom/sensor_raw_seed42
```

핵심 이력 비교는 다음 세 실험입니다. 각각 앞 실험이 완료된 뒤 실행합니다.

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_history_voltage.json \
  --run_dir runs/seungbeom/sensor_history_voltage_seed42
```

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_history_current.json \
  --run_dir runs/seungbeom/sensor_history_current_seed42
```

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_history_pressure.json \
  --run_dir runs/seungbeom/sensor_history_pressure_seed42
```

```bash
python scripts/compare_runs.py \
  runs/seungbeom/sensor_raw_seed42 \
  runs/seungbeom/sensor_history_voltage_seed42 \
  runs/seungbeom/sensor_history_current_seed42 \
  runs/seungbeom/sensor_history_pressure_seed42
```

나머지 후보도 실행하려면 아래 목록에서 선택해 순차 실행합니다. 기존 폴더나 학습 오류가 있으면 반복을 중단합니다.

```bash
sensor_candidates=(
  sensor_gas
  sensor_ambient
  sensor_thermal
  sensor_thermal_ambient
  sensor_raw_robust
  sensor_gas_robust
  sensor_no_current
  sensor_missing
  sensor_gas_missing
)
for sensor_config in "${sensor_candidates[@]}"; do
  sensor_run_dir="runs/seungbeom/${sensor_config}_seed42"
  if [ -e "$sensor_run_dir" ]; then
    printf 'Existing run directory: %s\n' "$sensor_run_dir"
    break
  fi
  python -u train.py \
    --data_dir /home/ubuntu/data/battery_data_light \
    --config "configs/${sensor_config}.json" \
    --run_dir "$sensor_run_dir" || break
done
```

완료된 후보만 지정해 서버에서 비교합니다. 첫 그룹은 추가 센서 효과, 둘째는 scaling 효과, 셋째는 결측 플래그·상수 전류 대조 실험입니다.

```bash
python scripts/compare_runs.py \
  runs/seungbeom/sensor_raw_seed42 \
  runs/seungbeom/sensor_gas_seed42 \
  runs/seungbeom/sensor_ambient_seed42 \
  runs/seungbeom/sensor_thermal_seed42 \
  runs/seungbeom/sensor_thermal_ambient_seed42
```

```bash
python scripts/compare_runs.py \
  runs/seungbeom/sensor_raw_seed42 \
  runs/seungbeom/sensor_raw_robust_seed42 \
  runs/seungbeom/sensor_gas_seed42 \
  runs/seungbeom/sensor_gas_robust_seed42
```

```bash
python scripts/compare_runs.py \
  runs/seungbeom/sensor_raw_seed42 \
  runs/seungbeom/sensor_no_current_seed42 \
  runs/seungbeom/sensor_missing_seed42 \
  runs/seungbeom/sensor_gas_seed42 \
  runs/seungbeom/sensor_gas_missing_seed42
```

## 선택과 전달

먼저 같은 Train·Validation의 Macro F1을 비교하고, 클래스별 F1 저하를 함께 확인합니다. 개선 후보와 기준을 seed 43·44에서도 같은 조건으로 다시 학습해 평균과 변동을 살핍니다. 추가 seed에는 해당 설정의 `seed`를 바꾼 별도 설정 파일과 새 run 디렉터리를 사용합니다. Validation을 여러 번 확인해 고른 결과는 최종 Test 성능이 아닙니다.

선택한 피처 순서, 전처리 방식·버전, 학습 조건과 checkpoint는 팀 내부 서버에서 결합 담당에게 전달합니다. 실제 데이터·분포·평가 결과·가중치를 이 문서나 공개 Git에 붙이지 않습니다. 에이전트에게 실제 집계 결과를 제공할 필요도 없습니다. 최종 결합 모델에서는 이미지와 센서를 함께 사용하고, 센서 단독 후보의 효과를 결합 Validation에서 다시 확인합니다.

로컬 검증은 독립적으로 만든 가상 CSV·이미지로 피처 공식, Train-only 통계, 전처리 저장·복원, 인과성 및 기존 동작 호환을 확인하는 범위입니다. 테스트 통과는 실제 데이터의 품질이나 성능을 입증하지 않습니다. 실행 여부·검증 개수·실제 점수는 확인 없이 기록하지 않습니다.

2026-11-14 구현 검증에서는 기존 파이프라인·센서·비교·EDA·시계열과 새 scaling·확장 피처의 가상 CPU 테스트 81개 및 명시한 코드의 Ruff 검사를 통과했습니다. 실제 대회 데이터 학습·평가는 실행하지 않았습니다.
