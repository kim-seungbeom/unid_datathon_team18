# 직전값 대체와 전압·전류·압력 시계열 실험

센서별 과거 피처 분리와 가스·주변 온도·열화상 수치·결측 표시·Robust scaling의 후속 설정은 [센서별 분리 실험 안내](sensor_ablation_experiments.md)를 참고하세요. 기존 설정은 같은 전처리를 유지하며, 최신 저장 형식은 scaling을 함께 기록하는 v4입니다.

사용자가 제공한 2026-11-14 OT 규정은 같은 실험의 현재·이전 센서와 이미지를 이용한 시간 윈도우 피처를 허용합니다. 미래 관측, 다른 실험, 실험 전체 평균·최댓값·길이는 사용할 수 없습니다. 실제 데이터 학습은 사용자가 제공 GPU 서버에서 실행합니다.

## 직전값 대체

설정의 `sensor_imputation`을 `forward_fill`로 지정합니다. 기존 설정에서 이 키를 생략하면 `median`을 유지하므로 이전 실험과 가중치를 재현할 수 있습니다.

1. 원본 센서를 숫자로 바꾸고 변환 실패·무한대를 결측으로 처리합니다.
2. 입력으로 받은 행을 실험별로 묶고 `relative_time_s`로 정렬합니다.
3. 원본 센서별로 현재 값이 없으면 같은 실험에서 가장 최근에 관측된 과거 유효값을 사용합니다. 연속 결측도 직전 유효값을 이어받습니다. 현재 구현에는 최대 경과 시간 제한이 없습니다.
4. 온도 파생 피처를 선택했다면 이 원본 센서 대체 후 계산합니다. 새 기본 비교 설정에는 온도 파생 피처를 넣지 않습니다.
5. 이전 관측이 없거나 시간 피처 계산에 필요한 관측이 없으면 해당 피처의 Train 중앙값으로 대체합니다. 이 중앙값은 forward fill로 복제된 값이 아닌, 대체 전 Train 피처의 유효값에서 계산합니다. 전부 결측이면 0입니다.
6. 이 과정을 거친 Train의 평균·표준편차로 표준화합니다. Validation·Test에는 저장한 기준을 적용합니다. 입력·출력 행 순서는 원래 순서로 유지합니다.

미래 값 대체, 보간, 실험 간 값 전달은 하지 않습니다. Train의 마지막 관측을 Validation이나 Test에 전달하지도 않습니다. 추론 호출 사이에 과거 값을 캐시하지 않고, 해당 입력 CSV의 행만 사용합니다.

시간은 유한한 0 이상의 초 단위 값이어야 합니다. 같은 실험에서 시간이 중복되면 같은 시각 관측의 선후 관계를 임의로 정하지 않고 오류를 냅니다. 이 경우 실제 값을 대화에 올리지 말고 공식 관측 규칙을 확인해야 합니다.

결측이 없는 원본 센서 입력은 대체 방식을 바꿔도 값이 그대로입니다. 직전값 대체 자체가 성능 개선을 보장하지 않습니다.

## 온도 외 추가 피처

원본 센서 5개에 아래 6개를 더해 입력이 11개가 됩니다. 피처마다 전압·전류·압력의 원래 단위를 유지합니다.

| 이름 | 계산 |
| --- | --- |
| `meas_v_mean30s` | 현재 시각 t에서 `(t-30, t]`에 있는 실제 유효 전압 관측의 평균 |
| `meas_v_delta30s` | 현재 전압 − `t-30` 이하의 가장 최근 유효 전압 관측 |
| `meas_a_mean30s` | 같은 방식의 전류 평균 |
| `meas_a_delta30s` | 같은 방식의 전류 변화량 |
| `meas_p_mean30s` | 같은 방식의 압력 평균 |
| `meas_p_delta30s` | 같은 방식의 압력 변화량 |

윈도우는 행 개수가 아닌 실제 초 기준입니다. 평균은 원본의 유효 관측만 사용하며, 직전값으로 채운 결측 지점을 새로운 실제 측정값처럼 평균에 추가하지 않습니다. 변화량의 현재 값에는 선택한 대체 방식을 적용합니다. 기준 시각과 정확히 일치하는 관측이 없어도 그 이전의 가장 최근 유효값을 사용하므로 엄밀한 초당 변화율은 아닙니다. 과거 관측이 없으면 결측으로 남긴 뒤 Train 중앙값을 적용합니다.

## 효과를 분리하는 비교 설정

| 설정 | 결측 대체 | 피처 수 | 비교 목적 |
| --- | --- | --- | --- |
| `configs/sensor_raw.json` | 중앙값 | 5 | 기존 기준 모델 |
| `configs/sensor_ffill_raw.json` | 직전값 + 중앙값 fallback | 5 | 대체 방식만의 효과 |
| `configs/sensor_history_median.json` | 중앙값 | 11 | 과거 피처만의 효과 |
| `configs/sensor_ffill_history.json` | 직전값 + 중앙값 fallback | 11 | 두 변경의 결합 효과 |

같은 Train·Validation, MLP 구조, seed, epoch 수, 학습률로 비교합니다. 점수가 가장 좋은 후보는 추가 seed에서도 확인하고, 클래스별 F1 저하도 함께 확인합니다. Validation에서 선택한 결과를 최종 Test 성능으로 해석하지 않습니다. 최적 설정은 실제 서버 결과로 판단하며 코드 구현만으로 확정할 수 없습니다.

## 서버에서 실행하기

이 변경이 commit·push되고 본인 서버 레포에서 같은 브랜치를 pull한 다음 실행합니다. 사용자가 별도로 요청하기 전에는 `main`에 merge하지 않습니다. 기존 코드가 실행 중이면 완료될 때까지 pull하지 않습니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/venv/bin/activate
```

GPU 사용을 조율하고 tmux 안에서 아래 실험을 하나씩 완료합니다. 같은 이름의 실험 폴더가 있으면 새 이름을 지정합니다.

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_ffill_raw.json \
  --run_dir runs/seungbeom/sensor_ffill_raw_seed42
```

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_history_median.json \
  --run_dir runs/seungbeom/sensor_history_median_seed42
```

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_ffill_history.json \
  --run_dir runs/seungbeom/sensor_ffill_history_seed42
```

기존 기준 실험까지 함께 비교합니다. 실제 사용한 실험 폴더명으로 지정하세요.

```bash
python scripts/compare_runs.py \
  runs/seungbeom/sensor_raw_seed42 \
  runs/seungbeom/sensor_ffill_raw_seed42 \
  runs/seungbeom/sensor_history_median_seed42 \
  runs/seungbeom/sensor_ffill_history_seed42
```

## 선택과 팀원 전달

첫 비교에서는 seed 42의 Macro F1과 클래스별 F1을 확인합니다. 개선된 후보는 전압·전류·압력 피처를 센서별로 빼 보는 실험으로 어떤 항목이 기여했는지 확인합니다. 설정의 `sensor_features` 목록만 바꾸면 되며, 나머지 학습 조건은 유지합니다. 상수인 센서의 평균·변화량은 새로운 정보를 제공하지 않으므로 무조건 남기지 않습니다. 후보와 기존 기준 모델을 추가 seed 43·44에서도 비교한 뒤 평균과 변동을 보고 선택합니다. 피처 수가 많다는 이유로 더 좋은 설정으로 판단하지 않습니다.

선택된 실험 폴더에서 아래 파일을 팀 내부 서버 작업 공간으로 전달합니다. 값이 들어 있는 결과 파일은 공개 Git에 커밋하지 않습니다.

| 파일 | 전달 내용 |
| --- | --- |
| `config.json` | 피처 이름·순서, 대체 방식, seed와 학습 조건 |
| `preprocessing.json` | Train에서 학습한 fallback 중앙값·평균·스케일·버전 |
| `best_metrics.json` | 선택 epoch, Validation Macro F1·클래스별 성능 |
| `split.json`, `environment.json` | 학습·검증 구성과 실행 환경 |
| `best.pt` | 선택된 센서 모델과 동일한 전처리 상태 |

코드는 이 브랜치의 `sensor_features.py`, `temporal.py`, `preprocessing.py`를 함께 사용합니다. 통합 담당은 저장된 전처리로 `SensorPreprocessor.from_dict(state)`를 만들고, CSV 전체의 센서를 `prep.transform(frame)`으로 변환합니다. 모델 입력 차원은 `prep.input_dim`입니다. 배치나 한 행으로 자르기 전에 과거 피처를 계산해야 합니다.

이미지 결합 실험은 선택한 설정을 복사하여 `mode`를 `multimodal`로 바꾸고 별도의 실험으로 학습합니다. 센서 단독에서 가장 좋았던 조합도 이미지와 결합하면 순위가 달라질 수 있으므로 결합 Validation으로 다시 확인합니다. 센서 단독 가중치가 그대로 최종 결합 모델이 되는 것은 아니며, 최종 모델은 이미지와 센서를 모두 사용해야 합니다.

## 추론과 검증의 의미

시계열 기능 도입 당시 체크포인트는 v3였으며 피처 목록, 대체 방식, 과거 계산 버전, Train fallback 중앙값·평균·스케일을 포함합니다. 최신 v4는 scaling 방식도 기록합니다. 기존 v1·v2는 중앙값·표준화 방식, v3는 저장된 결측 대체 방식과 표준화로 읽습니다. `temporal.py`도 제출 코드에 포함해야 합니다. 알고리즘을 바꾸면 `TEMPORAL_VERSION`을 올리고 새로 학습합니다.

직전값 대체와 과거 피처는 필요한 과거 행이 입력에서 빠지면 값이 달라질 수 있습니다. `check_submission.py`는 반복·순서 변경·실험 분리·미래 제거·임의 부분 입력 일관성을 계속 검사하며, 기본 실행은 어느 항목이든 달라지면 오류로 종료합니다. 연구용 `--diagnose_history_subset`을 명시하면 과거 기반 체크포인트의 임의 부분 입력 차이만 `DIAGNOSTIC`과 미확인 상태로 표시합니다. 이 옵션은 제출 적합 판정을 뜻하지 않습니다. 과거 이력 허용과 부분 입력 검사의 상세 관계는 운영진 확인 대상이며, 최종 제출 검사는 기본 엄격 모드로 실행합니다.

현재의 분포·시간축 보고서 기본 명령은 중앙값 또는 대체 전 피처를 보여주는 도구입니다. forward-fill 학습 결과를 확인하려면 해당 실험의 저장된 전처리를 복원하고 `prep.impute(frame)` / `prep.transform(frame)`을 사용해야 합니다. 과거가 필요한 전처리에 `frame.head()`만 먼저 전달하면 전체 이력을 제공한 결과와 다를 수 있으므로, 전체 실험 행을 변환한 뒤 표시할 행을 선택합니다.
