# 센서 파생 피처와 단독 성능 비교

이 문서는 초기 중앙값·현재 행 피처 실험의 설명입니다. 후속 직전값 대체와 전압·전류·압력 시계열 실험은 [시계열 센서 실험 안내](temporal_sensor_experiments.md)를 따릅니다. 기존 설정은 재현을 위해 유지하고, 새 설정은 대체 방식과 피처를 명시적으로 선택합니다.

승범의 담당 범위는 센서 전처리·파생 피처 구현, 센서 단독 기준 성능 측정, 결합 담당에게 입력 규격과 비교 결과 전달입니다. 코드 수정·가상 입력 CPU 검증은 로컬에서 하고, 실제 데이터 확인·학습·평가는 사용자가 주최 측 서버에서 실행합니다.

다른 세션에 이어서 맡길 때는 [핸드오프 문서](SESSION_HANDOFF.md), 사용자가 서버에서 분포를 확인할 때는 [센서 분포 안내](sensor_eda.md)를 따릅니다.

## 원본 컬럼과 모델 입력 피처

- `schema.py`의 `RAW_SENSOR_COLUMNS`: CSV에 필요한 원본 센서 5개입니다. 파생 피처를 이 목록에 추가하지 않습니다. 기존 `SENSOR_COLUMNS`는 같은 목록의 호환용 이름입니다.
- `sensor_features.py`: 현재 행만으로 계산하는 피처와 허용 이름을 정의합니다. `build_sensor_features(frame, feature_names)`는 지정된 순서의 표를 반환하며 어떤 통계도 학습하지 않습니다.
- `configs/*.json`의 `sensor_features`: 실험에서 사용할 모델 입력 목록입니다. 이 순서가 학습·추론에서 유지됩니다. 등록되지 않은 이름, 정답·식별자·시간 컬럼, 중복 이름은 거부합니다.
- `preprocessing.py`: 선택한 각 피처의 결측 대체·정규화를 Train에서 학습합니다. `preprocessor.input_dim`으로 모델 입력 차원을 정합니다.
- `model.py`: `mode`에 따라 이미지·센서 결합 또는 센서 단독 MLP를 실행합니다. 이미지 담당의 인코더는 기존 구조를 유지합니다.

현재 등록된 파생 피처는 공식 온도 컬럼 정의로부터 만든 실험 후보입니다. 실제 데이터의 통계값을 근거로 임계값을 정하거나 센서를 제거하지 않았습니다.

| 이름 | 계산 | 의미 |
| --- | --- | --- |
| `temp_mean` | `(meas_t1 + meas_t2) / 2` | 두 측정 위치 온도의 평균 |
| `temp_diff` | `meas_t2 - meas_t1` | 양극부와 표면의 온도 차이 |
| `temp_abs_diff` | `abs(meas_t2 - meas_t1)` | 두 온도의 차이 크기 |

파생 피처 추가는 성능 개선을 보장하지 않습니다. 현재 행의 계산만 사용하므로 과거 행·다른 실험·Test 전체 통계가 필요하지 않습니다.

## 전처리와 저장 계약

1. 원본 센서를 숫자로 변환합니다. 변환 실패·무한대는 결측으로 취급합니다.
2. 결측 대체 전에 파생 피처를 계산합니다. 두 온도 중 하나라도 없으면 관련 파생 피처도 결측입니다.
3. 선택한 피처마다 Train의 중앙값으로 결측을 대체합니다. Train에서 전부 결측인 피처의 대체값은 0입니다.
4. 대체한 Train의 평균·표준편차로 정규화합니다. 표준편차가 `1e-8` 미만이면 스케일을 1로 둡니다. 0이나 큰 값을 자동 삭제·클리핑하지 않습니다.
5. 피처 이름·순서·정의 버전, 원본 컬럼 목록, 중앙값·평균·스케일을 체크포인트에 저장합니다. 추론은 현재 설정 파일이 아닌 체크포인트의 목록과 통계를 사용합니다.

기존 v1의 원본 센서 5개 가중치도 읽을 수 있습니다. 새 가중치는 v2로 저장합니다. 이미 등록된 피처의 공식이나 결측 처리 의미를 바꾸면 `SENSOR_FEATURE_VERSION`을 올리고 해당 피처를 사용한 모델을 다시 학습해야 합니다. 이전 버전은 이전 코드와 함께 보관하세요.

새 파생 피처를 추가할 때는 `sensor_features.py`에 계산과 설명을 등록하고, 가상 입력으로 공식·결측·행 순서·부분 입력 검사를 추가한 뒤 설정 목록에 넣습니다. 원본 CSV를 수정하거나 센서 개수를 `model.py`에 직접 고정할 필요는 없습니다.

## 비교 설정

| 설정 | 입력 | 목적 |
| --- | --- | --- |
| `configs/sensor_raw.json` | 원본 센서 5개 | 센서 단독 기준 성능 |
| `configs/sensor_temperature.json` | 센서 5개 + 온도 파생 3개 | 센서 파생 피처 효과 |
| `configs/baseline.json` | 이미지 + 원본 센서 5개 | 기존 멀티모달 기준 |
| `configs/fusion_temperature.json` | 이미지 + 센서 피처 8개 | 결합 모델에서의 파생 피처 효과 |

모두 같은 seed·학습 횟수·학습률·배치 크기로 시작합니다. 제공 Train으로 학습하고 제공 Validation의 6클래스 Macro F1이 가장 높은 epoch를 선택합니다. 센서 단독 모드는 이미지 파일을 열거나 이미지 인코더를 생성하지 않습니다. `image_path` 컬럼도 센서 단독 실행에는 필수가 아닙니다.

센서 단독 모델은 분석용 비교 실험입니다. 대회 최종 모델은 이미지와 센서를 모두 사용해야 합니다. 센서 단독 실험에서는 `--model_out weights/model.pt`를 쓰지 않고 각 실험의 `best.pt`를 유지하세요.

## 서버 실행 순서

아래는 **로컬 변경을 `feat/seungbeom-sensor` 브랜치에 commit·push한 뒤**의 명령입니다. 문서와 코드가 아직 로컬에만 있다면 먼저 코드 전달을 완료해야 합니다. 서버 레포에 미커밋 변경이나 실행 중인 작업이 없는지 확인하고, 본인 폴더에서 받습니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git fetch origin
git switch feat/seungbeom-sensor
git pull --ff-only origin feat/seungbeom-sensor
source /home/ubuntu/venv/bin/activate
bash scripts/setup_env.sh --server
```

공용 GPU 사용을 팀원과 조율하고 [서버 실행 안내](server_quickstart.md)의 tmux 방법으로 실행합니다. 두 실험은 순서대로 실행하며, 새 tmux 화면에서는 본인 레포로 이동하고 가상환경을 다시 활성화합니다.

학습 전에 [센서 분포·시간축 변환 안내](sensor_eda.md)에 따라 `scripts.sensor_timeline`으로 한 실험의 전처리 전후를 볼 수 있습니다. 이 그래프는 현재 행 피처가 시간에 따라 어떻게 변하는지 확인하는 용도이며, 시계열 모델이나 과거 관측 입력을 새로 만드는 것은 아닙니다. 그래프용 개인 환경을 썼다면 학습 전 `source /home/ubuntu/venv/bin/activate`로 공용 학습 환경에 돌아옵니다.

첫 번째: 원본 센서만 사용합니다.

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_raw.json \
  --run_dir runs/seungbeom/sensor_raw_seed42
```

첫 번째가 완료된 뒤 두 번째: 온도 파생 피처를 추가합니다.

```bash
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/sensor_temperature.json \
  --run_dir runs/seungbeom/sensor_temperature_seed42
```

이미 존재하는 실험 폴더는 덮어쓰지 않습니다. 재실행에는 새 실험명을 사용하세요. 각 폴더의 `best_metrics.json`에 Macro F1·클래스별 F1·support·혼동행렬이 저장됩니다. 두 학습이 끝나면 서버에서 비교합니다.

```bash
python scripts/compare_runs.py \
  runs/seungbeom/sensor_raw_seed42 \
  runs/seungbeom/sensor_temperature_seed42
```

비교 도구는 Train·Validation CSV 해시가 다른 실험을 거부합니다. seed·최적 epoch·Macro F1·클래스별 F1을 출력합니다. 나머지 학습 설정도 각 `config.json`으로 확인하세요. 작은 점수 차이는 추가 seed 실험으로 확인하고, 특정 클래스의 저하가 있는지도 함께 살핍니다. 여기서 얻는 점수는 모델 선택에 사용한 Validation 점수이며 최종 Test 성능을 보장하지 않습니다.

결합 담당은 동일한 분할에서 `configs/baseline.json`과 `configs/fusion_temperature.json`을 각각 새 실험 폴더로 실행하여 파생 피처가 이미지와 함께 있을 때도 유효한지 비교합니다. 학습·실험 선택에 Test를 사용하지 않습니다.

## 공유할 내용의 구분

- AI와 작업할 수 있는 내용: 코드, 공식 컬럼 정의, 가상 입력과 그 결과, 데이터 값·식별자·경로·인증 정보가 없는 오류 메시지, Python·패키지·GPU 환경 정보.
- 외부 AI에 보내지 않을 내용: 실제 데이터 행·정답·이미지 및 이를 포함한 파일·화면·로그. 실제 데이터 파일은 서버에 둡니다.
- 결측률·분포·평가 점수 같은 집계 결과: 개별 행이 없다는 사실만으로 허용을 단정하지 않습니다. 현재 제공된 규정 외에 운영진의 별도 허용 안내는 확인되지 않았으므로 서버에서 직접 확인합니다. 터미널에서 출력했다는 사실은 공유 허용 여부를 바꾸지 않습니다.

구현을 위해 실제 통계값을 AI에 전달할 필요는 없습니다. 실제 결과는 팀 내부에서 비교하고, 결합 담당에게 선택한 피처 순서·설정·체크포인트 경로·검증 방법을 전달합니다.
