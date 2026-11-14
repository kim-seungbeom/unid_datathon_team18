# Uni-D 데이터톤 Team 18

열화상 이미지와 센서 정보를 함께 사용하여 현재 관측의 배터리 열폭주 단계를 `1..6`으로 분류하는 초기 구현입니다. 공식 베이스라인 v7의 입력·출력 계약을 참고하고 이미지 CNN + 센서 MLP 결합 모델로 구성했습니다. 실제 대회 데이터 학습이나 성능 검증은 아직 수행하지 않았습니다.

기본 작업 방식은 **맥 로컬 VS Code에서 코드 수정·가상 입력 검증 → GitHub에 commit·push → 서버에서 같은 브랜치를 clone/pull → 사용자가 학습·검증·추론 실행**입니다. 코드 변경은 로컬에서 하고 서버의 개인 레포는 실행용 사본으로 사용합니다. 실행 중 발견한 문제도 로컬 코드에서 수정해 같은 흐름으로 전달합니다.

## 데이터와 AI 작업 경계

- 대회 데이터가 들어 있는 `open/`은 Git에서 제외합니다. 원본 데이터·압축 파일·베이스라인 원본은 수정하지 않습니다.
- `.gitignore`와 `AGENTS.md`는 파일 접근을 기술적으로 차단하지 않습니다. AI를 사용하는 작업 공간에는 코드만 두고, 실제 데이터는 서버의 공용 경로에서 사용하세요. 현재 `open/`의 데이터는 AI가 읽지 않는 대상으로 취급합니다.
- AI가 확인한 것은 사용자가 지정한 베이스라인 `.py`·README·requirements와 공식 설명 문서입니다. 실제 CSV·이미지·노트북 출력·학습 로그는 확인하지 않았습니다.
- 맥에서는 임의로 생성한 가상 입력의 CPU 검증만 합니다. 대회 데이터 학습·추론은 사용자가 주최 측 서버에서 직접 실행합니다. 결과 로그·예측·가중치도 AI 대화에 올리지 마세요.
- 터미널 출력은 내용으로 구분합니다. 데이터가 없는 코드 오류·패키지 버전·GPU 환경 정보와 가상 입력 결과는 공유할 수 있습니다. 실제 데이터의 분포·결측률 등 집계 결과 공유는 운영진의 별도 허용이 확인되지 않았으므로 서버에서 직접 확인합니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `model.py` | 이미지 CNN + 센서 MLP + 6개 클래스 분류기 |
| `train.py` | Train 학습, Validation 모델 선택, 가중치·전처리 저장 |
| `test.py` | 저장된 모델로 입력 행만 예측하여 제출 CSV 생성 |
| `schema.py`, `dataset.py` | 컬럼 검사, split 기준 이미지 로딩, 라벨 변환 |
| `preprocessing.py` | Train에서만 학습하는 센서 결측 대체·정규화 |
| `sensor_features.py` | 원본 컬럼과 분리된 현재·과거 센서 피처 생성·이름 검증 |
| `temporal.py` | 실험별 시간 정렬, 과거 유효값 대체, 초 기준 평균·변화량 |
| `evaluate.py`, `runtime.py` | 6클래스 평가, 재현 설정, 메타데이터 기록 |
| `configs/baseline.json` | 모델·학습 설정 |
| `configs/sensor_*.json`, `configs/fusion_temperature.json` | 센서 단독 및 온도 파생 피처 비교 설정 |
| `check_submission.py` | 팀 자체 제출 재현·부분 입력 검사 도구 |
| `scripts/` | 로컬 환경 설치, 서버 환경 확인, staged 파일 검사 |
| `tests/` | 대회 데이터와 무관한 가상 입력 검증 |
| `CONTRIBUTING.md` | Git·코드·실험 컨벤션 |
| `baselines/organizer_v7/` | 비교용 운영진 원본 코드 5개. 데이터·노트북·실행 결과 제외 |
| `docs/rules_review.md` | 공식 안내 대조 결과와 아직 확인하지 않은 항목 |
| `docs/server_quickstart.md` | `main` 브랜치를 개인 서버 폴더에 받아 학습·추론하는 순서 |
| `docs/sensor_experiments.md` | 센서 담당의 피처 확장·단독 실험·서버 비교 순서 |
| `docs/temporal_sensor_experiments.md` | 직전값 대체와 전압·전류·압력 과거 피처의 분리 실험 |
| `scripts/compare_runs.py` | 동일 Train·Validation을 사용한 실험의 검증 지표 비교 |
| `scripts/sensor_eda.py`, `docs/sensor_eda.md` | 사용자가 서버에서 확인하는 센서 분포 요약·그래프 |
| `docs/SESSION_HANDOFF.md` | 다른 세션에 전달할 파일 구조·서버 상태·진행 단계 |

## 시작 순서: 주최 측 GPU 서버 가이드 기준

서버 초기 준비는 [주최 측 가이드의 3-1절](https://hkksi.notion.site/GPU-414b516e192380c9adb4df7d7b4d68b9)을 따릅니다. **데이터를 맥에 다운로드해 서버로 다시 업로드할 필요가 없다는 뜻입니다.** 팀 대표나 팀원 한 명이 공식 `setup.sh`를 실행하면 서버가 데이터를 직접 내려받아 공용 경로에 설치합니다. 이미 완료했다면 다른 팀원은 같은 데이터를 바로 사용합니다. SSH 접속 성공 자체가 데이터 설치 완료를 뜻하지는 않습니다.

| 파일/명령 | 역할 | 실행 위치 |
| --- | --- | --- |
| 운영진이 배포한 `setup.sh` | 공용 Python·데이터·베이스라인 설치, 팀에서 한 명이 진행 | GPU 서버 |
| 우리 `bash scripts/setup_env.sh --server` | 설치된 공용 경로·패키지·CUDA 확인만 수행 | GPU 서버의 코드 저장소 |
| 우리 `bash scripts/setup_env.sh --local` | 가상 입력 검증용 `.venv` 설치. 서버 사용을 위한 필수 단계는 아님 | 맥 로컬 코드 저장소 |

### 1. 팀 공용 세팅 완료 여부 확인

이미 연결된 **VS Code SSH 창의 터미널**에서 아래 명령으로 정해진 경로의 존재 여부만 확인할 수 있습니다. 실제 파일 내용은 출력하지 않습니다.

```bash
test -x /home/ubuntu/venv/bin/python && echo "공용 Python: 있음" || echo "공용 Python: 확인 필요"
test -d /home/ubuntu/data/battery_data_light && echo "공용 데이터 폴더: 있음" || echo "공용 데이터 폴더: 확인 필요"
test -d /home/ubuntu/data/baseline && echo "베이스라인 배포 폴더: 있음" || echo "베이스라인 배포 폴더: 없음 (팀 모델 실행에는 불필요)"
```

폴더가 있어도 압축 해제와 설치가 끝났는지는 별개입니다. 팀에서 공식 `setup.sh`를 실행한 사람에게 완료 여부를 확인하세요. 작업 중이면 중복 실행하지 않습니다.

아직 아무도 세팅하지 않은 경우에만 다음 순서로 진행합니다.

1. **맥:** 운영진 드라이브에서 `setup.sh`만 받습니다. 데이터 압축 파일과 베이스라인 압축 파일은 각각 **파일 공유 링크**를 복사해 스크립트의 `DATA_URL`, `BASELINE_URL`에 넣습니다. 링크·비밀번호·토큰은 레포나 대화에 저장하지 않습니다.
2. **맥 → 서버:** 수정한 공식 스크립트만 VS Code SSH 탐색기의 본인 폴더로 옮깁니다. 기존 같은 이름 파일을 덮어쓰지 마세요.
3. **서버:** 그 스크립트가 있는 폴더에서 `bash setup.sh`를 실행합니다. 공용 환경과 데이터는 서버가 직접 설치합니다.
4. **서버:** 설치 완료 후 `source /home/ubuntu/venv/bin/activate`로 공용 환경을 사용합니다. 이후 새 터미널마다 활성화합니다.

이미 맥의 `open/`에 둔 데이터는 이 서버 작업 흐름에 필요하지 않습니다. 서버 데이터는 `/home/ubuntu/data/battery_data_light`를 팀원이 함께 참조하고, 각자 폴더에 복제하지 않습니다.

### 2. 팀원마다 개인 코드 폴더 준비

서버에서는 **각자 `/home/ubuntu/<본인이름>/`을 만들고 그 아래에 팀 레포를 clone**합니다. GitHub 저장소는 함께 사용하되 서버의 실행용 사본은 개인별로 둡니다. `seungbeom`은 승범의 폴더명 예시이며, 모든 팀원이 이 폴더에 들어갈 필요는 없습니다.

```text
/home/ubuntu/
├── venv/                              # 팀 공용 Python 환경
├── data/                              # 팀 공용 데이터·운영진 원본
├── seungbeom/unid_datathon_team18/     # 승범의 실행용 레포·실험 결과
├── member_a/unid_datathon_team18/      # 팀원 A: 실제 이름으로 변경
└── member_b/unid_datathon_team18/      # 팀원 B: 실제 이름으로 변경
```

코드 편집은 맥 로컬 VS Code에서 진행합니다. 변경한 코드를 GitHub에 push한 뒤 **본인 서버 레포의 같은 작업 브랜치**에서 pull하고 실행합니다. 로컬 파일과 SSH 서버 파일은 자동 동기화되지 않습니다. 기존 서버 레포에 미커밋 변경이 있다면 먼저 보존·정리하며, Git 인증은 사용자가 직접 처리합니다. 데이터와 가상환경은 개인 폴더마다 다시 설치하지 않습니다.

이 팀 레포를 사용할 때는 베이스라인을 레포 위에 복사하지 않습니다. 원본 베이스라인을 따로 실행하고 싶을 때만 가이드 3-2절처럼 아직 존재하지 않는 개인 경로로 복사합니다. 공용 베이스라인 원본은 수정하지 않습니다.

초기 세팅과 운영진 베이스라인 소스는 공통 브랜치 `main`에서 받습니다. `initial-setup`은 초기 작업에 사용한 브랜치입니다. 서버 초기 설치를 완료했다면 [서버 실행 안내](docs/server_quickstart.md)에 따라 `main`을 clone하고, 환경 확인 → 학습 → 제출 검사를 진행하세요. 이미 `initial-setup`을 받은 팀원은 같은 안내의 5절에 따라 본인 레포를 `main`으로 전환합니다.

## 선택: 맥 로컬 코드 검증 환경

**맥 로컬 터미널, 저장소 루트에서 실행합니다.** Python 3.12를 기준으로 프로젝트 전용 `.venv`를 사용합니다. 시스템 Python이나 서버 환경을 변경하지 않습니다.

```bash
bash scripts/setup_env.sh --local
source .venv/bin/activate
python -m pytest tests
```

다른 Python 실행 파일을 쓰려면 `PROJECT_PYTHON=/실행파일/경로 bash scripts/setup_env.sh --local`로 지정합니다. 기존 `.venv`가 있으면 재사용합니다.

코드 스타일 검사도 데이터 폴더를 탐색하지 않도록 코드 경로를 명시합니다.

```bash
python -m ruff check schema.py sensor_features.py temporal.py dataset.py preprocessing.py model.py evaluate.py runtime.py train.py test.py check_submission.py scripts tests
python -m ruff format --check schema.py sensor_features.py temporal.py dataset.py preprocessing.py model.py evaluate.py runtime.py train.py test.py check_submission.py scripts tests
```

## GPU 서버 환경 확인

**이후 실제 데이터 관련 명령은 사용자가 SSH 서버 터미널에서 실행합니다.** 팀원이 주최 측 `setup.sh`를 실행했는지 먼저 확인하세요. 이 저장소의 `scripts/setup_env.sh`는 주최 측의 데이터 다운로드 스크립트와 별개입니다.

서버 개인 폴더의 코드 저장소로 이동한 다음:

```bash
source /home/ubuntu/venv/bin/activate
bash scripts/setup_env.sh --server
nvidia-smi
```

`--server`는 각 split의 CSV·이미지 폴더, 패키지·CUDA를 확인만 합니다. 운영진 베이스라인 배포 폴더는 참고용으로 확인하며 없어도 팀 모델의 환경 검사를 중단하지 않습니다. 압축 내부의 특정 폴더 이름은 가정하지 않습니다. 파일 내용을 읽거나 데이터 다운로드·설치·업그레이드를 수행하지 않습니다. 경로 존재 검사는 데이터 전체의 완전성을 보장하지 않습니다. 공용 환경에 NumPy·pandas·Pillow가 없을 때만 팀과 합의한 뒤 `python -m pip install -r requirements-common.txt`를 실행하세요. 이미 설치된 GPU용 PyTorch는 교체하지 않습니다. 제출 가이드에 맞춰 `requirements.txt`의 `torch`에는 버전을 붙이지 않고, 실행 시 실제 버전을 별도로 기록합니다. 코드의 최소 환경은 Python 3.10, PyTorch 2.7입니다.

주최 측 공용 환경은 `/home/ubuntu/venv`, 공식 데이터 기준 경로는 `/home/ubuntu/data/battery_data_light`입니다. 실제 설치 여부는 서버에서 확인해야 합니다. `sudo apt upgrade`는 실행하지 않습니다.

## 학습

서버에서 GPU 사용을 팀원과 조율한 뒤, 먼저 이 한 줄만 실행합니다. **이하 학습·추론 명령은 승범의 예시이므로 다른 팀원은 `seungbeom`을 본인 폴더명으로 바꿉니다.** 개인 폴더를 나눠도 GPU와 Python 환경은 공용입니다.

```bash
tmux new -s train-seungbeom
```

새 화면이 열리면 가상환경을 활성화하고 **서버의 본인 코드 저장소로 이동한 뒤** 아래 명령을 실행하세요. 같은 tmux 세션이 이미 있으면 `tmux attach -t train-seungbeom`으로 접속합니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/venv/bin/activate
PROJECT_DATA_ROOT=/home/ubuntu/data/battery_data_light
python -u train.py \
  --data_dir "$PROJECT_DATA_ROOT" \
  --config configs/baseline.json \
  --run_dir runs/seungbeom/exp001_fusion_seed42 \
  --model_out weights/model.pt
```

- `--data_dir` 아래 `train/train.csv`, `validation/validation.csv`를 읽습니다. Test는 학습 중 읽지 않습니다.
- 공식 CSV의 `image_path`가 `images/...png`이므로 각 CSV의 부모 폴더가 이미지 기준 경로입니다. `--image_root`에 `test/images`를 넣으면 `images/images`가 되어 잘못된 경로가 됩니다.
- 배포 구조가 다르거나 자체 분할을 쓸 때는 `--data_dir` 대신 `--train_csv`, `--val_csv`를 사용하고, 필요하면 `--train_image_root`, `--val_image_root`를 각각 지정하세요. 다른 폴더를 재귀 검색해 이미지를 찾지 않습니다.
- 학습 기본 장치는 CUDA입니다. 맥에서 이 명령으로 실제 데이터를 처리하지 마세요. `--device cpu`는 가상 입력 검증용으로 제공합니다.
- 기존 `run_dir`와 `model_out`은 덮어쓰지 않습니다. 다음 실험에서는 새 경로를 지정하세요. `--model_out`은 선택 사항이며 생략해도 `run_dir/best.pt`가 남습니다.
- tmux에서 나갈 때는 Ctrl+B 후 D입니다. Mac에서도 Cmd가 아닙니다.

실험 폴더에는 `best.pt`, `best_metrics.json`, `history.json`, `config.json`, `preprocessing.json`, `environment.json`, `split.json`이 저장됩니다. `best.pt`에는 모델 설정·전처리·클래스 매핑이 포함되므로 추론용 전처리 fit이 필요 없습니다. 코드 SHA, 코드 파일 해시, 실제 Python·라이브러리·CUDA·GPU 환경, 입력 CSV 해시를 함께 기록합니다. 이 기록은 실제 데이터에서 생성된 비공개 실험 산출물입니다.

## 추론과 제출 검사

**SSH 서버 터미널의 본인 레포에서 실행합니다.** 학습 후 해당 레포 안에 `weights/model.pt`가 준비되면 제출 필수 인자 세 개만으로 동작합니다. 아래 결과도 본인 레포의 `outputs/` 안에 저장됩니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/venv/bin/activate
PROJECT_DATA_ROOT=/home/ubuntu/data/battery_data_light
python test.py \
  --input "$PROJECT_DATA_ROOT/test/test.csv" \
  --image_root "$PROJECT_DATA_ROOT/test" \
  --output outputs/seungbeom/submission_exp001.csv

python check_submission.py \
  --data_dir "$PROJECT_DATA_ROOT" \
  --final outputs/seungbeom/submission_exp001.csv
```

`sample_submission.csv`가 데이터 루트 밖에 있으면 점검 도구의 `--sample_submission /실제/경로/sample_submission.csv`로 지정합니다. 이 도구는 **공식 가이드의 검사 항목을 구현한 팀 자체 도구**이며 운영진 배포 스크립트 자체가 아닙니다. 최종 선택 파일 재현, 반복 실행, 입력 행 섞기, 실험 분리, 미래 행 제거, 임의 부분 입력을 검사하고 불일치 시 실패합니다. 연구용 `--diagnose_history_subset`을 명시할 때만 과거 기반 모델의 임의 부분 입력 차이를 진단 상태로 출력하며, 제출 적합성을 뜻하지 않습니다. 최종 검사는 기본 엄격 모드와 제공받은 운영진 점검 도구로 실행하세요.

`test.py`는 입력 순서를 그대로 보존하며, 공식 자료집의 Test와 제출 양식 순서가 같다는 계약을 따릅니다. 점검 도구는 실제 템플릿의 식별자·순서와도 일치하는지 검사합니다. 결과는 `sample_id,label` 두 열, 정수 `1..6`이고 인덱스를 저장하지 않습니다. 출력 파일이 있으면 덮어쓰지 않으므로 새로운 파일명을 지정하세요.

다른 실험 가중치는 `--checkpoint runs/실험명/best.pt`로 선택할 수 있습니다(`--model`은 같은 인자의 별칭). 이전 베이스라인의 `.joblib`은 이 신경망의 `.pt`와 호환되지 않습니다. 최종 선택 모델은 반드시 기본 위치 `weights/model.pt`로 준비하고 필수 인자 세 개로 재현되는지 확인하세요.

## 모델·검증 설계

- **이미지:** 공식 설명의 RGB 의사색 PNG를 고정 크기 128×128로 변환하고 픽셀을 255로 나눕니다. CNN이 시각 특징을 학습합니다. 색을 물리적인 온도로 환산하지 않습니다. 학습·추론에 무작위 이미지 증강을 사용하지 않습니다.
- **센서:** 원본 CSV 계약 `RAW_SENSOR_COLUMNS`는 `meas_v`, `meas_a`, `meas_t1`, `meas_t2`, `meas_p`입니다. 모델 입력은 설정의 `sensor_features`로 따로 지정합니다. 기본은 원본 5개이고, 온도 평균·차이·절댓값 차이를 추가한 8개 피처 비교 설정도 제공합니다. 실제 추가 컬럼을 자동으로 입력에 넣지 않습니다.
- **전처리:** 기본 `median`은 피처별 Train 중앙값 대체입니다. `forward_fill`은 같은 실험의 원본 센서를 과거 유효값으로 먼저 채운 뒤 피처를 계산하며, 과거가 없으면 Train 중앙값을 사용합니다. Train에서 대체한 값의 평균·표준편차로 정규화합니다. 모두 결측인 열은 0, 상수 열의 스케일은 1로 처리합니다. 피처 이름·순서·버전·대체 방식과 통계를 저장하여 Validation·Test에 그대로 적용합니다.
- **결합:** 기본 `mode=multimodal`은 이미지 특징과 센서 MLP 출력을 연결한 뒤 6개 logits를 출력합니다. `mode=sensor_only`는 이미지를 읽지 않는 센서 분석용 비교 실험이며 최종 제출 모델로 사용하지 않습니다. 내부 타깃은 `0..5`, 제출 때만 `1..6`으로 변환합니다.
- **검증:** Train만 모델·전처리를 학습하고 제공 Validation의 6클래스 Macro F1으로 최적 epoch를 선택합니다. 두 분할의 sample ID·experiment ID 중복은 오류로 처리합니다. 자체 CV와 Train+Validation 최종 재학습은 현재 구현하지 않았습니다.
- **시계열:** 기존 설정은 현재 행만 사용합니다. `sensor_ffill_raw.json`은 같은 실험의 직전 유효값으로 결측을 채우며, `sensor_history_median.json`과 `sensor_ffill_history.json`은 전압·전류·압력의 과거 30초 평균·변화량도 사용합니다. 미래 관측·실험 전체 통계·다른 실험 관측은 사용하지 않습니다. 식별자는 그룹 구분에만 사용합니다.
- **추론 일관성:** `eval()`, inference mode, 시드 및 결정적 연산 설정을 사용합니다. 모델 forward는 한 샘플씩 수행하므로 `--batch_size`는 로딩에만 적용됩니다. 처리 속도보다 부분 입력·배치 구성 일관성을 우선한 초기 구현입니다.
- **사전학습:** 사용하지 않습니다. 외부 데이터나 외부 추론 API도 사용하지 않습니다.

센서 담당의 실행 명령과 피처 확장 방법은 [센서 실험 안내](docs/sensor_experiments.md), 후속 과거 기반 실험은 [시계열 실험 안내](docs/temporal_sensor_experiments.md)를 따릅니다. 새 가중치는 피처·대체 방식·과거 계산 버전을 포함하는 v3로 저장하고, 기존 v1·v2도 읽습니다. 제출 코드에는 `sensor_features.py`와 `temporal.py`를 함께 포함해야 합니다.

CPU와 CUDA, 서로 다른 버전·장치 간 결과까지 동일하다고 보장할 수는 없습니다. 최종 가중치와 함께 실제 서버 환경을 기록하고 동일 환경에서 최종 파일 재현 검사를 수행하세요. [PyTorch 재현성 안내](https://docs.pytorch.org/docs/2.7/notes/randomness.html)를 기준으로 난수와 결정적 연산을 설정했습니다.

## 로컬 검증 기록

2026-11-14 KST, macOS arm64 / Python 3.12.13 / PyTorch 2.15.0 / NumPy 2.5.4 / pandas 3.1.1 / Pillow 12.3.0의 로컬 환경에서 초기 가상 입력 CPU 테스트 13개를 통과했습니다. 센서 피처 확장 후에는 CPU 테스트 31개와 Ruff 검사·포맷 검사를 통과했습니다. 원본·파생 피처 각각의 센서 단독 및 멀티모달 학습 → 가중치 복원 → 추론, 부분 입력 일관성, 기존 v1 가중치 호환, Train·Validation이 다른 실험의 비교 거부를 확인했습니다. 이 기록은 주최 측 서버의 환경이나 실제 대회 성능을 나타내지 않습니다.

센서 분포 보고서 추가 후에는 가상 CPU 테스트 35개를 통과했습니다. 로컬에 Matplotlib 3.11.2를 설치하여 가상 입력의 PNG·HTML 생성, 결측·상수 분포, 기존 보고서 덮어쓰기 방지와 Test 분석 거부를 확인했습니다. Matplotlib은 `requirements-eda.txt`의 선택 의존성이며 서버 환경을 자동으로 바꾸지 않습니다.

## 최종 제출 전에 채울 정보

이 저장소는 초기 세팅이며 아직 최종 제출물이 아닙니다.

- 팀명: 미기입 / 최종 선택 파일: 미선택
- 실제 학습 OS·Python·GPU·패키지 버전: 실행 후 `environment.json` 기준으로 기록
- Validation Macro F1 및 클래스별 결과: 미측정
- 최종 학습·검증 구성과 모델 선택 근거: 실험 후 기록
- 가중치: 아직 없음. 학습 후 `weights/model.pt`로 배치
- 가중치 다운로드 링크·파일 SHA-256: 외부 배포 시 기록하고 정확한 설치 위치 명시

`weights/`는 기본적으로 Git에서 제외합니다. **최종 제출 시에는 가중치를 레포에 명시적으로 포함하거나 README의 다운로드 링크로 제공해야 합니다.** 작은 최종 모델 파일을 포함하기로 한 경우에만 `git add -f weights/model.pt` 후 `python scripts/check_staged.py --allow_final_weights`로 정확히 그 파일만 예외 처리할 수 있습니다. 데이터나 다른 실험 산출물은 예외가 아닙니다. DAKER 본문은 100MB 초과 가중치의 다운로드 링크를 요구하며, 웹 업로드의 파일당 25MB 제한은 별도입니다. 원활한 배포를 위해 25MB 초과 파일은 링크 방식으로 제공해도 됩니다.

최종 리더보드 파일은 반드시 `test.py`로 생성합니다. 리더보드는 **2026-11-15 13:00 KST**, 코드·PPT와 서버 사용은 **13:50 KST**가 기준입니다. 최종 제출 시 공개 레포, 필수 코드, 재현 가능한 가중치, README를 갖추어야 합니다. 이 코드 준비 작업은 Git push·공개 전환·리더보드 업로드·메일 발송을 자동으로 수행하지 않습니다.
