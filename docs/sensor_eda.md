# 서버에서 센서 분포 확인하기

실제 대회 데이터로 아래 명령을 실행하는 사람은 사용자입니다. 에이전트는 로컬에서 코드와 독립 가상 입력만 검증합니다. 다른 세션도 이 안내로 같은 분석을 준비할 수 있지만, 실제 보고서를 자동으로 열거나 업로드하지 않습니다.

## 준비

`scripts/sensor_eda.py`가 포함된 코드를 서버의 개인 레포에 받은 후 실행합니다. 아직 미커밋·미push 상태라면 [핸드오프 문서](SESSION_HANDOFF.md)의 코드 전달 단계가 먼저입니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/venv/bin/activate
```

요약표와 히스토그램 수치만 생성할 때는 기존 NumPy·pandas로 충분합니다. PNG 그래프를 원하면 Matplotlib이 필요합니다. 이미 설치되어 있는지 다음처럼 확인할 수 있습니다.

```bash
python -c "import importlib.util; print('Matplotlib installed:', importlib.util.find_spec('matplotlib') is not None)"
```

없다면 공용 환경을 사용하는 팀원과 합의한 뒤 설치합니다. PyTorch를 설치·교체하는 명령은 아닙니다.

```bash
python -m pip install -r requirements-eda.txt
```

## Train 보고서 만들기

아래는 원본 센서 5개와 온도 파생 3개를 분석하고 그래프도 저장하는 명령입니다. GPU나 이미지 파일을 사용하지 않습니다.

```bash
python -m scripts.sensor_eda \
  --data_dir /home/ubuntu/data/battery_data_light \
  --split train \
  --feature_set temperature \
  --output_dir runs/seungbeom/eda_train_v1 \
  --plots
```

Matplotlib 설치 없이 실행하려면 `--plots`를 빼세요. 원본 센서만 보려면 `--feature_set raw`를 사용합니다. 보고서 폴더가 이미 있으면 새 이름을 지정해야 하며 덮어쓰지 않습니다.

스크립트는 정해진 센서 컬럼만 읽습니다. 정답·식별자·이미지 경로·추가 컬럼은 읽지 않습니다. 원본 CSV나 전처리·모델 파일을 수정하지 않고, 터미널에는 실제 통계값을 출력하지 않습니다.

| 산출물 | 내용 |
| --- | --- |
| `summary.csv` | 결측·변환 실패·무한대·0·고유값 수, 최솟값·백분위·중앙값·최댓값 |
| `histograms.json` | 피처별 히스토그램 구간과 개수 |
| `metadata.json` | split·피처 목록·피처 정의 버전·설정 |
| `report.html` | 요약표와 생성한 그래프를 모은 문서. 외부 리소스 없음 |
| 각 피처의 `.png` | `--plots`에서 생성하는 전체 범위 및 중앙 1~99백분위 분포 |

1~99백분위 그래프에는 제외된 유효값 개수가 표시됩니다. 이는 분포를 보기 위한 확대이며 값을 제거하거나 모델의 클리핑 기준을 학습한 것이 아닙니다. 결측·비유한값은 히스토그램에서 제외하고 요약표에 개수로 남깁니다. 파생 피처는 결측 대체 전 값입니다.

## 결과 보기

VS Code SSH 탐색기에서 본인 레포의 `runs/seungbeom/eda_train_v1`을 열고 `summary.csv` 또는 PNG를 직접 확인합니다. AI에게 해당 폴더나 화면을 읽도록 요청하지 않습니다.

HTML로 모아 보려면 사용자가 서버의 별도 터미널에서 해당 보고서 폴더만 로컬 주소로 제공합니다.

```bash
python -m http.server 8765 --bind 127.0.0.1 \
  --directory /home/ubuntu/seungbeom/unid_datathon_team18/runs/seungbeom/eda_train_v1
```

VS Code SSH의 Ports 탭에서 서버 포트 `8765`를 로컬로 전달하고, 표시된 로컬 주소 뒤에 `/report.html`을 붙여 본인 브라우저에서 엽니다. 공개 링크나 공개 포트로 전환하지 않습니다. 보고 나면 서버 터미널에서 Ctrl+C로 종료합니다. 포트가 사용 중이면 빈 포트 번호를 선택합니다.

Validation은 별도 보고서로 만들 수 있습니다. `--split validation`과 새로운 `--output_dir`를 함께 지정합니다. Validation 통계로 전처리를 다시 학습하지 않습니다. 이 도구는 Test 분석 옵션을 제공하지 않습니다.

## 공유 범위

생성 결과는 실제 데이터에서 나온 비공개 산출물로 `runs/` 아래 보관하며 Git에 올리지 않습니다. 터미널에서 직접 출력하거나 HTML로 보더라도 외부 AI 공유 허용 여부가 달라지지 않습니다. 집계 통계의 외부 AI 공유에 대한 별도 운영진 안내는 아직 확인되지 않았습니다.

다른 세션에는 이 문서와 분석 코드, 실행 단계, 데이터가 없는 오류 설명을 전달합니다. 실제 데이터·보고서·분포 수치를 재현용 가상 입력으로 복사하지 않습니다.
