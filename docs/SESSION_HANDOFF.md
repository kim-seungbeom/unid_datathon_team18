# 다른 세션에 전달할 작업 상태

작성 기준: 2026-11-14 KST. 실제 데이터·정답·통계 수치·접속 비밀정보를 포함하지 않는 코드 작업 핸드오프입니다. 아래 상태는 작성 시점의 기록이므로 다음 세션은 먼저 로컬 `git status --short --branch`로 변경 여부를 확인합니다.

## 새 세션에 붙여 넣을 요청

```text
이 저장소의 AGENTS.md, docs/SESSION_HANDOFF.md, docs/sensor_experiments.md를 읽고 이어서 작업해 줘.
나는 김승범이며 센서 전처리·파생 피처·센서 단독 성능 비교를 담당해.
코드는 로컬에서 수정하고 GitHub로 전달하며, 실제 데이터 확인과 학습은 내가 GPU 서버에서 직접 실행해.
작업 브랜치는 feat/seungbeom-sensor야. 센서 확장·EDA·문서의 현재 커밋과 원격 동기화 상태를 확인해 줘.
기존 변경을 보존하고 코드/가상 입력으로만 점검해 줘. 현재 실행 위치와 Git 상태부터 확인해 줘.
사용자가 SSH 창을 열었다는 사실을 에이전트의 서버 연결 상태로 가정하지 마.
서버에서 실행할 명령을 단계별로 안내하고, 분포 보고서는 내가 서버에서 직접 확인하게 해 줘.
원본 데이터나 실제 집계 결과를 요청하지 말고, 필요한 정보는 코드·공식 정의·환경 정보로 먼저 해결해 줘.
현재 단계와 다음 단계를 설명한 뒤 필요한 코드 작업을 이어가 줘.
```

같은 로컬 작업 공간을 여는 다른 세션에서는 미커밋 파일도 볼 수 있습니다. 새 clone이나 서버 레포에는 commit·push·pull하기 전의 변경이 전달되지 않습니다. 이 문서는 SSH 연결을 생성하거나 인증을 전달하지 않습니다.

## 작업 위치와 역할

| 구분 | 위치 또는 역할 |
| --- | --- |
| 로컬 코드 저장소 | `/Users/kimseungbeom/Documents/unid_datathon/unid_datathon_team18` |
| 사용자 서버 실행용 레포 | `/home/ubuntu/seungbeom/unid_datathon_team18` |
| 공용 Python | `/home/ubuntu/venv/bin/python` |
| 공용 데이터 | `/home/ubuntu/data/battery_data_light` |
| 운영진 원본 배포 폴더 | `/home/ubuntu/data/baseline` 아래. 압축 내부 폴더명은 가정하지 않음 |
| 승범 | 센서 전처리·파생 피처·센서 단독 검증 |
| 다른 팀원 2명 | 이미지 분석 / 두 결과 결합·통합 |

현재 에이전트는 로컬 코드에서 작업했습니다. 사용자 터미널에는 서버 연결과 해당 레포 위치가 확인됐지만, 에이전트가 서버에 연결됐다는 뜻은 아닙니다. 서버 접속 정보나 개인 인증을 문서에 추가하지 않습니다.

코드 전달: 로컬 수정·가상 입력 검증 → commit·push → 사용자가 서버의 같은 브랜치에서 pull → 사용자 실행. 각 팀원은 서버 개인 폴더에 실행용 clone을 두며, 데이터·venv·GPU는 공용입니다. 실행 중인 레포에서는 pull·브랜치 전환을 하지 않습니다.

## 현재 파일 구조

```text
schema.py                       원본 센서 컬럼, 라벨, 실험 분할 계약
sensor_features.py              모델 입력 피처 이름·계산·버전
preprocessing.py               Train에서 학습한 결측 대체·정규화
dataset.py                     현재 행 로딩; 센서 단독은 이미지 생략
model.py                       이미지·센서 결합 / 센서 단독 모드
train.py                       Train 학습, Validation 모델 선택
test.py                        저장된 피처·전처리·가중치로 추론
evaluate.py                    6클래스 Macro F1·클래스별 지표
runtime.py                     설정·추론·재현 메타데이터
configs/baseline.json           이미지 + 원본 센서 5개
configs/sensor_raw.json         센서 5개 단독
configs/sensor_temperature.json 센서 5개 + 온도 파생 3개 단독
configs/fusion_temperature.json 이미지 + 센서 피처 8개
scripts/setup_env.sh            공용 서버 환경 확인 / 로컬 환경 설치
scripts/sensor_eda.py           서버에서 사람이 실행하는 센서 분포 보고서
scripts/sensor_timeline.py      실험별 시간축에서 파생·결측 대체·표준화 과정 비교
scripts/compare_runs.py         동일 분할 실험의 검증 지표 비교
requirements-eda.txt            선택 설치: 분포 그래프용 Matplotlib
tests/                         실제 데이터와 무관한 가상 입력 테스트
baselines/organizer_v7/         수정하지 않는 운영진 원본 소스 5개
```

`open/`, `data/`, `runs/`, `outputs/`, `weights/` 등 비공개 자료 경로는 에이전트가 열거나 검색하지 않습니다. `.gitignore`가 AI 접근을 차단해 주는 것은 아닙니다.

## 구현과 실행 상태

| 항목 | 확인된 상태 |
| --- | --- |
| 초기 코드와 협업 문서 | 이전 작업에서 PR #1을 `main`에 병합함 |
| 코드 전달 브랜치 | `feat/seungbeom-sensor`; 이 문서와 센서 확장·EDA를 함께 전달. 추가 미커밋 변경과 원격 동기화 상태는 Git으로 확인 |
| 센서 처리 | 원본 5개와 입력 피처 분리, 온도 평균·차이·절댓값 차이 3개 추가 |
| 모델 | 센서 단독 / 멀티모달, 피처 개수 자동 반영, v2 저장·v1 가중치 읽기 |
| 로컬 검증 | EDA 추가 후 가상 CPU 테스트 35개·Ruff 검사 통과. 가상 데이터 PNG도 시각 확인. 실제 데이터 검증 아님 |
| 공식 `setup.sh` | 사용자 설치 로그에서 완료 확인 |
| GPU | 이전 사용자 설치 로그에서 CUDA 사용 가능·RTX 5080 확인. 현재 점유·상태는 재확인 필요 |
| 서버 레포·venv | 사용자 터미널에서 본인 레포 위치와 `(venv)` 확인 |
| 센서 통계 조회 | 사용자가 Train 요약 코드를 실행함. 결과 수치는 이 문서에 기록하지 않음 |
| 환경 검사 오류 | 이전 코드가 특정 베이스라인 하위 폴더를 요구해 실패. 로컬 수정 완료, 서버 반영 여부 미확인 |
| 실제 학습과 성능 | 완료 보고 없음. 점수·가중치 생성 여부를 추정하지 않음 |

베이스라인 경로 오류는 `setup.sh` 파일을 레포 밖에 둔 것과 무관합니다. 수정한 검사기는 원본 베이스라인 폴더를 선택 항목으로 취급합니다. 아직 이전 코드를 사용하는 서버에서는 `PROJECT_BASELINE_ROOT=/home/ubuntu/data/baseline bash scripts/setup_env.sh --server`로 검사할 수 있습니다. 이후 전체 환경 검사가 통과했다는 보고는 아직 없습니다.

## 다음 실행 단계

1. 다음 에이전트는 로컬 실행 위치·브랜치·수정 파일을 확인하고 기존 변경을 보존합니다. 실제 데이터 디렉터리를 조사하지 않습니다.
2. 변경을 검토하고 가상 입력 검사 후 코드 전달을 진행합니다. 새 세션에 “push를 완료했다”고 가정하지 않습니다. 수정 파일만 명시적으로 stage하고 `scripts/check_staged.py`를 확인합니다.
3. 사용자가 서버에서 같은 브랜치를 받습니다. 아래 명령은 **변경이 GitHub에 push된 이후**, 서버에 미커밋 변경과 실행 중인 작업이 없을 때 실행합니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git fetch origin
git switch feat/seungbeom-sensor
git pull --ff-only origin feat/seungbeom-sensor
source /home/ubuntu/venv/bin/activate
bash scripts/setup_env.sh --server
```

4. 사용자가 [센서 분포·시간축 변환 안내](sensor_eda.md)에 따라 보고서를 만들고 서버에서 직접 확인합니다. `scripts.sensor_timeline`은 학습 전에 실행 가능하며, 실험 하나의 원본 온도→파생 피처와 피처별 결측 대체→표준화를 보여줍니다. Train에서만 전처리를 학습하고 ID·시간은 표시용으로만 사용합니다. 공용 환경에 Matplotlib이 없으면 문서의 개인 환경 설치 방법을 사용합니다.
5. GPU 사용을 조율하고 [센서 실험 안내](sensor_experiments.md)에 따라 원본 센서 단독 → 파생 피처 포함 단독 학습을 순서대로 실행합니다.
6. 같은 Train·Validation의 결과를 `scripts/compare_runs.py`로 비교하고, 피처 순서·설정·검증 방법을 결합 담당에게 전달합니다.

전처리 통계는 Train에서만 학습하고 Validation에는 적용만 합니다. 현재 피처는 한 행의 센서만 사용하며 식별자·미래·실험 전체 통계를 입력으로 넣지 않습니다. 센서 단독은 분석용이고 최종 대회 모델은 이미지와 센서를 모두 활용해야 합니다.

## 다른 세션에 공유할 수 있는 맥락

코드·공식 컬럼 정의·가상 입력 결과·데이터 값이 없는 오류 메시지·Python/패키지/GPU 환경 정보는 전달할 수 있습니다. 실제 데이터·정답·이미지와 이를 포함한 출력은 전달하지 않습니다.

집계 통계 공유에 대해 사용자는 “되는 것 같다”고 답했으나 운영진의 명시적 허용 문구는 확인되지 않았습니다. 이를 공식 승인으로 기록하거나 다른 세션의 데이터 접근 허가로 해석하지 않습니다. 분포 보고서는 사용자가 서버에서 확인합니다. 이 범위가 공식적으로 달라지면 데이터 값 대신 해당 안내 문구를 기준으로 문서와 작업 범위를 갱신합니다.
