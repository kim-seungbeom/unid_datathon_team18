# 팀 작업 컨벤션

## 기본 작업 위치

**코드 편집·가상 입력 검증은 각자의 로컬 VS Code에서 하고, 실제 데이터 실행은 GPU 서버에서 합니다.** Codex도 로컬 코드 전용 작업 공간에서 사용합니다.

```text
로컬 VS Code: 코드 수정 → 가상 입력 검증 → commit·push
                                              ↓
                                         팀 GitHub
                                              ↓
서버 SSH 터미널: 첫 clone / 이후 같은 브랜치 pull → 학습·검증·추론
```

서버에서 실행하다 코드 수정이 필요하면 로컬에서 수정하고 다시 push → pull합니다. 서버 개인 레포는 실행용 사본이며, 각자의 로컬 레포에서 변경을 만들고 PR로 통합합니다.

## 서버 폴더와 Git 저장소의 관계

서버에서는 환경·데이터를 공용으로 사용하고, 팀원별로 같은 팀 레포의 실행용 사본을 각각 clone합니다. **개인 폴더는 팀 레포 밖에 두고 그 안에 레포 전체를 받습니다.** 서로 다른 브랜치·실험을 실행할 때 다른 팀원의 파일이 바뀌는 것을 막기 위한 구분입니다. 아래 개인 레포 경로는 권장 예시이며 실제로 생성·배포됐다는 뜻은 아닙니다. `seungbeom`은 승범의 예시이고, `member_a`, `member_b`는 실제 팀원 이름으로 바꿉니다.

```text
/home/ubuntu/
├── venv/                              # 공용 Python 환경
├── data/
│   ├── battery_data_light/             # 공용 데이터: 복제·원본 수정하지 않음
│   └── baseline/uni_d_baseline_v7/     # 운영진 원본
├── seungbeom/
│   └── unid_datathon_team18/           # 승범이 로컬에서 push한 코드의 실행용 사본
├── member_a/
│   └── unid_datathon_team18/           # 팀원 A의 실행용 사본
└── member_b/
    └── unid_datathon_team18/           # 팀원 B의 실행용 사본
```

같은 서버 레포 디렉터리에서 여러 사람이 브랜치를 바꾸거나 pull하지 않습니다. 각자 실행용 사본에서 로컬 작업 브랜치의 코드를 받아 실행합니다. 같은 `ubuntu` 계정 안의 개인 폴더는 작업 충돌을 줄이는 구분이며 인증 정보나 파일 접근 권한을 격리하는 보안 경계는 아닙니다.

팀원이 합류할 때는 다음 순서로 준비합니다. 구체적인 명령은 [서버 실행 안내](docs/server_quickstart.md)를 따릅니다.

1. 다른 팀원과 겹치지 않는 영문 폴더명을 정하고 `/home/ubuntu/<본인이름>/`을 만듭니다.
2. 그 안에 같은 팀 GitHub 레포의 `main`을 clone합니다. 개인 실험에서는 본인이 로컬에서 push한 작업 브랜치를 사용하고, PR로 통합한 공통 코드는 `main`에서 받습니다. 개인 폴더명과 브랜치명은 별개입니다.
3. 공용 `/home/ubuntu/venv`를 활성화하고, 데이터는 `/home/ubuntu/data/battery_data_light`를 참조합니다. 공식 `setup.sh`를 팀원마다 다시 실행하거나 데이터를 개인 폴더에 복사하지 않습니다.
4. 실험 결과·예측·가중치는 본인 레포 안에 저장합니다. tmux 세션도 `train-<본인이름>`으로 구분하고, 공용 GPU 사용 시간은 팀원과 조율합니다.

예를 들어 승범의 `weights/model.pt`는 `/home/ubuntu/seungbeom/unid_datathon_team18/weights/model.pt`입니다. 다른 팀원은 본인 레포에 같은 상대 경로를 사용할 수 있습니다. 폴더를 나눠도 공용 Python 패키지 변경은 모두에게 영향을 주므로 팀과 합의합니다.

## 코드 저장소 내부 구조

현재 실행 코드의 루트 구조를 유지합니다. `configs/<담당자>/`는 개인 실험 설정이 생길 때 추가하고, `runs/`, `outputs/`, `weights/`는 서버에서 실행·배포 준비 시 생성합니다.

```text
unid_datathon_team18/
├── model.py                           # 팀 모델, 제출 필수 진입 파일
├── train.py                           # 팀 학습 실행
├── test.py                            # 팀 제출용 추론
├── schema.py                          # 컬럼·라벨·분할 계약
├── dataset.py                         # 공통 데이터 로더
├── preprocessing.py                   # 공통 전처리
├── evaluate.py                        # 공통 평가
├── runtime.py                         # 시드·설정·저장 공통 기능
├── check_submission.py                # 제출 일관성 점검
├── configs/
│   ├── baseline.json                  # 공통 시작 설정
│   └── <담당자>/exp001_fusion.json     # 새로 추가할 개인 실험 설정 예시
├── baselines/organizer_v7/             # 수정하지 않는 운영진 원본 코드
├── scripts/                           # 환경·Git 점검 도구
├── tests/                             # 가상 입력 검증
├── docs/                              # 설계·규정 문서, 실제 데이터 출력 제외
├── runs/<담당자>/<실험명>/            # 서버 전용 실험 산출물, Git 제외
├── outputs/<담당자>/                  # 서버 전용 예측 CSV, Git 제외
├── weights/                           # 최종 배포 가중치, 평소에는 Git 제외
├── README.md
├── CONTRIBUTING.md
└── requirements*.txt
```

개인 실험을 위해 `train_seungbeom.py`, `train_final2.py` 같은 실행 파일을 계속 복제하지 않습니다. 같은 코드에서 설정만 달라지는 실험은 `configs/`의 새 설정으로 표현합니다. 모델 구조가 달라지는 실험은 작업 브랜치에서 구현하고, 여러 구조를 함께 유지할 필요가 생기면 공통 `model.py` 진입점을 유지한 채 모듈을 분리합니다.

## 작업 흐름

- GitHub 기본 브랜치 `main`은 실행 가능한 공통 코드입니다. 초기 세팅 브랜치는 `initial-setup`, 이후 기능 작업은 `feat/담당자-작업`, 오류 수정은 `fix/담당자-작업`처럼 목적이 드러나는 이름을 사용합니다.
- 한 PR은 한 가지 목적에 집중하고 변경 내용·가상 입력 검증 결과를 적습니다.
- 커밋은 `feat:`, `fix:`, `docs:`, `test:`, `chore:` 접두사를 사용합니다.
- 코드와 실험 설정은 로컬 VS Code에서 수정합니다. 실제 데이터 학습·검증·추론은 사용자가 서버 SSH 터미널에서 직접 실행합니다.
- 코드 전달 방향은 로컬 commit·push → 서버의 같은 브랜치에서 pull입니다. 서버에 첫 레포를 준비할 때는 개인 폴더 아래 clone합니다. 데이터·venv는 Git으로 전달하지 않습니다.
- 실행에 쓸 코드는 학습 시작 전에 동기화합니다. 실행 중인 레포에서 pull하거나 브랜치를 바꾸지 않습니다.
- 수정 파일만 명시적으로 stage합니다. `git diff --cached --name-only`와 `python scripts/check_staged.py`를 확인합니다. 검사기는 알려진 경로·확장자를 검사하는 보조 수단이며 비밀값 검사나 접근 차단을 보장하지 않습니다.
- 이미지 특징 추출용 `image_features/01_extract_features.ipynb` 원본만 예외로 추적합니다. 이 파일을 stage한 PR은 `python scripts/check_staged.py --allow_clean_notebook`을 사용합니다. 검사기는 Git index의 실제 노트북을 읽어 실행 번호·출력·첨부·셀 메타데이터가 없는지 확인하며, 다른 노트북과 데이터 파일은 계속 차단합니다. 실행본은 커밋하지 않습니다.
- 서버에 미커밋 변경이 있으면 먼저 보존하고 합칩니다. 강제 push·`reset --hard`는 사용하지 않습니다.

## 담당 범위 예시

팀원 이름을 정해 실제 담당자를 합의합니다. 다음은 3명 협업을 위한 예시이며 확정 배정이 아닙니다.

| 담당 | 주로 변경하는 파일 | 함께 맞출 계약 |
| --- | --- | --- |
| 데이터·전처리 | `schema.py`, `dataset.py`, `preprocessing.py` | 센서 순서, 이미지 크기, 타깃 변환, 분할 |
| 모델·학습 | `model.py`, `train.py`, `configs/` | 모델 입력·출력, 체크포인트 형식 |
| 검증·제출 | `evaluate.py`, `test.py`, `check_submission.py` | 지표, 저장된 전처리 사용, 부분 입력 재현 |

공통 인터페이스를 바꾸는 PR에는 영향을 받는 담당자의 확인을 받습니다. 개인 브랜치 예시는 `feat/seungbeom-fusion`, `feat/member-a-preprocess`, `feat/member-b-validation`입니다. 팀 공통 코드가 필요한 변경은 담당자 폴더에 숨기지 않고 해당 공통 모듈에서 관리합니다.

## 코드와 인터페이스

- Python 3.10 이상 문법, 들여쓰기 4칸, 100자 줄 길이를 사용합니다. 로컬 검증 환경은 Python 3.12로 만듭니다.
- 함수·변수·파일: `snake_case`, 클래스: `PascalCase`, 상수: `UPPER_SNAKE_CASE`.
- 모델 계약: `model(images[B,3,H,W], sensors[B,5]) -> logits[B,6]`.
- 센서 순서: `meas_v`, `meas_a`, `meas_t1`, `meas_t2`, `meas_p`.
- 타깃은 내부에서 `0..5`, CSV 제출 시 `1..6`입니다.
- 경로는 CLI, 모델·학습 설정은 `configs/*.json`으로 전달합니다. 저장된 전처리와 클래스 매핑을 추론에서 그대로 사용합니다.
- 식별자·실험 ID·파일명은 모델 특징으로 넣지 않습니다. 현재 구현은 시간 정보나 과거 행을 사용하지 않습니다.

## 실험과 데이터

- 이름 예시: `exp001_fusion_seed42`. 저장 예시: `runs/seungbeom/exp001_fusion_seed42/`. 같은 실험 폴더는 덮어쓰지 않습니다.
- 개인 설정 예시: `configs/seungbeom/exp001_fusion_seed42.json`. 공통 `configs/baseline.json`을 복사해 필요한 값만 바꾸고, `train.py --config`에 지정합니다. 설정에 데이터 내용·공유 링크·인증 정보는 넣지 않습니다.
- Train 학습, Validation 검증을 기본으로 합니다. 자체 분할은 실험 단위로 분리합니다.
- 정규화·결측 대체는 Train에서만 fit합니다. Test를 학습·모델 선택·전처리 fit에 사용하지 않습니다.
- 공통 지표는 6개 클래스 전체 Macro F1이며 클래스별 F1·support·혼동행렬도 기록합니다.
- 커밋 SHA, 코드 해시, 설정, 실제 실행 환경, 분할 정보, 전처리, 가중치를 함께 보관합니다.
- 데이터·로그·예측·가중치는 Git에 자동 추가하지 않습니다. 최종 모델은 별도 배포하고 README에 획득 방법을 적습니다.
- 대회 데이터와 데이터가 포함된 출력은 외부 AI에 제공하지 않습니다. `open/`을 ignore해도 AI 접근이 차단되지는 않습니다. AI용 작업 공간은 코드만 두는 구성이 원칙입니다.
- 공용 GPU 사용은 팀원과 조율하며, 공용 venv 패키지는 합의 없이 바꾸지 않습니다.
