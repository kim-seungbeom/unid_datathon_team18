# 서버에서 코드 받아 실행하기

작업 방식: **로컬 VS Code에서 수정·검증·push → 서버에서 같은 브랜치를 pull → 사용자가 직접 실행**.

아래 명령은 모두 사용자가 **GPU 서버에 SSH로 연결된 VS Code 터미널**에서 실행합니다. 운영진 `setup.sh`를 완료한 상태를 전제로 하며, 데이터 재다운로드나 서버에 Codex를 설치하는 과정은 없습니다.

## 1. 코드 받기 — 최초 한 번

```bash
mkdir -p /home/ubuntu/seungbeom
cd /home/ubuntu/seungbeom
git -c credential.helper= clone --branch initial-setup https://github.com/kim-seungbeom/unid_datathon_team18.git
cd unid_datathon_team18
git branch --show-current
```

마지막 출력은 `initial-setup`이어야 합니다. 같은 이름의 디렉터리가 이미 있으면 삭제하거나 덮어쓰지 말고 기존 레포와 미커밋 변경을 먼저 확인합니다.

인증이 필요한 레포라면 사용자가 직접 GitHub 인증을 처리합니다. HTTPS 비밀번호 입력칸에는 GitHub 계정 비밀번호 대신 해당 레포에 접근할 수 있는 토큰을 사용합니다. 위 명령의 `credential.helper=`는 이 clone에서 토큰 저장용 helper를 사용하지 않게 합니다. 토큰을 URL·문서·대화에 넣거나 공용 서버에 저장하지 않습니다.

VS Code의 File → Open Folder에서 `/home/ubuntu/seungbeom/unid_datathon_team18`을 열면 실행할 파일을 볼 수 있습니다. 코드와 설정의 수정은 로컬에서 진행합니다.

## 2. 공용 환경 확인

```bash
source /home/ubuntu/venv/bin/activate
bash scripts/setup_env.sh --server
```

이 명령은 공용 경로·패키지·CUDA를 확인만 합니다. NumPy·pandas·Pillow가 없을 때만 팀과 설치를 조율한 뒤 아래 명령을 실행하고 다시 확인하세요.

```bash
python -m pip install -r requirements-common.txt
bash scripts/setup_env.sh --server
```

주최 측 GPU용 PyTorch는 그대로 사용합니다. 서버에서 `--local`을 실행해 새 `.venv`를 만들지 않습니다.

## 3. 팀 모델 학습

GPU를 사용하는 팀원이 없는지 확인합니다.

```bash
nvidia-smi
```

먼저 아래 한 줄만 실행하세요. 같은 이름의 tmux 세션이 이미 있으면 `tmux attach -t train-seungbeom`으로 들어갑니다.

```bash
tmux new -s train-seungbeom
```

새 화면이 열린 다음:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/venv/bin/activate
python -u train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config configs/baseline.json \
  --run_dir runs/seungbeom/exp001_fusion_seed42 \
  --model_out weights/model.pt
```

처음에는 10 epoch의 시작 설정을 사용합니다. `runs/...`와 `weights/model.pt`는 이미 존재하면 덮어쓰지 않고 중단합니다. 다음 실험에는 새 `--run_dir`를 지정하고 `--model_out`은 생략하거나 다른 파일명으로 바꾸세요. 각 실험의 최적 모델은 항상 해당 `run_dir/best.pt`에 남습니다.

학습을 유지한 채 tmux에서 나오려면 Ctrl+B를 누른 뒤 손을 떼고 D를 누릅니다. 실행 중에는 해당 레포에서 pull하거나 브랜치를 바꾸지 않습니다.

## 4. 학습 완료 후 추론

첫 실험의 `weights/model.pt`가 생성된 뒤 실행합니다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/venv/bin/activate
python test.py \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --output outputs/seungbeom/submission_exp001.csv
```

`image_path`는 split 폴더 기준이므로 `--image_root`는 `test` 폴더입니다. `test/images`를 지정하지 않습니다. 결과 파일이 이미 있으면 새 출력 이름을 사용합니다.

```bash
python check_submission.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --final outputs/seungbeom/submission_exp001.csv
```

이 도구는 팀이 구현한 재현·부분 입력 점검입니다. 최종 제출 전 운영진 배포 점검 도구도 실행하세요. 실제 데이터·로그·예측은 서버에서 직접 확인하고 AI 대화에 보내지 않습니다.

## 5. 로컬에서 수정한 코드를 다시 받기

로컬에서 `initial-setup` 브랜치의 변경을 commit·push한 뒤, 실행 중인 작업이 없는 서버 레포에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short --untracked-files=no
git branch --show-current
```

변경 파일이 없고 브랜치가 `initial-setup`이면 이어서:

```bash
git -c credential.helper= pull --ff-only origin initial-setup
```

서버에 소스 변경이 있거나 pull이 실패하면 강제로 덮어쓰지 않고 원인을 확인합니다. 이후 다른 기능 브랜치로 작업하면 로컬과 서버의 대상 브랜치를 함께 맞춥니다.

## 운영진 원본 베이스라인 위치

원본 소스는 `baselines/organizer_v7/`에 보관돼 있습니다. 위의 학습 명령은 저장소 루트의 팀 CNN·MLP 모델을 실행합니다. 원본 로지스틱 회귀를 비교 실행할 때는 해당 폴더의 README와 requirements를 따르고, 실행 산출물은 별도의 `runs/` 경로로 저장하세요. 원본 코드를 팀 모델 위에 복사하지 않습니다.
