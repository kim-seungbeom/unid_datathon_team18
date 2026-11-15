# 최고 제출물 확인 및 서버 내부 보존

사용자가 확인한 최고 제출물은 `submission_inner_best_40_60_prior_tau.csv`,
Public **0.9093891677**이다. 제출자는 다른 팀원이다. 점수는 사용자 보고이며,
서버 파일과 리더보드 업로드가 같은지는 아직 확인하지 않았다.
이전 30:70+tau 실험을 이 제출물과 동일한 모델로 취급하지 않는다.

확인한 생성 코드는 `feat/doyu-prior-correction`의 `2dae970`에 있는
`experiments/doyu/prior_correction.py`다. 신경망과 원본 ExtraTrees의 확률을
40:60으로 합치고 `argmax(p / prior**tau) + 1`을 적용한다.
**40:60의 tau는 실제 correction 파일에서 확인해야 한다. 30:70의 0.15를 대입하지 않는다.**
소스 해시가 다른 생성 기록은 자동 승인하지 않는다.

## 1. 서버에서 원본 찾기

아래 명령은 사람이 서버에서 실행한다. 에이전트가 실제 파일을 조사하지 않는다.
실행 중인 작업이 없는 본인 clone에서 다음 코드를 받는다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning

python3 scripts/locate_final_submission.py locate \
  --team_home /home/ubuntu \
  --output runs/seungbeom/final_prior40_location.json
```

Python 표준 라이브러리만 사용하므로 가상환경·GPU·추가 설치가 필요 없다.
현재 clone의 `runs`와 `/home/ubuntu/<팀원>/unid_datathon_team18/runs`만 탐색한다.
데이터 폴더·이미지·임베딩·확률 캐시·숨김 폴더·디렉터리 심볼릭 링크는 건너뛴다.
다른 clone 이름/위치라면 `--runs_root /실제/프로젝트/runs`를 추가한다.
탐색 범위 밖 파일이나 읽기 권한이 없는 파일은 찾지 못할 수 있다.

출력은 후보 번호·경로·검사 상태·확인된 tau·누락 소스 개수다.
CSV 행은 파싱하거나 출력하지 않고 해시만 계산한다. 모델도 로드하지 않는다.
`artifacts_matched`는 CSV 생성 manifest → correction/recipe → 저장된 모델의
파일 해시가 일치했다는 뜻이며, 실제 추론 재현 또는 리더보드 점수 검증은 아니다.

- 후보가 0개면 제출한 팀원이 가진 CSV와 생성 폴더 위치를 팀 내부에서 확인한다.
- 같은 이름의 후보가 여럿이면 제출자가 원본을 확인해야 한다. 최신 수정 시각으로 고르지 않는다.
- `unverified`이면 `reason`에 표시된 누락/불일치를 해결한다. 다른 모델로 대체하지 않는다.
- 보고서가 이미 있으면 `--output`에 새 파일명을 사용한다.

팀원에게 보낼 내용:

> 0.9093891677 나온 submission_inner_best_40_60_prior_tau.csv를 최종 보존하려고 해.
> 실제 업로드한 원본 CSV와 같은 이름의 .manifest.json, 사용한 correction.json과
> recipe.json의 서버 경로를 알려줘. 신경망 체크포인트와 원본 ExtraTrees가 있는
> runs 폴더도 삭제하지 말아줘. 파일은 팀 내부에서만 전달하면 돼.

## 2. 확인한 후보를 별도 폴더에 백업

예를 들어 출력의 `[1]`이 제출자가 확인한 원본이며 `artifacts_matched`라면:

```bash
python3 scripts/locate_final_submission.py archive \
  --report runs/seungbeom/final_prior40_location.json \
  --candidate 1 \
  --output_dir runs/seungbeom/final_prior40_backup
```

현재 파일 해시를 다시 검사하고 CSV·생성 기록·참조 모델·전처리/PCA가 들어 있는
bundle·사용 가능한 동일 해시 소스를 복사한다. 원본은 수정하지 않는다.
복사 후 원본/사본 해시를 다시 검사하며 기존 백업 폴더에 덮어쓰지 않는다.
`backup.json`에 원래 경로와 사본 경로를 기록한다. 실패하면 `FAILED.json`이 남으며
그 폴더를 완성된 백업으로 사용하면 안 된다.

**이 백업은 서버 내부 복구용이며 공개 제출 패키지가 아니다.** CSV와 실험별 진단
파일도 포함하므로 GitHub에 올리지 않는다. `missing_sources`가 있으면 해당 버전
코드 확보가 추가로 필요하다. 기존 참조 경로가 들어 있어 이동 후 바로 실행되는
`test.py` 패키지로 취급하면 안 된다.

## 3. main 반영 및 캐시 없는 서버 재현 검사

`main`의 `a7c5d64` (PR #9)에 팀원의 `2dae970` 코드가 들어왔고,
`feat/seungbeom-multimodal-tuning`에도 통합했다. 위 브랜치를 pull한 다음
locator를 다시 실행한다. 이전 보고서가 있으면 `final_prior40_location_v2.json`처럼
새 경로를 사용한다. 새 소스가 생기면 이전 보고서의 `missing_sources`도 달라질 수 있다.

제출자가 `[1]` 후보가 실제 최고 파일이라고 확인했고 `artifacts_matched`라면:

```bash
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
tmux new -s seungbeom-final-replay
```

새 tmux 터미널에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/final_prior40_verify.py \
  --location runs/seungbeom/final_prior40_location_v2.json \
  --candidate 1 \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --output_dir runs/seungbeom/final_prior40_replay \
  --device cuda
```

저장된 실제 tau를 그대로 쓰며 학습·tau 탐색·확률 캐시는 사용하지 않는다.
전체 입력 추론과 최대 32행의 역순 부분 입력 검사를 수행한다. 원본 CSV는 예측이
끝난 뒤 일치 검사와 출력 순서 정렬에만 사용한다. `changed=0`,
`reordered subset changed=0`을 확인한다. 원본 CSV 바이트 일치 여부도 별도로 기록한다.
`verification.json`의 `replay_passed`는 재현 검사 통과이며 대회 점수 검증은 아니다.
일부 행 검사는 모든 가능한 부분집합을 검사한 것은 아니다.
추론용 환경이 기존 기록과 다르면 중단하며 체크를 우회하지 않는다.

## 4. 아직 남은 최종 확인

제출자가 원본임을 확인한 뒤, 동일한 가중치·전처리·PCA·이미지 인코더·tau로
위 도구로 캐시 없이 CSV를 다시 생성해 원본 예측과 일치하는지 검사한다.
부분 입력·행 순서 변경에서도 같은 샘플이 같은 결과인지 확인하고, 이 경로를
필수 제출 CLI `python test.py --input ... --image_root ... --output ...`에 연결한다.
현재 저장소 루트의 `test.py`는 초기 모델용이므로 최고 앙상블 실행기로 간주하지 않는다.
원본 위치/생성 기록이 확정되기 전에는 다른 recipe로 패키지를 만들지 않는다.

리더보드에서는 제출자가 **해당 파일의 최종 평가 선택 및 저장**도 확인한다.
이 도구는 리더보드에 접근하거나 최종 선택을 변경하지 않는다.
