# 원본 CSV 없이 저장된 40:60+prior 모델로 제출 CSV 만들기

이 경로에는 예전 제출 CSV와 그 `.manifest.json`이 필요 없다.
**저장된 correction.json과 inner_best_ratios의 recipe.json, 그 recipe가 참조하는
가중치·전처리/PCA·실행 기록은 필요하다.** 이들까지 없으면 코드만으로 복원할 수 없다.
원본 생성 기록 없이 고른 모델/보정 조합이 기존 최고 파일의 조합과 같은지는 증명할 수 없다.

새 파일은 `submission_inner_best_40_60_prior_tau_regenerated.csv`다.
40:60과 저장된 tau/prior를 그대로 사용하고 학습·보정값 탐색·OOF 접근을 하지 않는다.
원본 CSV와 일치 또는 Public 0.9093891677 재현으로 표시하지 않는다.

## 서버에서 코드 받기

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
```

## CSV 대신 보정 설정과 모델 recipe 찾기

```bash
python experiments/seungbeom/final_prior40_export.py discover \
  --team_home /home/ubuntu \
  --output runs/seungbeom/prior40_saved_artifacts.json
```

현재 및 팀원별 표준 clone의 `runs`만 검색한다. 별도 위치는
`--runs_root /실제/프로젝트/runs`로 추가한다. 원본 제출 CSV는 열지 않는다.
`corrections: 1`, `recipes: 1`이면 아래 명령이 유일한 후보를 사용한다.
바이트가 같은 복사본은 하나로 취급한다. 후보가 여러 개면 최신 시각이나 점수로 고르지 않는다.
생성한 팀원의 확인을 받아 `--discovery ...` 대신 **둘 다** 명시한다:

```text
--correction /실제/correction.json --recipe /실제/40대60비교/recipe.json
```

둘 중 하나라도 0개면 그 저장 파일의 위치가 추가로 필요하다. 30:70 설정을 대신 쓰거나
tau를 추측하지 않는다. 설정은 발견돼도 연결된 가중치가 없으면 추론 시작 전에 중단한다.

검색기는 전체 `correction.json` 발견 수와 제외 이유도 출력한다.
`not_40_60_correction`은 다른 비율의 설정이며,
`generator_source_hash_missing_or_different`는 생성 코드 버전을 확인해야 한다는 뜻이다.
파일이 아예 없는 경우와 검사에서 제외된 경우를 구분하고, 검사 자체를 우회하지 않는다.
기존 검색 보고서가 있으면 `--output runs/seungbeom/prior40_saved_artifacts_v2.json`처럼
새 경로를 사용한다. 사용할 수 있는 40:60 보정 파일이 없다면 이전 모델을 정확히 복원할 수
없다. Train OOF에서 40:60용 보정을 새로 학습하는 작업은 별도의 새 실험이며, 과거 최고
제출물의 복원으로 표시하면 안 된다.

## tmux에서 추론

```bash
tmux new -s seungbeom-final40
```

같은 이름이 이미 있으면 `tmux attach -t seungbeom-final40`로 들어간다.
안에서 실행:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/final_prior40_export.py predict \
  --discovery runs/seungbeom/prior40_saved_artifacts.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output_dir runs/seungbeom/final_prior40_regenerated \
  --device cuda
```

이미지에서 직접 추론하므로 확률 캐시 기반 내보내기보다 시간이 걸린다.
전체 결과와 최대 32행 역순 부분 입력의 결과가 같은지 확인한 후에만 CSV를 저장한다.
이는 일부 행 검사이며 모든 부분집합에 대한 증명은 아니다.
템플릿은 sample_id 순서만 사용하고 예시 label은 사용하지 않는다.
기존 모델·파일은 바꾸지 않고 출력 폴더가 이미 있으면 중단한다.

완료 후 파일:

```text
runs/seungbeom/final_prior40_regenerated/submission_inner_best_40_60_prior_tau_regenerated.csv
```

같은 이름의 manifest에 실제 모델/설정 해시, tau, 실행 환경, 부분 입력 검사 결과를
기록한다. 원본 제출물 일치 여부는 false, 새 Public 점수는 null로 남긴다.
새 CSV는 별도 제출 후보이며 기존 리더보드 최고 파일을 자동 교체하지 않는다.

## 코드·가중치 최종 제출 시 남는 작업

이 도구는 현재 서버에 보존된 원본 run 경로를 사용한다. 다른 컴퓨터로 옮겨도
바로 실행되는 제출 패키지는 아니다. 실제 저장 아티팩트를 확보한 뒤 필요한
가중치·전처리·이미지 인코더를 묶고 `test.py --input --image_root --output` 형식으로
연결하는 패키징/검증은 별도로 필요하다. 루트의 초기 모델용 test.py를 이 모델의
추론기로 제출하면 안 된다. 원본 CSV가 없어도 이 패키징 작업은 가능하다.
