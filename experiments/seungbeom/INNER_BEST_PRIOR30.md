# 30:70 + Train OOF prior correction

기존 제출 모델은 보존한다. 로컬은 코드/합성 CPU 테스트만, 실제 데이터 실행은 제공 서버에서 사람이 한다.
팀원 브랜치 전체 머지는 필요 없다. `prior_oof_reference.py`는 도유의 `2dae970` 커밋
`experiments/doyu/prior_correction.py` 중 캐시 기반 OOF head 생성 부분을 출처를 명시해 가져왔다.
클래스 보정은 별도 `inner_best_prior30.py`에 구현한다.

- 고정 결합: `p = 0.3 * inner_best + 0.7 * original ExtraTrees`.
- 최종 클래스: `argmax(p / Train_class_prior**tau) + 1`.
- tau: Train OOF에서 0..2를 0.05 간격으로 선택, 동률은 작은 값 우선.
- Test 분포/라벨/리더보드는 tau 선택 입력이 아니다. Test에서 재학습하지 않는다.
- OOF가 없으면 기존 고정 ResNet18 임베딩과 트리 OOF를 재사용해 fold별 신경망 head만 학습한다.
  원본 이미지 추출/원본 제출 모델 재학습은 하지 않는다. 원본 모델/캐시 경로는 기존 recipe에서 읽는다.
- head는 실제 inner_best의 대체 모델이다. 기존 내부 선택 epoch도 재사용하므로 이 결과는
  완전한 중첩 CV나 새로운 독립 성능 추정이 아니다. 다른 fold로 tau를 정한 진단이 무보정보다
  나아야 `adopt=true`가 된다. 이 진단도 기본 OOF 모델을 재학습하지 않으므로 독립 nested CV가 아니다.
- 보정 점수는 확률로 보정된 값이 아니다. 각 행에 고정 prior/tau를 적용하므로 행별 연산이다.
- compare/predict는 기존 확률 캐시를 검증해 사용한다. 캐시가 없거나 일치하지 않으면 중단한다.
  현재 캐시 CLI는 동일한 전체 입력만 허용한다. 일부 입력 재현은 기존 모델 추론 경로로 확률을
  만들고 동일한 고정 보정을 적용하는 제출 진입점에서 별도 검증해야 한다.
- 이 파일은 실험 도구다. 최종 제출 전 공식 `test.py` CLI에 고정 보정/가중치를 연결하고,
  부분 입력·이미지 교체·모델 아티팩트 재현 검증을 마친다.

## 서버 실행

실행 중인 학습과 같은 checkout에서 pull하지 않는다. 별도 기존 작업이 끝난 뒤 진행한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short --branch
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-prior30
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail
```

OOF 파일은 Git으로 전달되지 않는다. 이번 도구의 OOF를 처음 한 번 생성한다.
기존 tree run의 `train_oof.npz`, `embeddings/train/{features.npy,metadata.json,sample_ids.csv}`가 필요하다.

```bash
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/inner_best_prior30.py oof \
  --baseline_recipe runs/seungbeom/inner_best_refine_seed42/recipe.json \
  --data_dir /home/ubuntu/data/battery_data_light \
  --output_dir runs/seungbeom/prior30_oof_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/prior30_oof_seed42.log
```

완료 후 선택한다. 이 단계는 신경망을 학습하지 않는다.

```bash
python -u experiments/seungbeom/inner_best_prior30.py select \
  --baseline_recipe runs/seungbeom/inner_best_refine_seed42/recipe.json \
  --oof_dir runs/seungbeom/prior30_oof_seed42 \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --output_dir runs/seungbeom/prior30_selection_seed42 \
  --device cuda
```

`adopt=false`면 기존 30:70을 유지한다. `adopt=true`면 고정 tau로 Validation을 비교한다.
`at_grid_edge=true`는 탐색 경계 선택이라는 뜻이며 임의로 범위를 계속 넓히지 않는다.

```bash
python -u experiments/seungbeom/inner_best_prior30.py compare \
  --correction runs/seungbeom/prior30_selection_seed42/correction.json \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/prior30_validation_seed42 \
  --device cuda
```

기존 30:70 재현 성공 후 Macro F1, 클래스별 F1, 3→2/5→4와 역방향 오류를 비교한다.
Validation 점수로 Public 점수를 예상하지 않는다. 다음 명령은 고정된 보정 후보 하나만 내보낸다.

```bash
python -u experiments/seungbeom/inner_best_prior30.py predict \
  --recipe runs/seungbeom/prior30_validation_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --cache_dir runs/seungbeom/inner_best_probability_cache/test \
  --output_dir runs/seungbeom/submissions_prior30_seed42 \
  --device cuda
```

생성 파일: `runs/seungbeom/submissions_prior30_seed42/submission_inner_best_30_70_prior_tau.csv`.
CSV/가중치/확률 캐시/데이터가 포함된 로그는 Git에 올리지 않는다. 기존 결과 폴더는 덮어쓰지 않는다.
