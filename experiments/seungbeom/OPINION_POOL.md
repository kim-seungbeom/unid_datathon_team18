# Fixed arithmetic / geometric pooling

기존 inner-best 신경망과 원본 ExtraTrees, 전처리 및 최고 제출 파일을 보존한다.
학습 없이 기존 Train OOF / Validation 확률 캐시에서 두 후보를 비교한다.

- 기준: `A = 0.3 * p_neural + 0.7 * p_tree`
- `geometric_30_70`: `G = normalize(exp(0.3*log(p_neural) + 0.7*log(p_tree)))`
- `arithmetic_geometric_half`: `0.5*A + 0.5*G`

로그 계산 전 각 확률을 최소 `1e-12`로 고정한다. Temperature와 prior 보정은
새 후보에 사용하지 않는다. 모든 비율과 clipping은 비교 전에 고정하며
OOF/Validation/Public/Test 결과로 코드가 자동 조정하지 않는다. 기하평균은 한
모델이 거의 0을 주면 해당 클래스 점수를 크게 낮출 수 있어 개선을 보장하지 않는다.
관련 연구: [Heskes (1997)](https://papers.neurips.cc/paper_files/paper/1997/hash/59f51fd6937412b7e56ded1ea2470c25-Abstract.html).

OOF는 기존 proxy head와 이전 epoch 선택을 재사용한다. 새 독립 검증이 아니며,
fold 평균과 pooled OOF F1은 다른 값이다. Validation 역시 반복 사용한 비교 자료다.
`diagnostic_improved`는 pooled OOF 기준일 뿐 제출 채택이나 Public 개선 보장이 아니다.
기하평균과 완만한 혼합 후보가 모두 OOF/Validation에서 나빠지면 제출을 보류한다.

## 서버 실행

다른 작업이 실행 중이지 않은 개인 clone에서 코드를 받은 뒤 실행한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-opinion-pool
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
set -o pipefail
python -u experiments/seungbeom/opinion_pool.py compare \
  --baseline_recipe runs/seungbeom/prior30_validation_seed42/recipe.json \
  --oof_dir runs/seungbeom/prior30_oof_seed42 \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/opinion_pool_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/opinion_pool_seed42.log
```

`--device cuda`는 기존 캐시 환경 검증용이다. 새 이미지 추출, 신경망/트리 학습,
파라미터 최적화를 하지 않는다. 이미지/모델 해시 검증에 I/O 시간이 들 수 있다.

```bash
python experiments/seungbeom/opinion_pool.py report \
  --recipe runs/seungbeom/opinion_pool_seed42/recipe.json --folds
```

선택적으로 CSV를 생성한다. 직접 제출하지 않으며 기존 파일을 덮어쓰지 않는다.

```bash
python -u experiments/seungbeom/opinion_pool.py predict \
  --recipe runs/seungbeom/opinion_pool_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --cache_dir runs/seungbeom/inner_best_probability_cache/test \
  --output_dir runs/seungbeom/submissions_opinion_pool_seed42 \
  --device cuda
```

출력 파일은 `submission_geometric_30_70.csv`,
`submission_arithmetic_geometric_half.csv`다. raw 30:70, prior30 또는 서로 간에
예측 라벨이 완전히 같은 후보는 건너뛴다. 전체 과거 제출물과의 중복 검사는 아니다.
Test에서는 고정된 수식만 적용한다. 정답이나 Test 분포에 맞춰 설정을 학습하지 않는다.
재실행 시 새 output_dir을 사용한다.

현재 실행기는 원본 전체 입력과 정확히 일치하는 캐시를 요구한다. 행 단위 결합
함수의 부분 입력/순서 불변성은 합성 입력으로 검사하지만, 최종 제출용 `test.py`에서
임의 CSV·이미지를 직접 추론하는 경로는 선택 모델 확정 후 별도 연결/검증해야 한다.
