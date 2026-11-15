# 기존 OOF로 40:60 조합의 tau 새로 선택하기

신경망:원본 ExtraTrees 비율은 **40:60 고정**이다. 기존 prior30 OOF에 들어 있는
두 모델의 예측을 재사용하지만, 합치는 비율과 tau 선택을 새로 계산한다.
OOF 생성·이미지 추론·신경망/트리 재학습은 하지 않는다.
기존 최고 제출물 복원이 아닌 새 실험이다. 원본 40:60 correction.json은 필요 없다.

Train OOF Macro F1 기준으로 tau=0..2, 간격 0.05의 사전 지정 후보를 비교한다.
동점은 작은 tau를 선택한다. 각 fold를 제외한 나머지에서 prior와 tau를 선택하여
제외 fold에 적용한 진단도 출력한다. 기존 OOF proxy와 epoch/구조 선택을 재사용하므로
독립 nested CV가 아니다. `fit_diagnostic`을 일반화 성능으로 해석하지 않는다.
OOF 자체를 좋은 점수가 나오는 분할로 교체하지 않으며 Validation/Test/Public 점수로
tau를 최적화하지 않는다.

## 서버 실행

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate

python -u experiments/seungbeom/inner_best_prior40.py select \
  --baseline_recipe runs/seungbeom/inner_best_refine_seed42/recipe.json \
  --oof_dir runs/seungbeom/prior30_oof_seed42 \
  --train_csv /home/ubuntu/data/battery_data_light/train/train.csv \
  --output_dir runs/seungbeom/prior40_new_selection \
  --device cuda

python -u experiments/seungbeom/inner_best_prior40.py compare \
  --correction runs/seungbeom/prior40_new_selection/correction.json \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/prior40_new_validation \
  --device cuda
```

`compare`는 앞서 `select`가 저장한 tau를 고정해서 적용한다.
`adopt=True`는 제외-fold 진단이 보정 전보다 높았다는 코드 기준이며 유의성 검정이 아니다.
False여도 명시적 predict는 탐색용 CSV를 만들며, 개선 검증이 안 됐다는 사실을 출력한다.
최대 후보 tau=2.0이 선택되면 `edge=True`로 표시하며 자동으로 탐색 범위를 늘리지 않는다.

## CSV 생성

```bash
python -u experiments/seungbeom/inner_best_prior40.py predict \
  --recipe runs/seungbeom/prior40_new_validation/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --cache_dir runs/seungbeom/inner_best_probability_cache/test \
  --output_dir runs/seungbeom/submissions_prior40_new \
  --device cuda
```

출력: `runs/seungbeom/submissions_prior40_new/submission_inner_best_40_60_prior_tau.csv`.
예전 제출 파일과 이름이 같아도 새 폴더에 저장된 새 후보이며, 0.9093891677의 재현을
의미하지 않는다. 기존 모든 모델·캐시·CSV를 보존하며 기존 출력 폴더를 덮어쓰지 않는다.
최종 심사용 코드에는 같은 고정 계산을 실제 모델 추론에 연결하는 별도 패키징이 필요하다.
