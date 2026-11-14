# 저장된 확률로 25:75·20:80 비교

신경망 : ExtraTrees 비율이다. 현재 기준 30:70과 기존 가중치·CSV는 보존한다.
`inner_best_refine.py`가 만든 Validation/Test 멤버 확률 캐시를 읽어 고정된 두 비율을
계산한다. 추가 학습·모델 로딩·이미지 추론이 없으며, 캐시가 없거나 다르면 중단한다.
파일 해시 검사를 위해 이미지·모델 파일은 읽지만 네트워크 순전파는 하지 않는다.

사람이 운영진 서버에서 실행한다. 캐시·CSV·가중치는 Git에 올리거나 외부 AI에 보내지 않는다.
기존 작업 종료 후 코드와 가상환경 준비:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
set -o pipefail
```

Validation 비교:

```bash
python -u experiments/seungbeom/inner_best_cached_ratios.py compare \
  --baseline_recipe runs/seungbeom/inner_best_refine_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --cache_dir runs/seungbeom/inner_best_probability_cache/validation \
  --output_dir runs/seungbeom/inner_best_lower_ratios_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/inner_best_lower_ratios_seed42.log
```

기존 멤버와 50:50·40:60·60:40·35:65·30:70 지표가 모두 재현되어야 완료된다.
`delta_vs_30_70`은 Validation Macro F1 차이이고, `changed_rows_vs_30_70`은 예측이
달라진 행 수다. 변경 행 수를 정답 개선 수로 해석하지 않는다. Public/Private 점수 상승은
보장하지 않는다. 저장된 결과만 다시 보기:

```bash
python experiments/seungbeom/inner_best_cached_ratios.py report \
  --recipe runs/seungbeom/inner_best_lower_ratios_seed42/recipe.json
```

비교가 오류 없이 완료되면 두 CSV를 생성한다:

```bash
python -u experiments/seungbeom/inner_best_cached_ratios.py predict \
  --recipe runs/seungbeom/inner_best_lower_ratios_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --cache_dir runs/seungbeom/inner_best_probability_cache/test \
  --output_dir runs/seungbeom/submissions_inner_best_lower_ratios_seed42 \
  --device cuda \
  2>&1 | tee runs/seungbeom/inner_best_lower_ratios_predict_seed42.log
```

결과 폴더 `runs/seungbeom/submissions_inner_best_lower_ratios_seed42/`에
`submission_inner_best_25_75.csv`, `submission_inner_best_20_80.csv`와 재현 manifest,
`status.json`이 생긴다. 완료 메시지와 `status.json`의 `complete`를 확인한다.
`Candidate disagreement rows: 0`이면 두 후보의 예측 라벨이 같으므로 하나만 제출한다.
30:70 대비 변경 행 수가 0인 후보도 동일 입력의 기존 30:70과 같은 라벨을 낸다.

`--device cuda`는 원래 캐시의 추론 환경 확인용이며 모델 추론을 시작하지 않는다.
출력 폴더가 존재하면 `_v2` 같은 새 이름을 사용한다. 캐시는 읽기 전용으로 재사용한다.
캐시가 없으면 [앞선 실행 안내](INNER_BEST_REFINE.md)의 명령을 새 출력 폴더로 실행해
생성해야 하며, 그 경우에만 이미지 추론이 필요하다.

이 도구는 기존 확률 재결합용이다. 일부 입력에 전체 Test 캐시를 조회해 사용하지 않는다.
입력 CSV·행 순서·이미지 내용이 달라지면 해당 입력으로 만든 별도 캐시가 필요하다.
Test 실행은 Train/Validation CSV 및 Validation 확률 캐시를 읽지 않는다.
