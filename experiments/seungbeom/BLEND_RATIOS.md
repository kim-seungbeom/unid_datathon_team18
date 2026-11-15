# 신경망·ExtraTrees 비율 비교

완료된 `midpoint_blend.py` v2 recipe를 사용한다. 기존 모델·전처리·이미지 계산 설정을
유지하고 **신경망:ExtraTrees = 50:50, 75:25, 25:75** 세 비율만 비교한다.
추가 학습과 자동 가중치 최적화는 없다. 비율을 비교해 선택하는 것도 Validation을
이용한 모델 선택이므로 이 점수는 독립적인 최종 성능 추정치가 아니다.

기존 `midpoint_blend.py`와 그 해시 대상 파일은 수정하지 않는다. 원래 제출 CSV,
모델, recipe는 그대로 보관한다. 아래 명령은 사용자가 운영진 GPU 서버에서 실행한다.

## 1. Validation 비교

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate

python -u experiments/seungbeom/blend_ratios.py compare \
  --recipe runs/seungbeom/midpoint_blend_replay_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --output_dir runs/seungbeom/blend_ratios_seed42 \
  --device cuda
```

이미지와 두 모델의 추론은 한 번만 수행하고 같은 확률로 모든 비율을 계산한다.
원래 신경망·ExtraTrees·50:50의 전체 지표와 혼동행렬이 재현돼야 완료된다.
이는 개별 예측 전체의 동일성을 증명하는 검사는 아니다. 실패 시 우회하지 않는다.

전체 Macro F1, 50:50 대비 변화, 단계별 F1을 함께 확인한다. 출력의 비율은 항상
신경망이 먼저다. 기존 코드·모델·패키지·GPU 환경이 바뀌면 중단한다.

저장된 비교표만 다시 출력하려면:

```bash
python experiments/seungbeom/blend_ratios.py report \
  --recipe runs/seungbeom/blend_ratios_seed42/recipe.json
```

## 2. 비교 완료 후 후보 CSV 생성

```bash
python -u experiments/seungbeom/blend_ratios.py predict \
  --recipe runs/seungbeom/blend_ratios_seed42/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output_dir runs/seungbeom/submissions_blend_ratios_seed42 \
  --candidates blend_75_25 blend_25_75 \
  --device cuda
```

출력 폴더에 `submission_blend_75_25.csv`, `submission_blend_25_75.csv`와 각각의
manifest가 생성된다. `status.json`이 `complete`인지 확인한다. 후보 하나만 지정할
수도 있고, 기존 조합의 재생성을 위해 `blend_50_50`을 지정할 수도 있다.

Test에서는 이미 검증한 비율을 적용하기만 하며 Validation CSV·이미지 캐시를 읽지
않는다. 현재 입력 행의 이미지·센서만 사용한다. `report`는 추론하지 않는다.
기존 출력 폴더가 있으면 덮어쓰지 않으므로 재실행 시 새 경로를 지정한다.
모든 후보가 기준보다 좋아지는 것은 아니며, 제출·최종 선택은 사용자가 수행한다.
