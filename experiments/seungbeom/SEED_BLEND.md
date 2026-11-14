# 신경망 seed 결합과 ExtraTrees 비교

기존 50:50의 원래 신경망·ExtraTrees·이미지 계산 설정을 보존하고, 학습이 완료된
seed 반복 실험에서 두 후보를 추가한다. 새로운 학습이나 가중치 최적화는 없다.

| 출력 이름 | 결합 |
| --- | --- |
| `baseline_50_50` | 원래 신경망 50% + ExtraTrees 50% |
| `seed44_50_50` | seed 44 신경망 50% + 같은 ExtraTrees 50% |
| `seed_mean_50_50` | seed 42·43·44의 예측 확률 평균 50% + 같은 ExtraTrees 50% |

세 seed 평균에서는 신경망 하나당 최종 비중이 1/6이고 ExtraTrees는 1/2이다.
로짓·라벨·F1을 평균하지 않는다. 새 seed 42 대조군을 학습한 실험이라면 평균에는 그
새 모델이 들어간다. 기준 후보는 계속 원래 체크포인트를 사용한다.

## 서버에서 Validation 비교

이 clone에서 실행 중인 작업이 끝난 뒤 코드를 받는다. 실제 실행은 사용자가 운영진
GPU 서버에서 한다. 이미 tmux 안이라면 현재 창을 사용할 수 있다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new-session -A -s seungbeom-seed-blend
```

tmux 안에서:

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate

python -u experiments/seungbeom/seed_blend.py compare \
  --recipe runs/seungbeom/midpoint_blend_replay_seed42/recipe.json \
  --experiment_dir runs/seungbeom/neural_seed_repeat \
  --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
  --image_root /home/ubuntu/data/battery_data_light/validation \
  --output_dir runs/seungbeom/seed_blend_validation \
  --device cuda
```

신경망·센서 전처리·학습 입력·코드·환경·이미지 계약을 확인하고, 현재 이미지에서
벡터를 한 번만 추출해 모든 모델이 공유한다. 원래 신경망, ExtraTrees, 기존 50:50,
각 seed 신경망의 전체 집계 지표와 혼동행렬이 기존 기록과 같아야 완료된다.
이 검사는 원래 개별 예측 전체의 동일성을 증명하는 것은 아니다. 실패 시 우회하지 않는다.

세 후보의 Macro F1, 기존 50:50 대비 변화, 단계별 F1을 출력한다. 개별 신경망과
확률 평균 신경망의 단독 결과도 보려면 `compare` 또는 `report`에 `--members`를 추가한다.

```bash
python experiments/seungbeom/seed_blend.py report \
  --recipe runs/seungbeom/seed_blend_validation/recipe.json
```

`report`는 저장된 결과만 읽는다. `Ctrl+B` 후 `D`로 tmux에서 나와도 실행은 계속되며,
복귀는 `tmux attach -t seungbeom-seed-blend`다.

## 비교 완료 후 별도 후보 CSV 생성

```bash
python -u experiments/seungbeom/seed_blend.py predict \
  --recipe runs/seungbeom/seed_blend_validation/recipe.json \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output_dir runs/seungbeom/submissions_seed_blend \
  --candidates seed44_50_50 seed_mean_50_50 \
  --device cuda
```

`submission_seed44_50_50.csv`, `submission_seed_mean_50_50.csv`와 재현용 manifest가
새 폴더에 생성된다. `status.json`이 `complete`인지 확인한다. 후보 하나만 지정해도 된다.
기존 후보의 별도 재생성은 `--candidates baseline_50_50`으로 한다.

Test 추론은 현재 입력 행의 이미지·센서만 사용하고 Train/Validation CSV·이미지 캐시는
읽지 않는다. 준비된 모델과 메타데이터 파일은 유지해야 한다. 출력 폴더를 덮어쓰지 않으므로
재실행할 때는 새 경로를 사용한다. 비교에 사용한 Validation 점수는 독립적인 최종 성능
추정치가 아니며, 제출과 최종 후보 선택은 사용자가 한다.
