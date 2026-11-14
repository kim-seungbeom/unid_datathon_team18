# 기존 신경망 seed 43·44 반복

사용자가 운영진 GPU 서버에서 실행한다. 기존 50:50 제출 파일·모델·recipe는 보관한다.
원본 신경망 run의 `config.json`에서 seed만 바꾼다. ResNet18 고정 벡터, 원본 센서
5개, 전처리, 결합 구조, 최대 epoch, 조기 종료, dropout, weight decay는 그대로다.
이미 저장된 Train/Validation 벡터를 사용하므로 이미지 인코더를 재학습하지 않는다.

기존 seed 42의 CSV·캐시는 반드시 같아야 한다. 코드와 학습 환경까지 같으면 기존
seed 42를 재사용하고 43·44만 학습한다. 코드나 환경이 다르거나 환경 기록이 없으면
새 폴더에서 seed 42도 대조군으로 학습한다. 이 경우 원래 seed 42는 참고값으로만
따로 출력하고, 같은 조건으로 새로 학습한 42·43·44로 평균·표준편차를 계산한다.

## 서버 준비와 tmux

이 clone에서 진행 중인 작업이 끝난 뒤 실행한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new-session -A -s seungbeom-seeds
```

tmux 안에서 폴더와 환경을 다시 지정한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
```

## 설정 생성

```bash
python experiments/seungbeom/neural_seed_repeat.py prepare \
  --reference_run runs/seungbeom/neural_fusion_seed42/embedding_concat_s42_3daac9bb72d0ba69 \
  --data_dir /home/ubuntu/data/battery_data_light \
  --train_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/train \
  --val_embeddings runs/seungbeom/classwise_resnet18_seed42/embeddings/validation \
  --output_dir runs/seungbeom/neural_seed_repeat \
  --device cuda
```

`Prepared seeds: 43, 44` 또는 대조군이 필요한 경우 `42, 43, 44`가 출력된다.
이 단계는 설정과 입력·코드·환경 일치 여부만 확인하고 학습하지 않는다.
폴더가 이미 있으면 덮어쓰지 않는다. 준비까지 성공한 경우 다음 단계부터 이어 간다.

## 순차 학습

```bash
python -u experiments/seungbeom/neural_seed_repeat.py run \
  --experiment_dir runs/seungbeom/neural_seed_repeat
```

GPU에서 한 모델씩 실행한다. `Ctrl+B`를 누르고 손을 뗀 뒤 `D`를 누르면 학습은
계속되고 tmux에서 빠져나온다. 복귀는 `tmux attach -t seungbeom-seeds`다.
학습 중에는 같은 clone에서 pull·브랜치 전환·패키지 변경을 하지 않는다.

같은 명령을 다시 실행하면 동일 조건으로 정상 완료한 run만 건너뛴다. 중단된 run의
가중치부터 재개하는 기능은 없다. 실패한 run이 있으면 결과를 보존하고 새 output_dir로
준비·실행한다. 기존 최고 모델을 덮어쓰는 방식으로 복구하지 않는다.

## 비교

```bash
python experiments/seungbeom/neural_seed_repeat.py compare \
  --experiment_dir runs/seungbeom/neural_seed_repeat
```

seed별 최고 epoch, Macro F1, 1~6단계 F1과 세 seed의 평균·표본표준편차(ddof=1)를
출력한다. 모든 계획이 완료되고 seed 외 설정·CSV·캐시·학습 코드·환경이 같아야 한다.
평균 점수는 각 신경망의 성능 요약이며 예측을 평균한 앙상블 성능이 아니다.
세 seed의 변동성은 새 실험에 대한 신뢰구간도 아니다.

`plan.json`은 전체 설정, `experiment.json`은 참조와 재현 정보, `runs/`는 새 seed의
모델·학습 이력을 보관한다. 신경망 반복 결과를 본 뒤 seed 평균 및 기존 ExtraTrees와의
결합 여부를 결정한다. 이 단계에는 Test 추론이나 기존 제출 후보의 자동 교체가 없다.
