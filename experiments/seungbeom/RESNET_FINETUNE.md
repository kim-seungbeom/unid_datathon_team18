# ResNet18 마지막 블록 미세조정

기존 모델·50:50 recipe·제출 CSV를 보존하고, 개인 실행 경로에서 두 모델을 순서대로
학습한다. `resnet_finetune.py`, `resnet_finetune_core.py`, `resnet_layer4.json`만 사용하며
기존 모델 소스와 체크포인트 형식은 변경하지 않는다.

| 항목 | frozen | layer4 |
| --- | --- | --- |
| 이미지 | ImageNet ResNet18 고정 | 같은 초기 가중치, 마지막 `layer4`만 학습 |
| 센서 | 현재 행 원본 5개, 중앙값 대체·표준화 | 동일 |
| 결합 | 이미지 512차원 + 센서 MLP → concat 분류기 | 동일 |
| BatchNorm | 모든 running mean/variance 고정 | 동일, layer4의 scale/bias는 학습 |
| 학습률 | 분류기·센서 MLP `1e-4` | 분류기·센서 MLP `1e-4`, layer4 `1e-5` |

두 모델은 초기 가중치, seed 42, batch 32, dropout 0.1, weight decay 1e-4,
일반 CE와 shuffle을 공유한다. 이미지 증강은 이번 비교에 추가하지 않는다.
기존 인코더와 같은 전체 이미지 224 letterbox·ImageNet 정규화를 사용한다.
최대 20 epoch, patience 5는 시작 설정이며 성능 개선을 보장하는 값이 아니다.

Train의 첫 `StratifiedGroupKFold(5, shuffle=True, random_state=42)` 분할을 고정한다.
각 모델은 Train 내부의 실험별 holdout에서 epoch 수만 선택한다. 내부 선택 모델을
버린 뒤 같은 초기 가중치에서 전체 Train으로 그 횟수만큼 새로 학습한다.
전처리도 각 단계의 학습 부분에서만 fit한다. 공식 Validation은 학습·epoch 선택에
사용하지 않고 최종 비교와 저장 모델 재현 검사에만 사용한다. 이것은 3-fold CV가
아니며, 공식 Validation은 이전 실험 설계에도 사용했으므로 새 blind test가 아니다.

`historical_neural`은 이전 고정형 모델의 저장 점수다. 새 실험과 CSV 해시는 같지만
학습률·epoch 선택 절차 등이 다르므로 **미세조정 효과는 새 `layer4`와 새 `frozen`의
차이로 판단**한다. 과거 점수와의 차이 전체를 미세조정 효과로 해석하지 않는다.
이번 비교에는 ExtraTrees를 새로 학습하거나 50:50을 새로 만드는 단계가 없다.

## 서버 실행

같은 clone에서 학습 중인 작업이 없을 때 코드를 받는다. 개인 환경은 그대로 사용한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
git status --short
git fetch origin
git switch feat/seungbeom-multimodal-tuning
git pull --ff-only origin feat/seungbeom-multimodal-tuning
tmux new -A -s seungbeom-resnet-ft
```

이미 tmux 안이라면 새 tmux를 중첩하지 말고 현재 창에서 실행한다.

```bash
cd /home/ubuntu/seungbeom/unid_datathon_team18
source /home/ubuntu/seungbeom/venv-ensemble/bin/activate
mkdir -p runs/seungbeom
set -o pipefail

LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u experiments/seungbeom/resnet_finetune.py run \
  --reference_run runs/seungbeom/neural_fusion_seed42/embedding_concat_s42_3daac9bb72d0ba69 \
  --data_dir /home/ubuntu/data/battery_data_light \
  --config experiments/seungbeom/resnet_layer4.json \
  --output_dir runs/seungbeom/resnet_layer4_seed42 \
  --device cuda \
  --resume \
  2>&1 | tee -a runs/seungbeom/resnet_layer4_seed42.log
```

`frozen` 선택 → 전체 Train 재학습 → Validation 평가/재현 → `layer4` 같은 절차로
진행한다. 최초에는 Train/Validation 이미지 지문을 계산한다. 실제 이미지를 매번
읽으므로 기존 캐시 벡터만 사용한 MLP 학습보다 오래 걸린다. 예측 중에는 이미지
처리 개수, 학습 중에는 처리 행 수와 epoch 결과를 표시한다.

출력 폴더는 프로그램이 생성한다. `--resume`은 완료 모델을 해시 확인 후 재사용하고,
실패·중단된 모델은 새로운 attempt 폴더에서 처음부터 다시 학습한다. epoch 중간부터
재개하는 기능은 아니다. 소스·환경·입력·설정이 바뀌면 새 output_dir가 필요하다.
실행 중인 clone에서 pull하지 않는다.

분리: `Ctrl+b`를 누르고 뗀 다음 `d`. 돌아오기:

```bash
tmux attach -t seungbeom-resnet-ft
```

완료 후:

```bash
python experiments/seungbeom/resnet_finetune.py report \
  --output_dir runs/seungbeom/resnet_layer4_seed42 \
  --class_details \
  | tee runs/seungbeom/resnet_layer4_seed42/comparison.txt
```

처음 세 행은 `historical_neural`, `frozen`, `layer4`이고, 각각의 Macro F1·클래스별
F1·선택 epoch를 표시한다. `delta_vs_matched_frozen`은 새 고정형 대조군 대비 차이다.
`layer4`의 단일 점수가 높아도 Public/Private 상승이나 기존 50:50보다 우수함을
보장하지 않는다. 새 후보를 비교할 때 기존 50:50은 그대로 남긴다.

## 저장·추론 계약

각 모델의 attempt 폴더에 전체 ResNet18 가중치를 포함한 `model.pt`, 결과·history·
체크섬을 저장한다. 저장 직후 offline 복원으로 Validation 확률을 정확히 재현해야
완료로 기록한다. `report` 마지막의 `layer4 model_dir`가 해당 모델 폴더다.

별도 추론 CLI는 다음 형태다. 실제 Test 추론은 결과 검토 후 서버에서 실행한다.

```bash
python experiments/seungbeom/resnet_finetune.py predict \
  --run_dir <report에_나온_layer4_model_dir> \
  --input <입력_CSV> --image_root <그_CSV의_이미지_폴더> \
  --output <새_submission.csv> --device cuda
```

선택 인자 `--sample_submission <양식_CSV>`를 주면 정확히 같은 ID 집합인지 확인한 뒤
양식 순서로 저장한다. 없으면 입력 순서로 저장한다. 각 샘플은 한 행씩 같은 수치
경로로 추론하므로 입력 배치·순서·부분집합에 영향을 받지 않도록 구성했다.
센서 전처리 재학습, Test 적응, 이미지 가중치 다운로드, Train/Validation 재접근은
하지 않는다. 소스·실행 환경이 저장 시점과 다르면 중단한다. root `test.py` 최종
제출 포장은 아직 이 개인 실험 CLI에 연결하지 않았다.

사전학습 출처는 [torchvision ResNet18 IMAGENET1K_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)이다.
Torchvision 버전은 실행 metadata의 runtime에 기록한다.
[Torchvision 코드 라이선스](https://github.com/pytorch/vision/blob/main/LICENSE)는 BSD-3-Clause이며,
가중치·사전학습 데이터 조건과 구분해 기록한다. 기존 [이미지 인코더 안내](../../image_features/README.md)도 참고한다.
실제 학습·추론은 주최 측 서버에서 사용자가 수행하며, 로컬 테스트는 독립 합성 입력의
코드 검증이다. 합성 입력의 점수를 실제 대회 Validation 성능으로 사용하지 않는다.
