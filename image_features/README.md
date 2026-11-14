# 고정 이미지 encoder로 특징 추출하기

이미지 모델은 학습하지 않아. 현재 이미지 한 장을 벡터 하나로 바꾸고,
팀원이 센서 특징과 연결할 수 있도록 sample_id와 원래 행 순서를 보존해.
label과 센서는 encoder 입력이나 특징 파일에 포함하지 않아.

| 실행 파일 | encoder | 이미지당 특징 |
|---|---|---:|
| `extract_resnet18.py` | ResNet18, IMAGENET1K_V1 | 512 |
| `extract_dinov2.py` | DINOv2 ViT-S/14, facebook/dinov2-small | 384 |
| `extract_convnext.py` | ConvNeXt-Tiny, IMAGENET1K_V1 | 768 |

`01_extract_features.ipynb`를 열면 각 모델을 별도 셀에서 실행할 수 있어.
기존 분류 실험은 수정하지 않았고, 이 폴더 전체를 서버 개인 폴더에 복사하면 돼.
노트북은 서버의 개인 Python 커널을 선택해. GPU 추출은 기본 CUDA를 사용해.

## 노트북 없이 모델별로 실행

아래 명령은 **서버 터미널**에서, 이 폴더로 이동한 다음 사용해.
CSV가 있는 split 폴더를 image_root로 넣으면 `images/...png` 경로를 연결해.
이미지 폴더 자체를 넣어도 파일명으로 연결할 수 있어.

```bash
source /home/ubuntu/seonghyeon/activate.sh
cd /home/ubuntu/seonghyeon/image_features

python extract_resnet18.py --input "$DATA_DIR/train/train.csv" --image_root "$DATA_DIR/train" --output outputs/resnet18/train
python extract_dinov2.py --input "$DATA_DIR/train/train.csv" --image_root "$DATA_DIR/train" --output outputs/dinov2/train
python extract_convnext.py --input "$DATA_DIR/train/train.csv" --image_root "$DATA_DIR/train" --output outputs/convnext_tiny/train
```

Validation은 입력 경로와 출력 마지막 부분을 `validation`으로 바꾸면 돼.
Test도 동일한 입력 형식으로 추출할 수 있어. 어떤 split에서도 모델을 학습하거나 전처리 통계를 fit하지 않아.
CLI는 이미 있는 출력 폴더를 덮어쓰지 않으므로 새 폴더를 지정해.
노트북의 순차 모델별 셀은 재실행 시 `train_retry_<임의코드>`처럼 새 split 폴더를 자동 선택하고
실제 저장 위치를 출력해. 이전 완료·실패 결과는 그대로 보존해. 이어받기나 기존 결과 재사용은 아니야.

## 세 모델을 한 번에 실행 / 병렬 실행

기본은 **모델 순차 처리**야. GPU 한 장을 효율적으로 활용하고 메모리 부족을 피하기 위한 기본값이야.
한 모델을 처리하는 동안 PNG 읽기·전처리는 DataLoader worker 4개가 병렬 처리해.

```bash
python extract_all.py --input "$DATA_DIR/train/train.csv" --image_root "$DATA_DIR/train" --output outputs/all_train
```

동시 모델 실행도 지원해. 아래는 최대 두 모델을 별도 프로세스에서 실행하고 끝난 자리에 다음 모델을 넣어.
**동시에 실행한다고 반드시 빨라지는 건 아니야.** 같은 GPU의 연산·메모리와 디스크를 공유하므로
실제 여유 메모리와 처리 시간을 비교해서 선택해. 다른 팀원의 작업은 건드리지 마.

```bash
python extract_all.py --input "$DATA_DIR/train/train.csv" --image_root "$DATA_DIR/train" --output outputs/parallel_train --parallel 2 --batch_size 16 --workers 2
```

`--parallel 3`도 가능하지만 기본 권장은 아니야. 메모리 부족이 나면 성공한 모델의 결과는 보존되고,
실패한 모델만 **새 출력 경로**에 더 작은 batch로 실행해. 자동 재시작·이어받기는 하지 않아.
긴 작업은 개인 tmux 세션에서 실행해. 커널 문제가 있는 경우 위 CLI를 사용하면 노트북 연결을 우회할 수 있어.

## 고정되는 것과 전처리

- optimizer, backward, loss, epoch가 없어. 파라미터는 requires_grad=False이고 eval/inference_mode를 사용해.
- BatchNorm 통계와 dropout/stochastic depth도 추론 모드로 고정해.
- ResNet은 fc를 제외하고, ConvNeXt는 마지막 Linear만 제외해 LayerNorm을 유지해.
- DINOv2는 마지막 hidden state의 CLS token을 사용해. patch 평균이나 분류 확률이 아니야.
- 공통 전처리: RGB → 비율 유지 bilinear letterbox 224×224 → 패딩 RGB (124,116,104) → 0~1.
- mean (0.485,0.456,0.406), std (0.229,0.224,0.225)로 정규화해.
- 전체 장면을 보존하려고 공식 가중치의 기본 center crop 대신 letterbox를 사용했어. 모델 비교 시 이 차이를 기록해.
- 224는 DINOv2 패치 크기 14로 나누어떨어져. 세 모델은 같은 공간 범위의 이미지를 받아.
- 색/밝기를 실제 온도로 환산하지 않아. 증강이나 이미지별 대비 보정은 없어.
- 기본은 FP32 특징, L2 정규화 없음. 필요하면 `--l2_normalize`를 명시해.

## 결과를 팀원에게 전달하기

각 출력 폴더에는 아래 세 파일이 생겨. `metadata.json`의 status가 complete인지 먼저 확인해.

- `features.npy`: N×D float32 배열. 대용량이므로 메모리 매핑으로 저장해.
- `sample_ids.csv`: 특징의 각 행과 대응하는 sample_id. CSV 입력 순서를 그대로 유지해.
- `metadata.json`: 모델·가중치·전처리·차원·버전·실제 소요 시간·ID 순서 해시.

팀원은 sample_id로 센서 데이터에 연결해. 세 모델 출력의 ID 순서 해시도 비교할 수 있어.
벡터 자체를 채팅이나 외부 서비스에 업로드하지 마. `features.partial.npy`는 실패·중단된 결과야.
완료된 결과만 사용하고, 기존 실패 폴더는 자동으로 삭제하거나 덮어쓰지 않아.

```python
import numpy as np

features = np.load("outputs/resnet18/train/features.npy", mmap_mode="r", allow_pickle=False)
print(features.shape)  # 숫자 벡터 전체를 출력하지 않아도 돼.
```

최종 추론 코드는 필요하면 이 encoder를 불러 입력 이미지에서 직접 벡터를 계산할 수 있어.
다른 모델과 비교할 때는 학습 데이터에서만 scaler/PCA 등을 fit하고 검증 데이터에는 적용만 해.

## 패키지와 출처

기존 서버 torch/torchvision은 유지해. DINOv2용 transformers와 safetensors가 추가로 필요해.
필요 패키지 목록은 requirements.txt이고, 설치는 개인 가상환경에서만 수행해.
첫 실행은 공개 사전학습 가중치를 내려받으며 이후 캐시를 사용해. 실제 이미지는 외부로 전송하지 않아.

- [ResNet18 공식 가중치](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18)
- [ConvNeXt-Tiny 공식 가중치](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny)
- [DINOv2-Small](https://huggingface.co/facebook/dinov2-small), 고정 revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
- DINOv2는 해당 저장소 표기 Apache-2.0. Torchvision 코드는 BSD-3-Clause이고 가중치·사전학습 데이터 조건은 별도 확인해.

## 로컬 합성 테스트

```bash
python -B -m unittest discover -s . -p test_features.py -v
```

테스트는 임의 가중치와 시스템 TEMP의 합성 이미지만 사용해. 다운로드나 실제 대회 데이터 처리가 없어.
기본 CLI는 항상 사전학습 가중치를 사용해. 테스트의 pretrained=False를 실제 특징 추출에 사용하면 안 돼.
병렬 GPU의 실제 속도·메모리와 실제 데이터 결과는 실행 전에는 확인된 것으로 간주하지 않아.
