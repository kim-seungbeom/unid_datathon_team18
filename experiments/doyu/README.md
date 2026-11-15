# 도유 실험: 최종 분류기 + 센서·이미지 결합

팀 공식 진입점(루트 `model.py`, `train.py`, `test.py`)과 별개인 개인 실험 코드입니다.

## 결합 구조 (`train_fusion.py`)

```
CSV 행 ─→ [승범] sensor_features.build_sensor_features (config의 sensor_features) ─┐
                                                                                   ├→ [도유] 결측 대체 → XGB / LGBM / MLP → 1~6
이미지 ─→ [성현] image_features 고정 인코더 → 벡터 → 표준화 → PCA ─────────────────┘
```

- **센서:** 승범님 `configs/*.json`의 `sensor_features`·`sensor_imputation`을 그대로 사용합니다(`--config`). 같은 실험의 현재·과거 행만 쓰며 fit이 없습니다.
- **이미지:** 성현님 추출 결과 폴더(`features.npy`, `sample_ids.csv`, `metadata.json`)를 `sample_id`로 연결합니다. 폴더를 주지 않으면 같은 `extract()` 함수로 즉시 추출합니다. 미리 뽑은 벡터와 즉시 계산한 벡터가 같은지는 테스트로 확인합니다.
- **누수 방지:** 결측 대체·표준화·PCA는 sklearn Pipeline 안에 있어 CV의 각 fold train 부분, 또는 최종 Train에서만 fit됩니다. 학습된 변환은 joblib에 함께 저장되어 test에 그대로 적용됩니다.
- **호환성 검사:** 저장 bundle에 센서 특징 버전, 시계열 버전, 인코더 가중치, 전처리, `l2_normalize`가 기록됩니다. 추론할 때 이 값이 다르면 오류를 냅니다.
- **규정:** `docs/rules_review.md`의 OT 반영 내용에 따라 같은 실험의 현재·과거 관측은 사용할 수 있습니다. 제출 전에는 `check_submission.py` 기본 엄격 모드로 확인하세요.

## 기존 실험

| 파일 | 내용 |
| --- | --- |
| `models/` | XGBoost / LightGBM / MLP / Cross-Attention 정의, `ShiftLabels`(0..5 ↔ 1..6) |
| `train_compare.py` | 운영진 v7 33개 특징으로 3종 비교 → `best_{모델}.joblib` |
| `train_cross_attention.py` | 64×64 이미지 + 과거 센서 시퀀스 Cross-Attention |
| `predict.py` | `train_compare.py` 모델로 제출 CSV 생성, 정답이 있으면 지표 계산 |
| `dataana.py` | validation 오분류 분석. `train_compare`와 `train_fusion` 모델을 모두 지원 |
| `organizer_v7_model.py` | `baselines/organizer_v7/model.py`를 수정 없이 별도 이름으로 로드 |

## 실행 (레포 루트, 서버에서 사용자가 직접)

```bash
DATA=/home/ubuntu/data/battery_data_light
EMB=<성현님 추출 결과 상위 폴더>   # 예: .../outputs/resnet18  (train/, validation/ 포함)

# 1) 결합 학습·비교 (인코더·센서 설정마다 새 out_dir)
python experiments/doyu/train_fusion.py --data_dir $DATA \
  --config configs/sensor_history_voltage.json --encoder resnet18 \
  --train_embeddings $EMB/train --val_embeddings $EMB/validation \
  --out_dir runs/doyu/fusion/resnet18_voltage

#    센서만 비교 기준: --encoder none (임베딩 인자 생략)
#    PCA 차원: --pca_dim 32 (기본), 0이면 PCA 없이 표준화만

# 2) 모델별 validation 분석
for m in xgboost lightgbm mlp; do
  python experiments/doyu/dataana.py --data_dir $DATA \
    --model runs/doyu/fusion/resnet18_voltage/best_${m}.joblib \
    --embeddings $EMB/validation --out_dir runs/doyu/dataana/resnet18_voltage/${m}
done

# 3) test 예측 (+정답이 있으면 precision/recall/f1). 이미지 벡터는 즉시 추출
for m in xgboost lightgbm mlp; do
  python experiments/doyu/predict_fusion.py --input $DATA/test/test.csv --image_root $DATA/test \
    --model runs/doyu/fusion/resnet18_voltage/best_${m}.joblib \
    --output outputs/doyu/fusion/resnet18_voltage/${m}/submission.csv
done
```

- **DINOv2:** `transformers`가 필요합니다. 공용 venv에는 없으므로, 즉시 추출하려면 성현님 개인 환경처럼 패키지가 있는 환경에서 실행하세요. 미리 추출한 폴더를 쓰면 필요 없습니다.
- **사전학습 가중치:** ImageNet, DINOv2 같은 외부 사전학습 가중치는 대회 허용 여부를 확인한 뒤 사용하세요.
- **refit 모델:** `--refit_train_val` 모델은 validation을 학습에 썼으므로 `dataana.py` 분석에 쓰지 않습니다.
- **결과 위치:** 결과 폴더 `runs/`, `outputs/`는 Git 제외 대상입니다.
