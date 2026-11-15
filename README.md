# Uni-DTHON Team 18 최종 블렌드 제출

센서 기반 HGB와 센서·열화상 ResNet18 임베딩 기반 HGB의 확률을 50:50으로 합친다. 저장된 PCA와 모델만 불러오며 추론 중 학습하지 않는다.

```bash
python test.py --input <CSV> --image_root <이미지 폴더> --output <저장 경로>
```

추가 옵션 없이 `weights/`의 두 모델과 `blend.json`을 사용한다. 기본 추론 장치는 CPU이며 대회 제공 서버에서는 `--device cuda`도 가능하다. CSV에 주어진 행과 해당 이미지만 읽고, 결과는 입력 순서의 `sample_id,label`로 저장한다. 사전 계산된 Test 피처는 필요 없다.

## 설치와 환경

```bash
python -m pip install -r requirements.txt
```

이미 설정된 대회 서버 환경을 우선 사용한다. torch·CUDA를 임의 교체하지 않는다. Python·OS·패키지 등 저장 당시 환경은 각 `weights/member_*/config.json`에 기록되어 있다. 저장 메타데이터의 환경은 Windows / Python 3.14.2 / numpy 2.5.4 / pandas 3.1.1 / scikit-learn 1.9.1 / joblib 1.6.0 / torch 2.15.0+cpu / torchvision 0.29.2+cpu / Pillow 13.0.0이다. joblib 모델 호환성을 위해 scikit-learn은 저장 당시 버전으로 고정하고 torch는 버전을 명시하지 않는다. 이번 정리 작업에서는 대회 서버의 현재 OS·Python·GPU를 직접 확인하지 않았다.

이미지 인코더는 torchvision의 `ResNet18_Weights.IMAGENET1K_V1`이다. 첫 실행 시 공식 공개 가중치를 내려받으므로 인터넷 연결 또는 torchvision 캐시가 필요하다. 대회 데이터는 외부로 전송하지 않는다. 이 공개 모델은 대회 이전 공개된 ImageNet 모델이며, 인코더는 동결하고 eval 모드로 사용한다. 전체 프레임 224 letterbox와 ImageNet 정규화를 적용한다.

출처: [ResNet18 공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [torchvision BSD-3-Clause 라이선스](https://github.com/pytorch/vision/blob/main/LICENSE). 소프트웨어 라이선스와 가중치·원 학습 데이터의 사용 조건은 구분한다.

## 저장소 구성

최종 제출 진입점은 저장소 루트의 `test.py`이며 기본 가중치 경로는 `weights/`다. 복원된 `experiments/doyu/`, `experiments/seungbeom/`, `configs/`, `docs/`, `image_features/` 등은 팀원별 실험과 개발 기록을 보존한다. 각 실험은 당시의 실행 환경과 설정을 참고하며, 최종 추론은 루트의 제출 코드로 실행한다.

최종 추론에 필요한 파일은 다음과 같다.

```text
test.py
model.py
features.py
postprocess.py
evaluate.py
encoders.py
requirements.txt
weights/
  blend.json
  member_base/
    config.json
    model.joblib
  member_base_img16_delta/
    config.json
    model.joblib
```

`model.py`는 같은 폴더의 `features.py`, `postprocess.py`, `evaluate.py`, `encoders.py`를 사용한다. 실행 시 다른 작업 폴더의 코드나 사전 계산한 Test 피처를 가져올 필요는 없다. `train.py`는 재학습용이며 위 추론 명령에서는 실행하지 않는다.

## 구성과 학습

- `model.py`, `features.py`, `postprocess.py`: 모델·과거 피처·인과 후처리.
- `encoders.py`: 같은 폴더에 포함된 이미지 인코더. 상위 작업 폴더에 의존하지 않는다.
- `train.py`: Train 및 Validation으로 최종 모델을 학습하고 `weights/`에 저장.
- `evaluate.py`: experiment_id 단위 GroupKFold 및 Train 학습 / Validation 평가.
- `weights/`: 센서 HGB와 이미지 PCA·HGB, 설정 및 블렌드 비율. 각 모델 파일은 25MB 미만이다.
- `tests/`: 실제 대회 데이터를 사용하지 않는 합성 테스트.

```bash
python train.py --data_dir <데이터 폴더> --out weights_retrained --device cuda
```

학습 시 데이터는 `train/train.csv`, `validation/validation.csv` 또는 루트의 `train.csv`, `validation.csv` 구조를 지원한다. CSV의 상위 폴더가 해당 이미지 루트다. 위 예시는 기존 제출 가중치를 보존하기 위해 별도 출력 폴더 `weights_retrained/`를 사용한다. 재학습 가중치를 추론하려면 `test.py`에 `--artifacts weights_retrained`를 지정한다. 운영진 재현에는 학습이 필요 없다.

기본 최종 학습 구성은 Train+Validation이다. 검증은 실험 단위 GroupKFold OOF와 Train만 학습한 공식 Validation으로 구분한다. PCA는 각 학습 부분에서만 fit한다. Validation을 사용해 선택한 설정의 Validation 성능은 낙관적일 수 있다. 이 배포 정리에서 실데이터 학습·성능 재측정은 실행하지 않았다.

## 시계열 및 후처리

같은 실험 안에서 실제 `relative_time_s`로 정렬한다. 전압·전류·두 온도·압력·thermal_t의 과거 30/120초 평균·최대·차분과 일부 기울기를 계산한다. 이미지 모델은 16차원 PCA와 30초 이전 이미지 대비 변화량을 추가한다. 미래 행, 실험 전체 평균·길이·종료 시각, label을 입력으로 사용하지 않는다. RGB 의사색 밝기를 실제 온도로 해석하지 않는다.

확률을 합친 뒤 `mask_cummax`를 적용한다. 과거까지 온도 도달·전압 강하 상태로 허용 단계를 제한하고 시간순 누적 최대를 사용한다. 행 순서 변경, 실험별 분할, 미래 행 제거를 검사한다. 필요한 과거 중간 행을 임의로 삭제했을 때까지 동일 예측을 보장하지는 않는다.

## 재현 점검

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_features.py tests/test_blend.py tests/test_image_path.py tests/test_submission.py -p no:cacheprovider
python check_submission.py --data_dir <데이터 폴더> --final <최종 CSV>
```

위 pytest 명령은 최종 블렌드 제출 코드의 합성 테스트 4개 파일을 지정한다. `tests/`에는 복원된 과거 실험용 테스트도 함께 있으므로 전체 테스트와 제출 모델용 테스트를 구분한다.

동봉 `check_submission.py`는 팀 자체 검사 도구이며 운영진 배포본이 아니다. 공식 스크립트를 확보하면 추가로 실행해야 한다. 실제 데이터 점검은 대회 서버에서 사용자가 실행한다. 실제 최종 CSV와의 일치 여부는 이 배포 작업에서 확인하지 않았다. 최종 선택 CSV는 반드시 이 폴더의 `test.py`로 생성해야 한다.

원본 CSV·이미지·예측 파일·임베딩·서버 접속 정보·비밀정보는 제출 저장소에 포함하지 않는다. `MANIFEST.json`은 명시된 제출 패키지 파일의 SHA256과 크기를 기록한다. 복원된 팀원 실험 파일 전체의 목록은 아니다.

검증 기록과 범위:

- 제출 패키지에서 실행한 위 합성 테스트: **30 passed (2026-11-15)**.
- 원본 최종 블렌드 코드의 CPU 추론 기록: 3,451행 생성, 출력 열·입력 순서·ID 중복·정수 라벨 범위 검사 통과.
- 복구 커밋 `346a2ae`: 삭제된 187개 파일을 기존 Git blob과 대조해 복원하고, 기존 제출 파일 28개가 바뀌지 않았음을 확인.
- 위 기록을 복구된 `main`의 실제 데이터 전체 재현 검사나 DAKER 최종 선택 CSV 일치 검사 완료로 해석하지 않는다. 최종 파일 비교, 반복 실행, 행 섞기, 실험 분리, 미래 행 제거는 해당 검사 결과로 별도 확인한다.
