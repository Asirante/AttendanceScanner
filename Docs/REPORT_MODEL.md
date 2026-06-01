# 프로젝트 보고서 — 모델 부분

> AI 안면인식 출퇴근 시스템 / 얼굴 인식 모델 설계·학습·평가
> (수치가 비어 있는 표는 `run_experiments.py` 실행 후 `runs/comparison.csv` 값으로 채우면 됩니다.)

---

## 1. 데이터셋

### 1.1 데이터 확보
직원별 얼굴 영상을 촬영해 데이터를 확보한다. 영상 한 개에서 여러 프레임을 추출하므로, 사진을 한 장씩 모으는 것보다 다양한 각도·표정을 효율적으로 수집할 수 있다. 수집은 `utils/data_collector.py`가 담당하며, 영상 파일(`collect_from_video`) 또는 웹캠 실시간 촬영(`collect_live`) 두 경로를 지원한다.

- 대상 인원: (예) N명
- 인당 영상: (예) 1~2개 × 5~10초
- 인당 확보 프레임: (예) 약 OO장

### 1.2 전처리
원본 프레임을 그대로 쓰지 않고 얼굴 영역만 정렬해 사용한다.

1. **얼굴 검출·정렬**: MTCNN으로 얼굴을 검출하고 160×160으로 정렬 크롭한다. 검출 신뢰도(`prob`)가 0.95 미만인 프레임은 버려, 흐리거나 얼굴이 작은 프레임이 학습에 섞이지 않게 한다.
2. **정규화**: 학습 시 픽셀값을 `(x - 127.5) / 128.0`으로 정규화해 InceptionResnet 입력 분포(약 -1~1)에 맞춘다. (증강을 쓰는 경우 0~1 정규화 → 증강 → -1~1 스케일 순서로 처리)

전처리 결과는 직원별 폴더에 `.pt` 텐서로 저장한다(`data/raw/{emp_id}/0001.pt …`).

### 1.3 증강 (Augmentation)
학습 데이터에만 적용하며, 검증·테스트에는 적용하지 않는다(`utils/dataset.py`).

- **RandomHorizontalFlip**: 좌우 반전으로 좌·우 얼굴 방향 변화에 강건하게.
- **ColorJitter** (brightness 0.2 / contrast 0.2 / saturation 0.1): 조명·화이트밸런스 변화에 강건하게.

증강의 효과는 실험 exp04(증강 off)와 비교해 확인한다.

### 1.4 라벨링
별도 수작업 라벨링은 필요 없다. **폴더명이 곧 직원 ID(라벨)** 이며, `utils/registry.py`의 `build_label_map`이 train 디렉토리의 직원 폴더를 정렬해 `{인덱스: 직원ID}` 매핑을 `data/label_map.json`으로 생성한다. 분류 모델의 출력 인덱스와 직원 ID가 이 매핑으로 연결된다.

### 1.5 데이터셋 구성 (train / val / test)
`utils/split_dataset.py`가 직원별로 프레임을 **70% / 15% / 15%** 로 분할한다(`random_state=42`로 재현 가능). 직원별로 프레임이 `MIN_FRAMES`(기본 10장) 미만이면 경고 후 제외해, 데이터가 너무 적은 직원이 평가를 왜곡하지 않게 한다.

| 분할 | 비율 | 용도 |
|------|------|------|
| train | 70% | 학습 |
| val | 15% | 에폭별 검증·조기 종료·최적 모델 선택 |
| test | 15% | 최종 성능 평가 (학습·검증에 미사용) |

---

## 2. 사용 모델

### 2.1 구조
사전학습된 **InceptionResnetV1**(facenet-pytorch, VGGFace2 가중치)을 백본으로 사용하고, 그 위에 직원 수만큼 출력하는 분류 헤드를 붙인다(`models/classifier.py`).

```
입력 얼굴 (3×160×160)
      │
 InceptionResnetV1 backbone  → 512차원 임베딩
      │
 Linear(512 → num_classes)   → 직원별 점수
```

- 백본: 얼굴을 512차원 특징 벡터로 변환 (VGGFace2 9,000명·330만 장으로 사전학습)
- 분류 헤드: 512차원 → 직원 수 만큼의 클래스 점수

### 2.2 전이학습 전략
데이터가 적어 백본을 처음부터 학습하면 과적합되므로, 사전학습 가중치를 활용하는 전이학습을 기본으로 한다.

- **동결(freeze)**: 백본을 고정하고 분류 헤드만 학습 → 적은 데이터로 빠르게 수렴.
- **부분 미세조정(fine-tune)**: 백본 마지막 일부 블록만 풀어 추가 학습(`unfreeze_last_blocks`) → 우리 데이터에 더 적응.
- **처음부터 학습(from scratch)**: 사전학습을 쓰지 않는 비교군 → 전이학습의 이점을 정량적으로 보이기 위함.

---

## 3. 성능 비교 방법

`run_experiments.py`로 아래 설정을 동일 데이터에서 학습·평가하고 결과를 `runs/comparison.csv`로 모은다. 비교 축은 **pretrained 사용 여부 / 미세조정 범위 / 증강 유무 / 학습률**이다.

| 실험 | pretrained | 백본 | 증강 | 학습률 | 목적 |
|------|-----------|------|------|--------|------|
| exp00 | ○ | 학습 없음 (임베딩+유사도) | - | - | 순수 VGGFace2 베이스라인 (실제 앱 방식) |
| exp01 | ○ | 전체 동결 | ○ | 1e-3 | 전이학습 기본 |
| exp02 | ○ | 마지막 2블록 미세조정 | ○ | 5e-4 | 미세조정 효과 |
| exp03 | ✕ | 처음부터 학습 | ○ | 1e-3 | 전이학습 vs 비전이 |
| exp04 | ○ | 전체 동결 | ✕ | 1e-3 | 증강 효과 |
| exp05 | ○ | 전체 동결 | ○ | 1e-4 | 학습률 영향 |

- exp00은 학습하지 않는다. train 이미지로 사람별 평균 임베딩(프로토타입)을 만들고, test 임베딩을 가장 가까운(코사인 유사도) 프로토타입으로 분류한다. 이는 실제 출퇴근 앱이 쓰는 방식과 동일하며, "사전학습 모델을 그대로 쓰면 어느 정도 나오는가"의 기준선이 된다.
- 공통: 옵티마이저 Adam, weight_decay 1e-4, CosineAnnealing 스케줄러, val 정확도 기준 조기 종료(patience 15~20), val 최고 시점 가중치 저장. 에폭은 충분히 크게(100~150) 두되 조기 종료로 과학습 전에 멈춘다.

---

## 4. 성능 및 측정 기준

### 4.1 측정 지표
- **정확도(Accuracy)**: 전체 맞춘 비율.
- **Macro-F1**: 직원별 F1의 평균 — 특정 인원에 데이터가 쏠려도 균형 있게 평가.
- **혼동 행렬(Confusion Matrix)**: 어떤 직원이 누구와 혼동되는지 확인.
- **추론 시간(Inference Time) / FPS**: 실시간 출퇴근용이므로 속도도 평가. 실시간 처리 기준으로 **FPS ≥ 15** 를 목표로 한다. (`evaluate.py`가 측정)

### 4.2 결과 (실험 후 채움)

| 실험 | val 정확도 | test 정확도 | macro-F1 | ms/frame | FPS |
|------|-----------|------------|----------|----------|-----|
| exp00 | - |  |  |  |  |
| exp01 |  |  |  |  |  |
| exp02 |  |  |  |  |  |
| exp03 |  |  |  |  |  |
| exp04 |  |  |  |  |  |
| exp05 |  |  |  |  |  |

> 위 표는 `runs/comparison.csv`의 값을 그대로 옮기면 된다. 가장 성능이 좋은 설정을 최종 모델로 선택하고, 그 근거(정확도·F1·속도 균형)를 서술한다.

---

## 5. 학습 로그
학습 과정은 TensorBoard로 기록한다(`train.py`의 `SummaryWriter`, `runs/{exp_name}/`).

- `Loss/train`: 에폭별 학습 손실
- `Acc/val`: 에폭별 검증 정확도
- `LR`: 학습률 변화(CosineAnnealing)

확인: `tensorboard --logdir runs` 실행 후 브라우저에서 곡선을 캡처해 보고서에 첨부한다.

---

## 6. 코드 구조 (모델 관련)

| 파일 | 역할 |
|------|------|
| `utils/download_lfw.py` | LFW 데이터셋 다운로드·압축 해제 (실험용 데이터 확보) |
| `utils/prepare_lfw.py` | LFW 이미지 → MTCNN 정렬 → `.pt` 저장 (파이프라인 형식 변환) |
| `utils/data_collector.py` | 영상/웹캠 → MTCNN 정렬 → 얼굴 `.pt` 저장 (실운영 등록용 데이터 경로) |
| `utils/split_dataset.py` | 직원별 프레임을 train/val/test 70:15:15 분할 |
| `utils/registry.py` | 직원 폴더 → label_map 생성/로드 (라벨링) |
| `utils/dataset.py` | `.pt` 로드 + 정규화/증강하는 PyTorch Dataset |
| `models/classifier.py` | InceptionResnet 백본 + 분류 헤드, freeze/미세조정/pretrained 옵션 |
| `train.py` | 학습 루프 (조기 종료·스케줄러·TensorBoard 로그) |
| `evaluate.py` | test 평가 (정확도·F1·혼동행렬·FPS) |
| `run_experiments.py` | 여러 설정 일괄 학습·평가 → 비교표 CSV 생성 |

---

## 7. 실행 순서 (재현 방법)

학습 실험은 공개 데이터셋 **LFW(Labeled Faces in the Wild)** 를 사용한다. (오픈 액세스 라이선스, 연구용으로 자유롭게 사용 가능)

```bash
# 1) LFW 다운로드 + 압축 해제 → data/lfw/
python utils/download_lfw.py

# 2) MTCNN 정렬 → data/raw/{인물}/*.pt
#    사진이 적은 인물은 자동 제외(기본 20장 미만). 필요 시 --min_images 조정
python utils/prepare_lfw.py --lfw_dir data/lfw --min_images 20

# 3) train/val/test 분할 (70:15:15)
python utils/split_dataset.py

# 4) 비교 실험 일괄 실행 → runs/comparison.csv
python run_experiments.py

# 5) 학습 곡선 확인
tensorboard --logdir runs
```

> 참고: LFW는 약 5,700명을 포함하지만 대부분 사진이 1~2장뿐이고, 20장 이상 가진 인물은 수십 명 수준이다. 분류 실험에는 이 "사진이 많은 인물"만 자동 선별되어 사용된다. 더 많은 클래스를 원하면 `--min_images` 값을 낮추되, 클래스당 표본이 적어지면 정확도 해석에 주의한다.

> 실제 운영 시스템은 이와 별개다. 직원 얼굴을 앱의 등록 기능(캡처/영상으로 등록)으로 직접 등록하며, 위 학습 과정 없이 사전학습 임베딩만으로 동작한다.
