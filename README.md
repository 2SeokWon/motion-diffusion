# Motion Transformer

인간 동작 생성을 위한 트랜스포머 기반 디퓨전 모델입니다. 이 프로젝트는 BVH 파일을 사용하여 인간의 동작 데이터를 학습하고, 새로운 동작을 생성할 수 있습니다.

## 🚀 빠른 시작

### 1. 환경 설정

먼저 conda를 사용하여 가상환경을 생성합니다:

```bash
# 가상환경 생성 및 활성화
conda env create -f environment.yml
conda activate pyopengl
```

### 2. 데이터 전처리

BVH 파일들을 모델 학습에 적합한 형태로 전처리합니다:

```bash
python preprocess_bvh.py
```

이 단계에서는:
- `dataset/` 폴더의 BVH 파일들을 읽어옵니다
- 관절 회전을 6D 표현으로 변환합니다
- 루트 모션 특징을 추출합니다
- 전처리된 데이터를 `processed_data/` 폴더에 저장합니다

### 3. 모델 학습

전처리된 데이터를 사용하여 모델을 학습합니다:

```bash
python train.py
```

학습 과정에서:
- 모델 체크포인트가 `checkpoints/` 폴더에 저장됩니다
- 기본적으로 10 에포크마다 체크포인트가 저장됩니다
- 총 500 에포크 동안 학습이 진행됩니다

### 4. 동작 생성

학습된 모델을 사용하여 새로운 동작을 생성합니다:

```bash
python generate.py --checkpoint_path checkpoints/model_epoch_250.pt
```

#### 생성 옵션

`generate.py`는 다음과 같은 인자를 받습니다:

- `--checkpoint_path` (필수): 사용할 모델 체크포인트 파일의 경로
  ```bash
  --checkpoint_path checkpoints/model_epoch_250.pt
  ```

- `--num_samples` (선택, 기본값: 5): 생성할 동작 샘플의 개수
  ```bash
  --num_samples 10
  ```

- `--seq_len` (선택, 기본값: 180): 생성할 동작 시퀀스의 길이 (프레임 단위)
  ```bash
  --seq_len 240
  ```

#### 예시 사용법

```bash
# 기본 설정으로 5개 샘플 생성 (180 프레임)
python generate.py --checkpoint_path checkpoints/model_epoch_250.pt

# 10개 샘플을 240 프레임 길이로 생성
python generate.py --checkpoint_path checkpoints/model_epoch_250.pt --num_samples 10 --seq_len 240

# 최신 체크포인트 사용
python generate.py --checkpoint_path checkpoints/model_epoch_500.pt --num_samples 3 --seq_len 120
```

생성된 BVH 파일들은 `results/` 폴더에 저장됩니다.

## 🔧 모델 설정

- **관절 수**: 23개
- **입력 특징**: 루트 모션(4차원) + 관절 회전(23 × 6차원)
- **시퀀스 길이**: 기본 180 프레임 (조정 가능)
- **디퓨전 타임스텝**: 1000단계

## ⚠️ 주의사항

- CUDA 지원 GPU가 있으면 자동으로 GPU를 사용합니다
- 학습은 상당한 시간이 소요될 수 있습니다 (GPU 권장)
- 충분한 디스크 공간이 필요합니다 (체크포인트 및 결과 파일)

## 📄 라이센스

이 프로젝트는 연구 및 교육 목적으로 사용할 수 있습니다.
