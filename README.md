# Motion Diffusion

Transformer 기반 DDPM(Denoising Diffusion Probabilistic Model)을 이용한 스타일 조건부 인간 동작 생성 모델입니다.

BVH 파일로부터 모션 특징을 추출하고, 사용자가 지정한 **절대 궤적(absolute trajectory)** 과 **동작 클래스** 를 조건으로 자연스러운 전신 모션을 생성합니다.

---

## 모델 구조

- **입력**: 213-dim 모션 특징 벡터 (프레임당)
  - Root: hipY(1) + local XZ velocity(2) + yaw angular velocity(1)
  - Joint positions: 22 joints × 3 = 66
  - Joint rotations: 23 joints × 6D = 138 (Gram-Schmidt 6D representation)
  - Foot contact: 2 (RightToe, LeftToe)
  - Condition: absolute trajectory (x, z, yaw) = 3
- **Backbone**: Transformer Encoder (8 layers, latent dim 1024)
- **Diffusion**: DDPM (1000 timesteps, linear beta schedule)
- **Guidance**: Classifier-Free Guidance (CFG) — class label 조건부/무조건부 동시 학습

---

## 프로젝트 구조

```
motion-diffusion/
├── config.yml                  # 모든 하이퍼파라미터 설정
├── core/
│   ├── model.py                # MotionTransformer
│   ├── gaussian_diffusion.py   # DDPM 구현
│   ├── dataset.py              # MotionDataset
│   ├── motion_features.py      # 특징 추출 / 궤적 복원 유틸리티
│   ├── kinematics.py           # 6D rotation 변환
│   ├── config.py               # config.yml 로더
│   └── utils.py                # BVH 저장
├── scripts/
│   ├── preprocess.py           # BVH → 전처리 데이터
│   ├── train.py                # 학습
│   ├── generate.py             # 모션 생성
│   └── make_control.py         # 제어 궤적 추출
├── bvh_viewer/                 # BVH 파싱 / 렌더링
├── data/
│   ├── raw/                    # 원본 BVH 파일
│   └── processed/              # 전처리된 데이터 (.npz, .npy)
├── checkpoints/                # 학습 체크포인트
└── results/                    # 생성 결과
```

---

## 시작하기

### 1. 환경 설정

```bash
conda env create -f environment.yml
conda activate pyopengl
```

### 2. 전처리

`data/raw/` 폴더에 BVH 파일을 넣고 실행합니다.

```bash
python scripts/preprocess.py
```

BVH 파일명은 `{ClassName}_{MotionType}.bvh` 형식이어야 합니다. (예: `Angry_FW.bvh`)

전처리 결과는 `data/processed/`에 저장됩니다:
- `clip_XXXX.npz` — 클립별 210-dim 특징
- `root_pos_mean/std.npy` — root 통계
- `position/rotation/foot_mean/std.npy` — joint 통계
- `abs_traj_mean/std.npy` — 절대 궤적 통계
- `metadata.json` — 클립 메타데이터

### 3. 학습

```bash
python scripts/train.py --config config.yml
```

체크포인트 재개:

```bash
python scripts/train.py --config config.yml --resume checkpoints/YYYYMMDD_HHMM/model_epoch_50.pt
```

학습 로그는 [Weights & Biases](https://wandb.ai)로 기록됩니다.

### 4. 제어 궤적 준비

생성 시 조건으로 사용할 절대 궤적을 BVH 파일에서 추출합니다.

```bash
python scripts/make_control.py --bvh data/raw/Angry_FW.bvh --start_frame 0
```

`data/control/absolute_3d_compatible.pt` 로 저장됩니다 (shape: `[180, 3]`, 원점 기준).

### 5. 모션 생성

```bash
# 기본 (궤적 조건만, unconditional)
python scripts/generate.py --checkpoint_path checkpoints/.../model_epoch_500.pt

# 클래스 조건 추가
python scripts/generate.py --checkpoint_path checkpoints/.../model_epoch_500.pt --class_idx 1

# 궤적 파일 직접 지정
python scripts/generate.py \
    --checkpoint_path checkpoints/.../model_epoch_500.pt \
    --abs_traj_path data/control/absolute_3d_compatible.pt \
    --class_idx 2 \
    --guidance_scale 3.0
```

생성 결과는 `results/generated_YYYYMMDD_HHMM/`에 저장됩니다:
- `sample.bvh` — 생성된 모션
- `sample.mp4` — 렌더링 영상
- `generated_traj_abs.pt` — 생성된 궤적

---

## 설정 (config.yml)

모든 실험 세팅은 `config.yml` 에서 관리합니다.

```yaml
model:
  njoints: 23
  seq_len: 180
  latent_dim: 1024
  num_layers: 8
  num_heads: 8

diffusion:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02

training:
  num_epochs: 500
  batch_size: 64
  learning_rate: 1e-4
  mask_prob: 0.1       # CFG dropout 확률
  feat_bias: 15.0      # position/traj 정규화 스케일

generation:
  guidance_scale: 3.0
```

---

## 동작 클래스

| index | class |
|---|---|
| 0 | Aeroplane |
| 1 | Angry |
| 2 | Cat |
| 3 | Chicken |
| 4 | Drunk |
| 5 | InTheDark |
| 6 | KarateChop |

각 클래스는 BR / BW / FR / FW / SR / SW / TR1 방향 변형을 포함합니다.

---

## Branch

| branch | 설명 |
|---|---|
| `master` | 절대 궤적 조건부 생성 (안정 버전) |
| `experiment/delta-interp` | coarse interp + style delta 분리 실험 |
