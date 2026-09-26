# scripts/split_stats.py
# 학습 구간(각 파일의 앞 1 - test_ratio)만으로 정규화 통계를 다시 계산한다.
# preprocess.py의 통계는 평가 구간까지 포함하므로, 평가 데이터 정보가 학습 입력의 스케일에 섞이지 않게 한다.
# 저장된 clip 특징에서 바로 계산하므로 BVH 전처리를 다시 돌릴 필요가 없다.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np

from core.config import load_config
from core.dataset import split_range
from core.motion_features import STRIDE, tensor_to_motion_object_root, integrate_root_velocity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    P, out_dir = cfg.data.processed_dir, cfg.data.stats_dir
    T, test_ratio = cfg.model.seq_len, cfg.data.test_ratio
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(P, "metadata.json")) as f:
        meta = json.load(f)
    meta = [c for sub in meta for c in sub] if meta and isinstance(meta[0], list) else meta

    feat_sum = np.zeros(210); feat_sq = np.zeros(210); feat_n = 0
    traj_sum = np.zeros(3);   traj_sq = np.zeros(3);   traj_n = 0
    check_windows = []

    for c in meta:
        feats = np.load(os.path.join(P, c['path']))['features'].astype(np.float64)
        lo, hi = split_range(c['length'], 'train', test_ratio)
        train = feats[lo:hi]

        # 모션 특징: preprocess.py와 같은 방식(파일 단위 특징을 그대로)으로 학습 구간만 집계
        feat_sum += train.sum(0); feat_sq += (train ** 2).sum(0); feat_n += len(train)

        # 궤적: preprocess.py와 같이 STRIDE 간격 창마다 첫 프레임 속도 0으로 두고 적분
        starts = np.arange(0, len(train) - T + 1, STRIDE)
        windows = np.stack([train[s:s + T, 1:4] for s in starts])  # [N, T, 3]
        windows[:, 0, :] = 0.0
        traj = integrate_root_velocity(windows)
        traj_sum += traj.sum((0, 1)); traj_sq += (traj ** 2).sum((0, 1)); traj_n += traj.shape[0] * T

        if len(check_windows) < 5:
            w = train[starts[len(starts) // 2]:starts[len(starts) // 2] + T].copy()
            w[0, 1:4] = 0.0
            check_windows.append((w, traj[len(starts) // 2]))

    # 벡터화 적분이 원래 함수와 같은지 검사
    max_diff = max(np.abs(tensor_to_motion_object_root(w) - t).max() for w, t in check_windows)
    print(f"Vectorized integration vs tensor_to_motion_object_root: max diff {max_diff:.2e}")
    assert max_diff < 1e-2, "vectorized trajectory integration does not match the reference"

    def mean_std(s, sq, n):
        m = s / n
        return m, np.sqrt(np.maximum(sq / n - m ** 2, 0.0))

    mean, std = mean_std(feat_sum, feat_sq, feat_n)
    tmean, tstd = mean_std(traj_sum, traj_sq, traj_n)

    stats = {
        'root_pos': (mean[0:4], std[0:4]),
        'position': (mean[4:70], std[4:70]),
        'rotation': (mean[70:208], std[70:208]),
        'foot':     (mean[208:210], std[208:210]),
        'abs_traj': (tmean, tstd),
    }
    for name, (m, s) in stats.items():
        np.save(os.path.join(out_dir, f"{name}_mean.npy"), m)
        np.save(os.path.join(out_dir, f"{name}_std.npy"), s)

    with open(os.path.join(out_dir, "split.json"), 'w') as f:
        json.dump({'test_ratio': test_ratio, 'seq_len': T, 'traj_window_stride': STRIDE,
                   'train_frames': int(feat_n), 'traj_windows': int(traj_n // T)}, f, indent=2)

    print(f"Train frames {feat_n} / traj windows {traj_n // T}")
    print(f"abs_traj mean {tmean.round(3)} std {tstd.round(3)}")
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()
