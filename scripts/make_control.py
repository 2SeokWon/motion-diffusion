# scripts/make_control.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from bvh_viewer.BVH_Parser import bvh_parser
from core.motion_features import CLIP_LENGTH, extract_features, tensor_to_motion_object_root

OUTPUT_DIR = "./data/control/"


def make_waypoint_mask(num_frames, waypoints=None, frames=None):
    """
    평가용 경유점 마스크 [T, 1]. 시작(0)과 도착(T-1)은 항상 포함한다.
    waypoints: 'goal'(도착만) | 'dense'(전체) | 정수 N(중간 경유점 N개를 균등 간격으로)
    frames:    중간 경유점 프레임을 직접 지정 (waypoints보다 우선)
    평가는 다시 돌려도 같은 조건이어야 하므로 무작위가 아니라 균등 간격으로 뽑는다.
    """
    mask = np.zeros((num_frames, 1), dtype=np.float32)
    mask[0] = mask[-1] = 1.0

    if frames is not None:
        idx = np.array(frames, dtype=int)
        if ((idx <= 0) | (idx >= num_frames - 1)).any():
            raise ValueError(f"--frames must be within 1..{num_frames - 2}")
        mask[idx] = 1.0
    elif waypoints == 'dense':
        mask[:] = 1.0
    elif waypoints not in (None, 'goal'):
        n = int(waypoints)
        idx = np.linspace(0, num_frames - 1, n + 2).round().astype(int)[1:-1]  # 양 끝 제외 N개
        mask[idx] = 1.0
    return mask


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract a waypoint condition from a BVH file.")
    parser.add_argument('--bvh', type=str, required=True, help="Path to the source BVH file.")
    parser.add_argument('--start_frame', type=int, default=0, help="Start frame for feature extraction.")
    parser.add_argument('--waypoints', type=str, default='goal',
                        help="'goal' (start+goal only), 'dense' (every frame), or N (N evenly spaced intermediate waypoints).")
    parser.add_argument('--frames', type=str, default=None,
                        help="Comma-separated intermediate waypoint frames, e.g. 45,90,135 (overrides --waypoints).")
    parser.add_argument('--out', type=str, default=os.path.join(OUTPUT_DIR, "waypoint_cond.pt"),
                        help="Output .pt path.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"\n--- Extracting waypoint condition from: {args.bvh} ---")
    root, motion = bvh_parser(args.bvh)
    motion.list_to_quaternion(root)
    motion.save_virtual_root_info(root)

    final_features = extract_features(motion, args.start_frame, CLIP_LENGTH)
    abs_traj = tensor_to_motion_object_root(final_features)  # [T, 3] 원점 기준 (cm, cm, rad)

    frames = [int(f) for f in args.frames.split(',')] if args.frames else None
    mask = make_waypoint_mask(len(abs_traj), args.waypoints, frames)

    torch.save({
        'traj': torch.from_numpy(abs_traj),  # 전체 정답 궤적 (평가용). 정규화는 generate.py에서 한다
        'mask': torch.from_numpy(mask),      # 1 = 경유점으로 주어진 프레임
        'source_bvh': os.path.basename(args.bvh),
        'start_frame': args.start_frame,
    }, args.out)

    wp = np.flatnonzero(mask[:, 0])
    print(f"traj: {abs_traj.shape}, waypoint frames ({len(wp)}): {wp.tolist() if len(wp) <= 20 else 'dense'}")
    print(f"Saved to: {args.out}")


if __name__ == '__main__':
    main()
