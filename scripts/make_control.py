# scripts/make_control.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from bvh_viewer.BVH_Parser import bvh_parser
from core.motion_features import CLIP_LENGTH, extract_features, tensor_to_motion_object_root

OUTPUT_DIR = "./data/control/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract control trajectory from a BVH file.")
    parser.add_argument('--bvh', type=str, required=True, help="Path to the template BVH file.")
    parser.add_argument('--start_frame', type=int, default=0, help="Start frame for feature extraction.")
    parser.add_argument('--out_dir', type=str, default=OUTPUT_DIR, help="Output directory for .pt files.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n--- Extracting control trajectory from: {args.bvh} ---")
    root, motion = bvh_parser(args.bvh)
    motion.list_to_quaternion(root)
    motion.save_virtual_root_info(root)

    final_features = extract_features(motion, args.start_frame, CLIP_LENGTH)
    abs_traj = tensor_to_motion_object_root(final_features)

    torch.save(torch.from_numpy(abs_traj), os.path.join(args.out_dir, "absolute_3d_compatible.pt"))

    print(f"abs_traj: {abs_traj.shape}")
    print(f"Saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
