#new_preprocess.py
import os
import numpy as np
import torch
import math
from tqdm import tqdm
from joblib import Parallel, delayed
import json
import torch.nn.functional as F 
import traceback
from bvh_viewer.BVH_Parser import bvh_parser, get_preorder_joint_list
from new_preprocess import extract_features, tensor_to_motion_object_root
# --- 1. 설정 (Configuration) ---
output_processed_dir = "./"
template_bvh_path = "./Dinosaur_FW.bvh" 
os.makedirs(output_processed_dir, exist_ok=True)
ROTATION_ORDER = 'yxz'
CLIP_LENGTH = 180

def main():
    print("\n--- Step 1: Extracting Features from BVH Files ---")

    root, motion = bvh_parser(template_bvh_path)
    motion.list_to_quaternion(root)
    motion.save_virtual_root_info(root)

    final_features = extract_features(motion, 2000, CLIP_LENGTH)
    vel = final_features[:, :4]
    abs = tensor_to_motion_object_root(final_features)
    # 저장
    out_dir = os.path.join(output_processed_dir, "traj_from_template")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.from_numpy(vel), os.path.join(out_dir, "velocity_compatible.pt"))
    print(f"vel: {vel.shape}")
    torch.save(torch.from_numpy(abs), os.path.join(out_dir, "absolute_3d_compatible.pt"))
    print(f"abs: {abs.shape}")
    print("Saved to:", out_dir)
    print("\nPreprocessing complete.")

if __name__ == '__main__':
    main()