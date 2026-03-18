# scripts/preprocess.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import traceback
from tqdm import tqdm
from joblib import Parallel, delayed

from bvh_viewer.BVH_Parser import bvh_parser
from core.motion_features import (
    CLIP_LENGTH, STRIDE, ROOT_DISP_DIM,
    extract_features, tensor_to_motion_object_root,
)

BVH_FOLDER_PATH = "./data/raw/"
OUTPUT_PROCESSED_DIR = "./data/processed/"
OUTPUT_METADATA_PATH = os.path.join(OUTPUT_PROCESSED_DIR, "metadata.json")
os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)


def process_single_file(idx, filename, class_name, class_name_idx, class_type, class_type_idx, feature_dim):
    filepath = os.path.join(BVH_FOLDER_PATH, filename)

    final_count = 0
    final_sum = np.zeros(feature_dim)
    final_sum_sq = np.zeros(feature_dim)

    abs_count = 0
    abs_sum = np.zeros(ROOT_DISP_DIM)
    abs_sum_sq = np.zeros(ROOT_DISP_DIM)

    clip_info = []

    try:
        root, motion = bvh_parser(filepath)
        motion.list_to_quaternion(root)
        motion.save_virtual_root_info(root)

        total_frames = motion.frame_len

        num_windows = (total_frames - CLIP_LENGTH) // STRIDE + 1
        for window_idx in range(num_windows):
            start_frame = window_idx * STRIDE
            if start_frame + CLIP_LENGTH > total_frames:
                break
            features_seg = extract_features(motion, start_frame, CLIP_LENGTH)
            abs_traj = tensor_to_motion_object_root(features_seg)  # [T, 3]

            if not np.isfinite(abs_traj).all():
                print(f"Warning: NaN/Inf in abs_traj of {filename}. Skipping window.")
            else:
                abs_count += abs_traj.shape[0]
                abs_sum += np.sum(abs_traj, axis=0)
                abs_sum_sq += np.sum(abs_traj ** 2, axis=0)

        total_final_features = extract_features(motion, 0, total_frames)

        if not np.isfinite(total_final_features).all():
            nan_count = np.isnan(total_final_features).sum()
            inf_count = np.isinf(total_final_features).sum()
            print(f"Warning: NaN({nan_count})/Inf({inf_count}) in features of {filename}. Skipping.")
        else:
            final_count += total_final_features.shape[0]
            final_sum += np.sum(total_final_features, axis=0)
            final_sum_sq += np.sum(total_final_features ** 2, axis=0)

        clip_filename = f"clip_{idx:04d}.npz"
        np.savez_compressed(
            os.path.join(OUTPUT_PROCESSED_DIR, clip_filename),
            features=total_final_features.astype(np.float32),
        )

        clip_info.append({
            "path": clip_filename,
            "length": int(total_frames),
            "source_file": filename,
            "class_name": class_name,
            "class_name_idx": int(class_name_idx),
            "class_type": class_type,
            "class_type_idx": int(class_type_idx),
        })

    except Exception as e:
        print(f"Error in {filename}: {e}")
        traceback.print_exc()

    return final_count, final_sum, final_sum_sq, abs_count, abs_sum, abs_sum_sq, clip_info


def main():
    print("\n--- Step 1: Extracting Features from BVH Files ---")
    bvh_files = [f for f in os.listdir(BVH_FOLDER_PATH) if f.endswith(".bvh")]
    class_names = sorted(set(f.split('_')[0] for f in bvh_files))
    class_types = sorted(set(f.split('_')[1].split('.')[0] for f in bvh_files))

    class_name_map = {name: i for i, name in enumerate(class_names)}
    class_type_map = {t: i for i, t in enumerate(class_types)}

    feature_dim = 210

    tasks_to_run = []
    for idx, filename in enumerate(bvh_files):
        class_name = filename.split('_')[0]
        class_type = filename.split('_')[1].split('.')[0]
        class_name_idx = class_name_map[class_name]
        class_type_idx = class_type_map[class_type]
        tasks_to_run.append(
            delayed(process_single_file)(idx, filename, class_name, class_name_idx, class_type, class_type_idx, feature_dim)
        )

    results = list(tqdm(
        Parallel(n_jobs=-1, return_as='generator')(tasks_to_run),
        total=len(tasks_to_run),
        desc="Processing BVH files"
    ))

    total_final_count = 0
    total_final_sum = np.zeros(feature_dim)
    total_final_sum_sq = np.zeros(feature_dim)
    total_abs_count = 0
    total_abs_sum = np.zeros(ROOT_DISP_DIM)
    total_abs_sum_sq = np.zeros(ROOT_DISP_DIM)
    all_motion_clips = []

    for final_count, final_sum, final_sum_sq, abs_count, abs_sum, abs_sum_sq, clip_info in results:
        total_final_count += final_count
        total_final_sum += final_sum
        total_final_sum_sq += final_sum_sq
        total_abs_count += abs_count
        total_abs_sum += abs_sum
        total_abs_sum_sq += abs_sum_sq
        if clip_info:
            all_motion_clips.append(clip_info)

    print("Calculating mean and std for the entire dataset...")
    if total_final_count == 0:
        raise ValueError("No valid data processed for final feature stats.")
    if total_abs_count == 0:
        raise ValueError("No valid data processed for abs_traj stats.")

    def calc_mean_std(total_sum, total_sum_sq, count):
        mean = total_sum / count
        variance = np.maximum((total_sum_sq / count) - (mean ** 2), 0)
        return mean, np.sqrt(variance)

    mean, std = calc_mean_std(total_final_sum, total_final_sum_sq, total_final_count)
    abs_mean, abs_std = calc_mean_std(total_abs_sum, total_abs_sum_sq, total_abs_count)

    out = OUTPUT_PROCESSED_DIR
    np.save(os.path.join(out, "root_pos_mean.npy"), mean[0:4])
    np.save(os.path.join(out, "root_pos_std.npy"),  std[0:4])
    np.save(os.path.join(out, "position_mean.npy"), mean[4:70])
    np.save(os.path.join(out, "position_std.npy"),  std[4:70])
    np.save(os.path.join(out, "rotation_mean.npy"), mean[70:208])
    np.save(os.path.join(out, "rotation_std.npy"),  std[70:208])
    np.save(os.path.join(out, "foot_mean.npy"),     mean[208:210])
    np.save(os.path.join(out, "foot_std.npy"),      std[208:210])
    np.save(os.path.join(out, "abs_traj_mean.npy"), abs_mean)
    np.save(os.path.join(out, "abs_traj_std.npy"),  abs_std)

    with open(OUTPUT_METADATA_PATH, 'w') as f:
        json.dump(all_motion_clips, f, indent=4)

    print(f"\nPreprocessing complete. Saved to '{OUTPUT_PROCESSED_DIR}'")


if __name__ == '__main__':
    main()
