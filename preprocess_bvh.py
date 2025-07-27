# 파일 이름: preprocess_bvh_no_pymo.py

import os
import numpy as np
import torch
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation

# utils.py에서 오일러 -> 6D 변환 함수를 가져옵니다.
from utils import euler_to_sixd

# --- 1. 설정 (Configuration) ---
bvh_folder_path = "./dataset/"
output_processed_dir = "./processed_data/"
output_metadata_path = os.path.join(output_processed_dir, "metadata.json")
os.makedirs(output_processed_dir, exist_ok=True)

ROTATION_ORDER = 'yxz'  # 오일러 각도 변환 순서
SEQ_LEN = 90

def parse_bvh_motion_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # "MOTION" 섹션 찾기
    motion_start_index = -1
    for i, line in enumerate(lines):
        if "MOTION" in line.upper():
            motion_start_index = i
            break
            
    if motion_start_index == -1:
        return None # MOTION 섹션이 없는 경우

    # 프레임 수와 데이터 시작점 찾기
    # Frames: [num_frames]
    # Frame Time: [time]
    # [data]
    # [data]
    # ...
    num_frames = int(lines[motion_start_index + 1].split(':')[1].strip())
    data_start_index = motion_start_index + 3
    
    motion_data = []
    for i in range(data_start_index, data_start_index + num_frames):
        # 공백으로 분리된 숫자들을 float으로 변환
        values = [float(v) for v in lines[i].strip().split()]
        motion_data.append(values)
        
    return np.array(motion_data)


# --- 2. 전처리 시작 ---
all_motion_clips = []
bvh_files = [f for f in os.listdir(bvh_folder_path) if f.endswith(".bvh")]

print(f"Found {len(bvh_files)} BVH files. Starting preprocessing...")

for idx, filename in enumerate(tqdm(bvh_files, desc="Processing BVH files")):
    filepath = os.path.join(bvh_folder_path, filename)
    
    try:
        # (1) BVH 파일에서 모션 데이터(오일러 각도 + 위치) 파싱
        motion_data_euler = parse_bvh_motion_data(filepath)
        if motion_data_euler is None:
            print(f"Skipping {filename}: No MOTION data found.")
            continue
            
        motion_data_euler = motion_data_euler[::2, :] #60fps -> 30fps로 다운샘플링
        #결국 offset 값은 못 담나?

        root_positions = motion_data_euler[:, 0:3]
        root_rotations_deg = motion_data_euler[:, 3:6]
        joint_rotations_deg = motion_data_euler[:, 6:]
        
        num_frames = motion_data_euler.shape[0]

        # (3) 오일러 각도를 Quaternion으로 변환
        root_rotations_quat = Rotation.from_euler(ROTATION_ORDER, root_rotations_deg, degrees=True).as_quat()
        
        #이전 프레임 대비 루트 위치 및 회전 변화량 계산
        root_velocity = root_positions[1:] - root_positions[:-1]

        # 루트 회전 변화량 계산
        root_rot_quat_diff = (
            Rotation.from_quat(root_rotations_quat[1:]) *
            Rotation.from_quat(root_rotations_quat[:-1]).inv()
        ).as_quat()

        #변화량을 루트의 로컬 좌표계 기준으로 변환
        inv_root_rotations = Rotation.from_quat(root_rotations_quat[:-1]).inv()
        root_velocity_local = inv_root_rotations.apply(root_velocity)
        
        #Y축 회전 속도만 추출
        root_angular_velocity_y = Rotation.from_quat(root_rot_quat_diff).as_euler('yxz', degrees=True)[:, 0]

        #오일러 각을 6d로 변환
        all_rotations_deg = np.concatenate([root_rotations_deg, joint_rotations_deg], axis=1)
        all_rotations_rad = np.deg2rad(all_rotations_deg)
        num_joints = all_rotations_rad.shape[1] // 3
        rotations_tensor = torch.from_numpy(all_rotations_rad).float()
        rotations_tensor = rotations_tensor.reshape(num_frames, num_joints, 3)
        
        # (c) euler_to_sixd 함수 호출
        sixd_rotations_tensor = euler_to_sixd(rotations_tensor, order=ROTATION_ORDER)
        
        # (d) 다시 NumPy 배열로 변환하고 평탄화 (Flatten)
        sixd_rotations_np = sixd_rotations_tensor.numpy()
        sixd_rotations_flat = sixd_rotations_np.reshape(num_frames, -1)
        
        # 첫 프레임은 속도를 계산할 수 없으므로, 두 번째 프레임부터 시작합니다.
        # 따라서 모든 데이터의 길이는 (num_frames - 1)이 됩니다.
        root_y_height = root_positions[1:, [1]]  # shape (N-1, 1)
        root_xz_velocity = root_velocity_local[:, [0, 2]] # shape (N-1, 2)
        root_y_angular_velocity = root_angular_velocity_y[:, np.newaxis] # shape (N-1, 1)

        all_joint_6d_rotations = sixd_rotations_flat[1:, :]  # shape (N-1, Joints * 6)
        
        final_features = np.concatenate([
            root_y_height,
            root_xz_velocity,
            root_y_angular_velocity,
            all_joint_6d_rotations
        ], axis=1)

        clip_filename = f"clip_{idx:04d}.npy"
        clip_filepath = os.path.join(output_processed_dir, clip_filename)
        np.save(clip_filepath, final_features)

        all_motion_clips.append({
            "path": clip_filename,
            "length": final_features.shape[0]
        })

    except Exception as e:
        print(f"Could not process file {filename}. Error: {e}")

# --- 3. 최종 데이터 취합 및 저장 ---
print("Calculating mean and std for the entire dataset...")
all_clips_for_stats = []
for metadata in tqdm(all_motion_clips, desc="Loading clips for stats"):
    clip_data = np.load(os.path.join(output_processed_dir, metadata['path']))
    all_clips_for_stats.append(clip_data)

full_dataset_np = np.concatenate(all_clips_for_stats, axis=0)
mean = np.mean(full_dataset_np, axis=0, keepdims=True)
std = np.std(full_dataset_np, axis=0, keepdims=True)
std[std == 0] = 1e-7

# 통계 정보 파일 저장
np.save(os.path.join(output_processed_dir, "mean.npy"), mean)
np.save(os.path.join(output_processed_dir, "std.npy"), std)

# 최종 메타데이터 파일 저장
with open(output_metadata_path, 'w') as f:
    json.dump(all_motion_clips, f, indent=4)

print("\nPreprocessing complete.")
print(f"Processed clips and metadata saved to '{output_processed_dir}'")