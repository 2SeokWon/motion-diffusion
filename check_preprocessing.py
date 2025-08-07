# check_preprocessing.py (최종 수정)

import torch
import numpy as np
import os
import json
import random
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# 필요한 유틸리티 함수들을 import
from kinematics import matrix_to_quaternion, sixd_to_rotation_matrix, euler_to_sixd

def reconstruct_bvh_from_features(
    features_unnormalized, 
    facing_rotations_quat,
    seq_len, 
    template_path, 
    output_path
):
    """
    특징 벡터와 저장된 Facing Rotation을 사용하여 BVH를 완벽하게 재구성합니다.
    """
    print(f"Reconstructing BVH file at: {output_path}")
    
    # --- 1. 특징 벡터 분해 ---
    root_y_height = features_unnormalized[:, 0]
    root_xz_velocity_local = features_unnormalized[:, 1:3]
    all_joint_local_6d_flat = features_unnormalized[:, 3:]
    num_all_joints = all_joint_local_6d_flat.shape[1] // 6
    
    # --- 2. Local 6D -> Local Quaternion 변환 ---
    all_local_6d_reshaped = torch.from_numpy(all_joint_local_6d_flat).float().reshape(seq_len, num_all_joints, 6)
    all_local_rotmats = sixd_to_rotation_matrix(all_local_6d_reshaped).numpy()
    all_local_quat = Rotation.from_matrix(all_local_rotmats.reshape(-1, 3, 3)).as_quat().reshape(seq_len, num_all_joints, 4)

    root_local_quat = all_local_quat[:, 0, :]
    joint_local_quat = all_local_quat[:, 1:, :]

    # --- 3. Global Orientation 복원 ---
    facing_rot_obj = Rotation.from_quat(facing_rotations_quat)
    local_rot_obj = Rotation.from_quat(root_local_quat)
    final_root_global_rot = facing_rot_obj * local_rot_obj
    final_root_global_quat = final_root_global_rot.as_quat()

    # --- 4. 위치 적분 ---
    final_root_positions = np.zeros((seq_len, 3))
    frame_time = 1.0 / 30.0
    
    final_root_positions[0, 1] = root_y_height[0]
    
    for i in range(seq_len - 1):
        current_facing_rot = Rotation.from_quat(facing_rotations_quat[i])
        local_vel = np.array([root_xz_velocity_local[i, 0], 0, root_xz_velocity_local[i, 1]])
        world_vel_increment = current_facing_rot.apply(local_vel) #* frame_time
        final_root_positions[i+1, [0,2]] = final_root_positions[i,[0,2]] + world_vel_increment[[0,2]]
        final_root_positions[i+1, 1] = root_y_height[i+1]  # Y 높이는 그대로 유지
    
    # --- 5. 최종 BVH 데이터 조립 ---
    all_bvh_rotations_quat = np.concatenate([
        final_root_global_quat[:, np.newaxis, :],
        joint_local_quat
    ], axis=1)
    
    final_all_euler_deg = Rotation.from_quat(
        all_bvh_rotations_quat.reshape(-1, 4)
    ).as_euler('yxz', degrees=True).reshape(seq_len, -1)
    
    motion_data_flat = np.concatenate([final_root_positions, final_all_euler_deg], axis=1)
    
    # --- 6. 파일 저장 ---
    with open(template_path, 'r') as f: lines = f.readlines()
    header_end_index = -1
    for i, line in enumerate(lines):
        if "MOTION" in line.upper(): header_end_index = i; break
    header_lines = lines[:header_end_index]
        
    with open(output_path, 'w') as f:
        f.writelines(header_lines)
        f.write("MOTION\n")
        f.write(f"Frames: {seq_len}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")
        motion_lines = [" ".join(f"{x:.6f}" for x in frame) for frame in motion_data_flat]
        f.write("\n".join(motion_lines))
        
    print("BVH reconstruction complete.")

def main():
    processed_data_path = "./processed_data"
    output_bvh_dir = "./verification"
    skeleton_template_path = "./dataset/Aeroplane_BR.bvh"
    os.makedirs(output_bvh_dir, exist_ok=True)
    
    print("Loading metadata and statistics...")

    with open(os.path.join(processed_data_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
        
    if not metadata:
        print("Error: metadata.json is empty or not found.")
        return
    clip_info = metadata[0]

    clip_path = os.path.join(processed_data_path, clip_info['path'])
    print(f"Verifying with random clip: {clip_info['path']}")

    with np.load(clip_path) as data:
        features_unnormalized = data['features']
        facing_rotations_quat = data['facing_rotations']

    seq_len = features_unnormalized.shape[0]

    # ==================== 핵심 수정 부분 ====================
    # 특징 벡터를 분리하여 각각 역정규화
    #pos_vel_part_norm = features_normalized[:, :3]
    #rotation_part_norm = features_normalized[:, 3:]

    #pos_vel_part_unnorm = (pos_vel_part_norm * pos_vel_std) + pos_vel_mean
    #rotation_part_unnorm = (rotation_part_norm * rotation_std) + rotation_mean

    # 다시 하나로 합쳐서 복원 함수에 전달
    #features_unnormalized = np.concatenate([pos_vel_part_unnorm, rotation_part_unnorm], axis=1)
    # =======================================================
    
    # 디버깅 프린트 (선택 사항이지만 유용함)
    print("-" * 50)
    print("Verifying Unnormalized Feature Statistics:")
    print(f"Y Height (mean, std): {features_unnormalized[:, 0].mean():.4f}, {features_unnormalized[:, 0].std():.4f}")
    print(f"XZ vel (mean, std):   {features_unnormalized[:, 1:3].mean():.4f}, {features_unnormalized[:, 1:3].std():.4f}")
    # ... (필요 시 다른 특징들도 출력)
    print("-" * 50)

    output_filename = f"reconstructed_{os.path.basename(clip_path).replace('.npz', '.bvh')}"
    output_path = os.path.join(output_bvh_dir, output_filename)

    reconstruct_bvh_from_features(
        features_unnormalized, 
        facing_rotations_quat,
        seq_len, 
        skeleton_template_path, 
        output_path
    )
    
if __name__ == '__main__':
    main()