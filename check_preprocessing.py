# check_preprocessing.py (정규화/역정규화 테스트 추가)

import torch
import numpy as np
import os
import json
from scipy.spatial.transform import Rotation

# 필요한 유틸리티 함수들을 import
from kinematics import matrix_to_quaternion, sixd_to_rotation_matrix

# reconstruct_bvh_from_features 함수는 변경 없이 그대로 사용합니다.
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
    all_joint_local_6d_flat = features_unnormalized[:, 4:]
    
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
        world_vel_increment = current_facing_rot.apply(local_vel)
        final_root_positions[i+1, [0,2]] = final_root_positions[i,[0,2]] + world_vel_increment[[0,2]]
        final_root_positions[i+1, 1] = root_y_height[i+1]

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

    print("--- Verifying Full Preprocessing Pipeline (including Norm/De-norm) ---")

    clip_filename = "clip_0000.npz"
    clip_path = os.path.join(processed_data_path, clip_filename)
    print(f"Verifying with fixed clip: {clip_path}")

    if not os.path.exists(clip_path):
        print(f"Error: Test clip '{clip_path}' not found. Please run preprocess_bvh.py first.")
        return

    with np.load(clip_path) as data:
        features_original = data['features']
        facing_rotations_quat = data['facing_rotations']

    seq_len = features_original.shape[0]

    # ==================== 핵심 추가 부분 ====================
    print("\n--- Step 1: Loading statistics and applying Normalization ---")
    try:
        # 통계 파일 로드
        pos_vel_mean = np.load(os.path.join(processed_data_path, "pos_vel_mean.npy"))
        pos_vel_std = np.load(os.path.join(processed_data_path, "pos_vel_std.npy"))
        rotation_mean = np.load(os.path.join(processed_data_path, "rotation_mean.npy"))
        rotation_std = np.load(os.path.join(processed_data_path, "rotation_std.npy"))

        # 통계 데이터를 하나의 벡터로 조합
        mean = np.concatenate([pos_vel_mean, rotation_mean], axis=1)
        std = np.concatenate([pos_vel_std, rotation_std], axis=1)

        # 정규화 수행 (Numpy의 브로드캐스팅 기능 활용)
        features_normalized = (features_original - mean) / std

        print("--- Step 2: Applying De-normalization ---")
        # 역정규화 수행
        features_reconstructed = features_normalized * std + mean
        
        # (선택적) 수학적 검증: 원본과 복원된 데이터가 거의 같은지 확인
        is_close = np.allclose(features_original, features_reconstructed)
        print(f"Mathematical integrity check (is data recovered?): {is_close}")
        if not is_close:
            print("WARNING: Mathematical integrity failed. There might be a serious issue in stats files.")


    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not load statistics files (.npy). Run preprocess_bvh.py first. Details: {e}")
        return
    # =======================================================

    print("\n--- Step 3: Reconstructing BVH from the de-normalized features ---")

    output_filename = f"reconstructed_{os.path.basename(clip_path).replace('.npz', '')}_norm_test.bvh"
    output_path = os.path.join(output_bvh_dir, output_filename)

    # 역정규화로 복원된 특징 벡터를 사용하여 BVH 재구성
    reconstruct_bvh_from_features(
        features_reconstructed, # <-- 원본 대신 복원된 데이터를 사용
        facing_rotations_quat,
        seq_len,
        skeleton_template_path,
        output_path
    )

if __name__ == '__main__':
    main()