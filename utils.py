# utils.py 의 generate_and_save_bvh 함수 (최종 수정 버전)

import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation, Slerp
from kinematics import sixd_to_rotation_matrix, matrix_to_quaternion_scipy

def generate_and_save_bvh(
    model, 
    diffusion, 
    dataset,
    num_samples, 
    seq_len,
    template_path, 
    output_dir, 
    device
):
    print("Starting sampling with the finalized and corrected generation logic...")
    model.eval()

    # --- 1. 데이터 생성 및 역정규화 (기존과 유사) ---
    input_feats = dataset.pos_vel_mean.shape[1] + dataset.rotation_mean.shape[1]
    sample_shape = (num_samples, seq_len, input_feats)
    with torch.no_grad():
        generated_motion_norm = diffusion.p_sample_loop(model, sample_shape)
        
        # 역정규화를 위해 mean, std를 numpy로 변환 (CPU에서 계산)
        mean_pos_vel = dataset.pos_vel_mean
        std_pos_vel = dataset.pos_vel_std
        mean_rotation = dataset.rotation_mean
        std_rotation = dataset.rotation_std
        
        mean = np.concatenate([mean_pos_vel, mean_rotation], axis=1)
        std = np.concatenate([std_pos_vel, std_rotation], axis=1)
        
        # 생성된 데이터를 CPU로 가져와 numpy로 변환 후 역정규화
        generated_motion = generated_motion_norm.cpu().numpy() * std + mean
    
    print("Converting generated motions to BVH files...")
    # --- 스켈레톤 템플릿 읽기 (기존과 동일) ---
    with open(template_path, 'r') as f: lines = f.readlines()
    header_end_index = -1
    for i, line in enumerate(lines):
        if "MOTION" in line.upper(): header_end_index = i; break
    header_lines = lines[:header_end_index]
    frame_time_line = f"Frame Time: {1.0/30.0:.8f}\n" # 30fps 기준
    
    # --- 각 샘플을 BVH로 변환 ---
    for i in range(num_samples):
        print(f"Processing sample {i+1}/{num_samples}...")
        single_motion = generated_motion[i]

        root_y_height = single_motion[:, 0]
        root_xz_velocity = single_motion[:, 1:3]
        root_y_angular_velocity = single_motion[:, 3] # Y축 각속도 추출!
        all_joint_6d = single_motion[:, 4:].reshape(seq_len, -1, 6)

        all_local_rotmats = sixd_to_rotation_matrix(torch.from_numpy(all_joint_6d))
        # [w,x,y,z] 형식의 numpy 배열 반환
        all_local_quat_wxyz = matrix_to_quaternion_scipy(all_local_rotmats).numpy().reshape(seq_len, -1, 4)
        root_local_quats_wxyz = all_local_quat_wxyz[:, 0, :]
        joint_local_quats_wxyz = all_local_quat_wxyz[:, 1:, :]

        final_root_positions = np.zeros((seq_len, 3), dtype=np.float32)
        final_root_global_quats_xyzw = np.zeros((seq_len, 4), dtype=np.float32)
        
        final_root_positions[0, 1] = root_y_height[0]
        current_facing_rot = Rotation.from_quat([0, 0, 0, 1]) # Scipy quat (x,y,z,w)

        for frame_idx in range(seq_len):
            # Y축 각속도를 이용해 현재 프레임의 facing rotation 업데이트
            if frame_idx > 0:
                ang_vel_rad = root_y_angular_velocity[frame_idx - 1]
                rot_change = Rotation.from_euler('y', ang_vel_rad, degrees=False)
                current_facing_rot = current_facing_rot * rot_change

            # 글로벌 회전 계산
            # [w,x,y,z] -> [x,y,z,w]로 변환하여 Scipy에 전달
            root_local_rot = Rotation.from_quat(root_local_quats_wxyz[frame_idx, [1, 2, 3, 0]])
            current_global_rot = current_facing_rot * root_local_rot
            final_root_global_quats_xyzw[frame_idx] = current_global_rot.as_quat()

            # 다음 프레임 위치 계산
            if frame_idx < seq_len - 1:
                local_vel_vec = np.array([root_xz_velocity[frame_idx, 0], 0, root_xz_velocity[frame_idx, 1]])
                world_increment = current_facing_rot.apply(local_vel_vec)
                final_root_positions[frame_idx + 1, [0, 2]] = final_root_positions[frame_idx, [0, 2]] + world_increment[[0, 2]]
                final_root_positions[frame_idx + 1, 1] = root_y_height[frame_idx + 1]

        # Scipy의 [x,y,z,w] quat -> Euler 변환
        joint_local_quats_xyzw = joint_local_quats_wxyz[:, :, [1, 2, 3, 0]]
        
        all_bvh_rotations_quat_xyzw = np.concatenate([
            final_root_global_quats_xyzw[:, np.newaxis, :],
            joint_local_quats_xyzw
        ], axis=1)
        
        final_all_euler_deg = Rotation.from_quat(
            all_bvh_rotations_quat_xyzw.reshape(-1, 4)
        ).as_euler('yxz', degrees=True).reshape(seq_len, -1)
        
        motion_data_flat = np.concatenate([final_root_positions, final_all_euler_deg], axis=1)
        
        output_path = os.path.join(output_dir, f"generated_motion_{i+1}.bvh")
        with open(output_path, 'w') as f:
            f.writelines(header_lines)
            f.write("MOTION\n")
            f.write(f"Frames: {seq_len}\n")
            f.write(frame_time_line)
            motion_lines = [" ".join(f"{x:.6f}" for x in frame) for frame in motion_data_flat]
            f.write("\n".join(motion_lines))
            
    print(f"\nAll {num_samples} motions saved to '{output_dir}' directory.")