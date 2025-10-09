import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation

from kinematics import sixd_to_rotation_matrix

def qmul_wxyz(q1, q2):
    # (w,x,y,z) -> (x,y,z,w)
    q1_scipy = q1[[1, 2, 3, 0]]
    q2_scipy = q2[[1, 2, 3, 0]]
    
    r1 = Rotation.from_quat(q1_scipy)
    r2 = Rotation.from_quat(q2_scipy)
    
    result_rot = r1 * r2
    result_quat_scipy = result_rot.as_quat()
    
    # (x,y,z,w) -> (w,x,y,z)
    return np.array([result_quat_scipy[3], result_quat_scipy[0], result_quat_scipy[1], result_quat_scipy[2]])

def reconstruct_bvh_from_features(
    features_unnormalized,
    seq_len,
    template_path,
    output_path
):

    print(f"Reconstructing BVH file at: {output_path}")

    # --- 1. 특징 벡터 분해 ---
    # 특징 순서: root(4) -> rotation(138) -> position(66)
    root_y_height = features_unnormalized[:seq_len, 0]
    root_xz_velocity_local = features_unnormalized[:seq_len, 1:3]
    root_angular_velocity_y = features_unnormalized[:seq_len, 3]
    all_local_6d_rotations = features_unnormalized[:seq_len, 70:208]
    # local_joint_positions_flat = features_unnormalized[:, 142:] # 재구성에는 사용 안 함

    num_joints = all_local_6d_rotations.shape[1] // 6

    # --- 2. 6D -> Quaternion 변환 ---
    all_local_6d_reshaped = torch.from_numpy(all_local_6d_rotations).float().reshape(seq_len, num_joints, 6)
    all_local_rotmats = sixd_to_rotation_matrix(all_local_6d_reshaped).numpy()
    # (seq_len * num_joints, 3, 3) 형태의 행렬들을 가져옵니다.
    reshaped_mats = all_local_rotmats.reshape(-1, 3, 3)

    # 각 3x3 행렬을 전치합니다.
    transposed_mats = reshaped_mats.transpose((0, 2, 1))

    # 전치된 행렬로부터 쿼터니언을 생성합니다.
    all_local_quats_scipy = Rotation.from_matrix(transposed_mats).as_quat().reshape(seq_len, num_joints, 4)
    # 루트와 나머지 관절의 로컬 회전을 분리
    root_local_quats_scipy = all_local_quats_scipy[:, 0, :]
    joint_local_quats_scipy = all_local_quats_scipy[:, 1:, :]

    # --- 3. 경로 및 전역 회전 적분 ---
    final_root_positions = np.zeros((seq_len, 3))
    final_root_global_quats_scipy = np.zeros((seq_len, 4))
    
    # 초기값 설정
    final_root_positions[0, 1] = root_y_height[0]
    current_facing_rot = Rotation.from_quat([0, 0, 0, 1]) # 초기 방향

    for i in range(seq_len):
        # 가상 루트(facing rotation) 업데이트: Y축 각속도를 적분
        if i > 0:
            ang_vel_rad = root_angular_velocity_y[i-1]
            rotation_vector = [0, ang_vel_rad, 0] 
            rot_change = Rotation.from_rotvec(rotation_vector)
            current_facing_rot = current_facing_rot * rot_change
        
        # 전역 루트 회전 계산: global = virtual * local
        root_local_rot = Rotation.from_quat(root_local_quats_scipy[i])
        current_global_rot = current_facing_rot * root_local_rot
        print(current_global_rot.as_quat())
        final_root_global_quats_scipy[i] = current_global_rot.as_quat()

        # 전역 위치 업데이트: 로컬 속도를 전역 속도로 변환하여 적분
        if i < seq_len - 1:
            local_vel = np.array([root_xz_velocity_local[i, 0], 0, root_xz_velocity_local[i, 1]])
            world_vel_increment = current_facing_rot.apply(local_vel)

            final_root_positions[i + 1] = final_root_positions[i] + world_vel_increment
            final_root_positions[i + 1, 1] = root_y_height[i + 1]

    # --- 4. 최종 BVH 데이터 조립 ---
    all_bvh_rotations_quat = np.concatenate([
        final_root_global_quats_scipy[:, np.newaxis, :],
        joint_local_quats_scipy
    ], axis=1)

    final_all_euler_deg = Rotation.from_quat(
        all_bvh_rotations_quat.reshape(-1, 4)
    ).as_euler('yxz', degrees=True).reshape(seq_len, -1)

    print("Applying Euler continuity filter to fix joint jitter...")
    num_joints_in_motion = final_all_euler_deg.shape[1] // 3
    for j in range(num_joints_in_motion):
        for axis in range(3):
            col_idx = j * 3 + axis
            for i in range(1, seq_len):
                diff = final_all_euler_deg[i, col_idx] - final_all_euler_deg[i-1, col_idx]
                if diff > 180:
                    final_all_euler_deg[i:, col_idx] -= 360
                elif diff < -180:
                    final_all_euler_deg[i:, col_idx] += 360

    motion_data_flat = np.concatenate([final_root_positions, final_all_euler_deg], axis=1)

    # --- 5. 파일 저장 (기존과 동일) ---
    with open(template_path, 'r') as f: lines = f.readlines()
    header_end_index = -1
    for i, line in enumerate(lines):
        if "MOTION" in line.upper(): header_end_index = i; break
    header_lines = lines[:header_end_index]
    frame_time = 1.0 / 60.0 

    with open(output_path, 'w') as f:
        f.writelines(header_lines)
        f.write("MOTION\n")
        f.write(f"Frames: {seq_len}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")
        motion_lines = [" ".join(f"{x:.6f}" for x in frame) for frame in motion_data_flat]
        f.write("\n".join(motion_lines))
        
    print("BVH reconstruction complete.")

def main():
    processed_data_path = "./processed_data_new"
    output_bvh_dir = "./verification"
    skeleton_template_path = "./dataset/Drunk_TR1.bvh" 
    os.makedirs(output_bvh_dir, exist_ok=True)

    print("--- Verifying Preprocessing Pipeline (Norm/De-norm -> BVH) ---")
    
    clip_filename = "clip_0047.npz"
    clip_path = os.path.join(processed_data_path, clip_filename)

    if not os.path.exists(clip_path):
        print(f"Error: Test clip '{clip_path}' not found. Run preprocess_bvh.py first.")
        return

    with np.load(clip_path) as data:
        features_original = data['features']
    
    seq_len = 180

    print("\n--- Step 1 & 2: Normalization & De-normalization ---")
    try:
        pos_vel_mean = np.load(os.path.join(processed_data_path, "pos_vel_mean.npy"))
        pos_vel_std = np.load(os.path.join(processed_data_path, "pos_vel_std.npy"))
        position_mean = np.load(os.path.join(processed_data_path, "position_mean.npy"))
        position_std = np.load(os.path.join(processed_data_path, "position_std.npy"))
        rotation_mean = np.load(os.path.join(processed_data_path, "rotation_mean.npy"))
        rotation_std = np.load(os.path.join(processed_data_path, "rotation_std.npy"))

        mean = np.concatenate([pos_vel_mean, position_mean, rotation_mean])
        std = np.concatenate([pos_vel_std, position_std, rotation_std])

        features_normalized = (features_original[:, :208] - mean) / std
        features_reconstructed = features_normalized * std + mean
        
        is_close = np.allclose(features_original[:, :208], features_reconstructed)
        print(f"Mathematical integrity check (is data recovered?): {is_close}")
        if not is_close:
            print("WARNING: Mathematical integrity failed.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not load statistics files. Details: {e}")
        return

    print("\n--- Step 3: Reconstructing BVH from de-normalized features ---")
    output_filename = f"reconstructed_{os.path.basename(clip_path).replace('.npz', '')}.bvh"
    output_path = os.path.join(output_bvh_dir, output_filename)

    reconstruct_bvh_from_features(
        features_reconstructed,
        seq_len,
        skeleton_template_path,
        output_path
    )

if __name__ == '__main__':
    main()