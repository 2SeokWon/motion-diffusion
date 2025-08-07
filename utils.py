import torch
import torch.nn.functional as F
import numpy as np
import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation
from kinematics import matrix_to_quaternion, sixd_to_rotation_matrix

def generate_and_save_bvh(model, diffusion, dataset, num_samples, seq_len, input_feats, root_motion_features, template_path, output_dir, device):
    print("Starting sampling...")
    model.eval()
    input_feats = input_feats
    
    with torch.no_grad():
        sample_shape = (num_samples, seq_len, input_feats)
        generated_motion_norm = diffusion.p_sample_loop(model, sample_shape)

        mean_pos_vel = torch.from_numpy(dataset.pos_vel_mean).float().to(device)
        std_pos_vel = torch.from_numpy(dataset.pos_vel_std).float().to(device)
        mean_rotation = torch.from_numpy(dataset.rotation_mean).float().to(device)
        std_rotation = torch.from_numpy(dataset.rotation_std).float().to(device)
        
        mean = torch.cat([mean_pos_vel, mean_rotation], dim=1)
        std = torch.cat([std_pos_vel, std_rotation], dim=1)

        generated_motion = generated_motion_norm * std + mean
    
    print("Converting generated motions to BVH files...")
    # --- 스켈레톤 템플릿 읽기 ---
    with open(template_path, 'r') as f: lines = f.readlines()
    
    header_end_index = -1
    for i, line in enumerate(lines):
        if "MOTION" in line.upper(): 
            header_end_index = i
            break
    
    header_lines = lines[:header_end_index + 1] # MOTION 라인까지 포함
    
    frame_time_line = f"Frame Time: {1.0/30.0:.8f}\n"
    
    # --- 각 샘플을 BVH로 변환 ---
    for i in range(num_samples):
        print(f"Processing sample {i+1}/{num_samples}...")

        single_motion = generated_motion[i].cpu()

        # [버그 수정 3] 새로운 141 feature 구조에 맞게 특징을 분해합니다.
        root_y_height = single_motion[:, 0]
        root_xz_velocity_local = single_motion[:, 1:3]
        all_joint_6d_rotations_flat = single_motion[:, 3:]

        num_all_joints = all_joint_6d_rotations_flat.shape[1] // 6
        all_joint_6d_reshaped = all_joint_6d_rotations_flat.reshape(seq_len, num_all_joints, 6)
        
        # 6D -> 오일러 각도로 변환 (Local Rotation)
        all_local_rotmats = sixd_to_rotation_matrix(all_joint_6d_reshaped)
        # 중요: 생성 시에는 Root의 회전도 Local로 간주하고 누적하여 Global을 만듭니다.
        all_local_quat = matrix_to_quaternion(all_local_rotmats.view(-1, 3, 3)).view(seq_len, num_all_joints, 4)

        # 위치 및 글로벌 회전 계산
        final_root_positions = torch.zeros(seq_len, 3)
        final_root_global_quats = torch.zeros(seq_len, 4)
        
        # 첫 프레임 초기화
        final_root_positions[0, 1] = root_y_height[0]
        final_root_global_quats[0] = torch.tensor([1.0, 0.0, 0.0, 0.0]) # w,x,y,z 순서, 정면을 보는 상태에서 시작

        for frame_idx in range(seq_len - 1):
            # [버그 수정 4] 위치 계산 로직을 check_preprocessing.py와 동일하게 수정합니다.
            
            # 1. 현재 프레임의 글로벌 방향을 가져옵니다.
            current_global_rot = Rotation.from_quat(final_root_global_quats[frame_idx].numpy()[[1,2,3,0]]) # scipy는 x,y,z,w
            
            # 2. 로컬 속도를 월드 이동량으로 변환합니다.
            local_vel = np.array([root_xz_velocity_local[frame_idx, 0], 0, root_xz_velocity_local[frame_idx, 1]])
            world_vel_increment = current_global_rot.apply(local_vel)

            # 3. 다음 프레임의 XZ 위치를 계산합니다.
            final_root_positions[frame_idx + 1, [0, 2]] = final_root_positions[frame_idx, [0, 2]] + torch.from_numpy(world_vel_increment)[[0, 2]]
            
            # 4. 다음 프레임의 Y 높이는 특징 벡터에서 직접 가져옵니다.
            final_root_positions[frame_idx + 1, 1] = root_y_height[frame_idx + 1]

            # 5. 다음 프레임의 글로벌 회전을 계산합니다. (현재 글로벌 회전 * 로컬 회전 변화량)
            local_rot_change = Rotation.from_quat(all_local_quat[frame_idx, 0].numpy()[[1,2,3,0]]) # scipy는 x,y,z,w
            next_global_rot = current_global_rot * local_rot_change
            final_root_global_quats[frame_idx + 1] = torch.tensor(next_global_rot.as_quat()[[3,0,1,2]]) # w,x,y,z로 저장

        # 최종 BVH 데이터 조립
        all_bvh_rotations_quat = torch.cat([
            final_root_global_quats.unsqueeze(1),
            all_local_quat[:, 1:, :] # 나머지 관절은 로컬 회전 그대로 사용
        ], dim=1)
        
        final_all_euler_deg = Rotation.from_quat(
            all_bvh_rotations_quat.reshape(-1, 4)[:, [1,2,3,0]] # Scipy quat (x,y,z,w)
        ).as_euler('yxz', degrees=True).reshape(seq_len, -1)
    
        motion_data_flat = torch.cat([final_root_positions, torch.from_numpy(final_all_euler_deg)], dim=1)
        
        # ... 파일 저장 로직 (기존과 거의 동일) ...
        output_path = os.path.join(output_dir, f"generated_motion_{i+1}.bvh")
        with open(output_path, 'w') as f:
            f.writelines(header_lines)
            f.write(f"Frames: {seq_len}\n")
            f.write(frame_time_line)
            motion_lines = [" ".join(f"{x:.6f}" for x in frame) for frame in motion_data_flat.cpu().numpy()]
            f.write("\n".join(motion_lines))
            
    print(f"\nAll motions saved to '{output_dir}' directory.")