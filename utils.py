import torch
import torch.nn.functional as F
import numpy as np
import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation, Slerp
from kinematics import matrix_to_quaternion, sixd_to_rotation_matrix

def generate_and_save_bvh(model, diffusion, dataset, num_samples, seq_len, input_feats, root_motion_features, template_path, output_dir, device):
    print("Starting sampling...")
    model.eval()
    input_feats = dataset.pos_vel_mean.shape[1] + dataset.rotation_mean.shape[1]

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

        # --- 4. 특징 분해 ---
        root_y_height = single_motion[:, 0]
        root_xz_velocity = single_motion[:, 1:3]
        all_joint_6d = single_motion[:, 3:].reshape(seq_len, -1, 6)
        all_local_rotmats = sixd_to_rotation_matrix(all_joint_6d)
        all_local_quat = matrix_to_quaternion(all_local_rotmats.view(-1, 3, 3)).view(seq_len, -1, 4)
        
        root_local_quats = all_local_quat[:, 0, :]
        joint_local_quats = all_local_quat[:, 1:, :]
        
        # 위치 및 글로벌 회전 계산
        final_root_positions = torch.zeros(seq_len, 3, dtype=torch.float32)
        final_root_global_quats = torch.zeros(seq_len, 4, dtype=torch.float32)

        facing_smoothing = 0.08
        movement_threshold = 0.1
        
        final_root_positions[0, 1] = root_y_height[0]
        current_facing_rot = Rotation.from_quat([0, 0, 0, 1])
        
        for frame_idx in range(seq_len - 1):
            root_local_rot = Rotation.from_quat(root_local_quats[frame_idx].numpy()[[1,2,3,0]])
            calculated_global_rot = current_facing_rot * root_local_rot
            
            final_root_global_quats[frame_idx] = torch.tensor(calculated_global_rot.as_quat()[[3,0,1,2]])

            if frame_idx < seq_len - 1:
                local_vel_vec = np.array([root_xz_velocity[frame_idx, 0], 0, root_xz_velocity[frame_idx, 1]])
                world_increment = calculated_global_rot.apply(local_vel_vec)
                final_root_positions[frame_idx + 1, [0, 2]] = final_root_positions[frame_idx, [0, 2]] + torch.from_numpy(world_increment).float()[[0, 2]]
                final_root_positions[frame_idx + 1, 1] = root_y_height[frame_idx + 1]

                move_direction = final_root_positions[frame_idx + 1] - final_root_positions[frame_idx]
                if torch.linalg.norm(move_direction) > movement_threshold:
                    move_dir_np = move_direction.numpy()
                    target_angle = np.arctan2(move_dir_np[0], move_dir_np[2])
                    target_rot = Rotation.from_euler('y', target_angle)
                    slerp = Slerp([0, 1], Rotation.from_quat([current_facing_rot.as_quat(), target_rot.as_quat()]))
                    current_facing_rot = slerp(facing_smoothing)
        # 최종 BVH 데이터 조립
        all_bvh_rotations_quat = torch.cat([final_root_global_quats.unsqueeze(1), joint_local_quats], dim=1)
        final_all_euler_deg = Rotation.from_quat(all_bvh_rotations_quat.reshape(-1, 4).numpy()[:, [1,2,3,0]]).as_euler('yxz', degrees=True)
        motion_data_flat = torch.cat([final_root_positions, torch.from_numpy(final_all_euler_deg).view(seq_len, -1).float()], dim=1)
        
        output_path = os.path.join(output_dir, f"generated_motion_{i+1}.bvh")
        with open(output_path, 'w') as f:
            f.writelines(header_lines)
            f.write(f"Frames: {seq_len}\n")
            f.write(frame_time_line)
            motion_lines = [" ".join(f"{x:.6f}" for x in frame) for frame in motion_data_flat.cpu().numpy()]
            f.write("\n".join(motion_lines))
            
    print(f"\nAll motions saved to '{output_dir}' directory.")