#new_preprocess.py
import os
import numpy as np
import torch
import math
from tqdm import tqdm
import json
import torch.nn.functional as F 
import traceback
from bvh_viewer.BVH_Parser import bvh_parser, get_preorder_joint_list

# --- 1. 설정 (Configuration) ---
bvh_folder_path = "./dataset/"
output_processed_dir = "./processed_data/"
template_bvh_path = "./dataset/Aeroplane_BR.bvh" 
output_metadata_path = os.path.join(output_processed_dir, "metadata.json")
os.makedirs(output_processed_dir, exist_ok=True)
ROTATION_ORDER = 'yxz'

def extract_features(motion, start_frame, clip_length):
    root_y_height = []
    root_xz_velocity = []
    root_y_angular_velocity = []
    local_joint_positions_flat = []
    all_joint_6d_rotations = []

    prev_yaw = None
    prev_global_pos_xz = None
    global_pos_xz = []

    ordered_joints = get_preorder_joint_list(motion.root)
    joints_to_process = [j for j in ordered_joints if "End" not in j.name]

    for i in range(start_frame, start_frame + clip_length):
        frame = motion.quaternion_frame[i] #MotionFrame 객체
        vr_global_matrix = np.array(frame.virtual_transform)
        vr_rot = vr_global_matrix[:3, :3]
        vr_pos = vr_global_matrix[:3, 3]

        vr_yaw = math.atan2(vr_rot[0, 2], vr_rot[2, 2])
        #y angular velocity 계산
        if prev_yaw is None:
            angular_velocity = 0.0
        else:
            angular_velocity = vr_yaw - prev_yaw
            if angular_velocity > math.pi:  angular_velocity -= 2 * math.pi
            if angular_velocity < -math.pi: angular_velocity += 2 * math.pi
        prev_yaw = vr_yaw

        #xz velocity 계산
        vr_pos_xz = np.array([vr_pos[0], vr_pos[2]])
        if prev_global_pos_xz is None:
            linear_velocity_local = np.zeros(2)
        else:
            linear_velocity_global = vr_pos_xz - prev_global_pos_xz
            world_vel_3d = np.array([linear_velocity_global[0], 0.0, linear_velocity_global[1]])
            inv_rot = np.linalg.inv(vr_rot)
            local_vel_3d = inv_rot @ world_vel_3d
            linear_velocity_local = np.array([local_vel_3d[0], local_vel_3d[2]])
 
        prev_global_pos_xz = vr_pos_xz
        global_pos_xz.append(vr_pos_xz)

        # 6D / position
        root_p_global = np.array(frame.joint_positions[motion.root.name])
        root_R_global = np.array(frame.joint_global_transforms[motion.root.name])[:3, :3]
        root_R_global_inv = np.linalg.inv(root_R_global)

        root_y_height.append(frame.hip_local_position.y)
        current_frame_rotations = []
        for joint in joints_to_process:
            T_local = np.array(frame.joint_local_transforms[joint.name])
            R_local = T_local[:3, :3]
            
            R_local_torch = torch.from_numpy(R_local).float()
            x_vec = F.normalize(R_local_torch[:, 0], dim=0)
            y_vec = F.normalize(R_local_torch[:, 1] - (x_vec * torch.dot(x_vec, R_local_torch[:, 1])), dim=0)
            sixd = np.concatenate([x_vec.numpy(), y_vec.numpy()])
            current_frame_rotations.append(sixd)
            
        all_joint_6d_rotations.append(np.concatenate(current_frame_rotations))

        local_posis = []
        for joint in joints_to_process:
            if joint.name == motion.root.name:
                continue      
            p_global = np.array(frame.joint_positions[joint.name])
            p_diff = p_global - root_p_global
            p_local = root_R_global_inv @ p_diff
            local_posis.append(p_local)

        local_joint_positions_flat.append(np.concatenate(local_posis))  # flat

        if i > 0:
            root_xz_velocity.append(linear_velocity_local)
            root_y_angular_velocity.append(angular_velocity)

    root_y_height = np.array(root_y_height).reshape(-1, 1)
    root_xz_velocity = np.pad(np.array(root_xz_velocity), ((1, 0), (0, 0)), mode='constant')
    root_y_angular_velocity = np.pad(np.array(root_y_angular_velocity), (1, 0), mode='constant')[:, np.newaxis]
    local_joint_positions_flat = np.array(local_joint_positions_flat)
    all_joint_6d_rotations = np.array(all_joint_6d_rotations)

    final_features = np.concatenate([
        root_y_height, root_xz_velocity, root_y_angular_velocity,
        local_joint_positions_flat, all_joint_6d_rotations
    ], axis=1)
    
    global_pos_xz = np.array(global_pos_xz)
    if len(global_pos_xz) > 0:
        global_pos_xz -= global_pos_xz[0]

    return final_features, global_pos_xz

# --- 3. 전처리 메인 로직 ---

def main():

    print("\n--- Step 1: Extracting Features from BVH Files ---")
    all_motion_clips = []
    bvh_files = [f for f in os.listdir(bvh_folder_path) if f.endswith(".bvh")]

    for idx, filename in enumerate(tqdm(bvh_files, desc="Processing BVH files")):
        filepath = os.path.join(bvh_folder_path, filename)
        try:
            root, motion = bvh_parser(filepath)
            motion.list_to_quaternion(root)
            motion.save_virtual_root_info(root)

            # 특징 추출 (새 함수 호출, 전체 모션)
            final_features, _ = extract_features(motion, 0, motion.frame_len)
            clip_filename = f"clip_{idx:04d}.npz"
            clip_filepath = os.path.join(output_processed_dir, clip_filename)

            np.savez(clip_filepath, features=final_features)

            all_motion_clips.append({
                "path": clip_filename,
                "length": final_features.shape[0]
            })

        except Exception as e:
            print(f"Could not process file {filename}. Error: {e}")
            traceback.print_exc()


    # --- 4. 최종 데이터 취합 및 저장 ---
    print("Calculating mean and std for the entire dataset...")
    all_clips_for_stats = []
    for metadata in tqdm(all_motion_clips, desc="Loading clips for stats"):
        # .npz 파일에서 'features' 키의 데이터만 불러와서 통계 계산
        with np.load(os.path.join(output_processed_dir, metadata['path'])) as data:
            clip_data = data['features']
            all_clips_for_stats.append(clip_data)

    print("\n--- Verifying final concatenated data before stats ---")
    full_dataset_np = np.concatenate(all_clips_for_stats, axis=0)
    print(f"Y Height Mean in full dataset: {full_dataset_np[:, 0].mean():.4f}")

    pos_vel_features = full_dataset_np[:, :4]  # Root position and velocity features
    position_features = full_dataset_np[:, 4:70]  # Joint positions
    rotation_features = full_dataset_np[:, 70:]  # Joint rotations

    pos_vel_mean = np.mean(pos_vel_features, axis=0, keepdims=True)
    pos_vel_std = np.std(pos_vel_features, axis=0, keepdims=True)
    pos_vel_std[pos_vel_std == 0] = 1e-7

    position_mean = np.mean(position_features, axis=0, keepdims=True)
    position_std = np.std(position_features, axis=0, keepdims=True)
    position_std[position_std == 0] = 1e-7

    rotation_mean = np.mean(rotation_features, axis=0, keepdims=True)
    rotation_std = np.std(rotation_features, axis=0, keepdims=True)
    rotation_std[rotation_std == 0] = 1e-7

    np.save(os.path.join(output_processed_dir, "pos_vel_mean.npy"), pos_vel_mean)
    np.save(os.path.join(output_processed_dir, "pos_vel_std.npy"), pos_vel_std)
    np.save(os.path.join(output_processed_dir, "position_mean.npy"), position_mean)
    np.save(os.path.join(output_processed_dir, "position_std.npy"), position_std)
    np.save(os.path.join(output_processed_dir, "rotation_mean.npy"), rotation_mean)
    np.save(os.path.join(output_processed_dir, "rotation_std.npy"), rotation_std)

    # 최종 메타데이터 파일 저장
    with open(output_metadata_path, 'w') as f:
        json.dump(all_motion_clips, f, indent=4)

    print("\nPreprocessing complete.")
    print(f"Processed clips and metadata saved to '{output_processed_dir}'")


if __name__ == '__main__':
    main()