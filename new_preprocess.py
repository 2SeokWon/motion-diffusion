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

# --- 1. 설정 (Configuration) ---
bvh_folder_path = "./dataset/"
output_processed_dir = "./processed_data_cond/"
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
    global_y_angular = []
    
    """
    foot_joints = {
        "RightAnkle" : 0,
        "RightToe" : 1,
        "LeftAnkle" : 2,
        "LeftToe" : 3
    }
    height_threshold = 5.0
    velocity_threshold = 2.0
    dt = 1 / 60.0

    foot_positions = [
        np.zeros((clip_length, 3)),  # right_heel
        np.zeros((clip_length, 3)),  # right_toe
        np.zeros((clip_length, 3)),  # left_heel
        np.zeros((clip_length, 3))   # left_toe
    ]
    
    height_contacts = np.zeros((clip_length, 4), dtype=int)
    """

    ordered_joints = get_preorder_joint_list(motion.root)
    joints_to_process = [j for j in ordered_joints if "End" not in j.name]

    for rel_i, abs_i in enumerate(range(start_frame, start_frame + clip_length)):
        frame = motion.quaternion_frame[abs_i] #MotionFrame 객체
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
        root_y_angular_velocity.append(angular_velocity)
        prev_yaw = vr_yaw
        #global_y_angular.append(vr_yaw) # global yaw for trajectory

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
        root_xz_velocity.append(linear_velocity_local)
        prev_global_pos_xz = vr_pos_xz
        global_pos_xz.append(vr_pos_xz) # global position for trajectory

        # Matrix to 6D
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
        
        # Local Position
        root_p_global = np.array(frame.joint_positions[motion.root.name])
        root_R_global = np.array(frame.joint_global_transforms[motion.root.name])[:3, :3]
        root_R_global_inv = np.linalg.inv(root_R_global)

        local_posis = []
        for joint in joints_to_process:
            if joint.name == motion.root.name:
                continue      
            p_global = np.array(frame.joint_positions[joint.name])
            p_diff = p_global - root_p_global
            p_local = root_R_global_inv @ p_diff
            local_posis.append(p_local)
            """
            #Foot contact 처리
            if joint.name in foot_joints:
                foot_idx = foot_joints[joint.name]
                foot_positions[foot_idx][rel_i] = p_global #[foot_idx, frame, xyz]
                height_contact = int(p_global[1] < height_threshold)
                height_contacts[rel_i, foot_idx] = height_contact
            """
        local_joint_positions_flat.append(np.concatenate(local_posis))  # flat
    """
    #Foot Velocity & Contact
    foot_velocities = []
    for foot_pos in foot_positions: #[frame, xyz]
        if foot_pos.shape[0] < 2:
            vel = np.zeros(foot_pos.shape[0])  # 클립이 너무 짧으면 0
        else:
            vel = np.linalg.norm(np.diff(foot_pos, axis=0), axis=1) / dt  # 속도 (magnitude / dt)
            vel = np.insert(vel, 0, 0.0)  # 첫 프레임 0 패딩
        foot_velocities.append(vel)
    
    foot_contacts = np.zeros((clip_length, 4), dtype=int)
    for idx in range(4):
        vel_contact = (foot_velocities[idx] < velocity_threshold).astype(int)
        foot_contacts[:, idx] = height_contacts[:, idx] & vel_contact
    """
    root_y_height = np.array(root_y_height).reshape(-1, 1)
    root_xz_velocity = np.array(root_xz_velocity)
    root_y_angular_velocity = np.array(root_y_angular_velocity)[:, np.newaxis]
    local_joint_positions_flat = np.array(local_joint_positions_flat)
    all_joint_6d_rotations = np.array(all_joint_6d_rotations)
    """
    global_pos_xz = np.array(global_pos_xz)
    if len(global_pos_xz) > 0:
        global_pos_xz -= global_pos_xz[0]

    global_y_angular = np.array(global_y_angular)[:, np.newaxis]
    if len(global_y_angular) > 0:
        global_y_angular -= global_y_angular[0]
    """
    final_features = np.concatenate([
        root_y_height, root_xz_velocity, root_y_angular_velocity,
        local_joint_positions_flat, all_joint_6d_rotations,
        #global_pos_xz, global_y_angular #Trajectory (global xz, global yaw)
        #foot_contacts.astype(np.float32)
    ], axis=1) #211
    
    return final_features, global_pos_xz

def process_single_file(idx, filename, class_name, class_name_idx, class_type, class_type_idx, feature_dim):
    filepath = os.path.join(bvh_folder_path, filename)
    local_count = 0
    local_sum = np.zeros(feature_dim)
    local_sum_sq = np.zeros(feature_dim)
    clip_info = None  # npz 저장용

    try:
        root, motion = bvh_parser(filepath)
        motion.list_to_quaternion(root)
        motion.save_virtual_root_info(root)
        # 전체 motion의 npz 저장
        final_features, _ = extract_features(motion, 0, motion.frame_len)
        if not np.isfinite(final_features).all():
            nan_count = np.isnan(final_features).sum()
            inf_count = np.isinf(final_features).sum()
            print(f"Warning: NaN({nan_count})/Inf({inf_count}) in {filename}. Skipping stats.")
        else:
            local_count += final_features.shape[0]
            local_sum += np.sum(final_features, axis=0)
            local_sum_sq += np.sum(final_features**2, axis=0)

        clip_filename = f"clip_{idx:04d}.npz"
        clip_filepath = os.path.join(output_processed_dir, clip_filename)
        np.savez(clip_filepath, features=final_features)
        clip_info = {"path": clip_filename, 
                     "length": final_features.shape[0], 
                     "class_name": class_name,
                     "class_name_idx": class_name_idx,
                     "class_type": class_type,
                     "class_type_idx": class_type_idx}

    except Exception as e:
        print(f"Error in {filename}: {e}")
        traceback.print_exc()

    return local_count, local_sum, local_sum_sq, clip_info

# --- 3. 전처리 메인 로직 ---

def main():
    print("\n--- Step 1: Extracting Features from BVH Files ---")
    bvh_files = [f for f in os.listdir(bvh_folder_path) if f.endswith(".bvh")]
    class_names = sorted(list(set([f.split('_')[0] for f in bvh_files])))
    class_types = sorted(list(set([f.split('_')[1].split('.')[0] for f in bvh_files])))

    class_name_map = {name: i for i, name in enumerate(class_names)}
    class_type_map = {type: i for i, type in enumerate(class_types)}

    feature_dim = 208

    tasks_to_run = []
    for idx, filename in enumerate(bvh_files):
        class_name = filename.split('_')[0]
        class_type = filename.split('_')[1].split('.')[0]
        if class_name in class_name_map:
            class_name_idx = class_name_map[class_name]
        if class_type in class_type_map:
            class_type_idx = class_type_map[class_type]

        tasks_to_run.append(
            delayed(process_single_file)(idx, filename, class_name, class_name_idx, class_type, class_type_idx, feature_dim)
        )

    n_jobs = -1
    results = Parallel(n_jobs=n_jobs)(
        tqdm(tasks_to_run, desc="Parallel processing BVH files")
    )

    # 결과 취합 (병렬 반환값 모아서 통계 계산)
    total_count = 0
    total_sum = np.zeros(feature_dim)
    total_sum_sq = np.zeros(feature_dim)
    all_motion_clips = []

    for local_count, local_sum, local_sum_sq, clip_info in results:
        total_count += local_count
        total_sum += local_sum
        total_sum_sq += local_sum_sq
        if clip_info:
            all_motion_clips.append(clip_info)

    print("Calculating mean and std for the entire dataset...")
    if total_count == 0:
        raise ValueError("No valid data processed for stats.")
    
    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - (mean ** 2)
    variance = np.maximum(variance, 0)
    std = np.sqrt(variance)

    pos_vel_mean = mean[:4]
    pos_vel_std = std[:4]
    position_mean = mean[4:70]
    position_std = std[4:70]
    rotation_mean = mean[70:208]
    rotation_std = std[70:208]
    #foot_mean = mean[208:]
    #foot_std = std[208:]
    #global_pos_mean = mean[208:211]
    #global_pos_std = std[208:211]

    np.save(os.path.join(output_processed_dir, "pos_vel_mean.npy"), pos_vel_mean)
    np.save(os.path.join(output_processed_dir, "pos_vel_std.npy"), pos_vel_std)
    np.save(os.path.join(output_processed_dir, "position_mean.npy"), position_mean)
    np.save(os.path.join(output_processed_dir, "position_std.npy"), position_std)
    np.save(os.path.join(output_processed_dir, "rotation_mean.npy"), rotation_mean)
    np.save(os.path.join(output_processed_dir, "rotation_std.npy"), rotation_std)
    #np.save(os.path.join(output_processed_dir, "foot_mean.npy"), foot_mean)
    #np.save(os.path.join(output_processed_dir, "foot_std.npy"), foot_std)
    #np.save(os.path.join(output_processed_dir, "global_pos_mean.npy"), global_pos_mean)
    #np.save(os.path.join(output_processed_dir, "global_pos_std.npy"), global_pos_std)

    # 최종 메타데이터 파일 저장
    with open(output_metadata_path, 'w') as f:
        json.dump(all_motion_clips, f, indent=4)

    print("\nPreprocessing complete.")
    print(f"Processed clips and metadata saved to '{output_processed_dir}'")


if __name__ == '__main__':
    main()