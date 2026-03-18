# core/motion_features.py
import numpy as np
import torch
import math
import torch.nn.functional as F
from pyglm import glm
from bvh_viewer.BVH_Parser import get_preorder_joint_list

ROTATION_ORDER = 'yxz'
CLIP_LENGTH = 180
STRIDE = 5
ROOT_DISP_DIM = 3


def extract_features(motion, start_frame, clip_length):
    root_y_height = []
    root_xz_velocity = []
    root_y_angular_velocity = []
    local_joint_positions_flat = []
    all_joint_6d_rotations = []

    prev_yaw = None
    prev_global_pos_xz = None

    foot_joints = {
        "RightToe": 0,
        "LeftToe": 1
    }
    height_threshold = 3.0

    foot_positions = [
        np.zeros((clip_length, 3)),  # right_toe
        np.zeros((clip_length, 3))   # left_toe
    ]

    foot_contacts = np.zeros((clip_length, 2), dtype=int)

    ordered_joints = get_preorder_joint_list(motion.root)
    joints_to_process = [j for j in ordered_joints if "End" not in j.name]

    for rel_i, abs_i in enumerate(range(start_frame, start_frame + clip_length)):
        frame = motion.quaternion_frame[abs_i]
        vr_global_matrix = np.array(frame.virtual_transform)
        vr_rot = vr_global_matrix[:3, :3]
        vr_pos = vr_global_matrix[:3, 3]

        vr_yaw = math.atan2(vr_rot[0, 2], vr_rot[2, 2])

        if prev_yaw is None:
            angular_velocity = 0.0
        else:
            angular_velocity = vr_yaw - prev_yaw
            if angular_velocity > math.pi:  angular_velocity -= 2 * math.pi
            if angular_velocity < -math.pi: angular_velocity += 2 * math.pi
        root_y_angular_velocity.append(angular_velocity)
        prev_yaw = vr_yaw

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

            if joint.name in foot_joints:
                foot_idx = foot_joints[joint.name]
                foot_positions[foot_idx][rel_i] = p_global
                height_contact = int(p_global[1] < height_threshold)
                foot_contacts[rel_i, foot_idx] = height_contact

        local_joint_positions_flat.append(np.concatenate(local_posis))

    root_y_height = np.array(root_y_height).reshape(-1, 1)
    root_xz_velocity = np.array(root_xz_velocity)
    root_y_angular_velocity = np.array(root_y_angular_velocity)
    local_joint_positions_flat = np.array(local_joint_positions_flat)
    all_joint_6d_rotations = np.array(all_joint_6d_rotations)

    final_features = np.concatenate([
        root_y_height,                          # 1
        root_xz_velocity,                       # 2
        root_y_angular_velocity[:, np.newaxis], # 1
        local_joint_positions_flat,             # 66
        all_joint_6d_rotations,                 # 138
        foot_contacts.astype(np.float32),       # 2
    ], axis=1)  # 210

    return final_features


def tensor_to_motion_object_root(generated_tensor: np.ndarray) -> np.ndarray:
    """
    로컬 root 속도/각속도를 적분해서 virtual root의 traj(x, z, yaw)를 반환합니다.
    shape (T, 3) numpy 배열을 반환합니다. 시작 상태: pos=(0,0), yaw=0.
    """
    num_frames = generated_tensor.shape[0]
    traj = np.zeros((num_frames, 3), dtype=np.float32)

    current_global_pos = glm.vec3(0.0, 0.0, 0.0)
    current_vr_rot = glm.quat(1.0, 0.0, 0.0, 0.0)

    for i in range(num_frames):
        frame_features = generated_tensor[i]
        root_xz_velocity_local = frame_features[1:3]
        root_y_angular_velocity = frame_features[3]

        rot_change = glm.angleAxis(root_y_angular_velocity, glm.vec3(0, 1, 0))
        current_vr_rot = glm.normalize(current_vr_rot @ rot_change)

        velocity_vec_local = glm.vec3(root_xz_velocity_local[0], 0, root_xz_velocity_local[1])
        world_increment = current_vr_rot * velocity_vec_local
        current_global_pos += glm.vec3(world_increment)
        current_global_pos.y = 0.0

        rot_mat = np.array(glm.mat4_cast(current_vr_rot))
        yaw = math.atan2(rot_mat[0][2], rot_mat[2][2])

        traj[i, 0] = current_global_pos.x
        traj[i, 1] = current_global_pos.z
        traj[i, 2] = yaw

    if num_frames > 1:
        traj[:, 2] = np.unwrap(traj[:, 2], period=2 * math.pi)
    return traj


def moving_average_path(raw_pos_xz, raw_yaw, radius=30):
    win = 2 * radius + 1
    pad_mode = "edge"

    pad_x = np.pad(raw_pos_xz[:, 0], radius, mode=pad_mode)
    pad_z = np.pad(raw_pos_xz[:, 1], radius, mode=pad_mode)
    pad_yaw = np.pad(np.unwrap(raw_yaw), radius, mode=pad_mode)

    csum_x = np.cumsum(np.insert(pad_x, 0, 0), dtype=np.float64)
    csum_z = np.cumsum(np.insert(pad_z, 0, 0), dtype=np.float64)
    csum_yaw = np.cumsum(np.insert(pad_yaw, 0, 0), dtype=np.float64)

    interp_x = (csum_x[win:] - csum_x[:-win]) / win
    interp_z = (csum_z[win:] - csum_z[:-win]) / win
    interp_yaw = (csum_yaw[win:] - csum_yaw[:-win]) / win
    interp_yaw = (interp_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]

    interp_pos_xz = np.stack([interp_x, interp_z], axis=1)
    return np.concatenate([interp_pos_xz, interp_yaw[:, np.newaxis]], axis=1)


def compute_delta_traj(raw_pos_xz: np.ndarray, raw_yaw: np.ndarray,
                       interp_pos_xz: np.ndarray, interp_yaw: np.ndarray) -> np.ndarray:
    """
    보간 궤적 대비 잔차(delta_x, delta_z, delta_yaw)를 반환합니다.
    delta_xz는 interp_yaw 좌표계 기준 로컬 값입니다.
    """
    delta_world = raw_pos_xz - interp_pos_xz
    cos_t = np.cos(-interp_yaw)
    sin_t = np.sin(-interp_yaw)
    delta_x = delta_world[:, 0] * cos_t - delta_world[:, 1] * sin_t
    delta_z = delta_world[:, 0] * sin_t + delta_world[:, 1] * cos_t
    delta_yaw = np.unwrap(raw_yaw - interp_yaw)

    return np.concatenate([
        np.stack([delta_x, delta_z], axis=1),
        delta_yaw[:, np.newaxis]
    ], axis=1)
