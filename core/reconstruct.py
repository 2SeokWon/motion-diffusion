# core/reconstruct.py
# 생성 특징 텐서 → Motion 객체 복원. bvh_viewer/render_video.py에서 옮겨 왔다.
# render_video.py는 pygame/OpenGL을 import하므로, 화면이 없는 GPU 서버에서도 생성·평가가 되도록 렌더링과 분리한다.
import numpy as np
import torch
from pyglm import glm
from tqdm import tqdm

from bvh_viewer.BVH_Parser import bvh_parser, Motion, MotionFrame, Joint, get_preorder_joint_list
from core.kinematics import sixd_to_rotation_matrix


def tensor_to_motion_object(generated_tensor: np.ndarray, template_bvh_path: str, FPS=60) -> (Joint, Motion):
    """
    모델이 생성하고 역정규화한 특징 텐서를 bvh_parser의 Motion 객체로 변환합니다.
    """
    print("Converting tensor to Motion object...")

    root, _ = bvh_parser(template_bvh_path)
    # quaternion_frame이 비어있을 수 있으므로, bvh_parser가 생성한 joint 계층구조에서 순서를 가져옵니다.
    joint_order = [j.name for j in get_preorder_joint_list(root) if "Site" not in j.name]

    num_joints = len(joint_order)
    sixd_dim = num_joints * 6
    pos_dim = (num_joints - 1) * 3
    sixd_start = 4 + pos_dim

    num_frames = generated_tensor.shape[0]
    motion_obj = Motion(frames=[], frame_time=1.0/FPS, frame_len=num_frames)

    current_global_pos_glm = glm.vec3(0.0, 0.0, 0.0)
    current_vr_rot_glm = glm.quat(1.0, 0.0, 0.0, 0.0)

    for i in tqdm(range(num_frames), desc="Reconstructing Motion"):
        frame_features = generated_tensor[i]

        root_y_height = frame_features[0]
        root_xz_velocity_local = frame_features[1:3]
        root_y_angular_velocity = frame_features[3]
        all_joint_6d = frame_features[sixd_start:sixd_start + sixd_dim].reshape(-1, 6)

        # 1. 6D를 쿼터니언으로 미리 변환
        all_joint_rotmats_torch = sixd_to_rotation_matrix(torch.from_numpy(all_joint_6d))
        all_joint_quats_glm = [glm.quat_cast(glm.mat3(rot.numpy())) for rot in all_joint_rotmats_torch]
        root_local_rot_glm = all_joint_quats_glm[0]

        # Virtual Root의 회전 처리
        rot_change = glm.angleAxis(root_y_angular_velocity, glm.vec3(0, 1, 0))
        current_vr_rot_glm = current_vr_rot_glm @ rot_change
        current_vr_rot_glm = glm.normalize(current_vr_rot_glm)

        current_hip_global_rot_glm = current_vr_rot_glm @ root_local_rot_glm
        velocity_vec_local = glm.vec3(root_xz_velocity_local[0], 0, root_xz_velocity_local[1])
        world_increment = current_vr_rot_glm * velocity_vec_local

        current_global_pos_glm += glm.vec3(world_increment) #virtual root의 위치
        current_global_pos_glm.y = 0.0

        motion_frame = MotionFrame()

        vr_translation = glm.translate(glm.mat4(1.0), current_global_pos_glm)
        vr_rotation = glm.mat4_cast(current_vr_rot_glm)
        motion_frame.virtual_transform = vr_translation @ vr_rotation

        t_hip_global = glm.translate(glm.mat4(1.0), current_global_pos_glm) @ glm.mat4_cast(current_hip_global_rot_glm)
        t_local_hip = glm.inverse(motion_frame.virtual_transform) @ t_hip_global
        motion_frame.hip_local_position = glm.vec3(t_local_hip[3])
        motion_frame.hip_local_position.y = root_y_height
        t_local_hip[3][1] = root_y_height
        motion_frame.joint_rotations[root.name] = root_local_rot_glm

        # 나머지 관절들의 지역 회전 저장
        for idx, joint_name in enumerate(joint_order):
            if idx > 0:
                motion_frame.joint_rotations[joint_name] = all_joint_quats_glm[idx]

        motion_obj.quaternion_frame.append(motion_frame)

    print("Performing Forward Kinematics for all frames...")
    for frame in tqdm(motion_obj.quaternion_frame, desc="Calculating FK"):
        motion_obj.compute_forward_kinematics(root, frame.virtual_transform, frame)
    print("Conversion complete.")
    return root, motion_obj
