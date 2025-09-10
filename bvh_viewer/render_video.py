import argparse
import os
import sys
import numpy as np
import torch
import imageio
from pyglm import glm
from tqdm import tqdm

os.environ['PYGAME_HIDE_SUPPORT_MPT'] = "1"
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

# --- 사용자님의 기존 모듈 임포트 ---
# 이 스크립트를 프로젝트 최상위 폴더에서 실행하거나,
# bvh_tools 폴더 등이 있는 경로를 sys.path에 추가해야 할 수 있습니다.
# 예: sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .BVH_Parser import bvh_parser, Motion, MotionFrame, Joint, get_preorder_joint_list
from .Rendering import draw_humanoid
from kinematics import sixd_to_rotation_matrix
from .Transforms import translation_matrix
from .utils import draw_axes, set_lights

# ----------------- 설정 -----------------
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
FPS = 60
# ----------------------------------------
prev_r_inv = glm.quat(1, 0, 0, 0)

def tensor_to_motion_object(generated_tensor: np.ndarray, template_bvh_path: str, FPS=60) -> (Joint, Motion):
    """
    모델이 생성하고 역정규화한 특징 텐서를 bvh_parser의 Motion 객체로 변환합니다.
    """
    print("Converting tensor to Motion object...")
    
    root, _ = bvh_parser(template_bvh_path)
    # quaternion_frame이 비어있을 수 있으므로, bvh_parser가 생성한 joint 계층구조에서 순서를 가져옵니다.
    joint_order = [j.name for j in get_preorder_joint_list(root) if "Site" not in j.name]

    num_joints = len(joint_order)  # <<<< 추가: 동적 계산
    sixd_dim = num_joints * 6
    pos_dim = (num_joints - 1) * 3  # local pos dim (root 제외)
    sixd_start = 4 + pos_dim

    num_frames = generated_tensor.shape[0]
    motion_obj = Motion(frames=[], frame_time=1.0/FPS, frame_len=num_frames)

    current_global_pos_glm = glm.vec3(0.0, 0.0, 0.0)
    current_vr_rot_glm = glm.quat(1.0, 0.0, 0.0, 0.0)
    
    global prev_r_inv

    for i in tqdm(range(num_frames), desc="Reconstructing Motion"):
        frame_features = generated_tensor[i]
        
        # 특징 분해 (차원은 N=23 기준)
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
        #여기서 joint_global_transform를 통해 root를 꺼내오고 나머지는 그냥.. joint_rotation 쓰면 되겠는데
    print("Conversion complete.")
    return root, motion_obj

def create_fbo(width, height):
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("FBO 생성 실패!", file=sys.stderr)
        return None, None, None
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return fbo, texture, rbo

def save_video(frames, filename, fps):
    print(f"\nSaving video... Total {len(frames)} frames")
    with imageio.get_writer(filename, fps=fps, quality=8, macro_block_size=16) as writer:
        for frame in tqdm(frames, desc="Encoding Video"):
            writer.append_data(frame)
    print(f"🎥 Video successfully saved to {filename}")

def render_movie(root, motion_obj, output_path):
    pygame.init()
    size = (WINDOW_WIDTH, WINDOW_HEIGHT)
    pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.HIDDEN)

    fbo, texture, rbo = create_fbo(*size)
    if fbo is None:
        pygame.quit()
        return

    camera_eye = glm.vec3(60, 180, 600)
    camera_center = glm.vec3(0, 0, 0) 
    camera_up = glm.vec3(0, 1, 0)

    recorded_frames = []
    num_frames = len(motion_obj.quaternion_frame)

    for i in tqdm(range(num_frames), desc="Rendering Frames"):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glViewport(0, 0, *size)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, size[0] / size[1], 1.0, 5000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(camera_eye.x, camera_eye.y, camera_eye.z,
                  camera_center.x, camera_center.y, camera_center.z,
                  camera_up.x, camera_up.y, camera_up.z)
        set_lights()
        draw_axes()
        
        current_frame = motion_obj.quaternion_frame[i]
        draw_humanoid(root, current_frame, color=(0.2, 0.6, 0.9))

        glReadBuffer(GL_COLOR_ATTACHMENT0)
        pixels = glReadPixels(0, 0, *size, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(size[1], size[0], 3)
        image = np.flipud(image)
        recorded_frames.append(image)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    glDeleteRenderbuffers(1, [rbo])
    glDeleteTextures(1, [texture])
    glDeleteFramebuffers(1, [fbo])
    
    save_video(recorded_frames, output_path, FPS)
    pygame.quit()