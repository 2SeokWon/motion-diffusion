#rendering.py
from OpenGL.GL import *
from pyglm import glm
import numpy as np
from utils import draw_colored_cube, draw_colored_sphere, bone_rotation, draw_arrow, draw_undercircle

joint_size = 3


# OpenGL_accelerate 사용하면 numpy로 변환해줘야함.
def glm_mat4_to_glf(m: glm.mat4) -> np.ndarray:
    return np.array(m.to_list(), dtype=np.float32).flatten()


def draw_humanoid(root, frame, color):
    glPushMatrix()
    # 1) virtual root transform
    glMultMatrixf(glm_mat4_to_glf(frame.virtual_transform))
    # 2) hip subtree
    draw_joint(root, frame, color)
    glPopMatrix()


def draw_joint(joint, frame, color):
    glPushMatrix()
    # 1) hip, 그 외 joint의 T_local 모두 적용
    glMultMatrixf(glm_mat4_to_glf(frame.joint_local_transforms[joint.name]))

    # 2) joint sphere
    draw_colored_sphere(joint_size)

    # 3) 각 자식 joint 간 뼈대 + 재귀
    for child in joint.children:
        draw_bone(frame, child, color)
        draw_joint(child, frame, color)

    glPopMatrix()

def draw_bone(frame, child, color):
    """
    parent_name에 해당하는 joint_local_transforms가 이미
    모델뷰 매트릭스에 곱해진 상태에서 호출해야 합니다.
    """
    # 1) offset 결정: hip은 hip_local_offsets, 나머지는 static offset
    if child.name == "hip":
        offset = glm.vec3(frame.hip_local_offsets)
    else:
        offset = glm.vec3(*child.offset)

    mid = [offset[0] / 2.0, offset[1] / 2.0, offset[2] / 2.0]
    rot_quat = bone_rotation(glm.vec3(*offset))
    rot_mat = glm.mat4_cast(rot_quat)
    glPushMatrix()
    glTranslatef(*mid)
    glMultMatrixf(np.array(rot_mat, dtype=np.float32).flatten())
    glScalef(joint_size, abs(glm.length(glm.vec3(*offset)) - 2 * joint_size) / 2, joint_size / 3)
    draw_colored_cube(1, color = color)
    glPopMatrix()

def draw_virtual_root_axis(kinematics, color, circle_radius=10, arrow_length=20):
    """
    root Transform에서 조그만한 3차원 축을 그리기 위함입니다.
    virtual root의 위치를 받아 회전만큼 회전하여 pelvis의 회전을 시각적으로 확인합니다.
    """
    glPushMatrix()
    glMultMatrixf(glm_mat4_to_glf(kinematics))
    draw_arrow(circle_radius, arrow_length, color)
    glRotatef(90, 1.0, 0.0, 0.0)
    glColor3f(1.0, 1.0, 1.0)
    draw_undercircle(10)
    glPopMatrix()