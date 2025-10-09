# transforms.py
import numpy as np
import math
from pyglm import glm

def get_rotation_matrix(channel, angle_deg):
    """
    채널에 존재하는 각도를 회전 행렬로 바꿔주는 함수입니다.
    :param channel: 바꿀 채널
    :param angle_deg: 각도 input
    :return: 회전 행렬
    """
    theta = glm.radians(angle_deg)
    if "Xrotation" in channel:
        q = glm.angleAxis(theta, glm.vec3(1,0,0))
    elif "Yrotation" in channel:
        q = glm.angleAxis(theta, glm.vec3(0, 1, 0))
    elif "Zrotation" in channel:
        q = glm.angleAxis(theta, glm.vec3(0, 0, 1))
    else:
        return glm.mat4(1.0)

    return glm.mat4_cast(q)

def translation_matrix(offset):
    """
    Translation (x,y,z)를 행렬로 바꿔주는 함수입니다.
    :param offset: (x,y,z) translation vector
    :return: translation 행렬
    """
    tx, ty, tz = offset
    return glm.translate(glm.mat4(1.0), glm.vec3(tx,ty,tz))

prev_r_inv = glm.quat(1, 0, 0, 0)

def lookrotation(v: glm.vec3, u: glm.vec3) -> glm.quat:
    v_hat = glm.normalize(v)
    u_hat = glm.normalize(u)
    t_hat = glm.normalize(glm.cross(u_hat, v_hat))
    up_corrected = glm.cross(v_hat, t_hat)
    rot_mat = glm.mat3(t_hat, up_corrected, v_hat)
    rot_q = glm.quat_cast(rot_mat)
    if rot_q.w < 0:
        rot_q = -rot_q
    return rot_q

def get_pelvis_virtual_safe(ap: glm.vec3, ar: glm.quat, fallback_forward=glm.vec3(0, 0, 1)):
    up = glm.vec3(0, 1, 0)
    p = ap - glm.dot(ap, up) / glm.dot(up, up) * up
    f = ar * glm.vec3(0, 0, 1)
    f_mod = f - glm.dot(f, up) / glm.dot(up, up) * up
    
    if glm.length(f_mod) < 1e-4:
        f_mod = fallback_forward
    
    r = lookrotation(f_mod, up)
    r_inv = glm.inverse(r)
    
    new_ap = r_inv * (ap - p)
    new_ar = r_inv * ar

    return new_ap, new_ar

def extract_vroot_transform(quat_rotation, position):
    """
    회전행렬에서 yaw값만을 추출하여, offset을 적용한 4x4 행렬을 반환합니다.
    glm 기반으로 스무딩 적용 (기존 함수 강화).
    """
    ap = glm.vec3(position)
    ar = quat_rotation  # glm.quat 가정

    #Virtual root 기준 hip의 local position(0,y,0), rotation(Pitch, Roll)
    new_ap, new_ar = get_pelvis_virtual_safe(ap, ar)
    
    ap_global = ap - new_ap  # global xz
    
    ar_mat = glm.mat4_cast(ar) 
    yaw = math.atan2(ar_mat[2][0], ar_mat[2][2]) #global yaw
    q_yaw = glm.angleAxis(yaw, glm.vec3(0,1,0))
    R_yaw = glm.mat4_cast(q_yaw)

    T_offset = glm.translate(glm.mat4(1.0), glm.vec3(ap_global.x, 0, ap_global.z))
    virtual_root_T = T_offset @ R_yaw
    return virtual_root_T, q_yaw