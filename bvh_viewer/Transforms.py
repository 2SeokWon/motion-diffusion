# transforms.py
import numpy as np
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

def extract_vroot_transform(quat_rotation, offset):
    """
    회전행렬에서 yaw값만을 추출하여, offset을 적용한 4x4 행렬을 반환합니다.
    기존 방식 대신 회전행렬의 특정 요소를 이용해 yaw를 안정적으로 계산합니다.
    
    :param kinetics: 4x4 변환 행렬 (회전 포함)
    :param offset: 3차원 offset 벡터 (예: [x, y, z])
    :return: yaw 회전만을 적용한 4x4 행렬
    """
    # 회전 행렬에서 yaw를 직접 추출 (현재 좌표계에 맞게 수정 필요)
    R_mat = glm.mat3_cast(quat_rotation)
    # 예: R_mat[0,2]와 R_mat[2,2]를 사용 (좌표계에 따라 부호나 순서가 달라질 수 있음)
    yaw = glm.atan(R_mat[2, 0], R_mat[2, 2])
    
    q_yaw = glm.angleAxis(yaw, glm.vec3(0,1,0))
    R_yaw = glm.mat4_cast(q_yaw)

    #T_offset = glm.translate(glm.mat4(1.0),glm.vec3(offset.x,0,offset.z)) 잠시 루트 고정
    T_offset = glm.mat4(1.0)
    virtual_root_T= T_offset @ R_yaw
    return virtual_root_T, q_yaw

'''
def motion_adapter(root, motion_frame):
    """
    root를 목표로 motion을 적용시키기 위한 함수입니다.
    add_motion 함수를 콜해 적용시키고 root position과 root를 return합니다.
    :param root: 적용할 root
    :param motion_frame: 모션 프레임값
    :return: root_position과 root를 return
    """
    root_position = motion_frame.joint_positions.get("virtual_root", glm.vec3(0,0,0))
    #add_motion(root, motion_frame)
    #compute_forward_kinematics(root, )
    print("=== Motion Frame joints ===")
    print(motion_frame.joint_positions.keys())

    return root_position, root

def add_motion(joint, motion_frame):
    """
    모션을 root에 재귀적으로 더해주는 함수입니다.
    :param node: 적용할 node, 재귀적으로 작동한다.
    :param motion_frame: 모션프레임
    :param idx: 인덱스
    """
    if joint.name in motion_frame.joint_rotations:
        print(f"Applying motion to {joint.name}")
        joint_rot = motion_frame.joint_rotations[joint.name] #quaternion
        joint_pos = motion_frame.joint_positions[joint.name] #vec3

        local_t = translation_matrix(joint_pos)
        local_r = glm.mat4_cast(joint_rot)

        joint.kinematics = local_t @ local_r

    for child in joint.children:
        add_motion(child, motion_frame)

    
    if not node:
        return

    if node.name != "Site":
        if len(node.channels) == 6:
            # 첫 3개가 position 다음 3개가 rotation
            idx[0] += 3
            rotation = list(map(float, motion_frame[idx[0]:idx[0] + 3]))
            node.kinetics = compute_forward_kinematics(node, rotation)
            idx[0] += 3
        elif len(node.channels) == 3:
            rotation = list(map(float, motion_frame[idx[0]:idx[0] + 3]))
            node.kinetics = compute_forward_kinematics(node, rotation)
            idx[0] += 3

    for child in node.children:
        add_motion(child, motion_frame, idx)
        


def inverse_matrix(T):
    """
    역행렬 구해주는 함수
    :param T: 역행렬을 구할 행렬
    :return: T의 역행렬
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def get_projection(v, onto):
    onto_norm = onto / np.linalg.norm(onto)
    proj = np.dot(v, onto_norm) * onto_norm
    return proj

def lookrotation(v, u):
    u_hat = u/np.linalg.norm(u)
    v_hat = v/np.linalg.norm(v)

    vxu = np.cross(u_hat, v_hat)
    t_hat = vxu/np.linalg.norm(vxu)

    R = np.array([t_hat, np.cross(v_hat, t_hat), v_hat]).T
    return R

def get_pelvis_virtual(kinetics):

    ap = kinetics[:3, 3]
    ar = kinetics[:3, :3]
    upVector = np.array([0,1,0], dtype=float)
    p = ap - get_projection(ap, upVector)
    f = ar[:, 2]
    r = lookrotation(f-get_projection(f, upVector), upVector)
    ap_transformed = r.T @ (ap - p)
    ar_transformed = r.T @ ar

    kinetics = np.eye(4)
    kinetics[:3, :3] = ar_transformed
    kinetics[:3, 3] = ap_transformed
    
    return kinetics
'''