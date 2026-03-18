# kinematics.py

import torch
import torch.nn.functional as F
import numpy as np
import math
from pyglm import glm


def sixd_to_rotation_matrix(sixd_vectors: torch.Tensor) -> torch.Tensor:
    """
    6D 벡터를 완전한 3x3 회전 행렬로 변환 (안정화 버전).
    Gram-Schmidt orthogonalization 적용 + full normalize.
    :param sixd_vectors: [..., 6]
    :return: [..., 3,3]
    """
    x_raw = sixd_vectors[..., :3]  # [..., 3]
    y_raw = sixd_vectors[..., 3:]  # [..., 3]

    # x normalize
    x = F.normalize(x_raw, dim=-1)

    # y에서 x proj 제거 후 normalize (Gram-Schmidt)
    dot_x_y = (x * y_raw).sum(dim=-1, keepdim=True)
    y = y_raw - dot_x_y * x
    y = F.normalize(y, dim=-1)

    # z = cross(x, y) 후 normalize
    z = torch.cross(x, y, dim=-1)
    z = F.normalize(z, dim=-1)

    # Stack to rotmat
    rotmat = torch.stack([x, y, z], dim=-1)  # [..., 3,3]

    # Zero handling: det=0 시 closest orthogonal (SVD)
    det = torch.det(rotmat)
    mask = torch.abs(det - 1.0) > 1e-4  # non-orthogonal mask
    if mask.any():
        u, s, vh = torch.linalg.svd(rotmat[mask], full_matrices=False)
        rotmat[mask] = u @ vh  # orthogonal approx

    return rotmat


def sixd_to_euler_angles(sixd_vectors, order='yxz'):
    """
    6D 회전 표현을 오일러 각도로 변환합니다.

    :param sixd_vectors: (..., 6) 모양의 6D 회전 텐서.
    :param order: 변환할 오일러 각도 순서 ('yxz').
    :return: (..., 3) 모양의 오일러 각도 텐서 (라디안 단위). (Y, X, Z) 순서.
    """
    if order.lower() != 'yxz':
        raise ValueError(f"Unsupported Euler angle order '{order}'. Only 'yxz' is implemented.")

    rotation_matrix = sixd_to_rotation_matrix(sixd_vectors)

    r13 = rotation_matrix[..., 0, 2]
    r21 = rotation_matrix[..., 1, 0]
    r22 = rotation_matrix[..., 1, 1]
    r23 = rotation_matrix[..., 1, 2]
    r33 = rotation_matrix[..., 2, 2]

    beta = -torch.asin(torch.clamp(r23, -1.0, 1.0))  # x축 회전 (pitch)
    cb = torch.cos(beta)

    singular = cb < 1e-6  # 짐벌락(Gimbal Lock) 체크

    alpha = torch.atan2(r13, r33)  # y축 회전 (yaw)
    gamma = torch.atan2(r21, r22)  # z축 회전 (roll)

    r11_singular = rotation_matrix[..., 0, 0]
    r12_singular = rotation_matrix[..., 0, 1]
    alpha_singular = torch.atan2(-r12_singular, r11_singular)
    gamma_singular = torch.zeros_like(gamma)

    final_alpha = torch.where(singular, alpha_singular, alpha)
    final_gamma = torch.where(singular, gamma_singular, gamma)

    return torch.stack([final_alpha, beta, final_gamma], dim=-1)


def quat_to_euler(quat, order='YXZ'):
    """
    쿼터니언을 YXZ 순서의 Euler 각도로 변환 (degrees).
    BVH 파일의 원래 값 복구에 적합.
    """
    if order != 'YXZ':
        raise ValueError("현재 YXZ만 지원됩니다.")

    mat = glm.mat3_cast(quat)  # 쿼터니언 → column-major 3x3 행렬

    epsilon = 1e-6

    # Gimbal lock: x ≈ +90°
    if abs(mat[2][1] + 1.0) < epsilon:
        x_deg = 90.0
        y_deg = math.degrees(math.atan2(mat[1][0], mat[0][0]))
        z_deg = 0.0
        return glm.vec3(y_deg, x_deg, z_deg)

    # Gimbal lock: x ≈ -90°
    elif abs(mat[2][1] - 1.0) < epsilon:
        x_deg = -90.0
        y_deg = math.degrees(math.atan2(-mat[1][0], mat[0][0]))
        z_deg = 0.0
        return glm.vec3(y_deg, x_deg, z_deg)

    # 일반 케이스
    else:
        x_rad = math.asin(-mat[2][1])
        y_rad = math.atan2(mat[2][0], mat[2][2])
        z_rad = math.atan2(mat[0][1], mat[1][1])
        return [y_rad, x_rad, z_rad]  # [Y, X, Z] 순서


def mat_to_euler_yxz(mat: np.ndarray) -> list:
    """
    3x3 회전 행렬을 Y-X-Z 순서의 오일러 각도(라디안)로 분해합니다.
    '최종행렬 = Ry * Rx * Rz' 연산의 역과정입니다.

    :param mat: 분해할 3x3 회전 행렬.
    :return: [y_rad, x_rad, z_rad]
    """
    epsilon = 1e-6

    # 짐벌 락 체크: x가 +90도일 때
    if abs(mat[1, 2] + 1.0) < epsilon:
        x_rad = math.pi / 2.0
        y_rad = math.atan2(mat[0, 1], mat[0, 0])
        z_rad = 0
        return [y_rad, x_rad, z_rad]

    # 짐벌 락 체크: x가 -90도일 때
    elif abs(mat[1, 2] - 1.0) < epsilon:
        x_rad = -math.pi / 2.0
        y_rad = math.atan2(-mat[0, 1], mat[0, 0])
        z_rad = 0
        return [y_rad, x_rad, z_rad]

    # 일반적인 경우
    else:
        x_rad = math.asin(-mat[1, 2])
        y_rad = math.atan2(mat[0, 2], mat[2, 2])
        z_rad = math.atan2(mat[1, 0], mat[1, 1])
        return [y_rad, x_rad, z_rad]
