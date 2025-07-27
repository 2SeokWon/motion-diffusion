# 파일 이름: utils.py

import torch
import torch.nn.functional as F
import numpy as np

def euler_to_sixd(euler_angles_rad, order='yxz'):
    """
    오일러 각도(라디안)를 6D 회전 표현으로 변환합니다.
    'zyx'와 'yxz' 순서를 지원합니다.

    :param euler_angles_rad: (..., 3) 모양의 오일러 각도 텐서.
                              order='zyx'일 경우 (Z, Y, X) 순서의 라디안 값.
                              order='yxz'일 경우 (Y, X, Z) 순서의 라디안 값.
    :param order: 오일러 각도 순서 ('zyx' 또는 'yxz').
    :return: (..., 6) 모양의 6D 회전 텐서.
    """
    # 각 축에 대한 코사인 및 사인 값 계산
    # order에 따라 각도의 의미가 달라짐
    # 예를 들어 'yxz'일 경우 euler_angles_rad[..., 0]은 Y축 회전값
    c1 = torch.cos(euler_angles_rad[..., 0])
    s1 = torch.sin(euler_angles_rad[..., 0])
    c2 = torch.cos(euler_angles_rad[..., 1])
    s2 = torch.sin(euler_angles_rad[..., 1])
    c3 = torch.cos(euler_angles_rad[..., 2])
    s3 = torch.sin(euler_angles_rad[..., 2])

    if order.lower() == 'zyx':
        # Z(1), Y(2), X(3) 순서
        # 3x3 회전 행렬의 첫 두 열 계산
        r11 = c1 * c2
        r21 = c2 * s1
        r31 = -s2
        
        r12 = c1 * s2 * s3 - c3 * s1
        r22 = c1 * c3 + s1 * s2 * s3
        r32 = c2 * s3
        
    elif order.lower() == 'yxz':
        # Y(1), X(2), Z(3) 순서
        # 3x3 회전 행렬의 첫 두 열 계산
        r11 = c1 * c3 + s1 * s2 * s3
        r21 = c2 * s3
        r31 = -c1 * s2 * s3 + c3 * s1

        r12 = c3 * s1 * s2 - c1 * s3
        r22 = c2 * c3
        r32 = c1 * s2 * c3 + s1 * s3

    else:
        raise ValueError(f"Unsupported Euler angle order '{order}'. Only 'zyx' and 'yxz' are implemented.")

    # 3x3 회전 행렬의 첫 두 열(6개 요소)을 반환
    return torch.stack([r11, r21, r31, r12, r22, r32], dim=-1)


def sixd_to_euler_angles(sixd_vectors, order='yxz'):
    """
    6D 회전 표현을 오일러 각도로 변환합니다.
    'zyx'와 'yxz' 순서를 지원합니다.

    :param sixd_vectors: (..., 6) 모양의 6D 회전 텐서.
    :param order: 변환할 오일러 각도 순서 ('zyx' 또는 'yxz').
    :return: (..., 3) 모양의 오일러 각도 텐서 (라디안 단위).
             order='zyx'일 경우 (Z, Y, X) 순서.
             order='yxz'일 경우 (Y, X, Z) 순서.
    """
    # 먼저 6D 벡터를 완전한 3x3 회전 행렬로 복원
    rotation_matrix = sixd_to_rotation_matrix(sixd_vectors)

    if order.lower() == 'zyx':
        # 회전 행렬에서 Z, Y, X 오일러 각도 추출
        sy = torch.sqrt(rotation_matrix[..., 0, 0]**2 + rotation_matrix[..., 1, 0]**2)
        singular = sy < 1e-6

        x = torch.atan2(rotation_matrix[..., 2, 1], rotation_matrix[..., 2, 2])
        y = torch.atan2(-rotation_matrix[..., 2, 0], sy)
        z = torch.atan2(rotation_matrix[..., 1, 0], rotation_matrix[..., 0, 0])

        # 짐벌락(Gimbal Lock) 특이 케이스 처리
        x_singular = torch.atan2(-rotation_matrix[..., 1, 2], rotation_matrix[..., 1, 1])
        y_singular = torch.atan2(-rotation_matrix[..., 2, 0], sy)
        z_singular = torch.zeros_like(z)

        x = torch.where(singular, x_singular, x)
        y = torch.where(singular, y_singular, y)
        z = torch.where(singular, z_singular, z)
        
        return torch.stack([z, y, x], dim=-1)
        
    elif order.lower() == 'yxz':
        # 회전 행렬에서 Y, X, Z 오일러 각도 추출
        sy = torch.sqrt(rotation_matrix[..., 0, 0]**2 + rotation_matrix[..., 1, 0]**2)
        singular = sy < 1e-6
        
        x = torch.atan2(sy, rotation_matrix[..., 2, 0]) * -1 #atan2(-y,x) 여기서 y는 -r20
        y = torch.atan2(rotation_matrix[..., 2, 1], rotation_matrix[..., 2, 2])
        z = torch.atan2(rotation_matrix[..., 1, 0], rotation_matrix[..., 0, 0])
        
        # 짐벌락(Gimbal Lock) 특이 케이스 처리
        x_singular = torch.atan2(-rotation_matrix[..., 1, 2], rotation_matrix[..., 1, 1])
        y_singular = torch.zeros_like(y)
        z_singular = torch.atan2(-rotation_matrix[..., 0, 1], rotation_matrix[..., 1, 1])
        
        x = torch.where(singular, x_singular, x)
        y = torch.where(singular, y_singular, y)
        z = torch.where(singular, z_singular, z)
        
        return torch.stack([y, x, z], dim=-1)
        
    else:
        raise ValueError(f"Unsupported Euler angle order '{order}'. Only 'zyx' and 'yxz' are implemented.")


def sixd_to_rotation_matrix(sixd_vectors):
    """
    6D 벡터를 완전한 3x3 회전 행렬로 변환하는 헬퍼 함수.
    이 함수는 순서에 독립적입니다.
    """
    a1 = sixd_vectors[..., 0:3]
    a2 = sixd_vectors[..., 3:6]

    # Gram-Schmidt 과정을 통해 직교 기저 벡터 생성
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    # 세 개의 기저 벡터를 쌓아 회전 행렬 구성
    return torch.stack([b1, b2, b3], dim=-2)