#utils.py
import torch
import torch.nn.functional as F

def sixd_to_euler_angles(sixd_vectors, order='zyx'):
    """
    6D 회전 표현을 오일러 각도로 변환합니다. (GPU 연산에 최적화)

    :param sixd_vectors: (..., 6) 모양의 6D 회전 텐서
    :param order: 오일러 각도 순서 (기본값은 BVH에서 가장 흔한 'zyx')
    :return: (..., 3) 모양의 오일러 각도 텐서 (라디안 단위)
    """
    # 6D 벡터를 3x3 회전 행렬의 첫 두 열로 해석
    a1 = sixd_vectors[..., 0:3]
    a2 = sixd_vectors[..., 3:6]

    # Gram-Schmidt 과정을 통해 완전한 3x3 직교 회전 행렬 복원
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    rotation_matrix = torch.stack([b1, b2, b3], dim=-2) # (..., 3, 3)

    # 회전 행렬에서 오일러 각도 추출 (ZYX 순서 기준)
    sy = torch.sqrt(rotation_matrix[..., 0, 0]**2 + rotation_matrix[..., 1, 0]**2)
    
    singular = sy < 1e-6

    x = torch.atan2(rotation_matrix[..., 2, 1], rotation_matrix[..., 2, 2])
    y = torch.atan2(-rotation_matrix[..., 2, 0], sy)
    z = torch.atan2(rotation_matrix[..., 1, 0], rotation_matrix[..., 0, 0])

    # 짐벌락(Gimbal Lock) 케이스 처리
    x_singular = torch.atan2(-rotation_matrix[..., 1, 2], rotation_matrix[..., 1, 1])
    y_singular = torch.atan2(-rotation_matrix[..., 2, 0], sy)
    z_singular = torch.zeros_like(z)

    # 특이 케이스일 경우 해당 값으로 대체
    x = torch.where(singular, x_singular, x)
    y = torch.where(singular, y_singular, y)
    z = torch.where(singular, z_singular, z)

    if order.lower() == 'zyx':
        return torch.stack([z, y, x], dim=-1)
    else:
        # 다른 순서가 필요하면 여기에 구현을 추가할 수 있습니다.
        raise ValueError(f"Unsupported Euler angle order '{order}'. Only 'zyx' is implemented.")


def euler_to_sixd(euler_angles_rad, order='zyx'):
    """
    오일러 각도(라디안)를 6D 회전 표현으로 변환합니다.
    (나중에 전처리 스크립트에서 유용하게 사용할 수 있습니다)

    :param euler_angles_rad: (..., 3) 모양의 오일러 각도 텐서 (라디안 단위, ZYX 순서)
    :return: (..., 6) 모양의 6D 회전 텐서
    """
    # 각 축에 대한 코사인 및 사인 값 계산
    c1 = torch.cos(euler_angles_rad[..., 0]) # Z
    s1 = torch.sin(euler_angles_rad[..., 0])
    c2 = torch.cos(euler_angles_rad[..., 1]) # Y
    s2 = torch.sin(euler_angles_rad[..., 1])
    c3 = torch.cos(euler_angles_rad[..., 2]) # X
    s3 = torch.sin(euler_angles_rad[..., 2])

    if order.lower() == 'zyx':
        # ZYX 순서에 대한 회전 행렬 계산
        r11 = c1 * c2
        r12 = c1 * s2 * s3 - c3 * s1
        r21 = c2 * s1
        r22 = c1 * c3 + s1 * s2 * s3
        r31 = -s2
        r32 = c2 * s3
    else:
        raise ValueError(f"Unsupported Euler angle order '{order}'. Only 'zyx' is implemented.")

    # 3x3 회전 행렬의 첫 두 열만 반환
    # (r11, r21, r31)는 첫 번째 열, (r12, r22, r32)는 두 번째 열
    return torch.stack([r11, r21, r31, r12, r22, r32], dim=-1)