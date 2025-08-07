# kinematics.py

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
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
        
    if order.lower() == 'yxz':
        # Y(1), X(2), Z(3) 순서
        # 3x3 회전 행렬의 첫 두 열 계산

        alpha = euler_angles_rad[..., 0]  # Y축 회전
        beta = euler_angles_rad[..., 1]   # X축 회전
        gamma = euler_angles_rad[..., 2] # Z축 회전

        ca, sa = torch.cos(alpha), torch.sin(alpha)
        cb, sb = torch.cos(beta), torch.sin(beta)
        cg, sg = torch.cos(gamma), torch.sin(gamma)
        
        # 3x3 회전 행렬의 첫 두 열 계산
        r11 = ca * cg + sa * sb * sg
        r21 = cb * sg
        r31 = cg * -sa + ca * sb * sg

        r12 = -ca * sg + cg * sa * sb
        r22 = cb * cg
        r32 = sa * sg + ca * cg * sb

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
    if order.lower() != 'yxz':
        raise ValueError(f"Unsupported Euler angle order '{order}'. Only 'yxz' are implemented.")
    
        # 먼저 6D 벡터를 완전한 3x3 회전 행렬로 복원
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
    alpha_singular = torch.atan2(-r12_singular, r11_singular)  # y축 회전 (yaw) for singular case
    gamma_singular = torch.zeros_like(gamma) # z축 회전 (roll) for singular case

    final_alpha = torch.where(singular, alpha_singular, alpha)
    final_gamma = torch.where(singular, gamma_singular, gamma)

    return torch.stack([final_alpha, beta, final_gamma], dim=-1)

def sixd_to_rotation_matrix(sixd_vectors):
    """
    6D 벡터를 완전한 3x3 회전 행렬로 변환하는 헬퍼 함수.
    이 함수는 순서에 독립적입니다.
    """
    x_raw = sixd_vectors[..., 0:3]
    y_raw = sixd_vectors[..., 3:6]

    x = F.normalize(x_raw, dim=-1)

    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)

    y = torch.cross(z, x, dim=-1)

    return torch.stack([x, y, z], dim=-1)

def qrot(q, v):
    """
    Rotate vector v by quaternion q.
    q: [..., 4]
    v: [..., 3]
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * 2.0 * q_w.unsqueeze(-1)
    c = q_vec * (torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0)
    return a + b + c

def matrix_to_quaternion(matrix):
    """
    Convert a batch of rotation matrices to quaternions.
    :param matrix: Rotation matrices as tensor of shape (..., 3, 3).
    :return: Quaternions with real part first (w, x, y, z), as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Input matrix must be a batch of 3x3 matrices, got {matrix.shape}")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    # 각 쿼터니언 성분의 제곱 * 4 계산
    w_sq = 1.0 + m00 + m11 + m22
    x_sq = 1.0 + m00 - m11 - m22
    y_sq = 1.0 - m00 + m11 - m22
    z_sq = 1.0 - m00 - m11 + m22

    # w, x, y, z 중 가장 큰 값을 찾아 계산의 안정성을 높임
    max_sq = torch.stack([w_sq, x_sq, y_sq, z_sq], dim=-1).argmax(dim=-1)

    q = torch.zeros(batch_dim + (4,), device=matrix.device, dtype=matrix.dtype)

    # Case 1: w is largest
    mask = max_sq == 0
    if mask.any():
        q[mask, 0] = 0.5 * torch.sqrt(w_sq[mask])
        q[mask, 1] = (m21[mask] - m12[mask]) / (4.0 * q[mask, 0])
        q[mask, 2] = (m02[mask] - m20[mask]) / (4.0 * q[mask, 0])
        q[mask, 3] = (m10[mask] - m01[mask]) / (4.0 * q[mask, 0])

    # Case 2: x is largest
    mask = max_sq == 1
    if mask.any():
        q[mask, 1] = 0.5 * torch.sqrt(x_sq[mask])
        q[mask, 0] = (m21[mask] - m12[mask]) / (4.0 * q[mask, 1])
        q[mask, 2] = (m10[mask] + m01[mask]) / (4.0 * q[mask, 1])
        q[mask, 3] = (m20[mask] + m02[mask]) / (4.0 * q[mask, 1])

    # Case 3: y is largest
    mask = max_sq == 2
    if mask.any():
        q[mask, 2] = 0.5 * torch.sqrt(y_sq[mask])
        q[mask, 0] = (m02[mask] - m20[mask]) / (4.0 * q[mask, 2])
        q[mask, 1] = (m10[mask] + m01[mask]) / (4.0 * q[mask, 2])
        q[mask, 3] = (m21[mask] + m12[mask]) / (4.0 * q[mask, 2])

    # Case 4: z is largest
    mask = max_sq == 3
    if mask.any():
        q[mask, 3] = 0.5 * torch.sqrt(z_sq[mask])
        q[mask, 0] = (m10[mask] - m01[mask]) / (4.0 * q[mask, 3])
        q[mask, 1] = (m20[mask] + m02[mask]) / (4.0 * q[mask, 3])
        q[mask, 2] = (m21[mask] + m12[mask]) / (4.0 * q[mask, 3])
        
    return q

def quat_to_rotmat(quat):
    """
    Convert a batch of quaternions to rotation matrices.
    quat: [..., 4] (w, x, y, z) format
    return: [..., 3, 3]
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # 회전 행렬의 각 원소 계산
    R = torch.stack([
        torch.stack([1 - 2 * (y2 + z2), 2 * (xy - wz),     2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz),     1 - 2 * (x2 + z2), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (x2 + y2)], dim=-1)
    ], dim=-2)

    return R

# kinematics.py 의 Skeleton 클래스 수정
class Skeleton:
    def __init__(self, offsets, parents, device):
        self.device = device
        self.offsets = torch.from_numpy(offsets).float().to(device)
        self.parents = parents
        self.num_joints = len(parents)

    def forward_kinematics(self, rotations_quat, root_positions):
        """
        올바른 Forward Kinematics 알고리즘.
        :param rotations_quat: [batch_size, seq_len, num_joints, 4]
        :param root_positions: [batch_size, seq_len, 3]
        :return: [batch_size, seq_len, num_joints, 3]
        """
        bs, seq_len, num_joints, _ = rotations_quat.shape
        
        # 1. 쿼터니언을 3x3 회전 행렬로 변환
        rotmats = quat_to_rotmat(rotations_quat.view(-1, 4)).view(bs, seq_len, num_joints, 3, 3)

        # 2. 글로벌 변환 행렬을 저장할 리스트
        global_positions = torch.zeros(bs, seq_len, num_joints, 3, device=self.device)
        global_rotations = torch.zeros(bs, seq_len, num_joints, 3, 3, device=self.device)
        
        # 3. 루트 관절(i=0)의 글로벌 위치와 회전을 먼저 설정
        #    루트의 글로벌 위치 = BVH의 루트 위치
        #    루트의 글로벌 회전 = BVH의 루트 회전
        global_positions[:, :, 0] = root_positions
        global_rotations[:, :, 0] = rotmats[:, :, 0]

        # 4. 자식 관절들의 글로벌 위치와 회전을 순차적으로 계산
        for i in range(1, self.num_joints):
            parent_idx = self.parents[i]
            
            # (A) 현재 관절의 위치 계산
            #     = 부모의 글로벌 위치 + (부모의 글로벌 회전 * 현재 관절의 오프셋)
            offset_rotated = torch.einsum('bsij,j->bsi', global_rotations[:, :, parent_idx], self.offsets[i])
            global_positions[:, :, i] = global_positions[:, :, parent_idx] + offset_rotated
            
            # (B) 현재 관절의 회전 계산
            #     = 부모의 글로벌 회전 * 현재 관절의 로컬 회전
            global_rotations[:, :, i] = torch.matmul(global_rotations[:, :, parent_idx], rotmats[:, :, i])

        return global_positions

def features_to_xyz(features, skeleton, mean, std):
    """
    학습된 특징 벡터 시퀀스를 3D 관절 위치 시퀀스로 변환 (Forward Kinematics).
    """
    device = features.device
    
    # 1. 역정규화
    features_unnormalized = features * std.to(device) + mean.to(device)
    
    # 2. 특징 벡터 분해
    root_y_height = features_unnormalized[..., 0:1]
    root_xz_velocity_local = features_unnormalized[..., 1:3]
    all_joint_6d_rotations = features_unnormalized[..., 3:]
    
    bs, seq_len, _ = features.shape
    num_joints = all_joint_6d_rotations.shape[-1] // 6
    
    # 3. 6D -> 쿼터니언
    rotations_6d = all_joint_6d_rotations.view(bs, seq_len, num_joints, 6)
    rotations_rotmat = sixd_to_rotation_matrix(rotations_6d)
    rotations_quat = matrix_to_quaternion(rotations_rotmat.view(-1, 3, 3)).view(bs, seq_len, num_joints, 4)

    root_rotations_quat = rotations_quat[:, :, 0, :]
    
    # 4. 속도를 위치로 적분
    final_root_positions = torch.zeros(bs, seq_len, 3, device=device)
    final_root_positions[:, 0, 1:2] = root_y_height[:, 0]
    frame_time = 1.0 / 30.0

    # 로컬 속도를 3D 벡터로 확장
    # shape: [bs, seq_len, 3]
    local_vel_3d = F.pad(root_xz_velocity_local, (0, 1, 0, 0)) # (x, z, 0)
    local_vel_3d = local_vel_3d[..., [0, 2, 1]] # (x, 0, z)
    
    # 모든 프레임에 대한 월드 속도를 한번에 계산
    # qrot은 브로드캐스팅을 지원해야 함
    world_vel_increment = qrot(root_rotations_quat, local_vel_3d) * frame_time
    
    # 누적합(cumsum)으로 모든 프레임의 위치를 한번에 계산
    # 첫 프레임은 [0,0,0] 이므로 pad 추가
    world_pos_offset = torch.cumsum(F.pad(world_vel_increment[:, :-1], (0,0,1,0)), dim=1)
    
    final_root_positions = torch.zeros(bs, seq_len, 3, device=device)
    final_root_positions[..., [0, 2]] = world_pos_offset[..., [0, 2]]
    final_root_positions[..., 1:2] = root_y_height

    # --- 최종 FK 수행 ---
    xyz_positions = skeleton.forward_kinematics(rotations_quat, final_root_positions)
    
    return xyz_positions