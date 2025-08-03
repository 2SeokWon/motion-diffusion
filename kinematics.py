# kinematics.py

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from utils import sixd_to_rotation_matrix
import numpy as np

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

class Skeleton:
    """
    PyTorch 기반으로 스켈레톤 구조를 정의하고 Forward Kinematics를 수행하는 클래스.
    """
    def __init__(self, offsets, parents, device):
        """
        :param offsets: [num_joints, 3] NumPy array. 각 관절의 부모로부터의 오프셋.
        :param parents: [num_joints] list. 각 관절의 부모 인덱스. 루트는 -1.
        """
        self.device = device
        self.offsets = torch.from_numpy(offsets).float().to(device)
        self.parents = parents
        self.num_joints = len(parents)

    def forward_kinematics(self, rotations_quat, root_positions):
        """
        In-place 연산을 제거하여 Autograd와 호환되도록 수정한 버전.
        """
        bs, seq_len, num_joints, _ = rotations_quat.shape
        
        # quat -> rotmat 변환 (이전과 동일)
        rotmats = quat_to_rotmat(rotations_quat.view(-1, 4)).view(bs, seq_len, num_joints, 3, 3)

        # 4x4 로컬 변환 행렬 준비 (이전과 동일)
        transforms_local = torch.eye(4, device=self.device).repeat(bs, seq_len, self.num_joints, 1, 1)
        transforms_local[..., :3, :3] = rotmats
        transforms_local[..., :3, 3] = self.offsets

        # ==================== 핵심 수정 부분 ====================
        # 미리 큰 텐서를 만들지 않고, 파이썬 리스트를 사용
        global_transforms_list = []
        
        # 루트 관절의 글로벌 변환을 리스트에 추가
        global_transforms_list.append(transforms_local[:, :, 0])

        for i in range(1, self.num_joints):
            parent_idx = self.parents[i]
            
            # 부모의 글로벌 변환 행렬을 리스트에서 가져옴
            parent_global_transform = global_transforms_list[parent_idx]
            
            # 현재 관절의 로컬 변환
            current_local_transform = transforms_local[:, :, i]
            
            # 새로운 글로벌 변환 계산 (Out-of-place)
            current_global_transform = torch.matmul(parent_global_transform, current_local_transform)
            
            # 계산된 결과를 리스트에 추가
            global_transforms_list.append(current_global_transform)
            
        # 모든 계산이 끝난 후, 리스트에 담긴 텐서들을 하나의 큰 텐서로 합침
        # dim=2는 num_joints 차원에 해당
        global_transforms = torch.stack(global_transforms_list, dim=2)
        # =======================================================
        
        # 루트 위치 적용 (이전과 동일)
        positions = global_transforms[..., :3, 3] + root_positions.unsqueeze(2)

        return positions

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
    all_joint_6d_rotations = features_unnormalized[..., 4:]
    
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