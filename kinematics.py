# kinematics.py

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from utils import sixd_to_rotation_matrix
import numpy as np

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
        쿼터니언 회전과 루트 위치로부터 모든 관절의 월드 좌표를 계산합니다.
        :param rotations_quat: [batch_size, seq_len, num_joints, 4] torch.Tensor
        :param root_positions: [batch_size, seq_len, 3] torch.Tensor
        :return: [batch_size, seq_len, num_joints, 3] torch.Tensor
        """
        bs, seq_len, _, _ = rotations_quat.shape
        
        # 쿼터니언 -> 회전 행렬 (scipy를 거치지만, loss 계산에는 문제가 없음)
        rotmats_np = Rotation.from_quat(rotations_quat.reshape(-1, 4).cpu().detach().numpy()).as_matrix()
        rotmats = torch.from_numpy(rotmats_np).float().to(self.device).view(bs, seq_len, self.num_joints, 3, 3)

        # 4x4 변환 행렬 준비 (로컬)
        transforms_local = torch.eye(4, device=self.device).repeat(bs, seq_len, self.num_joints, 1, 1)
        transforms_local[..., :3, :3] = rotmats
        transforms_local[..., :3, 3] = self.offsets #[32, 90, 23, 4, 4]

        # 글로벌 변환 행렬 계산
        global_transforms = torch.zeros_like(transforms_local)
        global_transforms[:, :, 0] = transforms_local[:, :, 0]

        for i in range(1, self.num_joints):
            parent_idx = self.parents[i]
            # global_transform(i) = global_transform(parent) @ local_transform(i)
            global_transforms[:, :, i] = torch.matmul(global_transforms[:, :, parent_idx], transforms_local[:, :, i])
        
        # 루트 위치 적용
        # FK 결과는 루트 기준 상대 위치이므로, 월드 루트 위치를 더해줌
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
    rotations_quat_np = Rotation.from_matrix(rotations_rotmat.reshape(-1, 3, 3).cpu().detach().numpy()).as_quat()
    rotations_quat = torch.from_numpy(rotations_quat_np).float().to(device).view(bs, seq_len, num_joints, 4)

    root_rotations_quat = rotations_quat[:, :, 0, :]
    
    # 4. 속도를 위치로 적분
    final_root_positions = torch.zeros(bs, seq_len, 3, device=device)
    final_root_positions[:, 0, 1:2] = root_y_height[:, 0]
    frame_time = 1.0 / 30.0

    for i in range(seq_len - 1):
        current_rot = Rotation.from_quat(root_rotations_quat[:, i].cpu().detach().numpy())
        local_vel_np = root_xz_velocity_local[:, i].cpu().detach().numpy()
        local_vel_3d_np = np.c_[local_vel_np[:, 0], np.zeros(bs), local_vel_np[:, 1]]
        
        world_vel_increment = current_rot.apply(local_vel_3d_np) * frame_time
        
        final_root_positions[:, i+1] = final_root_positions[:, i] + torch.from_numpy(world_vel_increment).float().to(device)
        final_root_positions[:, i+1, 1:2] = root_y_height[:, i+1]

    # 5. 최종 FK 수행
    xyz_positions = skeleton.forward_kinematics(rotations_quat, final_root_positions)
    
    return xyz_positions