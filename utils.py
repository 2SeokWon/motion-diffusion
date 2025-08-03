import torch
import torch.nn.functional as F
import numpy as np
import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation

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


def generate_and_save_bvh(model, diffusion, dataset, num_samples, seq_len, input_feats, root_motion_features, template_path, output_dir, device):
    print("Starting sampling...")
    model.eval()

    with torch.no_grad():
        sample_shape = (num_samples, seq_len, input_feats)
        generated_motion_norm = diffusion.p_sample_loop(model, sample_shape)

        mean = dataset.mean.to(device)  # dataset.mean도 numpy
        std = dataset.std.to(device)   # dataset.std도 numpy
        generated_motion = generated_motion_norm * std + mean
    
    print("Converting generated motions to BVH files...")
    # --- 스켈레톤 템플릿 읽기 ---
    try:
        with open(template_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Skeleton template file not found at '{template_path}'")
        return
    
    header_lines = []
    motion_header_found = False
    for line in lines:
        if "MOTION" in line.upper(): motion_header_found = True
        if motion_header_found and "Frame Time:" in line:
            header_lines.append(line); break
        else: header_lines.append(line)
        
    # --- 각 샘플을 BVH로 변환 ---
    for i in range(num_samples):
        print(f"Processing sample {i+1}/{num_samples}...")

        single_motion = generated_motion[i].cpu()

        root_y_height = single_motion[:,0]
        root_xz_velocity_local = single_motion[:, 1:3]
        root_y_angular_velocity_deg = single_motion[:, 3]
        all_joint_6d_rotations_flat = single_motion[:,4:]

        num_all_joints = all_joint_6d_rotations_flat.shape[1] // 6
        all_joint_6d_reshaped = all_joint_6d_rotations_flat.reshape(seq_len, num_all_joints, 6)

        all_joints_euler_rad = sixd_to_euler_angles(torch.from_numpy(all_joint_6d_reshaped.numpy()), order='yxz')
        all_joints_euler_deg = torch.rad2deg(all_joints_euler_rad)

        root_rotations_deg = all_joints_euler_deg[:, 0, :]  # 루트 관절 회전 (첫 번째 관절)
        other_joints_rotations_deg = all_joints_euler_deg[:, 1:, :]  # 나머지 관절 회전

        final_root_positions = torch.zeros(seq_len, 3)
        final_root_rotations_deg = root_rotations_deg
        final_root_positions[0,1] = root_y_height[0]


        for frame_idx in range(seq_len - 1):
            current_rot = Rotation.from_euler('yxz', final_root_rotations_deg[frame_idx].numpy(), degrees=True)

            local_vel = np.array([root_xz_velocity_local[frame_idx,0], 0, root_xz_velocity_local[frame_idx,1]])
            world_vel_increment = current_rot.apply(local_vel)

            final_root_positions[frame_idx + 1] = final_root_positions[frame_idx] + torch.from_numpy(world_vel_increment)

            final_root_positions[frame_idx + 1, 1] = root_y_height[frame_idx + 1]  # Y 높이는 그대로 유지
        
        # 최종 BVH 데이터 조립
        rotations_flat = torch.cat([
            final_root_rotations_deg, # 루트 회전
            other_joints_rotations_deg.reshape(seq_len, -1) # 나머지 관절 회전
        ], dim=1)
        
        motion_data_flat = torch.cat([final_root_positions, rotations_flat], dim=1)
        
        # MOTION 데이터 블록 생성
        motion_lines = [" ".join(f"{x:.6f}" for x in frame) for frame in motion_data_flat.cpu().numpy()]
        
        # 최종 파일 저장
        output_path = os.path.join(output_dir, f"generated_motion_{i+1}.bvh")
        with open(output_path, 'w') as f:
            for line in header_lines:
                if "Frames:" in line: f.write(f"Frames: {seq_len}\n")
                else: f.write(line)
            f.write("\n".join(motion_lines))
            
    print(f"\nAll motions saved to '{output_dir}' directory.")