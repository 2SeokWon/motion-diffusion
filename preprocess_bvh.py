import os
import numpy as np
import torch
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
import torch.nn.functional as F  # --- [추가] 이 줄만 넣어! ---
import traceback
# 필요한 유틸리티 및 클래스 import
from kinematics import Skeleton, euler_to_sixd, get_virtual_root_transform, qrot

# --- 1. 설정 (Configuration) ---
bvh_folder_path = "./dataset/"
output_processed_dir = "./processed_data/"
template_bvh_path = "./dataset/Aeroplane_BR.bvh" # 대표 파일 경로
output_metadata_path = os.path.join(output_processed_dir, "metadata.json")
os.makedirs(output_processed_dir, exist_ok=True)
ROTATION_ORDER = 'yxz'

# --- 2. 파싱 함수 정의 (수정 없음) ---
def parse_bvh_hierarchy(filepath):
    with open(filepath, 'r') as f: lines = f.readlines()
    parents, offsets, joint_names, joint_stack, current_joint_info = [], [], [], [], {}
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        keyword = parts[0].upper()
        if keyword in ["ROOT", "JOINT"]:
            if 'name' in current_joint_info and 'channels' in current_joint_info:
                joint_names.append(current_joint_info['name'])
                offsets.append(current_joint_info['offset'])
                parents.append(joint_stack[-1] if joint_stack else -1)
                joint_stack.append(len(joint_names) - 1)
            current_joint_info = {'name': parts[1]}
        elif keyword == "OFFSET":
            if 'name' in current_joint_info: current_joint_info['offset'] = [float(v) for v in parts[1:]]
        elif keyword == "CHANNELS":
            if 'name' in current_joint_info: current_joint_info['channels'] = parts[2:]
        elif keyword == "}":
            if 'name' in current_joint_info and 'channels' in current_joint_info:
                joint_names.append(current_joint_info['name'])
                offsets.append(current_joint_info['offset'])
                parents.append(joint_stack[-1] if joint_stack else -1)
                joint_stack.append(len(joint_names) - 1)
            current_joint_info = {}
            if joint_stack: joint_stack.pop()
        elif keyword == "MOTION": break
    if 'name' in current_joint_info and 'channels' in current_joint_info:
        joint_names.append(current_joint_info['name'])
        offsets.append(current_joint_info['offset'])
        parents.append(joint_stack[-1] if joint_stack else -1)
    return np.array(offsets), np.array(parents), joint_names

def parse_bvh_motion_data(filepath):
    with open(filepath, 'r') as f: lines = f.readlines()
    motion_start_index = -1
    for i, line in enumerate(lines):
        if "MOTION" in line.upper(): motion_start_index = i; break
    if motion_start_index == -1: return None
    num_frames = int(lines[motion_start_index + 1].split(':')[1].strip())
    data_start_index = motion_start_index + 3
    motion_data = [[float(v) for v in lines[i].strip().split()] for i in range(data_start_index, data_start_index + num_frames)]
    return np.array(motion_data)

# --- 3. 전처리 메인 로직 ---
print("--- Step 1: Parsing Skeleton Structure ---")
try:
    offsets, parents, joint_names = parse_bvh_hierarchy(template_bvh_path)
    np.save(os.path.join(output_processed_dir, "offsets.npy"), offsets)
    np.save(os.path.join(output_processed_dir, "parents.npy"), parents)
    skeleton = Skeleton(offsets=offsets, parents=parents.tolist(), device=torch.device('cpu'))
    print(f"Skeleton initialized with {len(parents)} joints.")
except Exception as e:
    print(f"FATAL: Could not parse skeleton. Error: {e}"); exit()

print("\n--- Step 2: Extracting Features from BVH Files ---")
all_motion_clips = []
bvh_files = [f for f in os.listdir(bvh_folder_path) if f.endswith(".bvh")]

try:
    r_hip_idx = joint_names.index('RightHip')
    l_hip_idx = joint_names.index('LeftHip')
    r_shoulder_idx = joint_names.index('RightShoulder')
    l_shoulder_idx = joint_names.index('LeftShoulder')
except ValueError as e:
    print(f"Error: Required joints not found in skeleton: {e}"); exit()

    

for idx, filename in enumerate(tqdm(bvh_files, desc="Processing BVH files")):
    filepath = os.path.join(bvh_folder_path, filename)
    try:
        motion_data_euler = parse_bvh_motion_data(filepath)
        if motion_data_euler is None or motion_data_euler.shape[0] < 20: continue
            
        #motion_data_euler = motion_data_euler[::2, :]
        num_frames = motion_data_euler.shape[0]

        # 1. 원본 BVH 데이터 로드
        root_positions_np_orig = motion_data_euler[:, :3]
        rotations_deg_orig = motion_data_euler[:, 3:]
        num_joints = rotations_deg_orig.shape[1] // 3
        
        rotations_quat_np_orig = Rotation.from_euler(
            ROTATION_ORDER, rotations_deg_orig.reshape(-1, 3), degrees=True
        ).as_quat().reshape(num_frames, num_joints, 4)
        
        # 2. FK를 통해 3D 관절 위치 계산
        positions_3d_orig = skeleton.forward_kinematics(
            torch.from_numpy(rotations_quat_np_orig).float().unsqueeze(0),
            torch.from_numpy(root_positions_np_orig).float().unsqueeze(0)
        ).squeeze(0).numpy()

        # 3. 데이터 정규화
        positions_3d_normalized = positions_3d_orig.copy() # 원본 보존을 위해 복사
        
        # 바닥에 놓기
        floor_height = positions_3d_normalized.reshape(-1, 3)[:, 1].min()
        positions_3d_normalized[:, :, 1] -= floor_height
        
        # 초기 위치 원점 맞추기
        root_pos_init_xz = positions_3d_normalized[0, 0, :] * np.array([1, 0, 1])
        positions_3d_normalized = positions_3d_normalized - root_pos_init_xz

        # 4. 정규화된 위치로부터 회전 정보 재추출 (Inverse Kinematics)
        rotations_quat_np = rotations_quat_np_orig
        
        # 원본 루트의 전역 회전을 Scipy 객체로 변환
        root_global_rotations_obj = Rotation.from_quat(rotations_quat_np_orig[:, 0, :])
        
        # 새로 추가된 함수를 호출하여 로컬 루트 회전, 가상 루트 회전/위치를 한 번에 계산
        root_local_quats, virtual_root_quats, virtual_root_positions = get_virtual_root_transform(
            root_positions_np_orig, root_global_rotations_obj
        )

        # 7. 모든 관절의 Local Orientation을 6D로 변환 (기존과 유사)
        joint_rotations_quat_np = rotations_quat_np_orig[:, 1:, :]
        # 쿼터니언 순서가 (x,y,z,w)이므로 그대로 사용
        all_local_rotations_quat = np.concatenate(
            [root_local_quats[:, np.newaxis, :], joint_rotations_quat_np], axis=1
        )
        all_local_rotations_rad = Rotation.from_quat(all_local_rotations_quat.reshape(-1, 4)).as_euler('yxz', degrees=False)
        all_local_rotations_6d = euler_to_sixd(torch.from_numpy(all_local_rotations_rad).float()).numpy().reshape(num_frames, -1)
        
        # ### [수정] 8단계: 루트 속도 계산 (가상 루트 정보 사용) ###
        # 가상 루트의 전역 속도 계산
        root_velocity_global = virtual_root_positions[1:] - virtual_root_positions[:-1]
        
        # 이전 프레임의 가상 루트 회전으로 변환하여 로컬 속도 계산
        virtual_root_rots_prev_obj = Rotation.from_quat(virtual_root_quats[:-1])
        root_velocity_local = virtual_root_rots_prev_obj.inv().apply(root_velocity_global)
        
        # 가상 루트의 각속도 계산
        virtual_root_rots_obj = Rotation.from_quat(virtual_root_quats)
        virtual_root_rot_diff = (virtual_root_rots_obj[1:] * virtual_root_rots_obj[:-1].inv())
        root_angular_velocity_y = virtual_root_rot_diff.as_euler('yxz', degrees=False)[:, 0]

        # 9. 최종 특징 벡터 조립
        #    (local_joint_position을 사용하지 않는 원래 버전 기준)
        root_y_height = positions_3d_orig[1:, 0, [1]] # 루트의 Y 높이는 원본 위치 사용
        root_xz_velocity = root_velocity_local[:, [0, 2]]
        root_y_angular_velocity = root_angular_velocity_y[:, np.newaxis]
        all_joint_6d_rotations = all_local_rotations_6d[1:, :]

        root_positions_subset = positions_3d_orig[1:, 0, :]
        all_joint_positions_subset = positions_3d_orig[1:, :, :]
        
        # SciPy 대신 쿼터니언 배열로 변환 (w,x,y,z 순서로 맞춤)
        virtual_root_quats_prev = torch.from_numpy(virtual_root_quats[:-1]).float()  # shape (M, 4), M = num_frames-1
        
        # 입력 벡터 준비: 월드 위치 - 루트 위치 (shape (M, num_joints, 3))
        world_pos_diff = torch.from_numpy(all_joint_positions_subset - root_positions_subset[:, np.newaxis, :]).float()
        
        # 가상 루트의 inverse 쿼터니언 계산 (conjugate: w 그대로, xyz 부호 반전)
        virtual_root_quats_prev_inv = virtual_root_quats_prev.clone()
        virtual_root_quats_prev_inv[:, 1:] = -virtual_root_quats_prev_inv[:, 1:]
        virtual_root_quats_prev_inv = F.normalize(virtual_root_quats_prev_inv, dim=-1)  # 안전하게 정규화
        
        # qrot로 회전 적용: qrot(inv_quat, v) (브로드캐스팅으로 한 번에)
        # virtual_root_quats_prev_inv: (M, 4)
        # world_pos_diff: (M, num_joints, 3)
        # -> qrot는 마지막 차원이 맞아야 하니, unsqueeze 등으로 브로드캐스트
        local_joint_positions = qrot(
            virtual_root_quats_prev_inv.unsqueeze(1),  # (M, 1, 4)로 확장
            world_pos_diff  # (M, num_joints, 3)
        ).numpy()  # Torch -> NumPy로 변환
        
        # 루트(0번 관절)를 제외하고 1차원으로 펼침
        local_joint_positions_flat = local_joint_positions[:, 1:, :].reshape(num_frames - 1, -1)

        # 최종 특징 벡터에 local_joint_positions_flat를 추가
        final_features = np.concatenate([
            root_y_height,                 # 1
            root_xz_velocity,              # 2
            root_y_angular_velocity,       # 1
            local_joint_positions_flat,    # 22 * 3 = 66
            all_joint_6d_rotations,        # 23 * 6 = 138
        ], axis=1)
        
        clip_filename = f"clip_{idx:04d}.npz"
        clip_filepath = os.path.join(output_processed_dir, clip_filename)

        np.savez(clip_filepath, 
                 features=final_features)

        all_motion_clips.append({
            "path": clip_filename,
            "length": final_features.shape[0]
        })

    except Exception as e:
        print(f"Could not process file {filename}. Error: {e}")
        traceback.print_exc()  # --- [추가] 이게 상세 에러 스택을 출력해줌! ---


# --- 3. 최종 데이터 취합 및 저장 ---
print("Calculating mean and std for the entire dataset...")
all_clips_for_stats = []
for metadata in tqdm(all_motion_clips, desc="Loading clips for stats"):
    # .npz 파일에서 'features' 키의 데이터만 불러와서 통계 계산
    with np.load(os.path.join(output_processed_dir, metadata['path'])) as data:
        clip_data = data['features']
        all_clips_for_stats.append(clip_data)

print("\n--- Verifying final concatenated data before stats ---")
full_dataset_np = np.concatenate(all_clips_for_stats, axis=0)
print(f"Y Height Mean in full dataset: {full_dataset_np[:, 0].mean():.4f}")

pos_vel_features = full_dataset_np[:, :4]  # Root position and velocity features
position_features = full_dataset_np[:, 4:70]  # Joint positions
rotation_features = full_dataset_np[:, 70:]  # Joint rotations

pos_vel_mean = np.mean(pos_vel_features, axis=0, keepdims=True)
pos_vel_std = np.std(pos_vel_features, axis=0, keepdims=True)
pos_vel_std[pos_vel_std == 0] = 1e-7

position_mean = np.mean(position_features, axis=0, keepdims=True)
position_std = np.std(position_features, axis=0, keepdims=True)
position_std[position_std == 0] = 1e-7

rotation_mean = np.mean(rotation_features, axis=0, keepdims=True)
rotation_std = np.std(rotation_features, axis=0, keepdims=True)
rotation_std[rotation_std == 0] = 1e-7

np.save(os.path.join(output_processed_dir, "pos_vel_mean.npy"), pos_vel_mean)
np.save(os.path.join(output_processed_dir, "pos_vel_std.npy"), pos_vel_std)
np.save(os.path.join(output_processed_dir, "position_mean.npy"), position_mean)
np.save(os.path.join(output_processed_dir, "position_std.npy"), position_std)
np.save(os.path.join(output_processed_dir, "rotation_mean.npy"), rotation_mean)
np.save(os.path.join(output_processed_dir, "rotation_std.npy"), rotation_std)

# 최종 메타데이터 파일 저장
with open(output_metadata_path, 'w') as f:
    json.dump(all_motion_clips, f, indent=4)

print("\nPreprocessing complete.")
print(f"Processed clips and metadata saved to '{output_processed_dir}'")