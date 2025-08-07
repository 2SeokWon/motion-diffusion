import os
import numpy as np
import torch
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation

# 필요한 유틸리티 및 클래스 import
from kinematics import Skeleton, euler_to_sixd

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

        # 1. 원본 BVH 데이터 로드 (이름에 _orig를 붙여 구분)
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
        if idx == 47:
            print(positions_3d_orig)
        # 3. 데이터 정규화 (이 코드는 올바르게 적용되었습니다)
        positions_3d_normalized = positions_3d_orig.copy() # 원본 보존을 위해 복사
        
        # 바닥에 놓기
        floor_height = positions_3d_normalized.reshape(-1, 3)[:, 1].min()

        positions_3d_normalized[:, :, 1] -= floor_height
        
        # 초기 위치 원점 맞추기
        root_pos_init_xz = positions_3d_normalized[0, 0, :] * np.array([1, 0, 1])
        positions_3d_normalized = positions_3d_normalized - root_pos_init_xz

        if idx == 47:
            print(f"\n--- Debugging {filename} ---")
            print(f"Original Y Height Mean: {root_positions_np_orig[:, 1].mean():.4f}")
            print(f"Normalized Y Height Mean: {positions_3d_normalized[:, :, 1].mean():.4f}")
        # --- 이제부터 모든 계산은 'positions_3d_normalized'를 기반으로 해야 합니다 ---

        # 4. 정규화된 위치로부터 회전 정보 재추출 (Inverse Kinematics)
        #    이것이 가장 정확하지만, 계산이 복잡하고 느릴 수 있음.
        #    대안: 원본 회전 정보를 사용하되, 이것이 오차의 원인이 될 수 있음을 인지.
        #    여기서는 일단 원본 회전(rotations_quat_np_orig)을 그대로 사용하겠습니다.
        rotations_quat_np = rotations_quat_np_orig
        
        # 5. 정규화된 위치로부터 Facing Rotation 계산
        facing_rotations_quat = np.zeros((num_frames, 4))
        for i in range(num_frames):
            across = (positions_3d_normalized[i, r_hip_idx] - positions_3d_normalized[i, l_hip_idx]) + (positions_3d_normalized[i, r_shoulder_idx] - positions_3d_normalized[i, l_shoulder_idx])
            across[1] = 0
            norm = np.linalg.norm(across)
            if norm < 1e-6:
                facing_rotations_quat[i] = facing_rotations_quat[i-1] if i > 0 else np.array([0, 0, 0, 1])
            else:
                forward = np.cross([0, 1, 0], across / norm)
                facing_rot = Rotation.align_vectors([[0, 0, 1]], forward[np.newaxis, :])[0]
                facing_rotations_quat[i] = facing_rot.as_quat()
        
        # 6. Global -> Local Root Orientation 계산
        root_global_rotations = Rotation.from_quat(rotations_quat_np[:, 0, :])
        facing_rotations = Rotation.from_quat(facing_rotations_quat)
        root_local_rotations = facing_rotations.inv() * root_global_rotations
        root_local_rotations_quat = root_local_rotations.as_quat()
        
        # 7. 모든 관절의 Local Orientation을 6D로 변환
        joint_rotations_quat_np = rotations_quat_np[:, 1:, :]
        all_local_rotations_quat = np.concatenate(
            [root_local_rotations_quat[:, np.newaxis, :], joint_rotations_quat_np], axis=1
        )
        all_local_rotations_rad = Rotation.from_quat(all_local_rotations_quat.reshape(-1, 4)).as_euler('yxz', degrees=False)
        all_local_rotations_6d = euler_to_sixd(torch.from_numpy(all_local_rotations_rad).float()).numpy().reshape(num_frames, -1)
        
        # 8. 루트 속도 계산 (정규화된 위치 사용!)
        normalized_root_positions = positions_3d_normalized[:, 0, :]
        print(f"Normalized Root Positions Mean: {normalized_root_positions}")
        root_velocity_global_normalized = normalized_root_positions[1:] - normalized_root_positions[:-1]
        
        inv_facing_rotations_prev = Rotation.from_quat(facing_rotations_quat[:-1]).inv()
        root_velocity_local = inv_facing_rotations_prev.apply(root_velocity_global_normalized)
        
        #root_rot_quat_diff = (root_global_rotations[1:] * root_global_rotations[:-1].inv()).as_quat()
        #root_angular_velocity_y = Rotation.from_quat(root_rot_quat_diff).as_euler('yxz', degrees=False)[:, 0]

        # 9. 최종 특징 벡터 조립 (정규화된 값 사용!)
        root_y_height = normalized_root_positions[1:, [1]] # <-- 핵심 수정
        if idx == 47:
            print(f"Root Y Height Mean: {root_y_height.mean():.4f}")
        root_xz_velocity = root_velocity_local[:, [0, 2]]
        #root_y_angular_velocity = root_angular_velocity_y[:, np.newaxis]
        all_joint_6d_rotations = all_local_rotations_6d[1:, :]
        
        final_features = np.concatenate([
            root_y_height,
            root_xz_velocity,
            #root_y_angular_velocity,
            all_joint_6d_rotations
        ], axis=1)   

        if idx == 47:
            print(f"Final Features Y Height Mean: {final_features[:, 0].mean():.4f}")
            print("-" * 30)     
        '''
        root_rotations_deg = motion_data_euler[:, 3:6] # [num_frames, 3]
        joint_rotations_deg = motion_data_euler[:, 6:] # [num_frames, 22 * 3]
        

        # (3) 오일러 각도를 Quaternion으로 변환, 회전 계산을 위해서 [num_frames, 4]
        root_rotations_quat = Rotation.from_euler(ROTATION_ORDER, root_rotations_deg, degrees=True).as_quat()
        
        #이전 프레임 대비 루트 위치 및 회전 변화량 계산 [num_frames - 1, 3] 근데 월드좌표계 기준임
        root_velocity = root_positions[1:] - root_positions[:-1]

        # 루트 회전 변화량 계산 t-1에서 t로 넘어갈 때 캐릭터의 회전 변화량 
        root_rot_quat_diff = (
            Rotation.from_quat(root_rotations_quat[1:]) * #현재 프레임의 root 회전
            Rotation.from_quat(root_rotations_quat[:-1]).inv() #이전 프레임의 root 역회전
        ).as_quat()

        #변화량을 루트의 로컬 좌표계 기준으로 변환
        inv_root_rotations = Rotation.from_quat(root_rotations_quat[:-1]).inv() #이전 프레임의 역회전
        root_velocity_local = inv_root_rotations.apply(root_velocity)
        
        #Y축 회전 변화량만 추출, 캐릭터가 어느 방향으로 나아가는지?
        root_angular_velocity_y = Rotation.from_quat(root_rot_quat_diff).as_euler('yxz', degrees=True)[:, 0]

        #오일러 각을 6d로 변환, 짐벌락 없애야 하니까
        all_rotations_deg = np.concatenate([root_rotations_deg, joint_rotations_deg], axis=1) #root position만 제외
        all_rotations_rad = np.deg2rad(all_rotations_deg) #degree to radian [100, 69]
        num_joints = all_rotations_rad.shape[1] // 3
        rotations_tensor = torch.from_numpy(all_rotations_rad).float() #pytorch tensor로 변환
        rotations_tensor = rotations_tensor.reshape(num_frames, num_joints, 3)
        
        # (c) euler_to_sixd 함수 호출 
        # 여기서부터 다시 확인
        sixd_rotations_tensor = euler_to_sixd(rotations_tensor, order=ROTATION_ORDER) #오일러 각도를 6D로
        
        # (d) 다시 NumPy 배열로 변환하고 평탄화 (Flatten)
        sixd_rotations_np = sixd_rotations_tensor.numpy()
        sixd_rotations_flat = sixd_rotations_np.reshape(num_frames, -1)

        # (e) Root Local Orientation 계산
        facing_rotations_deg_y = root_rotations_deg[:, 0] #루트 관절의 Y축 회전만 추출
        facing_rotations_quat = Rotation.from_euler('y', facing_rotations_deg_y, degrees=True).as_quat() #쿼터니언으로 변환

        inv_facing_rotations = Rotation.from_quat(facing_rotations_quat).inv()
        root_global_rotations = Rotation.from_quat(root_rotations_quat)

        root_local_rotations_quat = (inv_facing_rotations * root_global_rotations).as_quat() #루트 관절의 로컬 회전

        root_local_rotations_rad = Rotation.from_quat(root_local_rotations_quat).as_euler(ROTATION_ORDER, degrees=False)
        root_local_rotations_6d = euler_to_sixd(torch.from_numpy(root_local_rotations_rad).float().unsqueeze(0), order='yxz').squeeze(0).numpy()

        all_rotations_6d_np = sixd_rotations_flat.copy()
        all_rotations_6d_np[:, :6] = root_local_rotations_6d #루트 관절의 6D 회전으로 대체
        
        # 첫 프레임은 속도를 계산할 수 없으므로, 두 번째 프레임부터 시작합니다.
        # 따라서 모든 데이터의 길이는 (num_frames - 1)이 됩니다.
        root_y_height = root_positions[1:, [1]]  # shape (N-1, 1)
        root_xz_velocity = root_velocity_local[:, [0, 2]] # shape (N-1, 2)
        root_y_angular_velocity = root_angular_velocity_y[:, np.newaxis] # shape (N-1, 1)

        all_joint_6d_rotations = all_rotations_6d_np[1:, :]  # shape (N-1, Joints * 6)
        
        final_features = np.concatenate([ #결국 학습에선 이걸 사용
            root_y_height,
            root_xz_velocity,
            root_y_angular_velocity,
            all_joint_6d_rotations
        ], axis=1)
        '''

        print(f"\n\n--- Debugging Preprocessing for: {filename} ---")
        frame_to_check = 50
        
        # 1. 원본 Global Rotation
        original_global_rot_deg = motion_data_euler[frame_to_check, 3:6]
        print(f"\n[Frame {frame_to_check}] Original Global Rotation (YXZ, deg):")
        print(f"  Y(facing): {original_global_rot_deg[0]:.2f}, X(pitch): {original_global_rot_deg[1]:.2f}, Z(roll): {original_global_rot_deg[2]:.2f}")

        # 2. 계산된 Local Rotation
        calculated_local_rot_deg = Rotation.from_quat(root_local_rotations_quat[frame_to_check]).as_euler('yxz', degrees=True)
        print(f"\n[Frame {frame_to_check}] Calculated Local Rotation (YXZ, deg):")
        print(f"  Y(facing): {calculated_local_rot_deg[0]:.2f}, X(pitch): {calculated_local_rot_deg[1]:.2f}, Z(roll): {calculated_local_rot_deg[2]:.2f}")
        
        # 3. 검증
        facing_rot_obj = Rotation.from_quat(facing_rotations_quat[frame_to_check])
        local_rot_obj = Rotation.from_quat(root_local_rotations_quat[frame_to_check])
        reconstructed_global_rot_obj = facing_rot_obj * local_rot_obj
        original_global_rot_obj = Rotation.from_quat(rotations_quat_np[frame_to_check, 0, :])
        
        dot_product = np.abs(np.dot(reconstructed_global_rot_obj.as_quat(), original_global_rot_obj.as_quat()))
        
        print("\n--- Verification ---")
        print(f"  -> Dot product of quaternions: {dot_product:.6f} (1.0에 가까워야 함)")
        if np.isclose(dot_product, 1.0):
            print("  => SUCCESS: Local orientation was calculated correctly!")
        else:
            print("  => FAILURE: There is an issue in the local orientation calculation.")
            exit(1)
        print("-" * 50)
        clip_filename = f"clip_{idx:04d}.npz"
        clip_filepath = os.path.join(output_processed_dir, clip_filename)

        facing_rotations_to_save= facing_rotations_quat[1:]

        np.savez(clip_filepath, 
                 features=final_features,
                 facing_rotations=facing_rotations_to_save)

        all_motion_clips.append({
            "path": clip_filename,
            "length": final_features.shape[0]
        })

    except Exception as e:
        print(f"Could not process file {filename}. Error: {e}")

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

pos_vel_features = full_dataset_np[:, :3]  # Root position and velocity features
rotation_features = full_dataset_np[:, 3:]  # Joint rotations

pos_vel_mean = np.mean(pos_vel_features, axis=0, keepdims=True)
pos_vel_std = np.std(pos_vel_features, axis=0, keepdims=True)
pos_vel_std[pos_vel_std == 0] = 1e-7

rotation_mean = np.mean(rotation_features, axis=0, keepdims=True)
rotation_std = np.std(rotation_features, axis=0, keepdims=True)
rotation_std[rotation_std == 0] = 1e-7

np.save(os.path.join(output_processed_dir, "pos_vel_mean.npy"), pos_vel_mean)
np.save(os.path.join(output_processed_dir, "pos_vel_std.npy"), pos_vel_std)
np.save(os.path.join(output_processed_dir, "rotation_mean.npy"), rotation_mean)
np.save(os.path.join(output_processed_dir, "rotation_std.npy"), rotation_std)

# 최종 메타데이터 파일 저장
with open(output_metadata_path, 'w') as f:
    json.dump(all_motion_clips, f, indent=4)

print("\nPreprocessing complete.")
print(f"Processed clips and metadata saved to '{output_processed_dir}'")