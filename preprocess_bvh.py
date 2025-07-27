# 파일 이름: preprocess_bvh_no_pymo.py

import os
import numpy as np
import torch
from tqdm import tqdm
import json

# utils.py에서 오일러 -> 6D 변환 함수를 가져옵니다.
from utils import euler_to_sixd

# --- 1. 설정 (Configuration) ---
bvh_folder_path = "./dataset/"
output_processed_dir = "./processed_data/"
output_metadata_path = os.path.join(output_processed_dir, "metadata.json")
os.makedirs(output_processed_dir, exist_ok=True)

SEQ_LEN = 90

def parse_bvh_motion_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # "MOTION" 섹션 찾기
    motion_start_index = -1
    for i, line in enumerate(lines):
        if "MOTION" in line.upper():
            motion_start_index = i
            break
            
    if motion_start_index == -1:
        return None # MOTION 섹션이 없는 경우

    # 프레임 수와 데이터 시작점 찾기
    # Frames: [num_frames]
    # Frame Time: [time]
    # [data]
    # [data]
    # ...
    num_frames = int(lines[motion_start_index + 1].split(':')[1].strip())
    data_start_index = motion_start_index + 3
    
    motion_data = []
    for i in range(data_start_index, data_start_index + num_frames):
        # 공백으로 분리된 숫자들을 float으로 변환
        values = [float(v) for v in lines[i].strip().split()]
        motion_data.append(values)
        
    return np.array(motion_data)


# --- 2. 전처리 시작 ---
all_motion_clips = []
bvh_files = [f for f in os.listdir(bvh_folder_path) if f.endswith(".bvh")]

print(f"Found {len(bvh_files)} BVH files. Starting preprocessing...")

for idx, filename in enumerate(tqdm(bvh_files, desc="Processing BVH files")):
    filepath = os.path.join(bvh_folder_path, filename)
    
    try:
        # (1) BVH 파일에서 모션 데이터(오일러 각도 + 위치) 파싱
        motion_data_euler = parse_bvh_motion_data(filepath)
        if motion_data_euler is None:
            print(f"Skipping {filename}: No MOTION data found.")
            continue
            
        motion_data_euler = motion_data_euler[::2, :] #60fps -> 30fps로 다운샘플링
        # (2) 전역 위치(3)를 제외하고 회전값(오일러 각도)만 추출
        #     이 부분은 BVH 채널 순서에 따라 달라질 수 있으므로 주의해야 합니다.
        #     (가장 흔한 구조: 힙 위치 3, 힙 회전 3, 나머지 관절 회전 3...)
        joint_rotations_deg = motion_data_euler[:, 3:]
        
        # (3) 오일러 각도를 6D 회전 표현으로 변환
        # (a) Degree -> Radian 변환
        joint_rotations_rad = np.deg2rad(joint_rotations_deg)
        num_frames = joint_rotations_rad.shape[0]

        # (b) NumPy -> PyTorch Tensor 변환
        #      utils의 함수가 PyTorch 기반이므로 텐서로 바꿔줍니다.
        #      데이터를 [Frames, Joints, 3] 형태로 reshape합니다.
        num_joints = joint_rotations_rad.shape[1] // 3
        rotations_tensor = torch.from_numpy(joint_rotations_rad).float()
        rotations_tensor = rotations_tensor.reshape(num_frames, num_joints, 3)
        
        # (c) euler_to_sixd 함수 호출
        sixd_rotations_tensor = euler_to_sixd(rotations_tensor, order='yxz')
        
        # (d) 다시 NumPy 배열로 변환하고 평탄화 (Flatten)
        #     [Frames, Joints, 6] -> [Frames, Joints * 6]
        sixd_rotations_np = sixd_rotations_tensor.numpy()
        sixd_rotations_flat = sixd_rotations_np.reshape(num_frames, -1)
        
        clip_filename = f"clip_{idx:04d}.npy"
        clip_filepath = os.path.join(output_processed_dir, clip_filename)
        np.save(clip_filepath, sixd_rotations_flat)

        all_motion_clips.append({
            "path": clip_filename,
            "length": num_frames
        })

    except Exception as e:
        print(f"Could not process file {filename}. Error: {e}")

# --- 3. 최종 데이터 취합 및 저장 ---
print("Calculating mean and std for the entire dataset...")
all_clips_for_stats = []
for metadata in tqdm(all_motion_clips, desc="Loading clips for stats"):
    clip_data = np.load(os.path.join(output_processed_dir, metadata['path']))
    all_clips_for_stats.append(clip_data)

full_dataset_np = np.concatenate(all_clips_for_stats, axis=0)
mean = np.mean(full_dataset_np, axis=0, keepdims=True)
std = np.std(full_dataset_np, axis=0, keepdims=True)
std[std == 0] = 1e-7

# 통계 정보 파일 저장
np.save(os.path.join(output_processed_dir, "mean.npy"), mean)
np.save(os.path.join(output_processed_dir, "std.npy"), std)

# 최종 메타데이터 파일 저장
with open(output_metadata_path, 'w') as f:
    json.dump(all_motion_clips, f, indent=4)

print("\nPreprocessing complete.")
print(f"Processed clips and metadata saved to '{output_processed_dir}'")