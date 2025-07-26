# 파일 이름: preprocess_bvh_no_pymo.py

import os
import numpy as np
import torch
from tqdm import tqdm

# utils.py에서 오일러 -> 6D 변환 함수를 가져옵니다.
from utils import euler_to_sixd

# --- 1. 설정 (Configuration) ---
bvh_folder_path = "./dataset/"
output_npy_path = "./processed_motion_data.npy"
SEQ_LEN = 30
STEP_SIZE = 10

def parse_bvh_motion_data(filepath):
    """
    pymo 없이 BVH 파일에서 MOTION 데이터만 수동으로 읽어오는 함수
    """
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

print(f"Found {len(bvh_files)} BVH files. Starting preprocessing (without pymo)...")

for filename in tqdm(bvh_files, desc="Processing BVH files"):
    filepath = os.path.join(bvh_folder_path, filename)
    
    try:
        # (1) BVH 파일에서 모션 데이터(오일러 각도 + 위치) 파싱
        motion_data_euler = parse_bvh_motion_data(filepath)
        if motion_data_euler is None:
            print(f"Skipping {filename}: No MOTION data found.")
            continue
            
        num_frames, total_features = motion_data_euler.shape
        
        # (2) 전역 위치(3)를 제외하고 회전값(오일러 각도)만 추출
        #     이 부분은 BVH 채널 순서에 따라 달라질 수 있으므로 주의해야 합니다.
        #     (가장 흔한 구조: 힙 위치 3, 힙 회전 3, 나머지 관절 회전 3...)
        joint_rotations_deg = motion_data_euler[:, 3:]
        
        # (3) 오일러 각도를 6D 회전 표현으로 변환
        # (a) Degree -> Radian 변환
        joint_rotations_rad = np.deg2rad(joint_rotations_deg)
        
        # (b) NumPy -> PyTorch Tensor 변환
        #      utils의 함수가 PyTorch 기반이므로 텐서로 바꿔줍니다.
        #      데이터를 [Frames, Joints, 3] 형태로 reshape합니다.
        num_joints = joint_rotations_rad.shape[1] // 3
        rotations_tensor = torch.from_numpy(joint_rotations_rad).float()
        rotations_tensor = rotations_tensor.reshape(num_frames, num_joints, 3)
        
        # (c) euler_to_sixd 함수 호출
        sixd_rotations_tensor = euler_to_sixd(rotations_tensor, order='zyx')
        
        # (d) 다시 NumPy 배열로 변환하고 평탄화 (Flatten)
        #     [Frames, Joints, 6] -> [Frames, Joints * 6]
        sixd_rotations_np = sixd_rotations_tensor.numpy()
        sixd_rotations_flat = sixd_rotations_np.reshape(num_frames, -1)
        
        # (4) 슬라이딩 윈도우를 사용해 고정 길이 클립으로 자르기
        if num_frames >= SEQ_LEN:
            for start_frame in range(0, num_frames - SEQ_LEN + 1, STEP_SIZE):
                end_frame = start_frame + SEQ_LEN
                clip = sixd_rotations_flat[start_frame:end_frame, :]
                all_motion_clips.append(clip)
                
    except Exception as e:
        print(f"Could not process file {filename}. Error: {e}")

# --- 3. 최종 데이터 취합 및 저장 ---
if not all_motion_clips:
    print("\nNo motion clips were generated. Please check your BVH files or settings.")
else:
    final_data = np.array(all_motion_clips, dtype=np.float32)
    
    print(f"\nPreprocessing complete.")
    print(f"Total motion clips generated: {final_data.shape[0]}")
    print(f"Data shape: {final_data.shape}")
    
    np.save(output_npy_path, final_data)
    print(f"Data saved to {output_npy_path}")