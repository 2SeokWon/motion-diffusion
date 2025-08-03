import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self, processed_data_path, seq_len=90):
        self.processed_data_path = processed_data_path
        self.seq_len = seq_len

        metadata_path = os.path.join(processed_data_path, "metadata.json")
        mean_path = os.path.join(processed_data_path, "mean.npy")
        std_path = os.path.join(processed_data_path, "std.npy")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.mean_np = np.load(mean_path)
        self.std_np = np.load(std_path)

         # 2. 가중 샘플링을 위한 준비
        #    - 각 클립의 길이가 SEQ_LEN보다 짧으면 제외
        #    - 각 클립의 길이를 가중치로 사용
        self.sampleable_clips = []
        self.weights = []
        for clip_info in self.metadata:
            if clip_info['length'] >= self.seq_len:
                self.sampleable_clips.append(clip_info)
                # 가중치는 클립의 길이
                self.weights.append(clip_info['length'])
        
        self.weights = np.array(self.weights, dtype=np.float32)
        self.weights /= self.weights.sum() # 전체 합이 1이 되도록 정규화

        self.mean = torch.from_numpy(self.mean_np).float()
        self.std = torch.from_numpy(self.std_np).float()

        self.virtual_dataset_size = 0
        for clip_info in self.sampleable_clips:
            # 각 클립에서 (길이 - seq_len + 1) 만큼의 고유한 시작점을 가질 수 있습니다.
            num_possible_clips = clip_info['length'] - self.seq_len + 1
            self.virtual_dataset_size += num_possible_clips
            
        print(f"Total possible unique clips (virtual dataset size): {self.virtual_dataset_size}")
    
    def __len__(self):
        return self.virtual_dataset_size

    def __getitem__(self, index):
        # 'index'는 무시하고, 매번 가중치에 따라 랜덤하게 클립을 선택
        
        # 1. 클립 길이에 비례하여 랜덤하게 클립 하나를 선택
        selected_clip_info = random.choices(self.sampleable_clips, weights=self.weights, k=1)[0]
        
        # 2. 선택된 클립의 .npy 파일 로드
        clip_path = os.path.join(self.processed_data_path, selected_clip_info['path'])
        clip_data = np.load(clip_path)
        
        # 3. 클립 내에서 랜덤한 시작 프레임 선택
        clip_length = selected_clip_info['length']
        max_start_frame = clip_length - self.seq_len
        start_frame = random.randint(0, max_start_frame)
        
        # 4. SEQ_LEN 길이만큼 클립을 잘라냄
        motion_segment = clip_data[start_frame : start_frame + self.seq_len]
        
        # 5. 정규화 및 텐서로 변환
        normalized_segment = (motion_segment - self.mean_np) / self.std_np
        return torch.from_numpy(normalized_segment).float()
