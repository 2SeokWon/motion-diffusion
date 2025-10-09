#dataset.py
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class MotionDataset(Dataset):
    def __init__(self, processed_data_path, seq_len=180, feat_bias=15.0):
        self.processed_data_path = processed_data_path
        self.seq_len = seq_len
        self.feat_bias = feat_bias
        metadata_path = os.path.join(processed_data_path, "metadata.json")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.classes = sorted(set(clip_info['class_name'] for clip_info in self.metadata))
        self.num_classes = len(self.classes)
        self.class_map = {name : i for i , name in enumerate(self.classes)}

        self.pos_vel_mean = np.load(os.path.join(processed_data_path, "pos_vel_mean.npy"))
        pos_vel_std = np.load(os.path.join(processed_data_path, "pos_vel_std.npy"))
        pos_vel_std /= self.feat_bias
        pos_vel_std = np.maximum(pos_vel_std, 1e-8)
        self.pos_vel_std = pos_vel_std

        self.position_mean = np.load(os.path.join(processed_data_path, "position_mean.npy"))
        self.position_std = np.load(os.path.join(processed_data_path, "position_std.npy"))
        self.position_std = np.maximum(self.position_std, 1e-8)

        self.rotation_mean = np.load(os.path.join(processed_data_path, "rotation_mean.npy"))
        self.rotation_std = np.load(os.path.join(processed_data_path, "rotation_std.npy"))
        self.rotation_std = np.maximum(self.rotation_std, 1e-8)
        """
        self.foot_mean = np.load(os.path.join(processed_data_path, "foot_mean.npy"))
        foot_std = np.load(os.path.join(processed_data_path, "foot_std.npy"))
        foot_std /= self.feat_bias
        self.foot_std = foot_std
        """
        # 2. 가중 샘플링을 위한 준비
        #    - 각 클립의 길이가 SEQ_LEN보다 짧으면 제외
        self.sampleable_clips = []
        for clip_info in self.metadata:
            if clip_info['length'] >= self.seq_len:
                self.sampleable_clips.append(clip_info)

        self.index_map = []

        for clip_idx, clip_info in enumerate(self.sampleable_clips):
            # 각 클립에서 (길이 - seq_len + 1) 만큼의 고유한 시작점을 가질 수 있습니다.
            num_possible_clips = clip_info['length'] - self.seq_len + 1
            for start_frame in range(num_possible_clips):
                self.index_map.append((clip_idx, start_frame))

        print(f"Total possible unique clips (virtual dataset size): {len(self.index_map)}")

        self.clip_cache = {} # 데이터셋이 많아지면 메모리 에러날 듯? 그때 수정해야함.
        for clip_info in self.sampleable_clips:
            clip_path = os.path.join(self.processed_data_path, clip_info['path'])
            with np.load(clip_path, mmap_mode = 'r') as data:
                self.clip_cache[clip_info['path']] = data['features'].copy() #메모리 복사
        
        print(f"Loaded {len(self.clip_cache)} clips into cache. Ready for training!")
    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        # 1. 가중치에 따라 샘플링된 클립 정보 가져오기
        clip_idx, start_frame = self.index_map[index]
        selected_clip_info = self.sampleable_clips[clip_idx]
        clip_path = selected_clip_info['path']

        class_idx = selected_clip_info['class_idx']

        # Cache 사용
        clip_data = self.clip_cache[clip_path]        
                
        # 3. SEQ_LEN 길이만큼 클립을 잘라냄
        motion_segment = clip_data[start_frame : start_frame + self.seq_len]
        
        # 4. 정규화 및 텐서로 변환
        pos_vel_part = (motion_segment[:, :4] - self.pos_vel_mean) / self.pos_vel_std
        position_part = (motion_segment[:, 4:70] - self.position_mean) / self.position_std
        rotation_part = (motion_segment[:, 70:208] - self.rotation_mean) / self.rotation_std
        #foot_part = (motion_segment[:, 208:212] - self.foot_mean) / self.foot_std

        #normalized_segment = np.concatenate([pos_vel_part, position_part, rotation_part, foot_part], axis=1)
        normalized_segment = np.concatenate([pos_vel_part, position_part, rotation_part], axis=1)
        motion_tensor = torch.from_numpy(normalized_segment).float()
        label_one_hot = F.one_hot(torch.tensor(class_idx), num_classes=self.num_classes).float()

        return {
            'motion': motion_tensor, 
            'label': label_one_hot
        }
