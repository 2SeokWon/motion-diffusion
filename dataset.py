#dataset.py
import os
import json
import random
import numpy as np
import torch
import math
from torch.utils.data import Dataset
import torch.nn.functional as F
from new_preprocess import tensor_to_motion_object_root, moving_average_path, compute_delta_traj

class MotionDataset(Dataset):
    def __init__(self, processed_data_path, seq_len=180, feat_bias=15.0):
        self.processed_data_path = processed_data_path
        self.seq_len = seq_len
        self.feat_bias = feat_bias
        metadata_path = os.path.join(processed_data_path, "metadata.json")

        with open(metadata_path, 'r') as f:
            meta_raw = json.load(f)

        if len(meta_raw) > 0 and isinstance(meta_raw[0], list):
            self.metadata = [item for sub in meta_raw for item in sub]
        else:
            self.metadata = meta_raw

        self.name_classes = sorted(set(clip_info['class_name'] for clip_info in self.metadata))
        self.num_name_classes = len(self.name_classes)

        self.root_pos_mean = np.load(os.path.join(processed_data_path, "root_pos_mean.npy"))
        self.root_pos_std = np.load(os.path.join(processed_data_path, "root_pos_std.npy"))
        self.root_pos_std = np.maximum(self.root_pos_std / feat_bias, 1e-8)

        self.position_mean = np.load(os.path.join(processed_data_path, "position_mean.npy"))
        self.position_std = np.load(os.path.join(processed_data_path, "position_std.npy"))
        self.position_std = np.maximum(self.position_std, 1e-8)

        self.rotation_mean = np.load(os.path.join(processed_data_path, "rotation_mean.npy"))
        self.rotation_std = np.load(os.path.join(processed_data_path, "rotation_std.npy"))
        self.rotation_std = np.maximum(self.rotation_std, 1e-8)
        
        self.foot_mean = np.load(os.path.join(processed_data_path, "foot_mean.npy"))
        foot_std = np.load(os.path.join(processed_data_path, "foot_std.npy"))
        self.foot_std = np.ones_like(foot_std)

        self.interp_mean = np.load(os.path.join(processed_data_path, "interp_mean.npy"))
        self.interp_std = np.load(os.path.join(processed_data_path, "interp_std.npy"))
        self.interp_std = np.maximum(self.interp_std / self.feat_bias, 1e-8)

        self.delta_mean = np.load(os.path.join(processed_data_path, "delta_mean.npy"))
        self.delta_std = np.load(os.path.join(processed_data_path, "delta_std.npy"))
        self.delta_std = np.maximum(self.delta_std / self.feat_bias, 1e-8)

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

        class_name_idx = selected_clip_info['class_name_idx']

        # Cache 사용
        clip_data = self.clip_cache[clip_path]  
                
        # 3. SEQ_LEN 길이만큼 클립을 잘라냄
        end_frame = start_frame + self.seq_len
        features = clip_data[start_frame:end_frame].copy()      # [180, 210]

        abs_traj = tensor_to_motion_object_root(features)  # [180,3]
        interp_feat = moving_average_path(abs_traj[:, :2], abs_traj[:, 2], radius=30)
        delta_feat = compute_delta_traj(
            abs_traj[:, :2], abs_traj[:, 2],
            interp_feat[:, :2], interp_feat[:, 2]
        )

        # Normalize motion features
        root_hip_part = (features[:, 0] - self.root_pos_mean[0]) / self.root_pos_std[0] # [180,]
        root_delta_part = (delta_feat - self.delta_mean) / self.delta_std # [180,3]
        position_part = (features[:, 4:70] - self.position_mean) / self.position_std # [180,66]
        rotation_part = (features[:, 70:208] - self.rotation_mean) / self.rotation_std  # [180,138]
        foot_part = (features[:, 208:210] - self.foot_mean) / self.foot_std # [180,2]
        cond_part = (interp_feat - self.interp_mean) / self.interp_std # [180,3]

        normalized_segment = np.concatenate([root_hip_part[:, np.newaxis], root_delta_part, position_part, rotation_part, foot_part, cond_part], axis=1) #213
        
        motion_tensor = torch.from_numpy(normalized_segment).float() # [seq_len, 213]
        label_one_hot_name = F.one_hot(torch.tensor(class_name_idx), num_classes=self.num_name_classes).float()

        return {
            'motion': motion_tensor,
            'label_name': label_one_hot_name,
        }
