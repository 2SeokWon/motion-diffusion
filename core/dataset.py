# core/dataset.py
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .motion_features import tensor_to_motion_object_root


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

        self.abs_traj_mean = np.load(os.path.join(processed_data_path, "abs_traj_mean.npy"))
        self.abs_traj_std = np.load(os.path.join(processed_data_path, "abs_traj_std.npy"))
        self.abs_traj_std = np.maximum(self.abs_traj_std / self.feat_bias, 1e-8)

        self.sampleable_clips = [c for c in self.metadata if c['length'] >= self.seq_len]

        self.index_map = []
        for clip_idx, clip_info in enumerate(self.sampleable_clips):
            num_possible_clips = clip_info['length'] - self.seq_len + 1
            for start_frame in range(num_possible_clips):
                self.index_map.append((clip_idx, start_frame))

        print(f"Total possible unique clips (virtual dataset size): {len(self.index_map)}")

        self.clip_cache = {}
        for clip_info in self.sampleable_clips:
            clip_path = os.path.join(self.processed_data_path, clip_info['path'])
            with np.load(clip_path, mmap_mode='r') as data:
                self.clip_cache[clip_info['path']] = data['features'].copy()

        print(f"Loaded {len(self.clip_cache)} clips into cache. Ready for training!")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        clip_idx, start_frame = self.index_map[index]
        selected_clip_info = self.sampleable_clips[clip_idx]
        clip_path = selected_clip_info['path']
        class_name_idx = selected_clip_info['class_name_idx']

        clip_data = self.clip_cache[clip_path]
        features = clip_data[start_frame:start_frame + self.seq_len].copy()  # [180, 210]

        abs_traj = tensor_to_motion_object_root(features)  # [180, 3]

        # Normalize
        root_hip_part = (features[:, 0:1] - self.root_pos_mean[0]) / self.root_pos_std[0]      # [180, 1]
        root_vel_part = (features[:, 1:4] - self.root_pos_mean[1:4]) / self.root_pos_std[1:4]  # [180, 3]
        position_part = (features[:, 4:70] - self.position_mean) / self.position_std            # [180, 66]
        rotation_part = (features[:, 70:208] - self.rotation_mean) / self.rotation_std          # [180, 138]
        foot_part     = (features[:, 208:210] - self.foot_mean) / self.foot_std                 # [180, 2]
        cond_part     = (abs_traj - self.abs_traj_mean) / self.abs_traj_std                     # [180, 3]

        normalized_segment = np.concatenate(
            [root_hip_part, root_vel_part, position_part, rotation_part, foot_part, cond_part],
            axis=1
        )  # [180, 213]

        motion_tensor = torch.from_numpy(normalized_segment).float()
        label_one_hot = F.one_hot(torch.tensor(class_name_idx), num_classes=self.num_name_classes).float()

        return {
            'motion': motion_tensor,
            'label_name': label_one_hot,
        }
