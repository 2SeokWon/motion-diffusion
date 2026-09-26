# core/dataset.py
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .motion_features import tensor_to_motion_object_root


def split_range(length, split, test_ratio):
    """
    한 클립에서 split에 해당하는 프레임 범위 [lo, hi).
    각 파일의 앞부분은 학습, 뒤 test_ratio는 평가용이다. 창은 이 범위 안에서만 뽑으므로
    학습 창과 평가 창은 한 프레임도 겹치지 않는다.
    """
    n_test = int(round(length * test_ratio))
    if split == 'train':
        return 0, length - n_test
    if split == 'test':
        return length - n_test, length
    if split == 'all':
        return 0, length
    raise ValueError(f"Unknown split: {split}")


class MotionDataset(Dataset):
    def __init__(self, processed_data_path, seq_len=180, feat_bias=15.0, max_waypoints=10, dense_prob=0.1,
                 split='train', test_ratio=0.15, stats_dir=None):
        self.processed_data_path = processed_data_path
        self.seq_len = seq_len
        self.feat_bias = feat_bias
        self.max_waypoints = max_waypoints #시작/도착 외 중간 경유점 최대 개수
        self.dense_prob = dense_prob #전체 궤적을 조건으로 주는 확률
        self.split = split
        self.test_ratio = test_ratio
        # 정규화 통계는 학습 구간만으로 계산한 값을 쓴다 (scripts/split_stats.py). 없으면 전처리가 만든 전체 통계.
        stats_dir = stats_dir or processed_data_path
        metadata_path = os.path.join(processed_data_path, "metadata.json")

        with open(metadata_path, 'r') as f:
            meta_raw = json.load(f)

        if len(meta_raw) > 0 and isinstance(meta_raw[0], list):
            self.metadata = [item for sub in meta_raw for item in sub]
        else:
            self.metadata = meta_raw

        self.name_classes = sorted(set(clip_info['class_name'] for clip_info in self.metadata))
        self.num_name_classes = len(self.name_classes)

        self.root_pos_mean = np.load(os.path.join(stats_dir, "root_pos_mean.npy"))
        self.root_pos_std = np.load(os.path.join(stats_dir, "root_pos_std.npy"))
        self.root_pos_std = np.maximum(self.root_pos_std / feat_bias, 1e-8)

        self.position_mean = np.load(os.path.join(stats_dir, "position_mean.npy"))
        self.position_std = np.load(os.path.join(stats_dir, "position_std.npy"))
        # hip에 고정 offset으로 붙은 관절(Chest, RightHip, LeftHip)은 hip 기준 위치가 상수이다.
        # 반올림 오차(std≈4e-6)를 표준편차 1로 키우지 않도록, std가 매우 작으면 나누지 않는다(→ 정규화 후 ≈ 0).
        self.position_std = np.where(self.position_std < 1e-3, 1.0, self.position_std)

        self.rotation_mean = np.load(os.path.join(stats_dir, "rotation_mean.npy"))
        self.rotation_std = np.load(os.path.join(stats_dir, "rotation_std.npy"))
        self.rotation_std = np.maximum(self.rotation_std, 1e-8)

        self.foot_mean = np.load(os.path.join(stats_dir, "foot_mean.npy"))
        foot_std = np.load(os.path.join(stats_dir, "foot_std.npy"))
        self.foot_std = np.ones_like(foot_std)

        self.abs_traj_mean = np.load(os.path.join(stats_dir, "abs_traj_mean.npy"))
        self.abs_traj_std = np.load(os.path.join(stats_dir, "abs_traj_std.npy"))
        # 궤적은 노이즈를 섞는 생성 대상이 아니라 조건 입력이므로 feat_bias를 적용하지 않는다(표준정규화)
        self.abs_traj_std = np.maximum(self.abs_traj_std, 1e-8)

        self.sampleable_clips = [
            c for c in self.metadata
            if np.diff(split_range(c['length'], split, test_ratio))[0] >= self.seq_len
        ]

        self.index_map = []
        for clip_idx, clip_info in enumerate(self.sampleable_clips):
            lo, hi = split_range(clip_info['length'], split, test_ratio)
            for start_frame in range(lo, hi - self.seq_len + 1): #창 전체가 [lo, hi) 안에 있어야 한다
                self.index_map.append((clip_idx, start_frame))

        print(f"[{split}] Total possible unique clips (virtual dataset size): {len(self.index_map)}")

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
        # 창의 첫 프레임은 직전 프레임이 없으므로 root 속도/각속도를 0으로 둔다.
        # (make_control.py, preprocess.py의 abs_traj 통계와 같은 규약 → 조건 궤적이 항상 원점에서 시작)
        features[0, 1:4] = 0.0

        abs_traj = tensor_to_motion_object_root(features)  # [180, 3]

        # Normalize
        root_hip_part = (features[:, 0:1] - self.root_pos_mean[0]) / self.root_pos_std[0]      # [180, 1]
        root_vel_part = (features[:, 1:4] - self.root_pos_mean[1:4]) / self.root_pos_std[1:4]  # [180, 3]
        position_part = (features[:, 4:70] - self.position_mean) / self.position_std            # [180, 66]
        rotation_part = (features[:, 70:208] - self.rotation_mean) / self.rotation_std          # [180, 138]
        foot_part     = (features[:, 208:210] - self.foot_mean) / self.foot_std                 # [180, 2]
        traj_part     = (abs_traj - self.abs_traj_mean) / self.abs_traj_std                     # [180, 3]

        normalized_segment = np.concatenate(
            [root_hip_part, root_vel_part, position_part, rotation_part, foot_part],
            axis=1
        )  # [180, 210] ← 궤적은 정답에 넣지 않는다 (넣으면 조건을 그대로 베끼는 문제가 생김)

        motion_tensor = torch.from_numpy(normalized_segment).float()
        mask = self._sample_waypoint_mask()  # [180, 1]
        cond = torch.cat([torch.from_numpy(traj_part).float() * mask, mask], dim=1)  # [180, 4] = 경유점 값 + 마스크
        label_one_hot = F.one_hot(torch.tensor(class_name_idx), num_classes=self.num_name_classes).float()

        return {
            'motion': motion_tensor,
            'cond': cond,
            'label_name': label_one_hot,
        }

    def _sample_waypoint_mask(self):
        """
        매 샘플마다 어떤 프레임의 궤적을 보여줄지 무작위로 정한다 (빈칸 채우기 학습).
        DataLoader worker마다 numpy 난수 상태가 복제되는 문제를 피하려고 torch 난수를 쓴다.
        """
        T = self.seq_len
        mask = torch.zeros(T, 1)
        if torch.rand(()) < self.dense_prob: #전체 궤적 (dense 경로 추종 능력 유지)
            return mask.fill_(1.0)

        mask[0] = 1.0  #시작 (항상 원점)
        mask[-1] = 1.0 #도착
        k = int(torch.randint(0, self.max_waypoints + 1, ()))
        mask[1 + torch.randperm(T - 2)[:k]] = 1.0 #중간 경유점 0~max_waypoints개
        return mask
