import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import json
import random

from tqdm import tqdm

from model import MotionTransformer
from gaussian_diffusion import GaussianDiffusion
from utils import sixd_to_euler_angles

learning_rate = 1e-4
num_epochs = 50
batch_size = 32
save_interval = 10

data_path = ''
njoints = 23
nfeats = 6
seq_len = 90

num_timesteps = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

save_dir = "./checkpoints"
output_bvh_dir = "./results"
processed_data_path = "./processed_data"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(output_bvh_dir, exist_ok=True)

skeleton_template_path = "./dataset/Aeroplane_BR.bvh"  # 스켈레톤 템플릿 파일 경로
num_samples_to_generate = 5  # 생성할 샘플 수

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

def generate_and_save_bvh(model, diffusion, dataset, num_samples, seq_len, njoints, nfeats, template_path, output_dir, device):
    print("Starting sampling...")
    model.eval()

    with torch.no_grad():
        sample_shape = (num_samples, seq_len, njoints * nfeats)
        generated_motion = diffusion.p_sample_loop(model, sample_shape)

        mean = dataset.mean.to(device)
        std = dataset.std.to(device)
        unnormalized_motion = generated_motion * std + mean
    
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
        
        single_motion = unnormalized_motion[i]
        
        # 6D -> Euler 변환 (utils.py의 함수 사용)
        sixd_per_joint = single_motion.reshape(seq_len, njoints, nfeats)
        euler_angles_rad = sixd_to_euler_angles(sixd_per_joint, order='yxz')
        euler_angles_deg = torch.rad2deg(euler_angles_rad)
        
        # 전역 위치(0)와 회전값(오일러) 결합
        positions = torch.zeros(seq_len, 3, device=device)
        rotations_flat = euler_angles_deg.reshape(seq_len, -1)
        motion_data_flat = torch.cat([positions, rotations_flat], dim=1)
        
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

# --- 데이터셋 및 데이터로더 인스턴스 생성 ---
print("Loading preprocessed dataset...")
 
dataset = MotionDataset(processed_data_path=processed_data_path, seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Dataset loaded successfully.")

print("Initializing model...")
model = MotionTransformer(
    njoints=njoints,
    nfeats=nfeats,
    seq_len=seq_len,
    latent_dim=256,
    ff_size=1024,
).to(device)

betas = np.linspace(0.0001, 0.02, num_timesteps)

diffusion = GaussianDiffusion(betas=betas).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("Model initialized successfully.")

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch in dataloader:
        x_start = batch.to(device)

        t = torch.randint(0, num_timesteps, (x_start.shape[0],), device=device)

        optimizer.zero_grad()

        loss_dict = diffusion.training_losses(model, x_start, t)
        loss = loss_dict['loss']

        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()

        current_avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix({'loss': f'{current_avg_loss:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    if (epoch + 1) % save_interval == 0:
        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        generate_and_save_bvh(model, diffusion, dataset, num_samples_to_generate, seq_len, njoints, nfeats, skeleton_template_path, output_bvh_dir, device)

print("Training and Generating completed.")