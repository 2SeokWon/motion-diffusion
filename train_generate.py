import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os

from tqdm import tqdm

from model import MotionTransformer
from gaussian_diffusion import GaussianDiffusion
from utils import sixd_to_euler_angles

learning_rate = 1e-4
num_epochs = 500
batch_size = 32
save_interval = 50

data_path = ''
njoints = 23
nfeats = 6
seq_len = 30

num_timesteps = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

save_dir = "./checkpoints"
output_bvh_dir = "./results"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(output_bvh_dir, exist_ok=True)

skeleton_template_path = "./dataset/Aeroplane_BR.bvh"  # 스켈레톤 템플릿 파일 경로
num_samples_to_generate = 5  # 생성할 샘플 수

class MotionDataset(Dataset):
    def __init__(self, npy_path):
        all_data = np.load(npy_path)

        self.mean = np.mean(all_data, axis=(0,1), keepdims=True)
        self.std = np.std(all_data, axis=(0,1), keepdims=True)
        self.std[self.std == 0] = 1e-7  # 0으로 나누는 것을 방지

        normalized_data = (all_data - self.mean) / self.std

        self.data = torch.from_numpy(normalized_data).float()

        self.mean = torch.from_numpy(self.mean).float()
        self.std= torch.from_numpy(self.std).float()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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
        euler_angles_rad = sixd_to_euler_angles(sixd_per_joint, order='zyx')
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
# data_path는 이제 전처리된 .npy 파일을 가리킵니다.
data_path = "./processed_motion_data.npy" 
dataset = MotionDataset(npy_path=data_path)
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
        # 모델의 학습 가능한 파라미터만 저장하는 것이 효율적입니다.
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        generate_and_save_bvh(model, diffusion, dataset, num_samples_to_generate, seq_len, njoints, nfeats, skeleton_template_path, output_bvh_dir, device)

print("Training and Generating completed.")