#train.py
import model
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import random
import argparse

from tqdm import tqdm
from datetime import datetime
from model import MotionTransformer
from gaussian_diffusion import GaussianDiffusion
from dataset import MotionDataset
from kinematics import Skeleton

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train():
    set_seed(42)

    parser = argparse.ArgumentParser(description="Train a Motion Diffusion Model.")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to the checkpoint file to resume training from.")
    args = parser.parse_args()

    learning_rate = 1e-4
    weight_decay = 0.05
    lr_anneal_steps = 200000
    num_epochs = 500
    batch_size = 64
    num_workers = 16
    save_interval = 10

    njoints = 23
    position_features = 3
    rotation_features = 6
    root_motion_features = 4 # 루트 Y높이(1) + 수평속도(2) + y축 각속도(1) 

    joint_position_features = (njoints - 1) * position_features #66
    joint_rotation_features = njoints * rotation_features #138
    
    input_feats = root_motion_features + joint_position_features + joint_rotation_features #208

    seq_len = 180
    num_timesteps = 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"./checkpoints/{timestamp}"
    processed_data_path = "./processed_data"
    os.makedirs(save_dir, exist_ok=True)

    dataset = MotionDataset(processed_data_path=processed_data_path, seq_len=seq_len)
    sampler = WeightedRandomSampler(weights=dataset.sampler_weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    print("Dataset loaded successfully.")
    
    print("Initializing model...")
    model = MotionTransformer(
        njoints=njoints,
        input_feats=input_feats,
        seq_len=seq_len,
        latent_dim=512,
        ff_size=3072,
        time_integration_method='concat'
    ).to(device)

    betas = torch.linspace(0.0001, 0.02, num_timesteps)

    diffusion = GaussianDiffusion(betas=betas).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_anneal_steps)
    print("Model, Optimizer, and Scheduler initialized successfully.")

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # 다음 에포크부터 시작
        
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")

    # --- 훈련 Loop ---
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        #total_vel_loss = 0.0
        #total_fk_loss = 0.0
        #total_simple_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            x_start = batch.to(device)

            t = torch.randint(0, num_timesteps, (x_start.shape[0],), device=device)

            optimizer.zero_grad()

            loss_dict = diffusion.training_losses(model, x_start, t, noise=None)
            loss = loss_dict['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            #total_vel_loss += loss_dict['loss_vel']
            #total_fk_loss += loss_dict['loss_fk']
            #total_simple_loss += loss_dict['loss_simple']

            progress_bar.set_postfix({
                'loss': f'{total_loss / (progress_bar.n + 1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
            })
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f} | ")
        
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")
    
    print("Training completed.")

if __name__ == '__main__':
    train()
