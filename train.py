#train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import argparse
import wandb

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime
from model import MotionTransformer
from gaussian_diffusion import GaussianDiffusion
from dataset import MotionDataset

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
    lr_anneal_steps = 500000 
    num_epochs = 500
    batch_size = 64 
    num_workers = 16
    save_interval = 5
    mask_prob = 0.1 #unconditional training

    njoints = 23
    position_features = 3
    rotation_features = 6
    root_features = 4 # 루트 Y높이(1) + 수평속도(2) + y축 각속도(1) 
    foot_features = 2
    traj_features = 3
    joint_position_features = (njoints - 1) * position_features #66
    joint_rotation_features = njoints * rotation_features #138

    input_feats = root_features + joint_position_features + joint_rotation_features + foot_features + traj_features  #213
    seq_len = 180
    num_timesteps = 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"./checkpoints/traj/{timestamp}"
    processed_data_path = "./processed_data_position_1125/"
    os.makedirs(save_dir, exist_ok=True)

    # --- WandB 초기화 (훈련 시작 전에) ---
    config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lr_anneal_steps": lr_anneal_steps,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_timesteps": num_timesteps,
        "latent_dim": 1024,
        "ff_size": 4096,
        "num_layers": 8,
        "time_integration_method": "concat",
        "mask_prob": mask_prob,
    }
    wandb.init(project="motion-diffusion", config=config, resume="allow")

    dataset = MotionDataset(processed_data_path=processed_data_path, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("Dataset loaded successfully.")
    
    print("Initializing model...")
    model = MotionTransformer(
        input_feats=input_feats,
        seq_len=seq_len,
    ).to(device)

    betas = torch.linspace(0.0001, 0.02, num_timesteps)

    diffusion = GaussianDiffusion(betas=betas).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_anneal_steps)
    print("Model, Optimizer, and Scheduler initialized successfully.")

    scaler = GradScaler()
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # 다음 에포크부터 시작

        if 'wandb_run_id' in checkpoint:
            wandb.init(project="motion-diffusion", id=checkpoint['wandb_run_id'], resume="must")
        
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")

    # --- 훈련 Loop ---
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_root = 0.0
        total_joint = 0.0
        total_foot = 0.0
        total_traj = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            x_start = batch['motion'].to(device) #[B, seq_len, input_feats]
            labels_name = batch['label_name'].to(device)
            
            cond_vel = x_start[:, :, :4]
            cond_traj = x_start[:, :, 210:213]
            cond = torch.cat([cond_vel, cond_traj], dim=-1)  #[B, seq_len, 7]
            
            B = x_start.size(0)

            batch_size = x_start.shape[0]
            classes_name = torch.argmax(labels_name, dim=1)
            rand_name = torch.rand(batch_size, device=device)
            mask_name = rand_name < mask_prob
            classes_name = classes_name.clone()
            classes_name[mask_name] = -1

            model_kwargs = {
                'classes_name' : classes_name,
            }

            t = torch.randint(0, num_timesteps, (B,), device=device)

            optimizer.zero_grad()
            with autocast():
                loss_dict = diffusion.training_losses_cond(model, x_start, t, cond=cond, cond_drop_prob=mask_prob, model_kwargs=model_kwargs)
                loss = loss_dict['loss']
            
            scaler.scale(loss).backward()  # AMP: scale and backward
            scaler.step(optimizer)  # step
            scaler.update() 
            scheduler.step()

            total_loss += loss.item()
            total_root += loss_dict.get('loss_root', torch.tensor(0.0)).item()
            total_joint += loss_dict.get('loss_joint', torch.tensor(0.0)).item()
            total_foot += loss_dict.get('loss_foot', torch.tensor(0.0)).item()
            total_traj += loss_dict.get('loss_traj', torch.tensor(0.0)).item()

            progress_bar.set_postfix({
                'loss': f'{total_loss / (progress_bar.n + 1):.4f}',
                'root': f'{total_root / (progress_bar.n + 1):.4f}',
                'joint': f'{total_joint / (progress_bar.n + 1):.4f}',
                'foot': f'{total_foot / (progress_bar.n + 1):.4f}',
                'traj': f'{total_traj / (progress_bar.n + 1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
            })

        avg_loss = total_loss / len(dataloader)
        avg_root = total_root / len(dataloader)
        avg_joint = total_joint / len(dataloader)
        avg_foot = total_foot / len(dataloader)
        avg_traj = total_traj / len(dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Root: {avg_root:.4f}, Joint: {avg_joint:.4f}, Foot: {avg_foot:.4f}, Traj: {avg_traj:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_root": avg_root,
            "avg_joint": avg_joint,
            "avg_foot": avg_foot,
            "avg_traj": avg_traj,
            "learning_rate": scheduler.get_last_lr()[0],
        })

        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")
            wandb.save(save_path)
    
    print("Training completed.")
    wandb.finish()
    
if __name__ == '__main__':
    train()
