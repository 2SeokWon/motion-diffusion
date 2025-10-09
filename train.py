#train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import argparse
import wandb

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
    lr_anneal_steps = 500000 # 나중에 길게 학습할 때는 동적으로 변경하는게 좋을 듯
    num_epochs = 500
    batch_size = 64
    num_workers = 16
    save_interval = 5
    mask_prob = 0.1 #unconditional training

    njoints = 23
    position_features = 3
    rotation_features = 6
    root_motion_features = 4 # 루트 Y높이(1) + 수평속도(2) + y축 각속도(1) 
    #foot_features = 4

    joint_position_features = (njoints - 1) * position_features #66
    joint_rotation_features = njoints * rotation_features #138

    input_feats = root_motion_features + joint_position_features + joint_rotation_features #+ foot_features #212

    seq_len = 180
    num_timesteps = 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"./checkpoints/{timestamp}"
    processed_data_path = "./processed_data"
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
        "latent_dim": 512,
        "ff_size": 3072,
        "time_integration_method": "concat",
        "mask_prob": mask_prob,
    }
    wandb.init(project="motion-diffusion", config=config, resume="allow")

    dataset = MotionDataset(processed_data_path=processed_data_path, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("Dataset loaded successfully.")
    
    print("Initializing model...")
    model = MotionTransformer(
        njoints=njoints,
        input_feats=input_feats,
        seq_len=seq_len,
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

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            x_start = batch['motion'].to(device)
            labels = batch['label'].to(device)

            classes = torch.argmax(labels, dim=1)
            if random.random() < mask_prob: 
                classes = None

            model_kwargs = {
                'classes' : classes,
            }

            t = torch.randint(0, num_timesteps, (x_start.shape[0],), device=device)

            optimizer.zero_grad()

            loss_dict = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs, noise=None)
            loss = loss_dict['loss']

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_root += loss_dict.get('loss_root', torch.tensor(0.0)).item()
            total_joint += loss_dict.get('loss_joint', torch.tensor(0.0)).item()

            progress_bar.set_postfix({
                'loss': f'{total_loss / (progress_bar.n + 1):.4f}',
                'root': f'{total_root / (progress_bar.n + 1):.4f}',
                'joint': f'{total_joint / (progress_bar.n + 1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_root = total_root / len(dataloader)
        avg_joint = total_joint / len(dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Root: {avg_root:.4f}, Joint: {avg_joint:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_root": avg_root,
            "avg_joint": avg_joint,
            "learning_rate": scheduler.get_last_lr()[0]
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
