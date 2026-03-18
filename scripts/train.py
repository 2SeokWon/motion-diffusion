# scripts/train.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse
import wandb

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime
from core.config import load_config
from core.model import MotionTransformer
from core.gaussian_diffusion import GaussianDiffusion
from core.dataset import MotionDataset


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
    parser = argparse.ArgumentParser(description="Train a Motion Diffusion Model.")
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to the checkpoint file to resume training from.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(cfg.training.checkpoint_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    wandb.init(project="motion-diffusion", config={
        "learning_rate":   cfg.training.learning_rate,
        "weight_decay":    cfg.training.weight_decay,
        "lr_anneal_steps": cfg.training.lr_anneal_steps,
        "num_epochs":      cfg.training.num_epochs,
        "batch_size":      cfg.training.batch_size,
        "seq_len":         cfg.model.seq_len,
        "num_timesteps":   cfg.diffusion.num_timesteps,
        "latent_dim":      cfg.model.latent_dim,
        "ff_size":         cfg.model.ff_size,
        "num_layers":      cfg.model.num_layers,
        "mask_prob":       cfg.training.mask_prob,
    }, resume="allow")

    dataset = MotionDataset(
        processed_data_path=cfg.data.processed_dir,
        seq_len=cfg.model.seq_len,
        feat_bias=cfg.training.feat_bias,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    print("Dataset loaded successfully.")

    print("Initializing model...")
    model = MotionTransformer(
        input_feats=cfg.model.input_feats,
        seq_len=cfg.model.seq_len,
        latent_dim=cfg.model.latent_dim,
        ff_size=cfg.model.ff_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
    ).to(device)

    betas = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end, cfg.diffusion.num_timesteps)
    diffusion = GaussianDiffusion(betas=betas).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.lr_anneal_steps)
    print("Model, Optimizer, and Scheduler initialized successfully.")

    scaler = GradScaler()
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'wandb_run_id' in checkpoint:
            wandb.init(project="motion-diffusion", id=checkpoint['wandb_run_id'], resume="must")
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")

    cond_start = cfg.model.input_feats - cfg.model.cond_features  # 210

    print("Starting training...")
    for epoch in range(start_epoch, cfg.training.num_epochs):
        model.train()
        total_loss = total_root = total_joint = total_foot = total_cond = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}", leave=False)

        for batch in progress_bar:
            x_start = batch['motion'].to(device)
            labels_name = batch['label_name'].to(device)
            cond = x_start[:, :, cond_start:]

            B = x_start.size(0)
            classes_name = torch.argmax(labels_name, dim=1)
            mask_name = torch.rand(B, device=device) < cfg.training.mask_prob
            classes_name = classes_name.clone()
            classes_name[mask_name] = -1

            t = torch.randint(0, cfg.diffusion.num_timesteps, (B,), device=device)

            optimizer.zero_grad()
            with autocast():
                loss_dict = diffusion.training_losses_cond(
                    model, x_start, t,
                    cond=cond,
                    cond_drop_prob=cfg.training.mask_prob,
                    model_kwargs={'classes_name': classes_name},
                )
                loss = loss_dict['loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss  += loss.item()
            total_root  += loss_dict.get('loss_root',  torch.tensor(0.0)).item()
            total_joint += loss_dict.get('loss_joint', torch.tensor(0.0)).item()
            total_foot  += loss_dict.get('loss_foot',  torch.tensor(0.0)).item()
            total_cond  += loss_dict.get('loss_cond',  torch.tensor(0.0)).item()

            n = progress_bar.n + 1
            progress_bar.set_postfix({
                'loss':  f'{total_loss  / n:.4f}',
                'root':  f'{total_root  / n:.4f}',
                'joint': f'{total_joint / n:.4f}',
                'foot':  f'{total_foot  / n:.4f}',
                'cond':  f'{total_cond  / n:.4f}',
                'lr':    f'{scheduler.get_last_lr()[0]:.6f}',
            })

        N = len(dataloader)
        avg = {k: v / N for k, v in {
            'loss': total_loss, 'root': total_root,
            'joint': total_joint, 'foot': total_foot, 'cond': total_cond,
        }.items()}

        print(f"Epoch [{epoch+1}/{cfg.training.num_epochs}] "
              f"Loss: {avg['loss']:.4f}  Root: {avg['root']:.4f}  "
              f"Joint: {avg['joint']:.4f}  Foot: {avg['foot']:.4f}  Cond: {avg['cond']:.4f}")
        wandb.log({"epoch": epoch + 1, "learning_rate": scheduler.get_last_lr()[0], **{f"avg_{k}": v for k, v in avg.items()}})

        if (epoch + 1) % cfg.training.save_interval == 0:
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
