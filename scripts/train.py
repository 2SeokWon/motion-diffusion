# scripts/train.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import math
import shutil
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse
import wandb
import yaml

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


def cycle(loader):
    #겹치는 창 때문에 에폭은 의미가 약하므로, 스텝 기준으로 끝없이 배치를 뽑는다
    while True:
        for batch in loader:
            yield batch


def warmup_cosine(step, warmup_steps, total_steps, lr, min_lr):
    #학습률 배율: 워밍업 동안 0 → 1 선형 증가, 이후 total_steps까지 코사인으로 한 번만 min_lr까지 감소
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
    return (min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))) / lr


@torch.no_grad()
def update_ema(ema_model, model, decay, step):
    #학습 초반에는 EMA가 초기 가중치에 묶이지 않도록 decay를 서서히 올린다
    d = min(decay, (1 + step) / (10 + step))
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.lerp_(p, 1.0 - d)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def build_eval_set(cfg, n):
    """
    평가 손실용 고정 평가 창. 창 선택, 경유점 마스크, t, 노이즈를 모두 시드로 고정해
    같은 모델이면 항상 같은 값이 나오게 한다 (모델이 바뀔 때만 손실이 변한다).
    """
    test_set = MotionDataset(
        processed_data_path=cfg.data.processed_dir, seq_len=cfg.model.seq_len,
        feat_bias=cfg.training.feat_bias, max_waypoints=cfg.waypoint.max_waypoints,
        dense_prob=cfg.waypoint.dense_prob, split='test',
        test_ratio=cfg.data.test_ratio, stats_dir=cfg.data.stats_dir,
    )
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(test_set), generator=g)[:n].tolist()
    with torch.random.fork_rng(devices=[]):  #마스크 샘플링은 전역 torch 난수를 쓰므로 학습 난수와 분리
        torch.manual_seed(0)
        items = [test_set[i] for i in idx]
    motion = torch.stack([it['motion'] for it in items])
    return {
        'motion': motion,
        'cond': torch.stack([it['cond'] for it in items]),
        'classes_name': torch.stack([torch.argmax(it['label_name']) for it in items]),
        't': torch.randint(0, cfg.diffusion.num_timesteps, (len(idx),), generator=g),
        'noise': torch.randn(motion.shape, generator=g),
    }


@torch.no_grad()
def evaluate(model, diffusion, eval_set, device, use_amp, batch_size=256):
    model.eval()
    totals = {'loss': 0.0, 'loss_root': 0.0, 'loss_joint': 0.0, 'loss_foot': 0.0}
    n = eval_set['motion'].size(0)
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        with torch.autocast(device_type=device, enabled=use_amp):
            out = diffusion.training_losses_cond(
                model, eval_set['motion'][sl].to(device), eval_set['t'][sl].to(device),
                cond=eval_set['cond'][sl].to(device),
                model_kwargs={'classes_name': eval_set['classes_name'][sl].to(device)},
                noise=eval_set['noise'][sl].to(device),
            )
        bs = eval_set['motion'][sl].size(0)
        for k in totals:
            totals[k] += out[k].item() * bs
    model.train()
    return {k: v / n for k, v in totals.items()}


def train():
    parser = argparse.ArgumentParser(description="Train a Motion Diffusion Model.")
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to last.pt to resume training from (continues in the same directory).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(args.config, 'r', encoding='utf-8') as f:
        config_text = f.read()
    tc = cfg.training
    set_seed(tc.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"Using device: {device}")

    resume_ckpt = torch.load(args.resume, map_location=device) if args.resume else None
    if resume_ckpt is not None:
        save_dir = os.path.dirname(os.path.abspath(args.resume))
    else:
        save_dir = os.path.join(tc.checkpoint_dir, datetime.now().strftime("%Y%m%d_%H%M"))
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(save_dir, "config.yml"))  #이 체크포인트를 만든 설정을 함께 보관

    wandb_id = resume_ckpt.get('wandb_run_id') if resume_ckpt else None
    wandb.init(project="motion-diffusion", config=yaml.safe_load(config_text),
               id=wandb_id, resume="must" if wandb_id else "allow")

    dataset = MotionDataset(
        processed_data_path=cfg.data.processed_dir,
        seq_len=cfg.model.seq_len,
        feat_bias=tc.feat_bias,
        max_waypoints=cfg.waypoint.max_waypoints,
        dense_prob=cfg.waypoint.dense_prob,
        split='train',
        test_ratio=cfg.data.test_ratio,
        stats_dir=cfg.data.stats_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=tc.num_workers,
        drop_last=True,
        persistent_workers=tc.num_workers > 0,
    )
    eval_set = build_eval_set(cfg, tc.eval_windows)
    print(f"Dataset loaded. Eval windows fixed: {eval_set['motion'].size(0)}")

    print("Initializing model...")
    model = MotionTransformer(
        input_feats=cfg.model.input_feats,
        seq_len=cfg.model.seq_len,
        latent_dim=cfg.model.latent_dim,
        ff_size=cfg.model.ff_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        cond_dim=cfg.model.cond_dim,
    ).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    betas = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end, cfg.diffusion.num_timesteps)
    diffusion = GaussianDiffusion(betas=betas).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: warmup_cosine(s, tc.warmup_steps, tc.total_steps, tc.learning_rate, tc.min_lr))
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    step, best_val = 0, float('inf')
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model_state_dict'])
        ema_model.load_state_dict(resume_ckpt['ema_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        scaler.load_state_dict(resume_ckpt['scaler_state_dict'])
        step, best_val = resume_ckpt['step'], resume_ckpt['best_val_loss']
        print(f"Resumed from step {step} (best eval loss {best_val:.4f})")
        del resume_ckpt

    def save_checkpoint(path):
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val,
            'config': config_text,
            'wandb_run_id': wandb.run.id if wandb.run else None,
        }, path)

    # CFG 드롭 구간: [0, a) 클래스만 / [a, b) 경유점만 / [b, c) 둘 다 / 나머지는 드롭 없음
    drop_a = tc.class_drop
    drop_b = drop_a + tc.cond_drop
    drop_c = drop_b + tc.joint_drop

    print("Starting training...")
    model.train()
    batches = cycle(dataloader)
    running = {'loss': 0.0, 'loss_root': 0.0, 'loss_joint': 0.0, 'loss_foot': 0.0}
    # nohup으로 파일에 기록할 때는 진행 막대가 한 줄에 이어 붙으므로 끄고, log_interval마다 한 줄씩 남긴다
    is_tty = sys.stdout.isatty()
    progress_bar = tqdm(total=tc.total_steps, initial=step, desc="Training", disable=not is_tty)
    t_last, step_last = time.time(), step

    while step < tc.total_steps:
        batch = next(batches)
        x_start = batch['motion'].to(device)
        cond = batch['cond'].to(device)  # [B, T, 4]
        B = x_start.size(0)
        classes_name = torch.argmax(batch['label_name'].to(device), dim=1).clone()

        # CFG 드롭: 클래스만 / 경유점만 / 둘 다 (둘 다 드롭 = 샘플링의 uncond 분기와 같은 입력)
        u = torch.rand(B, device=device)
        drop_class = (u < drop_a) | ((u >= drop_b) & (u < drop_c))
        drop_cond  = (u >= drop_a) & (u < drop_c)
        classes_name[drop_class] = -1
        cond = cond.masked_fill(drop_cond.view(B, 1, 1), 0.0)  # 마스크까지 0 → "경유점 없음"

        t = torch.randint(0, cfg.diffusion.num_timesteps, (B,), device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=use_amp):
            loss_dict = diffusion.training_losses_cond(
                model, x_start, t,
                cond=cond,
                model_kwargs={'classes_name': classes_name},
            )
        scaler.scale(loss_dict['loss']).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema_model, model, tc.ema_decay, step)
        step += 1
        progress_bar.update(1)

        for k in running:
            running[k] += loss_dict[k].item()

        if step % tc.log_interval == 0:
            avg = {k: v / tc.log_interval for k, v in running.items()}
            progress_bar.set_postfix({'loss': f"{avg['loss']:.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
            wandb.log({'step': step, 'learning_rate': scheduler.get_last_lr()[0],
                       **{f'train/{k}': v for k, v in avg.items()}}, step=step)
            running = {k: 0.0 for k in running}
            if not is_tty:
                rate = (step - step_last) / max(time.time() - t_last, 1e-9)
                eta_h = (tc.total_steps - step) / max(rate, 1e-9) / 3600
                print(f"[step {step}/{tc.total_steps}] loss {avg['loss']:.4f} (root {avg['loss_root']:.4f}) "
                      f"lr {scheduler.get_last_lr()[0]:.2e} | {rate:.1f} it/s | ETA {eta_h:.1f}h", flush=True)
                t_last, step_last = time.time(), step

        if step % tc.eval_interval == 0 or step == tc.total_steps:
            ev = evaluate(ema_model, diffusion, eval_set, device, use_amp)  #생성에 쓰는 EMA 가중치로 평가
            wandb.log({f'eval/{k}': v for k, v in ev.items()}, step=step)
            tqdm.write(f"[step {step}] eval loss {ev['loss']:.4f} (root {ev['loss_root']:.4f}, "
                       f"joint {ev['loss_joint']:.4f}, foot {ev['loss_foot']:.4f})")
            if ev['loss'] < best_val:
                best_val = ev['loss']
                save_checkpoint(os.path.join(save_dir, "best.pt"))
                tqdm.write(f"  new best -> {os.path.join(save_dir, 'best.pt')}")

        if step % tc.save_interval == 0 or step == tc.total_steps:
            save_checkpoint(os.path.join(save_dir, "last.pt"))
        if step % tc.snapshot_interval == 0:
            save_checkpoint(os.path.join(save_dir, f"step_{step:06d}.pt"))

    progress_bar.close()
    print(f"Training completed. Best eval loss {best_val:.4f}. Checkpoints in {save_dir}")
    wandb.finish()


if __name__ == '__main__':
    train()
