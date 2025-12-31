# -*- coding: utf-8 -*-
import argparse
import os
import math
import json
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime

def wrap_angle(a):
    """Wrap angle to [-π, π]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_traj(path):
    """Load trajectory from .pt file: [T, 3] = [x, z, yaw]"""
    arr = torch.load(path, map_location="cpu")
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    arr = np.asarray(arr, dtype=np.float32)

    # 허용 포맷:
    # - [T,3]: x,z,yaw 그대로 사용
    # - [T,4]: x,y,z,yaw 형태라면 x,z,yaw만 추출
    # - [T,>4]: 첫 컬럼 x, z는 두 번째/세 번째, yaw는 마지막 컬럼으로 가정
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Trajectory tensor must be [T,3+] shape, got {arr.shape}")
    if arr.shape[1] == 3:
        return arr
    if arr.shape[1] == 4:
        return arr[:, [0, 2, 3]]
    # arr.shape[1] > 4
    return arr[:, [0, 2, -1]]


def compute_metrics(condition_traj, generated_traj):
    """
    Compute ADE, FDE, Heading MAE
    
    Args:
        condition_traj: [T, 3] - target trajectory
        generated_traj: [T, 3] - generated trajectory
    
    Returns:
        dict with metrics
    """
    T = min(len(condition_traj), len(generated_traj))
    cond = condition_traj[:T]
    gen = generated_traj[:T]
    
    # Position error
    pos_error = np.linalg.norm(gen[:, :2] - cond[:, :2], axis=1)  # [T]
    ade = float(pos_error.mean())
    fde = float(pos_error[-1])
    
    # Heading error
    yaw_error = wrap_angle(gen[:, 2] - cond[:, 2])
    heading_mae_rad = float(np.mean(np.abs(yaw_error)))
    heading_mae_deg = math.degrees(heading_mae_rad)
    
    return {
        "T": int(T),
        "ADE": ade,
        "FDE": fde,
        "Heading_MAE_rad": heading_mae_rad,
        "Heading_MAE_deg": heading_mae_deg,
        "pos_error": pos_error,
        "yaw_error": yaw_error
    }


def draw_heading_arrows(ax, traj, step=20, scale=2.0, **kwargs):
    """Draw heading arrows on trajectory"""
    for i in range(0, len(traj), step):
        x, z, yaw = traj[i]
        dx = math.cos(yaw) * scale
        dz = math.sin(yaw) * scale
        ax.arrow(x, z, dx, dz, 
                head_width=scale*0.3, 
                head_length=scale*0.4,
                length_includes_head=True, 
                **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Compare condition and generated trajectories")
    parser.add_argument("--condition", default="./traj_from_template/absolute_3d_compatible.pt",
                       help="Condition trajectory .pt file (absolute)")
    parser.add_argument("--generated", required=True, 
                       help="Generated trajectory .pt file (absolute)")
    parser.add_argument("--out_dir", default="./traj_comparison",
                       help="Output directory")
    parser.add_argument("--arrow_step", type=int, default=20,
                       help="Frame step for heading arrows")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./traj_comparison/generated_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    # Load trajectories
    print(f"Loading condition trajectory from: {args.condition}")
    cond_traj = load_traj(args.condition)
    
    print(f"Loading generated trajectory from: {args.generated}")
    gen_traj = load_traj(args.generated)
    
    print(f"Condition trajectory: {cond_traj.shape}")
    print(f"Generated trajectory: {gen_traj.shape}")

    # Compute metrics
    metrics = compute_metrics(cond_traj, gen_traj)
    
    print(f"\n=== Trajectory Metrics ===")
    print(f"Frames: {metrics['T']}")
    print(f"ADE: {metrics['ADE']:.3f} m")
    print(f"FDE: {metrics['FDE']:.3f} m")
    print(f"Heading MAE: {metrics['Heading_MAE_deg']:.2f}°")

    # ===== Plot 1: XZ Trajectory =====
    fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=args.dpi)
    
    # Plot trajectories
    ax1.plot(cond_traj[:, 0], cond_traj[:, 1], 
            'b--', linewidth=2.5, label='Condition', alpha=0.8)
    ax1.plot(gen_traj[:, 0], gen_traj[:, 1], 
            'r-', linewidth=2.5, label='Generated', alpha=0.8)
    
    # Start/End markers
    ax1.scatter(cond_traj[0, 0], cond_traj[0, 1], 
               s=150, c='green', marker='o', label='Start', zorder=10, edgecolors='black')
    ax1.scatter(cond_traj[-1, 0], cond_traj[-1, 1], 
               s=150, c='blue', marker='s', label='End (Cond)', zorder=10, edgecolors='black')
    ax1.scatter(gen_traj[-1, 0], gen_traj[-1, 1], 
               s=150, c='red', marker='x', linewidths=3, label='End (Gen)', zorder=10)
    
    # Styling
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Z (m)', fontsize=12)
    ax1.set_title(f'Trajectory Comparison\n'
                  f'ADE: {metrics["ADE"]:.3f}m | FDE: {metrics["FDE"]:.3f}m | '
                  f'Heading MAE: {metrics["Heading_MAE_deg"]:.2f}°',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='best')
    ax1.set_aspect('equal', adjustable='box')
    
    fig1.tight_layout()
    out_traj_png = os.path.join(output_dir, "trajectory_comparison.png")
    fig1.savefig(out_traj_png)
    plt.close(fig1)
    print(f"Saved: {out_traj_png}")

    # ===== Plot 2: Position Error over Time =====
    fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=args.dpi)
    
    frames = np.arange(metrics['T'])
    ax2.plot(frames, metrics['pos_error'], 'r-', linewidth=2)
    ax2.axhline(metrics['ADE'], color='b', linestyle='--', 
               label=f'ADE: {metrics["ADE"]:.3f}m', linewidth=1.5)
    
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Position Error (m)', fontsize=12)
    ax2.set_title('Position Error over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    fig2.tight_layout()
    out_pos_error = os.path.join(output_dir, "position_error.png")
    fig2.savefig(out_pos_error)
    plt.close(fig2)
    print(f"Saved: {out_pos_error}")

    # ===== Plot 3: Heading Error over Time =====
    fig3, ax3 = plt.subplots(figsize=(10, 4), dpi=args.dpi)
    
    yaw_error_deg = np.degrees(metrics['yaw_error'])
    ax3.plot(frames, yaw_error_deg, 'g-', linewidth=2)
    ax3.axhline(metrics['Heading_MAE_deg'], color='b', linestyle='--',
               label=f'MAE: {metrics["Heading_MAE_deg"]:.2f}°', linewidth=1.5)
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax3.set_xlabel('Frame', fontsize=12)
    ax3.set_ylabel('Heading Error (°)', fontsize=12)
    ax3.set_title('Heading Error over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    fig3.tight_layout()
    out_yaw_error = os.path.join(output_dir, "heading_error.png")
    fig3.savefig(out_yaw_error)
    plt.close(fig3)
    print(f"Saved: {out_yaw_error}")

    # ===== Save Metrics =====
    metrics_out = {
        "T": metrics["T"],
        "ADE": metrics["ADE"],
        "FDE": metrics["FDE"],
        "Heading_MAE_rad": metrics["Heading_MAE_rad"],
        "Heading_MAE_deg": metrics["Heading_MAE_deg"]
    }
    
    out_json = os.path.join(output_dir, "metrics.json")
    with open(out_json, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved: {out_json}")

    print(f"\n✅ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
