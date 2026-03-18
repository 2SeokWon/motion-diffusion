# scripts/generate.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from datetime import datetime

from core.config import load_config
from core.model import MotionTransformer
from core.gaussian_diffusion import GaussianDiffusion
from core.dataset import MotionDataset
from core.motion_features import tensor_to_motion_object_root
from core.utils import write_bvh
from bvh_viewer.render_video import render_movie, tensor_to_motion_object


def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          type=str,   default='config.yml')
    parser.add_argument('--checkpoint_path', type=str,   required=True)
    parser.add_argument('--abs_traj_path',   type=str,   default=None,
                        help="Path to abs trajectory .pt (default: data.control_dir/absolute_3d_compatible.pt)")
    parser.add_argument('--guidance_scale',  type=float, default=None,
                        help="CFG scale (overrides config.yml)")
    parser.add_argument('--class_idx',       type=int,   default=None,
                        help="Class label for conditional generation (0 ~ 6). Omit for unconditional.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    guidance_scale = args.guidance_scale if args.guidance_scale is not None else cfg.generation.guidance_scale
    abs_traj_path  = args.abs_traj_path  if args.abs_traj_path  is not None \
                     else os.path.join(cfg.data.control_dir, "absolute_3d_compatible.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join(cfg.generation.output_dir, f"generated_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ─ 모델 / 디퓨전 ─
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

    betas     = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end, cfg.diffusion.num_timesteps)
    diffusion = GaussianDiffusion(betas=betas).to(device)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ─ 통계 로드 ─
    print("Loading dataset statistics...")
    dataset     = MotionDataset(processed_data_path=cfg.data.processed_dir, seq_len=cfg.model.seq_len)
    class_names = dataset.name_classes

    full_mean = np.hstack([dataset.root_pos_mean,
                           dataset.position_mean, dataset.rotation_mean,
                           dataset.foot_mean, dataset.abs_traj_mean])   # [213]
    full_std  = np.hstack([dataset.root_pos_std,
                           dataset.position_std,  dataset.rotation_std,
                           dataset.foot_std,  dataset.abs_traj_std])    # [213]

    # ─ 조건 궤적 준비 ─
    abs_pt    = torch.load(abs_traj_path, map_location='cpu').numpy().astype(np.float32)  # [T, 3]
    cond_norm = torch.from_numpy(
        (abs_pt - dataset.abs_traj_mean) / dataset.abs_traj_std
    ).float().unsqueeze(0).to(device)  # [1, T, 3]

    T = abs_pt.shape[0]

    # ─ 클래스 설정 ─
    if args.class_idx is not None:
        if not (0 <= args.class_idx < len(class_names)):
            raise ValueError(f"Invalid class_idx {args.class_idx} (must be 0-{len(class_names)-1})")
        classes    = torch.tensor([args.class_idx], device=device)
        class_name = class_names[args.class_idx]
        print(f"Generating with class: {class_name} (index {args.class_idx})")
    else:
        classes    = None
        class_name = "unconditional"
        print("Generating unconditionally (no class)")

    # ─ 샘플링 ─
    print(f"Sampling ... (T={T}, guidance_scale={guidance_scale})")
    with torch.no_grad():
        generated_norm = diffusion.p_sample_loop_cond(
            model,
            shape=(1, T, cfg.model.input_feats),
            cond=cond_norm,
            guidance_scale=guidance_scale,
            model_kwargs={'classes_name': classes},
        )  # [1, T, 213]

    # ─ 역정규화 ─
    generated = generated_norm.cpu().numpy()[0] * full_std + full_mean  # [T, 213]

    # ─ 궤적 저장 (velocity 적분) ─
    cond_start   = cfg.model.input_feats - cfg.model.cond_features  # 210
    gen_traj_abs = tensor_to_motion_object_root(generated[:, :cond_start])  # [T, 3]
    torch.save(torch.from_numpy(gen_traj_abs), os.path.join(output_dir, "generated_traj_abs.pt"))

    root, motion_obj = tensor_to_motion_object(generated, cfg.generation.skeleton_template)
    write_bvh(root, motion_obj, os.path.join(output_dir, "sample.bvh"))
    render_movie(root, motion_obj, os.path.join(output_dir, "sample.mp4"))

    print(f"\nGeneration complete. Saved to {output_dir}")


if __name__ == '__main__':
    generate()
