# scripts/generate.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from datetime import datetime

from core.config import load_config
from core.model import MotionTransformer
from core.gaussian_diffusion import GaussianDiffusion
from core.dataset import MotionDataset
from core.motion_features import tensor_to_motion_object_root
from core.metrics import waypoint_metrics, foot_positions, foot_skating
from core.utils import write_bvh
from bvh_viewer.BVH_Parser import bvh_parser
from core.reconstruct import tensor_to_motion_object


def gt_foot_skating(raw_dir, source_bvh, start_frame, num_frames):
    """같은 구간의 원본 동작에서 잰 발 미끄러짐 (모션캡처 노이즈 수준의 기준값)."""
    root, motion = bvh_parser(os.path.join(raw_dir, source_bvh))
    motion.list_to_quaternion(root)
    motion.save_virtual_root_info(root)
    return foot_skating(foot_positions(motion, start_frame, num_frames))


def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          type=str,   default='config.yml')
    parser.add_argument('--checkpoint_path', type=str,   required=True)
    parser.add_argument('--cond_path',       type=str,   default=None,
                        help="Waypoint condition .pt from make_control.py (default: data.control_dir/waypoint_cond.pt)")
    parser.add_argument('--guidance_scale',  type=float, default=None,
                        help="CFG scale (overrides config.yml)")
    parser.add_argument('--class_idx',       type=int,   default=None,
                        help="Class label for conditional generation (0 ~ 6). Omit for unconditional.")
    parser.add_argument('--no_render',       action='store_true',
                        help="Skip sample.mp4 (needs a display/OpenGL; use on headless GPU servers).")
    args = parser.parse_args()

    cfg = load_config(args.config)

    guidance_scale = args.guidance_scale if args.guidance_scale is not None else cfg.generation.guidance_scale
    cond_path      = args.cond_path if args.cond_path is not None \
                     else os.path.join(cfg.data.control_dir, "waypoint_cond.pt")

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
        cond_dim=cfg.model.cond_dim,
    ).to(device)

    betas     = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end, cfg.diffusion.num_timesteps)
    diffusion = GaussianDiffusion(betas=betas).to(device)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    weights = 'ema_state_dict' if 'ema_state_dict' in ckpt else 'model_state_dict'  #생성에는 EMA 가중치를 우선 사용
    model.load_state_dict(ckpt[weights])
    print(f"Loaded {weights} (step {ckpt.get('step', '?')})")
    model.eval()

    # ─ 통계 로드 ─
    print("Loading dataset statistics...")
    dataset     = MotionDataset(processed_data_path=cfg.data.processed_dir, seq_len=cfg.model.seq_len,
                                feat_bias=cfg.training.feat_bias, split='test',
                                test_ratio=cfg.data.test_ratio, stats_dir=cfg.data.stats_dir)  # 학습과 같은 통계
    class_names = dataset.name_classes

    full_mean = np.hstack([dataset.root_pos_mean,
                           dataset.position_mean, dataset.rotation_mean,
                           dataset.foot_mean])   # [210]
    full_std  = np.hstack([dataset.root_pos_std,
                           dataset.position_std,  dataset.rotation_std,
                           dataset.foot_std])    # [210]

    # ─ 경유점 조건 준비 (dataset.py와 같은 규약: [값 × 마스크 | 마스크]) ─
    wp       = torch.load(cond_path, map_location='cpu')
    gt_traj  = wp['traj'].numpy().astype(np.float32)  # [T, 3] 전체 정답 궤적 (평가용)
    mask     = wp['mask'].numpy().astype(np.float32)  # [T, 1]
    traj_norm = (gt_traj - dataset.abs_traj_mean) / dataset.abs_traj_std
    cond_norm = torch.from_numpy(
        np.concatenate([traj_norm * mask, mask], axis=1)
    ).float().unsqueeze(0).to(device)  # [1, T, 4]

    T = gt_traj.shape[0]
    print(f"Waypoint condition: {cond_path} ({int(mask.sum())} frames given)")

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
    generated = generated_norm.cpu().numpy()[0] * full_std + full_mean  # [T, 210]

    # ─ 생성된 root 속도를 적분해 실제로 간 경로를 구하고 채점 ─
    gen_traj_abs = tensor_to_motion_object_root(generated)  # [T, 3]
    torch.save(torch.from_numpy(gen_traj_abs), os.path.join(output_dir, "generated_traj_abs.pt"))

    root, motion_obj = tensor_to_motion_object(generated, cfg.generation.skeleton_template)

    metrics = waypoint_metrics(gen_traj_abs, gt_traj, mask)
    metrics.update({f'gen_{k}': v for k, v in foot_skating(foot_positions(motion_obj)).items()})
    if 'source_bvh' in wp:
        gt = gt_foot_skating(cfg.data.raw_dir, wp['source_bvh'], wp['start_frame'], T)
        metrics.update({f'gt_{k}': v for k, v in gt.items()})
    metrics.update({'class': class_name, 'guidance_scale': guidance_scale,
                    'cond_path': cond_path, 'checkpoint': args.checkpoint_path})
    with open(os.path.join(output_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Waypoint error {metrics['waypoint_pos_err_cm']:.2f} cm | Goal error {metrics['goal_pos_err_cm']:.2f} cm | "
          f"Waypoint yaw error {metrics['waypoint_yaw_err_deg']:.2f} deg | (ref) full-path ADE {metrics['full_path_ADE_cm']:.2f} cm")
    print(f"Foot skating ratio {metrics['gen_skating_ratio']:.3f}"
          + (f" (GT {metrics['gt_skating_ratio']:.3f})" if 'gt_skating_ratio' in metrics else ""))
    write_bvh(root, motion_obj, os.path.join(output_dir, "sample.bvh"))
    if not args.no_render:
        from bvh_viewer.render_video import render_movie  #pygame/OpenGL은 렌더링할 때만 불러온다
        render_movie(root, motion_obj, os.path.join(output_dir, "sample.mp4"))

    print(f"\nGeneration complete. Saved to {output_dir}")


if __name__ == '__main__':
    generate()
