import os
import argparse
import numpy as np
import torch
import math
from datetime import datetime

from model import MotionTransformer
from gaussian_diffusion import GaussianDiffusion
from dataset import MotionDataset
from bvh_viewer.render_video import tensor_to_motion_object, render_movie, tensor_to_motion_object_traj
from new_preprocess import tensor_to_motion_object_root, moving_average_path, compute_delta_traj
from utils import write_bvh

def check_root_vs_cond(generated_motion, pos_vel_mean, pos_vel_std, cond_norm):
    gen_root = generated_motion[:, 1:4]  # [vx,vz,yaw_rate]
    cond_den = cond_norm.cpu().numpy()[0] * pos_vel_std[1:4] + pos_vel_mean[1:4]
    mse = float(np.mean((gen_root - cond_den) ** 2))
    print(f"[Root-vs-Cond MSE] {mse:.6f}")

def reconstruct_traj_from_interp_delta(interp_pos_xz, interp_yaw, delta_xz, delta_yaw):
    """
    보간된 궤적 + 잔차(delta_x/z/yaw)를 합쳐 월드 좌표의 (x, z, yaw)로 복원합니다.
    delta_xz는 interp_yaw 좌표계의 로컬 값이어야 합니다.
    """
    c = np.cos(interp_yaw)
    s = np.sin(interp_yaw)
    world_x = interp_pos_xz[:, 0] + delta_xz[:, 0] * c - delta_xz[:, 1] * s
    world_z = interp_pos_xz[:, 1] + delta_xz[:, 0] * s + delta_xz[:, 1] * c
    world_yaw = np.unwrap(interp_yaw + delta_yaw)
    return np.stack([world_x, world_z, world_yaw], axis=1)

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    parser.add_argument('--abs_traj_path', type=str, default='./traj_from_template/absolute_3d_compatible.pt')
    parser.add_argument('--processed_data_path', type=str, default='./processed_data_interp30_1226')
    parser.add_argument('--skeleton_template_path', type=str, default='./dataset/Aeroplane_BR.bvh')
    parser.add_argument('--class_idx', type=int, default=None,
                        help="Class label for conditional generation (optional, 0 ~ 6).")
    args = parser.parse_args()

    # ─ 기본 설정 ─
    njoints = 23
    position_features = 3
    rotation_features = 6
    root_features = 4
    foot_features = 2
    traj_features = 3
    input_feats = root_features + ((njoints - 1) * position_features) + (njoints * rotation_features) + foot_features + traj_features
    num_timesteps = 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./results/generated_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ─ 모델/디퓨전 ─
    print("Initializing model...")
    model = MotionTransformer(njoints=njoints, input_feats=input_feats, dropout=0.1).to(device)
    betas = torch.linspace(0.0001, 0.02, num_timesteps)
    diffusion = GaussianDiffusion(betas=betas).to(device)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ─ 통계 ─
    print("Loading dataset statistics (mean/std)...")
    dataset = MotionDataset(processed_data_path=args.processed_data_path)
    class_names = dataset.name_classes

    root_mean, root_std = dataset.root_pos_mean, dataset.root_pos_std #4
    position_mean, position_std = dataset.position_mean, dataset.position_std #66
    rotation_mean, rotation_std = dataset.rotation_mean, dataset.rotation_std #138
    foot_mean, foot_std = dataset.foot_mean, dataset.foot_std #2
    interp_mean, interp_std = dataset.interp_mean, dataset.interp_std #3
    delta_mean, delta_std = dataset.delta_mean, dataset.delta_std #3
    
    full_mean = np.hstack([root_mean[0], delta_mean, position_mean, rotation_mean, foot_mean, interp_mean])  # 213
    full_std  = np.hstack([root_std[0],  delta_std,  position_std,  rotation_std,  foot_std, interp_std])   # 213

    abs_pt = torch.load(args.abs_traj_path, map_location='cpu').numpy().astype(np.float32)  # [T,3]
    interp_cond = moving_average_path(abs_pt[:, :2], abs_pt[:, 2], radius=30)  # [T,3]
    cond_traj_norm = (interp_cond - interp_mean) / interp_std
    cond_norm = torch.from_numpy(cond_traj_norm).float().unsqueeze(0).to(device)  # [1, T, 3]
    
    T = abs_pt.shape[0]
    if args.class_idx is not None:
        if 0 <= args.class_idx < len(class_names):
            classes = torch.tensor([args.class_idx], device=device)  # [1] tensor (배치 1)
            class_name = class_names[args.class_idx]  # 파일 이름용
            print(f"Generating with class: {class_name} (index {args.class_idx})")
        else:
            raise ValueError(f"Invalid class_idx: {args.class_idx} (must be 0-{len(class_names)-1})")
    else:
        classes = None  # unconditional
        class_name = "unconditional"
        print("Generating unconditionally (no class)")
        
    
    model_kwargs = {'classes_name': classes}  # model_kwargs 구성

    sample_shape = (1, T, input_feats)

    print(f"Sampling ... (T={T})")
    with torch.no_grad():
        generated_motion_norm = diffusion.p_sample_loop_cond(
            model,
            sample_shape,
            cond=cond_norm,
            model_kwargs=model_kwargs
        )  # [1,T,213]

    # ─ 역정규화 & 루트 신호 투영 ─

    generated_motion = generated_motion_norm.cpu().numpy()[0] * full_std + full_mean  # [T,213] generated_motion_norm 에는 local x,z position, yaw
    
    gen_traj_abs =  reconstruct_traj_from_interp_delta(generated_motion[:, 210:212], generated_motion[:, 212],
                                                       generated_motion[:, 1:3], generated_motion[:, 3])  # [T,3]
    
    generated_motion[: , 1:4] = gen_traj_abs  # generated_motion 의 root pos_xz, yaw 채널을 abs traj로 교체
    torch.save(torch.from_numpy(gen_traj_abs), os.path.join(output_dir, "generated_traj_abs.pt"))
    
    motion_feature = generated_motion  #[T, 213]

    root, motion_obj = tensor_to_motion_object_traj(motion_feature, args.skeleton_template_path) #여기도 Local Position으로 복구
    out_bvh = os.path.join(output_dir, "sample.bvh")
    write_bvh(root, motion_obj, out_bvh)

    out_mp4 = os.path.join(output_dir, "sample.mp4")
    render_movie(root, motion_obj, out_mp4)
    print(f"\nGeneration complete. Saved to {output_dir}")

if __name__ == '__main__':
    generate()
