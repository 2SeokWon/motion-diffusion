import os
import argparse
import numpy as np
import torch
from datetime import datetime

from model import MotionTransformer
from gaussian_diffusion import GaussianDiffusion
from dataset import MotionDataset
from bvh_viewer.render_video import tensor_to_motion_object, render_movie
from utils import write_bvh

def vel_tensor_to_abs_traj(gen_tensor, start_x, start_z, start_yaw):
    """
    gen_tensor: [T, F] 역정규화된 생성 특징 (0:root_y, 1:vx_local, 2:vz_local, 3:dyaw, ...)
    return: [T,3] = [x_world, z_world, yaw_world]
    """
    T = gen_tensor.shape[0]
    x, z, yaw = float(start_x), float(start_z), float(start_yaw)
    traj = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        vx_l = float(gen_tensor[t, 1])
        vz_l = float(gen_tensor[t, 2])
        dyaw = float(gen_tensor[t, 3])   # 라디안 가정

        yaw += dyaw
        c, s = np.cos(yaw), np.sin(yaw)
        # local (vx, vz) → world 증분
        dx = c * vx_l + s * vz_l
        dz = -s * vx_l + c * vz_l

        x += dx
        z += dz
        traj[t] = (x, z, yaw)
    return traj

def cond_from_relative_pt(rel_pt, abs_pt, pos_vel_mean, pos_vel_std):
    """
    rel_pt: [T,3] = [Δx_global, Δz_global, Δyaw]  (Drawer가 저장한 relative .pt)
    pos_vel_mean/std: 전처리에서 저장한 root 4D 통계 (mean[:4], std[:4])
                      여기서 cond 정규화엔 [1:4] (vx,vz,yaw_rate)만 사용
    return: cond_norm [1,T,3] = normed [vx_local, vz_local, yaw_rate]
    """
    rel = rel_pt.astype(np.float32)

    yaw = np.cumsum(rel[:, 2]) + abs_pt[:, 2][0]
    c, s = np.cos(yaw), np.sin(yaw)

    dx, dz, dyaw = rel[:, 0], rel[:, 1], rel[:, 2]
    vx_local =  c*dx - s*dz
    vz_local =  s*dx + c*dz

    cond_local = np.stack([vx_local, vz_local, dyaw], axis=1)  

    mean_c = pos_vel_mean[1:4]  
    std_c  = np.maximum(pos_vel_std[1:4], 1e-8)
    cond_norm = (cond_local - mean_c) / std_c 
    
    return torch.from_numpy(cond_norm).float().unsqueeze(0)  

def check_root_vs_cond(generated_motion, pos_vel_mean, pos_vel_std, cond_norm):
    gen_root = generated_motion[:, 1:4]  # [vx,vz,yaw_rate]
    cond_den = cond_norm.cpu().numpy()[0] * pos_vel_std[1:4] + pos_vel_mean[1:4]
    mse = float(np.mean((gen_root - cond_den) ** 2))
    print(f"[Root-vs-Cond MSE] {mse:.6f}")

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    parser.add_argument('--rel_traj_path', type=str, default='./traj_from_template/relative_3d_compatible.pt')
    parser.add_argument('--abs_traj_path', type=str, default='./traj_from_template/absolute_3d_compatible.pt')
    parser.add_argument('--processed_data_path', type=str, default='./processed_data_traj_gait')
    parser.add_argument('--skeleton_template_path', type=str, default='./dataset/Aeroplane_BR.bvh')
    parser.add_argument('--class_idx', type=int, default=None,
                        help="Class label for conditional generation (optional, 0 ~ 6).")
    
    args = parser.parse_args()

    # ─ 기본 설정 ─
    njoints = 23
    position_features = 3
    rotation_features = 6
    root_motion_features = 4
    foot_features = 2
    input_feats = root_motion_features + ((njoints - 1) * position_features) + (njoints * rotation_features) + foot_features
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

    pos_vel_mean, pos_vel_std = dataset.pos_vel_mean, dataset.pos_vel_std
    position_mean, position_std = dataset.position_mean, dataset.position_std
    rotation_mean, rotation_std = dataset.rotation_mean, dataset.rotation_std
    foot_mean, foot_std = dataset.foot_mean, dataset.foot_std
    
    full_mean = np.hstack([pos_vel_mean, position_mean, rotation_mean, foot_mean])  # 210
    full_std  = np.hstack([pos_vel_std,  position_std,  rotation_std,  foot_std])

    rel_pt = torch.load(args.rel_traj_path, map_location='cpu').numpy().astype(np.float32)  # [T,3]
    abs_pt = torch.load(args.abs_traj_path, map_location='cpu').numpy().astype(np.float32)  # [T,3]
    
    cond = cond_from_relative_pt(rel_pt, abs_pt, pos_vel_mean, pos_vel_std).to(device)      # [1,T,3]
    T = cond.shape[1]

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
        generated_motion_norm = diffusion.p_sample_loop_cond_overwrite(
            model,
            sample_shape,
            cond=cond,
            #guidance_scale=args.guidance_scale,
            model_kwargs=model_kwargs
        )  # [1,T,210]

    # ─ 역정규화 & 루트 신호 투영 ─

    generated_motion = generated_motion_norm.cpu().numpy()[0] * full_std + full_mean  # [T,210]

    #check_root_vs_cond(generated_motion, pos_vel_mean, pos_vel_std, cond)
    gen_traj_abs = vel_tensor_to_abs_traj(
        generated_motion, 
        start_x=abs_pt[0,0], start_z=abs_pt[0,1], start_yaw=abs_pt[0,2]
    )
    torch.save(torch.from_numpy(gen_traj_abs), os.path.join(output_dir, "generated_traj_abs.pt"))
    
    motion_feature = generated_motion[:, :208]

    root, motion_obj = tensor_to_motion_object(motion_feature, args.skeleton_template_path, start_yaw=abs_pt[0,2], start_x=abs_pt[0,0], start_z=abs_pt[0,1])
    out_bvh = os.path.join(output_dir, "sample.bvh")
    write_bvh(root, motion_obj, out_bvh)

    out_mp4 = os.path.join(output_dir, "sample.mp4")
    render_movie(root, motion_obj, out_mp4)
    print(f"\nGeneration complete. Saved to {output_dir}")

if __name__ == '__main__':
    generate()
