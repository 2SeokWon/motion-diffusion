import torch
import os
import argparse
import numpy as np
from datetime import datetime

from model import MotionTransformer
from gaussian_diffusion import GaussianDiffusion
from dataset import MotionDataset  # mean/std 로드를 위해 필요
from utils import generate_and_save_bvh
from bvh_viewer.render_video import tensor_to_motion_object, render_movie

def generate():
    parser = argparse.ArgumentParser(description="Generate Human Motion from a trained MDM model")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the model checkpoint (.pt) file.")
    parser.add_argument('--num_samples', type=int, default=5,
                        help="Number of motion samples to generate.")
    parser.add_argument('--seq_len', type=int, default=180,
                        help="Length of the generated motion sequence in frames.")
    args = parser.parse_args()

    # --- 고정된 설정값들 ---
    njoints = 23
    position_features = 3
    rotation_features = 6
    root_motion_features = 4
    input_feats = root_motion_features + ((njoints-1) * position_features) + (njoints * rotation_features)
    num_timesteps = 1000
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 현재 시간을 포함한 고유한 출력 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./results/generated_{timestamp}"
    processed_data_path = "./processed_data"
    skeleton_template_path = "./dataset/Aeroplane_BR.bvh"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- 모델 및 Diffusion 초기화 ---
    print("Initializing model...")
    model = MotionTransformer(
        njoints=njoints,
        input_feats=input_feats, # 모델은 관절당 특징 수를 받음
        latent_dim=512,
        ff_size=3072,
        num_layers=12,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    betas = torch.linspace(0.0001, 0.02, num_timesteps)
    diffusion = GaussianDiffusion(betas=betas).to(device)

    # --- 체크포인트 로드 ---
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at '{args.checkpoint_path}'")
        return
        
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 평가 모드로 설정

    # --- 생성에 필요한 mean/std 로드 ---
    # MotionDataset 객체를 생성하여 mean, std를 쉽게 가져옴
    print("Loading dataset statistics (mean/std)...")
    dataset = MotionDataset(processed_data_path=processed_data_path, seq_len=args.seq_len)
    
    for i in range(args.num_samples):
        print(f"\n--- Generating and Rendering Sample {i+1}/{args.num_samples} ---")
        
        # 4-1. 텐서 생성 및 역정규화
        sample_shape = (1, args.seq_len, input_feats)
        with torch.no_grad():
            generated_motion_norm = diffusion.p_sample_loop(model, sample_shape)
        
        mean = np.concatenate([dataset.pos_vel_mean, dataset.position_mean, dataset.rotation_mean], axis=1)
        std = np.concatenate([dataset.pos_vel_std, dataset.position_std, dataset.rotation_std], axis=1)
        generated_motion = generated_motion_norm.cpu().numpy()[0] * std + mean
        
        # 4-2. 텐서를 Motion 객체로 변환
        root, motion_obj = tensor_to_motion_object(generated_motion, skeleton_template_path)
        
        # 4-3. Motion 객체를 영상으로 렌더링
        output_path = os.path.join(output_dir, f"sample_{i+1}.mp4")
        render_movie(root, motion_obj, output_path)

    print(f"\nGeneration complete. All videos saved in '{output_dir}'")

if __name__ == '__main__':
    generate()