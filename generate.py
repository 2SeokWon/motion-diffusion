import torch
import os
import argparse

from model import MotionTransformer
from gaussian_diffusion import GaussianDiffusion
from dataset import MotionDataset  # mean/std 로드를 위해 필요
from utils import generate_and_save_bvh

def generate():
    parser = argparse.ArgumentParser(description="Generate Human Motion from a trained MDM model")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the model checkpoint (.pt) file.")
    parser.add_argument('--num_samples', type=int, default=5,
                        help="Number of motion samples to generate.")
    parser.add_argument('--seq_len', type=int, default=90,
                        help="Length of the generated motion sequence in frames.")
    args = parser.parse_args()

    # --- 고정된 설정값들 ---
    njoints = 23
    rotation_features = 6
    root_motion_features = 4
    input_feats = root_motion_features + (njoints * rotation_features)
    num_timesteps = 1000
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_bvh_dir = "./results_1"
    processed_data_path = "./processed_data"
    skeleton_template_path = "./dataset/Aeroplane_BR.bvh"
    os.makedirs(output_bvh_dir, exist_ok=True)

    # --- 모델 및 Diffusion 초기화 ---
    print("Initializing model...")
    model = MotionTransformer(
        njoints=njoints,
        input_feats=input_feats, # 모델은 관절당 특징 수를 받음
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1
    ).to(device)
    
    betas = torch.linspace(0.0001, 0.02, num_timesteps)
    diffusion = GaussianDiffusion(betas=betas).to(device)

    # --- 체크포인트 로드 ---
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at '{args.checkpoint_path}'")
        return
        
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    
    # --- 생성에 필요한 mean/std 로드 ---
    # MotionDataset 객체를 생성하여 mean, std를 쉽게 가져옴
    print("Loading dataset statistics (mean/std)...")
    dataset = MotionDataset(processed_data_path=processed_data_path, seq_len=args.seq_len)
    
    # --- 생성 실행 ---
    generate_and_save_bvh(
        model, diffusion, dataset,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        input_feats=input_feats,
        root_motion_features=root_motion_features,
        template_path=skeleton_template_path,
        output_dir=output_bvh_dir,
        device=device
    )

if __name__ == '__main__':
    generate()