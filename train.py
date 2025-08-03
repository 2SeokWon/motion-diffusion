import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

from tqdm import tqdm

from model import MotionTransformer
from gaussian_diffusion import GaussianDiffusion
from dataset import MotionDataset
from kinematics import Skeleton

def train():
    learning_rate = 1e-4
    num_epochs = 100
    batch_size = 32
    save_interval = 10

    njoints = 23
    rotation_features = 6
    root_motion_features = 4 # 루트 Y높이(1) + 수평속도(2) + Y회전속도(1)

    joint_rotation_features = njoints * rotation_features
    input_feats = root_motion_features + joint_rotation_features

    seq_len = 90
    num_timesteps = 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    save_dir = "./checkpoints"
    output_bvh_dir = "./results"
    processed_data_path = "./processed_data"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_bvh_dir, exist_ok=True)

    dataset = MotionDataset(processed_data_path=processed_data_path, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataset loaded successfully.")

    try:
        offsets = np.load(os.path.join(processed_data_path, "offsets.npy"))
        parents = np.load(os.path.join(processed_data_path, "parents.npy"))
    except FileNotFoundError:
        print("Error: 'offsets.npy' or 'parents.npy' not found. Please run preprocess.py first.")
        return

    skeleton = Skeleton(offsets=offsets, parents=parents, device=device)
    
    mean_tensor = dataset.mean.to(device)
    std_tensor = dataset.std.to(device)   

    print("Initializing model...")
    model = MotionTransformer(
        njoints=njoints,
        input_feats=input_feats,
        seq_len=seq_len,
        latent_dim=256,
        ff_size=1024,
    ).to(device)

    betas = torch.linspace(0.0001, 0.02, num_timesteps)

    diffusion = GaussianDiffusion(betas=betas).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Model initialized successfully.")

    # --- 훈련 Loop ---
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_fk_loss = 0.0
        total_simple_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            x_start = batch.to(device)

            t = torch.randint(0, num_timesteps, (x_start.shape[0],), device=device)

            optimizer.zero_grad()

            loss_dict = diffusion.training_losses(model, x_start, t, skeleton=skeleton, mean=mean_tensor, std=std_tensor)
            loss = loss_dict['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            total_fk_loss += loss_dict['loss_fk']
            total_simple_loss += loss_dict['loss_simple']

            progress_bar.set_postfix({
                'loss': f'{total_loss / (progress_bar.n + 1):.4f}',
                'fk_loss': f'{total_fk_loss / (progress_bar.n + 1):.4f}',
                'simple_loss': f'{total_simple_loss / (progress_bar.n + 1):.4f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    print("Training completed.")

if __name__ == '__main__':
    train()
