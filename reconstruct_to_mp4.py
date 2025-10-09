import numpy as np
import os
# 각 스크립트에서 필요한 함수들을 가져옵니다.
from new_preprocess import extract_features
from bvh_viewer.render_video import tensor_to_motion_object, render_movie
from bvh_viewer.BVH_Parser import bvh_parser

def main():
    # --- 설정 ---
    original_bvh_path = "./dataset/Aeroplane_TR1.bvh"
    template_bvh_path = "./dataset/Drunk_TR1.bvh" # 스켈레톤 구조가 동일하므로 같은 파일 사용
    output_video_path = "./verification/perfect_round_trip_2.mp4"
    stats_dir = "./processed_data/"
    # ------------
    print(stats_dir)
    print("Step 1: Loading normalization statistics...")
    try:
        pos_vel_mean = np.load(os.path.join(stats_dir, "pos_vel_mean.npy"))
        pos_vel_std = np.load(os.path.join(stats_dir, "pos_vel_std.npy"))
        position_mean = np.load(os.path.join(stats_dir, "position_mean.npy"))
        position_std = np.load(os.path.join(stats_dir, "position_std.npy"))
        rotation_mean = np.load(os.path.join(stats_dir, "rotation_mean.npy"))
        rotation_std = np.load(os.path.join(stats_dir, "rotation_std.npy"))
        print("Statistics loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Statistics files not found in '{stats_dir}'. Please check the path.")
        return
    
    # 1. 원본 BVH 파일을 파싱하고, 전처리하여 '정답' 특징 벡터를 추출합니다.
    print("Step 1: Extracting features from original BVH...")
    root, original_motion = bvh_parser(original_bvh_path)
    features_original, _ = extract_features(original_motion, 0, 1800)
    print("--- Checking Input Tensor Integrity ---")
    if np.isnan(features_original).any():
        print("FATAL ERROR: NaN (Not a Number) detected in features!")
    elif np.isinf(features_original).any():
        print("FATAL ERROR: Infinity detected in features!")
    else:
        print("Tensor integrity check passed (No NaN/Inf).")
        print(f"Tensor Stats: Min={np.min(features_original):.4f}, Max={np.max(features_original):.4f}, Mean={np.mean(features_original):.4f}")

    root_features = features_original[:, :4]
    position_features = features_original[:, 4:70]
    rotation_features = features_original[:, 70:208]

    root_normalized = (root_features - pos_vel_mean)  / pos_vel_std
    position_normalized = (position_features - position_mean) / position_std
    rotation_normalized = (rotation_features - rotation_mean) / rotation_std
    
    root_denormalized = root_normalized * pos_vel_std + pos_vel_mean
    positions_denormalized = position_normalized * position_std + position_mean
    rotations_denormalized = rotation_normalized * rotation_std + rotation_mean

    # 비정규화된 특징들을 다시 하나의 텐서로 결합
    features_denormalized = np.concatenate([
        root_features, 
        position_features, 
        rotation_features
    ], axis=1).astype(np.float32)
    print("Denormalization complete.")
    # 2. 방금 추출한 '정답' 특징 벡터를 가지고 바로 Motion 객체로 복원합니다.
    #    (모델이나 정규화/역정규화 과정을 완전히 건너뜁니다)
    print("\nStep 2: Reconstructing Motion object from original features...")
    recon_root, recon_motion= tensor_to_motion_object(features_denormalized, template_bvh_path)

    # 3. 복원된 Motion 객체를 사용해 영상을 렌더링합니다.
    print("\nStep 3: Rendering the reconstructed motion to video...")
    render_movie(recon_root, recon_motion, output_video_path)

    print("\nVerification process complete.")

if __name__ == '__main__':
    main()