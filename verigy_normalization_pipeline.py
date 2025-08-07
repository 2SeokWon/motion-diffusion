# verify_normalization_pipeline.py

import numpy as np
import os

def main():
    processed_data_path = "./processed_data"
    print("--- Verifying the integrity of the Normalization -> De-normalization pipeline ---")

    try:
        # 1. 테스트할 원본 특징 데이터 로드 (정규화되지 않은 상태)
        clip_path = os.path.join(processed_data_path, "clip_0000.npz")
        with np.load(clip_path) as data:
            features_original = data['features']
        print(f"Loaded original features from '{clip_path}' with shape {features_original.shape}")

        # 2. 통계 파일 로드
        pos_vel_mean = np.load(os.path.join(processed_data_path, "pos_vel_mean.npy"))
        pos_vel_std = np.load(os.path.join(processed_data_path, "pos_vel_std.npy"))
        rotation_mean = np.load(os.path.join(processed_data_path, "rotation_mean.npy"))
        rotation_std = np.load(os.path.join(processed_data_path, "rotation_std.npy"))

        # 통계 데이터를 하나의 벡터로 조합
        mean = np.concatenate([pos_vel_mean, rotation_mean], axis=1)
        std = np.concatenate([pos_vel_std, rotation_std], axis=1)
        print(f"Loaded and combined statistics. Mean shape: {mean.shape}, Std shape: {std.shape}")

        # 3. 정규화 수행
        # 브로드캐스팅을 위해 features_original의 차원과 맞춰줌
        if mean.shape[0] == 1:
             mean = np.broadcast_to(mean, (features_original.shape[0], mean.shape[1]))
        if std.shape[0] == 1:
             std = np.broadcast_to(std, (features_original.shape[0], std.shape[1]))
        
        features_normalized = (features_original - mean) / std

        # 4. 역정규화 수행
        features_reconstructed = features_normalized * std + mean
        
        # 5. 원본과 복원된 데이터 비교
        error = np.abs(features_original - features_reconstructed)

        print("\n" + "="*50)
        print(" VERIFICATION RESULTS ")
        print("="*50)
        print("Comparing original data with (normalized -> de-normalized) data.\n")
        
        # 샘플 데이터 출력
        print("Original data sample (first 3 features of first 2 frames):")
        print(features_original[:2, :3])
        print("\nReconstructed data sample (first 3 features of first 2 frames):")
        print(features_reconstructed[:2, :3])

        print("\n--- Error Analysis ---")
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        print(f"Maximum Absolute Error: {max_error}")
        print(f"Mean Absolute Error:    {mean_error}")
        print("="*50)

        # 6. 최종 판정
        # 부동소수점 연산 오차를 고려하여 1e-6 정도는 정상으로 간주
        if max_error < 1e-5:
            print("\n✅ SUCCESS: The normalization/de-normalization pipeline is mathematically CORRECT.")
            print("This means the statistics files (mean/std) and the logic are sound.")
            print("The bug is likely in how this logic is integrated into the BVH generation script.")
        else:
            print("\n❌ FAILURE: The pipeline is FLAWED. The reconstructed data does not match the original.")
            print("This points to a fundamental issue in the statistics calculation (in preprocess_bvh.py).")

    except FileNotFoundError as e:
        print(f"\nError: Could not find data files. Did you run preprocess_bvh.py first?")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    main()