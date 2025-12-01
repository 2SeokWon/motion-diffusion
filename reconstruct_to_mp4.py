import argparse
import numpy as np
import os
import sys

# Add current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

from new_preprocess import extract_features, CLIP_LENGTH, tensor_to_motion_object_root
from bvh_viewer.render_video import (
    tensor_to_motion_object,
    tensor_to_motion_object_traj,
    render_movie,
)
from bvh_viewer.BVH_Parser import bvh_parser

try:
    import matplotlib.pyplot as plt
except Exception:  # optional dependency
    plt = None


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def get_root_traj_xzyaw(motion, start=0, length=None):
    traj = []
    frames = motion.quaternion_frame
    if length is not None:
        frames = frames[start:start + length]
    else:
        frames = frames[start:]
    for frame in frames:
        vr_global = np.array(frame.virtual_transform)
        rot = vr_global[:3, :3]
        pos = vr_global[:3, 3]
        yaw = np.arctan2(rot[0, 2], rot[2, 2])
        traj.append([pos[0], pos[2], yaw])
    return np.array(traj)


def plot_trajs(trajs, labels, out_path, title):
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return
    plt.figure(figsize=(7, 7))
    colors = ["black", "orange", "blue", "green"]
    for i, (traj, label) in enumerate(zip(trajs, labels)):
        c = colors[i % len(colors)]
        plt.plot(traj[:, 0], traj[:, 1], color=c, linewidth=2 if i == 0 else 1.5, label=label)
        plt.scatter(traj[0, 0], traj[0, 1], c=c, s=25, alpha=0.6)
        plt.scatter(traj[-1, 0], traj[-1, 1], c=c, s=25, alpha=0.9, marker="x")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct motion from traj-format features and compare to original.")
    parser.add_argument("--file", default="./dataset/Aeroplane_FW.bvh", help="Path to BVH file")
    parser.add_argument("--template", default="./dataset/Aeroplane_FW.bvh", help="Template BVH for reconstruction")
    parser.add_argument("--length", type=int, default=CLIP_LENGTH, help="Clip length (frames)")
    parser.add_argument("--start", type=int, default=-1, help="Start frame (default: random)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random start")
    parser.add_argument("--plot-out", default="./verification/root_traj_compare.png", help="Output path for plot")
    parser.add_argument("--render", action="store_true", help="Render mp4 for original and reconstructed")
    parser.add_argument("--render-out-recon", default="./verification/recon_traj.mp4", help="Reconstructed mp4 path")
    parser.add_argument("--render-out-orig", default="./verification/orig.mp4", help="Original mp4 path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.plot_out), exist_ok=True)
    if args.render:
        os.makedirs(os.path.dirname(args.render_out_recon), exist_ok=True)
        os.makedirs(os.path.dirname(args.render_out_orig), exist_ok=True)
    
    print(f"Loading BVH: {args.file}")
    root, original_motion = bvh_parser(args.file)
    total_frames = original_motion.frame_len
    clip_len = args.length
    if clip_len > total_frames:
        raise ValueError(f"clip length {clip_len} exceeds total frames {total_frames}")
    if args.start >= 0:
        start_frame = args.start
        if start_frame + clip_len > total_frames:
            raise ValueError(f"start {start_frame} + length {clip_len} exceeds total frames {total_frames}")
    else:
        rng = np.random.default_rng(args.seed)
        start_frame = int(rng.integers(0, total_frames - clip_len + 1))
    print(f"Using clip frames: {start_frame} .. {start_frame + clip_len - 1} (len={clip_len})")

    # 1. Extract Raw Features for the clip (velocity-based)
    features_raw = extract_features(original_motion, start_frame, clip_len)  # [180, 210]
    print(f"Extracted raw features shape: {features_raw.shape}")
    # Convert to traj-format features by replacing channels 1:4
    traj_from_root = tensor_to_motion_object_root(features_raw)
    features_traj = features_raw.copy()
    features_traj[:, 1:4] = traj_from_root  # dataset/generation format: [root_y, abs_traj(xz,yaw), ...]

    orig_root, orig_motion = tensor_to_motion_object(features_raw, args.template)
    # 2. Reconstruct Motion from traj-format features
    recon_root, recon_motion = tensor_to_motion_object_traj(
        features_traj,
        args.template,
    )
    #[180, 210]
    # 3. Extract trajectories
    orig_traj = get_root_traj_xzyaw(original_motion, start=start_frame, length=clip_len)
    recon_traj = get_root_traj_xzyaw(recon_motion)

    # 4. Metrics
    def report(name, a, b):
        pos_err = a[:, :2] - b[:, :2]
        pos_err_norm = np.linalg.norm(pos_err, axis=1)
        yaw_err = wrap_to_pi(a[:, 2] - b[:, 2])
        print(f"\nMetrics ({name}):")
        print(f"  rmse_pos:      {np.sqrt(np.mean(pos_err_norm**2)):.6f}")
        print(f"  max_pos_error: {np.max(pos_err_norm):.6f}")
        print(f"  rmse_yaw:      {np.sqrt(np.mean(yaw_err**2)):.6f}")
        print(f"  mean_abs_yaw:  {np.mean(np.abs(yaw_err)):.6f}")
        print(f"  max_abs_yaw:   {np.max(np.abs(yaw_err)):.6f}")

    report("recon_from_traj vs orig", recon_traj, orig_traj)
    report("traj_from_root vs orig", traj_from_root, orig_traj)
    report("recon vs traj_from_root", recon_traj, traj_from_root)

    # 5. Plot
    plot_trajs(
        [orig_traj[:, :2], recon_traj[:, :2], traj_from_root[:, :2]],
        ["orig", "recon_from_traj", "traj_from_tensor_to_root"],
        args.plot_out,
        "root traj comparison (xz)",
    )

    # 6. Optional rendering
    if args.render:
        print("Rendering reconstructed motion...")
        render_movie(recon_root, recon_motion, args.render_out_recon)
        render_movie(orig_root, orig_motion, args.render_out_orig)

    print("\nDone.")


if __name__ == '__main__':
    main()
