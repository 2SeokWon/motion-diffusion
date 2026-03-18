# scripts/verify_trajectory.py
"""
Verification script: compare raw hip XZ vs virtual root XZ vs local hip offset.

Answers the question: "Does the raw BVH root trajectory show lateral sway (zigzag),
and is it preserved or removed by the virtual root transform?"

Usage:
    python scripts/verify_trajectory.py --file ./data/raw/Aeroplane_FW.bvh
    python scripts/verify_trajectory.py --file ./data/raw/Aeroplane_FW.bvh --start 0 --length 180 --render
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from bvh_viewer.BVH_Parser import bvh_parser
from core.motion_features import moving_average_path


def extract_raw_hip_xz(root, motion_obj):
    """
    Extract raw hip XYZ position directly from the unparsed frame data.
    The root joint channels are typically [Xpos, Ypos, Zpos, Xrot, Yrot, Zrot].
    Returns array of shape [T, 3] = (x, y, z).
    """
    channels = root.channels  # e.g. ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']
    pos_indices = {}
    for i, ch in enumerate(channels):
        ch_l = ch.lower()
        if 'xposition' in ch_l:
            pos_indices['x'] = i
        elif 'yposition' in ch_l:
            pos_indices['y'] = i
        elif 'zposition' in ch_l:
            pos_indices['z'] = i

    if len(pos_indices) != 3:
        raise ValueError(f"Could not find 3 position channels in root channels: {channels}")

    traj = []
    for frame in motion_obj.frames:
        x = frame[pos_indices['x']]
        y = frame[pos_indices['y']]
        z = frame[pos_indices['z']]
        traj.append([x, y, z])
    return np.array(traj)  # [T, 3]


def extract_virtual_root_xz(motion_obj):
    """
    Extract virtual root XZ position from processed frames.
    Returns array of shape [T, 2] = (x, z).
    """
    traj = []
    for frame in motion_obj.quaternion_frame:
        vr_pos = frame.joint_positions.get("virtual_root")
        traj.append([vr_pos.x, vr_pos.z])
    return np.array(traj)  # [T, 2]


def extract_hip_local_offset(motion_obj):
    """
    Extract hip local position relative to virtual root (lateral sway = x, z offsets).
    Returns array of shape [T, 3] = (local_x, local_y, local_z).
    """
    offsets = []
    for frame in motion_obj.quaternion_frame:
        lp = frame.hip_local_position
        offsets.append([lp.x, lp.y, lp.z])
    return np.array(offsets)  # [T, 3]


def main():
    parser = argparse.ArgumentParser(description="Verify raw hip vs virtual root trajectory.")
    parser.add_argument("--file",   default="./data/raw/Aeroplane_FW.bvh", help="Path to BVH file")
    parser.add_argument("--start",  type=int, default=0,   help="Start frame")
    parser.add_argument("--length", type=int, default=180, help="Number of frames to visualize (0 = all)")
    parser.add_argument("--out",    default="./verification/traj_verify.png", help="Output plot path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Parsing BVH: {args.file}")
    root, motion_obj = bvh_parser(args.file)
    total = motion_obj.frame_len
    print(f"Total frames: {total}")

    length = args.length if args.length > 0 else total
    end = min(args.start + length, total)
    sl = slice(args.start, end)
    print(f"Analyzing frames {args.start} .. {end-1}")

    # ── Extract trajectories ──
    raw_hip   = extract_raw_hip_xz(root, motion_obj)[sl]          # [T, 3]
    vroot_xz  = extract_virtual_root_xz(motion_obj)[sl]           # [T, 2]
    hip_local = extract_hip_local_offset(motion_obj)[sl]          # [T, 3]

    raw_xz = raw_hip[:, [0, 2]]  # drop Y

    # Moving-average smoothed version of virtual root trajectory
    vroot_yaw = np.arctan2(
        np.diff(vroot_xz[:, 0], prepend=vroot_xz[0, 0]),
        np.diff(vroot_xz[:, 1], prepend=vroot_xz[0, 1]),
    )
    smooth_xz = moving_average_path(vroot_xz, vroot_yaw, radius=30)[:, :2]

    T = len(raw_xz)
    frames = np.arange(T)

    # ── Statistics ──
    print("\n=== Trajectory Statistics ===")
    lat_sway_raw   = np.std(raw_xz[:, 0] - np.interp(frames, [0, T-1], [raw_xz[0, 0], raw_xz[-1, 0]]))
    lat_sway_vroot = np.std(vroot_xz[:, 0] - np.interp(frames, [0, T-1], [vroot_xz[0, 0], vroot_xz[-1, 0]]))
    print(f"Raw hip X stddev from linear trend:    {lat_sway_raw:.4f}  cm")
    print(f"Virtual root X stddev from linear:     {lat_sway_vroot:.4f}  cm")
    print(f"Hip local X (sway):  mean={hip_local[:,0].mean():.4f}  std={hip_local[:,0].std():.4f}")
    print(f"Hip local Z (depth): mean={hip_local[:,2].mean():.4f}  std={hip_local[:,2].std():.4f}")
    print(f"Hip local Y (height):mean={hip_local[:,1].mean():.4f}  std={hip_local[:,1].std():.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{os.path.basename(args.file)}  frames {args.start}–{end-1}", fontsize=13)

    # Panel 1: XZ overhead view
    ax = axes[0]
    ax.plot(raw_xz[:, 0],  raw_xz[:, 1],  color='black',  lw=1.5, label='raw hip XZ')
    ax.plot(vroot_xz[:, 0], vroot_xz[:, 1], color='orange', lw=1.5, label='virtual root XZ')
    ax.plot(smooth_xz[:, 0], smooth_xz[:, 1], color='blue', lw=1, ls='--', label='smoothed (MA)')
    ax.scatter(*raw_xz[0],  c='black',  s=40, zorder=5)
    ax.scatter(*raw_xz[-1], c='black',  s=40, marker='x', zorder=5)
    ax.set_title("XZ overhead (world)")
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.axis('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Panel 2: X over time (lateral sway visible as oscillation)
    ax = axes[1]
    ax.plot(frames, raw_xz[:, 0],   color='black',  lw=1.2, label='raw hip X')
    ax.plot(frames, vroot_xz[:, 0], color='orange', lw=1.2, label='virtual root X')
    ax.set_title("X over time  (lateral sway)")
    ax.set_xlabel("frame"); ax.set_ylabel("X position")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Panel 3: Hip local offset (sway relative to virtual root)
    ax = axes[2]
    ax.plot(frames, hip_local[:, 0], color='steelblue', lw=1.2, label='local X (lateral sway)')
    ax.plot(frames, hip_local[:, 2], color='tomato',    lw=1.2, label='local Z (fore-aft)')
    ax.plot(frames, hip_local[:, 1], color='green',     lw=1.0, label='local Y (height)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_title("Hip local position relative to virtual root")
    ax.set_xlabel("frame"); ax.set_ylabel("offset (cm)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"\nSaved plot: {args.out}")


if __name__ == '__main__':
    main()
