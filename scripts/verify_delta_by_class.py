# scripts/verify_delta_by_class.py
"""
클래스별 delta 패턴이 실제로 다른지 확인하는 스크립트.

delta = virtual root trajectory - moving average (interp)
     → 보행 주기의 lateral sway 패턴을 담음

클래스마다 이 패턴이 다르면 delta는 class-discriminative한 신호 → 설계 유효
모두 비슷하면 delta는 style 정보를 담지 못함 → 재설계 필요

Usage:
    python scripts/verify_delta_by_class.py
    python scripts/verify_delta_by_class.py --raw_dir ./data/raw --radius 30 --out ./verification/delta_by_class.png
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict

from bvh_viewer.BVH_Parser import bvh_parser
from core.motion_features import moving_average_path, compute_delta_traj


def extract_delta_from_bvh(file_path, radius=30):
    """
    BVH 파일 하나에서 delta trajectory를 추출.
    Returns:
        delta_xz : [T, 2]  — 로컬 좌표계에서의 XZ 잔차
        delta_yaw: [T]     — yaw 잔차
    """
    root, motion_obj = bvh_parser(file_path)

    # Virtual root XZ 추출
    vr_xz  = np.array([[f.joint_positions["virtual_root"].x,
                         f.joint_positions["virtual_root"].z]
                        for f in motion_obj.quaternion_frame], dtype=np.float32)  # [T, 2]

    # yaw 추출: arctan2(forward_x, forward_z) from virtual root rotation
    vr_yaw = np.array([
        np.arctan2(
            float(f.joint_rotations["virtual_root"].x),  # approximation — use mat directly
            float(f.joint_rotations["virtual_root"].w),
        ) * 2  # quat → angle (rough)
        for f in motion_obj.quaternion_frame
    ], dtype=np.float32)

    # 더 정확한 yaw: virtual_transform 의 rotation 열로부터
    vr_yaw = np.array([
        np.arctan2(
            float(f.virtual_transform[2][0]),  # row2 col0 = sin(yaw) ... glm column-major
            float(f.virtual_transform[2][2]),
        )
        for f in motion_obj.quaternion_frame
    ], dtype=np.float32)

    # Moving average (interp)
    interp = moving_average_path(vr_xz, vr_yaw, radius=radius)  # [T, 3]

    # Delta (local frame residual)
    delta = compute_delta_traj(vr_xz, vr_yaw, interp[:, :2], interp[:, 2])  # [T, 3]

    return delta[:, :2], delta[:, 2]  # delta_xz [T,2], delta_yaw [T]


def class_from_filename(path):
    return os.path.basename(path).split('_')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="./data/raw")
    parser.add_argument("--radius",  type=int, default=30, help="Moving average radius")
    parser.add_argument("--out",     default="./verification/delta_by_class.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    bvh_files = sorted(glob(os.path.join(args.raw_dir, "*.bvh")))
    if not bvh_files:
        print(f"No BVH files found in {args.raw_dir}")
        return

    # 클래스별로 파일 묶기
    class_files = defaultdict(list)
    for f in bvh_files:
        class_files[class_from_filename(f)].append(f)

    classes = sorted(class_files.keys())
    print(f"Classes ({len(classes)}): {classes}")

    # 클래스별 delta 통계 수집
    class_stats = {}   # class → {x_std, z_std, yaw_std, x_abs_mean, ...}
    class_deltas = {}  # class → list of delta_xz arrays (for time-series plot)

    for cls in classes:
        files = class_files[cls]
        x_stds, z_stds, yaw_stds = [], [], []
        x_abs_means, z_abs_means = [], []
        sample_delta = None

        for fpath in files:
            try:
                dxz, dyaw = extract_delta_from_bvh(fpath, radius=args.radius)
                x_stds.append(dxz[:, 0].std())
                z_stds.append(dxz[:, 1].std())
                yaw_stds.append(dyaw.std())
                x_abs_means.append(np.abs(dxz[:, 0]).mean())
                z_abs_means.append(np.abs(dxz[:, 1]).mean())
                if sample_delta is None:
                    sample_delta = dxz
                print(f"  [{cls}] {os.path.basename(fpath):25s}  dx_std={dxz[:,0].std():.4f}  dz_std={dxz[:,1].std():.4f}  dyaw_std={dyaw.std():.4f}")
            except Exception as e:
                print(f"  [{cls}] ERROR {fpath}: {e}")

        class_stats[cls] = {
            'x_std':      np.mean(x_stds),
            'z_std':      np.mean(z_stds),
            'yaw_std':    np.mean(yaw_stds),
            'x_abs_mean': np.mean(x_abs_means),
            'z_abs_mean': np.mean(z_abs_means),
        }
        class_deltas[cls] = sample_delta  # first file as representative

    # ── Print summary ──
    print("\n=== Per-class delta statistics (averaged over files) ===")
    print(f"{'Class':15s}  {'dx_std':>8}  {'dz_std':>8}  {'dyaw_std':>10}  {'|dx|_mean':>10}  {'|dz|_mean':>10}")
    for cls in classes:
        s = class_stats[cls]
        print(f"{cls:15s}  {s['x_std']:8.4f}  {s['z_std']:8.4f}  {s['yaw_std']:10.4f}  {s['x_abs_mean']:10.4f}  {s['z_abs_mean']:10.4f}")

    x_stds_all   = [class_stats[c]['x_std']   for c in classes]
    z_stds_all   = [class_stats[c]['z_std']    for c in classes]
    yaw_stds_all = [class_stats[c]['yaw_std']  for c in classes]
    print(f"\nX-delta  inter-class std: {np.std(x_stds_all):.4f}  (within-class mean: {np.mean(x_stds_all):.4f})")
    print(f"Z-delta  inter-class std: {np.std(z_stds_all):.4f}  (within-class mean: {np.mean(z_stds_all):.4f})")
    print(f"Yaw-delta inter-class std:{np.std(yaw_stds_all):.4f}  (within-class mean: {np.mean(yaw_stds_all):.4f})")
    print("\ninter-class std >> within-class mean 이면 delta가 class-discriminative함.")

    # ── Plot ──
    n_cls = len(classes)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cls))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Delta trajectory analysis by class  (radius={})".format(args.radius), fontsize=13)

    # 1. Bar chart: dx std per class
    ax = axes[0, 0]
    ax.bar(classes, [class_stats[c]['x_std'] for c in classes], color=colors)
    ax.set_title("delta X std per class  (lateral sway residual)")
    ax.set_ylabel("std (cm)")
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)

    # 2. Bar chart: dz std per class
    ax = axes[0, 1]
    ax.bar(classes, [class_stats[c]['z_std'] for c in classes], color=colors)
    ax.set_title("delta Z std per class  (fore-aft residual)")
    ax.set_ylabel("std (cm)")
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)

    # 3. Time-series of delta X (one representative file per class)
    ax = axes[1, 0]
    for i, cls in enumerate(classes):
        d = class_deltas[cls]
        if d is not None:
            T = len(d)
            ax.plot(np.arange(T), d[:, 0], color=colors[i], lw=1.0, alpha=0.8, label=cls)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_title("delta X over time  (one file per class)")
    ax.set_xlabel("frame"); ax.set_ylabel("delta X (cm)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # 4. Time-series of delta Z
    ax = axes[1, 1]
    for i, cls in enumerate(classes):
        d = class_deltas[cls]
        if d is not None:
            T = len(d)
            ax.plot(np.arange(T), d[:, 1], color=colors[i], lw=1.0, alpha=0.8, label=cls)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_title("delta Z over time  (one file per class)")
    ax.set_xlabel("frame"); ax.set_ylabel("delta Z (cm)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
