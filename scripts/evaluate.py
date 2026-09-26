# scripts/evaluate.py
# 일괄 평가: 평가 구간(각 파일의 뒤 15%)의 고정 창 × 경유점 개수(goal/2/5/10/dense)로 생성하고 채점해 결과표를 만든다.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import time
import numpy as np
import torch
from datetime import datetime

from core.config import load_config
from core.model import MotionTransformer
from core.gaussian_diffusion import GaussianDiffusion
from core.dataset import MotionDataset, split_range
from core.motion_features import integrate_root_velocity
from core.metrics import waypoint_metrics, foot_positions, foot_skating
from core.reconstruct import tensor_to_motion_object
from core.utils import write_bvh
from scripts.make_control import make_waypoint_mask

# 결과표에 올리는 지표 (per_sample.csv에는 전부 저장)
SUMMARY_KEYS = [
    'waypoint_pos_err_cm', 'goal_pos_err_cm', 'waypoint_yaw_err_deg', 'full_path_ADE_cm',
    'gen_skating_ratio', 'gt_skating_ratio', 'gen_weighted_skate_cm', 'gt_weighted_skate_cm',
]


def select_windows(dataset, seq_len, per_clip):
    """
    파일마다 평가 구간 앞에서부터 겹치지 않는 창 per_clip개. 무작위가 아니므로 다시 돌려도 같은 창이다.
    반환: [(clip_info, start_frame), ...]
    """
    windows = []
    for clip in dataset.sampleable_clips:
        lo, hi = split_range(clip['length'], 'test', dataset.test_ratio)
        n = min(per_clip, (hi - lo) // seq_len)
        windows += [(clip, lo + i * seq_len) for i in range(n)]
    return windows


def mean_std(rows, key):
    v = np.array([r[key] for r in rows if key in r], dtype=np.float64)
    return (float(v.mean()), float(v.std())) if len(v) else (float('nan'), float('nan'))


def write_summary(rows, settings, class_names, out_dir, header):
    # 조건별 평균 ± 표준편차
    summary = []
    for s in settings:
        sub = [r for r in rows if r['waypoints'] == s]
        entry = {'waypoints': s, 'n': len(sub)}
        for k in SUMMARY_KEYS:
            entry[k + '_mean'], entry[k + '_std'] = mean_std(sub, k)
        summary.append(entry)

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    fmt = lambda e, k: f"{e[k + '_mean']:.2f} ± {e[k + '_std']:.2f}"
    lines = [header, '',
             '| 경유점 | n | 경유점 오차 (cm) | 도착 오차 (cm) | yaw 오차 (deg) | (참고) ADE (cm) | skating 비율 (생성 / 원본) |',
             '|---|---|---|---|---|---|---|']
    for e in summary:
        lines.append(f"| {e['waypoints']} | {e['n']} | {fmt(e, 'waypoint_pos_err_cm')} | {fmt(e, 'goal_pos_err_cm')} | "
                     f"{fmt(e, 'waypoint_yaw_err_deg')} | {fmt(e, 'full_path_ADE_cm')} | "
                     f"{e['gen_skating_ratio_mean']:.3f} / {e['gt_skating_ratio_mean']:.3f} |")

    # 클래스별: 경유점 오차 (cm)
    lines += ['', '클래스별 경유점 오차 (cm, 평균)', '',
              '| 클래스 | ' + ' | '.join(settings) + ' |', '|---' * (len(settings) + 1) + '|']
    for c in class_names:
        cells = [f"{mean_std([r for r in rows if r['waypoints'] == s and r['class'] == c], 'waypoint_pos_err_cm')[0]:.2f}"
                 for s in settings]
        lines.append(f"| {c} | " + ' | '.join(cells) + ' |')

    # 클래스별: 발 미끄러짐 비율 (생성, 원본은 조건과 무관)
    lines += ['', '클래스별 skating 비율 (생성, 평균)', '',
              '| 클래스 | ' + ' | '.join(settings) + ' | 원본 |', '|---' * (len(settings) + 2) + '|']
    for c in class_names:
        cells = [f"{mean_std([r for r in rows if r['waypoints'] == s and r['class'] == c], 'gen_skating_ratio')[0]:.3f}"
                 for s in settings]
        gt = mean_std([r for r in rows if r['waypoints'] == settings[0] and r['class'] == c], 'gt_skating_ratio')[0]
        lines.append(f"| {c} | " + ' | '.join(cells) + f" | {gt:.3f} |")

    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return '\n'.join(lines)


def evaluate():
    parser = argparse.ArgumentParser(description="Batch evaluation over held-out windows × waypoint settings.")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--config', type=str, default=None,
                        help="Default: config.yml next to the checkpoint (train.py copies it), else ./config.yml")
    parser.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm')
    parser.add_argument('--steps', type=int, default=50, help="DDIM steps (ignored for ddpm)")
    parser.add_argument('--eta', type=float, default=0.0, help="DDIM eta (0 = deterministic)")
    parser.add_argument('--guidance_scale', type=float, default=None, help="CFG scale (default: config)")
    parser.add_argument('--waypoints', type=str, default='goal,2,5,10,dense',
                        help="Comma-separated settings: goal | N | dense")
    parser.add_argument('--windows_per_clip', type=int, default=2)
    parser.add_argument('--max_windows', type=int, default=None, help="Use only the first N windows (smoke test)")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bvh_per_class', type=int, default=1, help="Save BVH for the first N windows of each class")
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
    config_path = args.config or (os.path.join(ckpt_dir, 'config.yml')
                                  if os.path.exists(os.path.join(ckpt_dir, 'config.yml')) else 'config.yml')
    cfg = load_config(config_path)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else cfg.generation.guidance_scale
    settings = [s.strip() for s in args.waypoints.split(',')]
    T = cfg.model.seq_len

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | config: {config_path}")

    # ─ 모델 / 디퓨전 ─
    model = MotionTransformer(
        input_feats=cfg.model.input_feats,
        seq_len=cfg.model.seq_len,
        latent_dim=cfg.model.latent_dim,
        ff_size=cfg.model.ff_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        cond_dim=cfg.model.cond_dim,
    ).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    weights = 'ema_state_dict' if 'ema_state_dict' in ckpt else 'model_state_dict'  #생성에는 EMA 가중치를 우선 사용
    model.load_state_dict(ckpt[weights])
    model.eval()
    step = ckpt.get('step', '?')
    print(f"Loaded {weights} (step {step}) from {args.checkpoint_path}")
    del ckpt

    betas = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end, cfg.diffusion.num_timesteps)
    diffusion = GaussianDiffusion(betas=betas).to(device)

    # ─ 평가 창 (학습과 같은 통계로 정규화) ─
    dataset = MotionDataset(processed_data_path=cfg.data.processed_dir, seq_len=T,
                            feat_bias=cfg.training.feat_bias, split='test',
                            test_ratio=cfg.data.test_ratio, stats_dir=cfg.data.stats_dir)
    windows = select_windows(dataset, T, args.windows_per_clip)
    if args.max_windows:
        windows = windows[:args.max_windows]
    class_names = [c for c in dataset.name_classes if any(clip['class_name'] == c for clip, _ in windows)]
    N = len(windows)
    print(f"Eval windows: {N} ({args.windows_per_clip} per clip, non-overlapping, held-out 15%)")

    full_mean = np.hstack([dataset.root_pos_mean, dataset.position_mean, dataset.rotation_mean, dataset.foot_mean])
    full_std  = np.hstack([dataset.root_pos_std,  dataset.position_std,  dataset.rotation_std,  dataset.foot_std])

    gt_feats = []
    for clip, start in windows:
        f = dataset.clip_cache[clip['path']][start:start + T].copy()
        f[0, 1:4] = 0.0  # 학습 창과 같은 규약: 첫 프레임 root 속도 0 → 궤적이 원점에서 시작
        gt_feats.append(f)
    gt_feats = np.stack(gt_feats)                             # [N, T, 210]
    gt_traj = integrate_root_velocity(gt_feats[:, :, 1:4])    # [N, T, 3] (cm, cm, rad)
    traj_norm = (gt_traj - dataset.abs_traj_mean) / dataset.abs_traj_std
    classes = torch.tensor([clip['class_name_idx'] for clip, _ in windows], device=device)

    # 원본의 발 미끄러짐: 생성과 똑같은 복원 경로(특징 → Motion → FK)로 잰다
    print("Measuring GT foot skating ...")
    template = cfg.generation.skeleton_template
    gt_skate = [foot_skating(foot_positions(tensor_to_motion_object(g, template, verbose=False)[1])) for g in gt_feats]

    # 클래스마다 앞쪽 창 bvh_per_class개는 BVH로 남긴다 (로컬 뷰어로 눈으로 확인)
    bvh_idx, seen = set(), {}
    for i, (clip, _) in enumerate(windows):
        if seen.get(clip['class_name'], 0) < args.bvh_per_class:
            bvh_idx.add(i)
            seen[clip['class_name']] = seen.get(clip['class_name'], 0) + 1

    sampler_tag = f"ddpm{cfg.diffusion.num_timesteps}" if args.sampler == 'ddpm' else f"ddim{args.steps}"
    run_name = os.path.basename(ckpt_dir) + '_' + os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    out_dir = args.out_dir or os.path.join(cfg.generation.output_dir,
                                           f"eval_{run_name}_{sampler_tag}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    os.makedirs(os.path.join(out_dir, 'bvh'), exist_ok=True)
    for i in sorted(bvh_idx):
        clip, start = windows[i]
        root, motion = tensor_to_motion_object(gt_feats[i], template, verbose=False)
        write_bvh(root, motion, os.path.join(out_dir, 'bvh', f"{clip['source_file'][:-4]}_{start}_gt.bvh"))

    rows, timing = [], {}
    for setting in settings:
        mask = make_waypoint_mask(T, setting)                                        # [T, 1]
        cond = torch.from_numpy(np.concatenate([traj_norm * mask, np.broadcast_to(mask, (N, T, 1))], axis=2)).float()

        # 조건마다 같은 시드 → 같은 초기 노이즈. 조건 사이 차이는 경유점 개수에서만 온다.
        torch.manual_seed(args.seed)
        gen_norm = []
        t0 = time.time()
        with torch.no_grad():
            for b in range(0, N, args.batch_size):
                c = cond[b:b + args.batch_size].to(device)
                kw = {'classes_name': classes[b:b + args.batch_size]}
                shape = (c.size(0), T, cfg.model.input_feats)
                if args.sampler == 'ddpm':
                    x = diffusion.p_sample_loop_cond(model, shape, c, guidance_scale=guidance_scale, model_kwargs=kw)
                else:
                    x = diffusion.ddim_sample_loop_cond(model, shape, c, num_steps=args.steps, eta=args.eta,
                                                        guidance_scale=guidance_scale, model_kwargs=kw)
                gen_norm.append(x.cpu().numpy())
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        timing[setting] = {'seconds': elapsed, 'sec_per_sample': elapsed / N}
        gen = np.concatenate(gen_norm) * full_std + full_mean                        # [N, T, 210]
        gen_traj = integrate_root_velocity(gen[:, :, 1:4])                           # [N, T, 3]

        for i, (clip, start) in enumerate(windows):
            root, motion = tensor_to_motion_object(gen[i], template, verbose=False)
            r = {'waypoints': setting, 'source_file': clip['source_file'], 'class': clip['class_name'],
                 'start_frame': start}
            r.update(waypoint_metrics(gen_traj[i], gt_traj[i], mask))
            r.update({f'gen_{k}': v for k, v in foot_skating(foot_positions(motion)).items()})
            r.update({f'gt_{k}': v for k, v in gt_skate[i].items()})
            rows.append(r)
            if i in bvh_idx:
                write_bvh(root, motion, os.path.join(out_dir, 'bvh', f"{clip['source_file'][:-4]}_{start}_{setting}.bvh"))

        sub = rows[-N:]
        print(f"[{setting:>5}] waypoint {mean_std(sub, 'waypoint_pos_err_cm')[0]:7.2f} cm | "
              f"goal {mean_std(sub, 'goal_pos_err_cm')[0]:7.2f} cm | "
              f"yaw {mean_std(sub, 'waypoint_yaw_err_deg')[0]:6.2f} deg | "
              f"skating {mean_std(sub, 'gen_skating_ratio')[0]:.3f} (GT {mean_std(sub, 'gt_skating_ratio')[0]:.3f}) | "
              f"{elapsed:.1f}s")

    with open(os.path.join(out_dir, 'per_sample.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    header = (f"## 평가 결과: {run_name} (step {step}, {weights})\n\n"
              f"샘플러 {sampler_tag}{f' eta={args.eta}' if args.sampler == 'ddim' else ''}, CFG {guidance_scale}, "
              f"평가 창 {N}개(파일당 {args.windows_per_clip}개, 학습에 안 쓴 뒤 15%), 시드 {args.seed}. "
              f"창 1개 생성 시간 {np.mean([t['sec_per_sample'] for t in timing.values()]):.3f}s ({device})")
    print('\n' + write_summary(rows, settings, class_names, out_dir, header))

    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': args.checkpoint_path, 'step': step, 'weights': weights, 'config': config_path,
                   'sampler': args.sampler, 'steps': cfg.diffusion.num_timesteps if args.sampler == 'ddpm' else args.steps, 'eta': args.eta,
                   'guidance_scale': guidance_scale, 'seed': args.seed, 'settings': settings,
                   'num_windows': N, 'windows_per_clip': args.windows_per_clip, 'device': device,
                   'timing': timing}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_dir}")


if __name__ == '__main__':
    evaluate()
