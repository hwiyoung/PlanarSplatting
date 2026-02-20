#!/usr/bin/env python3
"""Evaluate a PlanarSplatting checkpoint: PSNR, Depth MAE → JSON.

Usage (inside Docker container):
    python scripts/evaluate.py \
        --checkpoint planarSplat_ExpRes/demo/exp_example/2026_.../checkpoints/Parameters/latest.pth \
        --metrics psnr depth_mae \
        --output results.json

    python scripts/evaluate.py \
        --checkpoint path/to/latest.pth \
        --compare_with path/to/other_results.json
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'planarsplat'))

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger


def find_experiment_dir(checkpoint_path):
    """Walk up from checkpoint to find the experiment directory (contains input_data.pth)."""
    p = Path(checkpoint_path).resolve()
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent,
                   p.parent.parent.parent.parent]:
        if (parent / 'input_data.pth').exists():
            return parent
    return None


def find_config(exp_dir):
    """Find the run config .conf file in the experiment directory."""
    conf_files = list(exp_dir.glob('run_conf_*.conf'))
    if conf_files:
        return max(conf_files, key=lambda f: f.stat().st_mtime)
    return None


def load_model_and_data(checkpoint_path):
    """Load model from checkpoint + data from experiment directory."""
    exp_dir = find_experiment_dir(checkpoint_path)
    if exp_dir is None:
        raise FileNotFoundError(
            f"Cannot find experiment directory (with input_data.pth) for {checkpoint_path}")

    # Load config
    conf_path = find_config(exp_dir)
    if conf_path is None:
        raise FileNotFoundError(f"No run_conf_*.conf found in {exp_dir}")

    from pyhocon import ConfigFactory
    conf = ConfigFactory.parse_file(str(conf_path))

    # Load input data
    data_path = exp_dir / 'input_data.pth'
    data = torch.load(str(data_path), map_location='cpu')
    logger.info(f"Loaded input data: {data_path}")

    # Build dataset
    from utils.misc_util import get_class
    dataset_class = get_class(conf.get_string('train.dataset_class'))
    dataset = dataset_class(data, **conf.get_config('dataset'))

    # Build model
    from run.net_wrapper import PlanarRecWrapper
    net = PlanarRecWrapper(conf, '')
    net = net.cuda()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    plane_num = ckpt['model_state_dict']['planarSplat._plane_center'].shape[0]
    # Backward compatibility: add missing semantic features (Phase 2-B)
    sem_key = 'planarSplat._plane_semantic_features'
    if sem_key not in ckpt['model_state_dict']:
        num_classes = net.planarSplat.semantic_num_classes
        ckpt['model_state_dict'][sem_key] = torch.zeros(plane_num, num_classes)
    net.planarSplat.initialize_as_zero(plane_num)
    net.build_optimizer_and_LRscheduler()
    net.reset_plane_vis()
    net.reset_grad_stats()
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    logger.info(f"Loaded checkpoint: iter={ckpt.get('iter', '?')}, planes={plane_num}")

    return net, dataset, conf, ckpt.get('iter', -1)


def compute_psnr(rendered_rgb, gt_rgb):
    """Compute PSNR between two [H*W, 3] tensors in [0, 1] range."""
    mse = F.mse_loss(rendered_rgb, gt_rgb)
    if mse < 1e-10:
        return 100.0
    return -10.0 * torch.log10(mse).item()


def compute_miou(pred_classes, gt_classes, num_classes=4, ignore_index=0):
    """Compute mean IoU over non-ignored classes."""
    ious = {}
    class_names = {1: 'roof', 2: 'wall', 3: 'ground'}
    for c in range(num_classes):
        if c == ignore_index:
            continue
        pred_c = (pred_classes == c)
        gt_c = (gt_classes == c)
        intersection = (pred_c & gt_c).sum().item()
        union = (pred_c | gt_c).sum().item()
        if union > 0:
            ious[class_names.get(c, f'class_{c}')] = intersection / union
        else:
            ious[class_names.get(c, f'class_{c}')] = float('nan')
    valid_ious = [v for v in ious.values() if v == v]  # exclude NaN
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return miou, ious


def evaluate_checkpoint(checkpoint_path, metrics):
    if 'psnr' in metrics:
        logger.warning("PSNR: PlanarSplatting uses random colors (not learnable), "
                        "so PSNR reflects color randomness rather than reconstruction quality. "
                        "Use depth_mae and normal_cos for meaningful evaluation.")

    # Fix random seed for reproducible color assignment across runs
    torch.manual_seed(42)

    net, dataset, conf, ckpt_iter = load_model_and_data(checkpoint_path)
    H = conf.get_list('dataset.img_res')[0]
    W = conf.get_list('dataset.img_res')[1]

    results = {
        'checkpoint': str(checkpoint_path),
        'iter': ckpt_iter,
        'plane_count': net.planarSplat.get_plane_num(),
        'n_views': dataset.n_images,
    }

    psnr_list = []
    depth_mae_list = []
    normal_cos_list = []
    miou_list = []
    per_class_ious_accum = {}

    with torch.no_grad():
        for idx in range(dataset.n_images):
            view_info = dataset.view_info_list[idx]
            raster_cam_w2c = view_info.raster_cam_w2c

            rendered_rgb, allmap = net.planarSplat(view_info, -1, return_rgb=True)
            depth = allmap[0:1].squeeze().view(-1)
            normal_local = allmap[2:5]
            normal_global = (normal_local.permute(1, 2, 0) @ (raster_cam_w2c[:3, :3].T)).view(-1, 3)

            vis_weight = allmap[1:2].squeeze().view(-1)
            valid_mask = vis_weight > 0.00001
            # Match training mask: include depth>0 and normal!=0 (trainer.py L434-437)
            valid_depth_mask = view_info.mono_depth.abs() > 0
            valid_normal_mask = view_info.mono_normal_global.abs().sum(dim=-1) > 0
            valid_mask = valid_mask & valid_depth_mask & valid_normal_mask

            # PSNR
            if 'psnr' in metrics:
                gt_rgb = view_info.rgb  # (H*W, 3)
                rendered_rgb_flat = rendered_rgb.permute(1, 2, 0).reshape(-1, 3)
                psnr_val = compute_psnr(rendered_rgb_flat, gt_rgb)
                psnr_list.append(psnr_val)

            # Depth MAE
            if 'depth_mae' in metrics:
                gt_depth = view_info.mono_depth  # (H*W,)
                if valid_mask.sum() > 0:
                    mae = torch.abs(depth[valid_mask] - gt_depth[valid_mask]).mean().item()
                    depth_mae_list.append(mae)

            # Normal cosine similarity
            if 'normal_cos' in metrics:
                gt_normal = view_info.mono_normal_global
                normal_pred = F.normalize(normal_global, dim=-1)
                normal_gt = F.normalize(gt_normal, dim=-1)
                valid_normal = valid_mask & (gt_normal.abs().sum(dim=-1) > 0)
                if valid_normal.sum() > 0:
                    cos_sim = (normal_pred * normal_gt).sum(dim=-1)[valid_normal].mean().item()
                    normal_cos_list.append(cos_sim)

            # Semantic mIoU
            if 'semantic_miou' in metrics and view_info.seg_map is not None:
                # rendered_rgb is (4, H, W) = semantic features when enable_semantic
                sem_pred = rendered_rgb.argmax(dim=0).reshape(-1)  # (H*W,)
                gt_seg = view_info.seg_map  # (H*W,)
                # Only evaluate where GT has non-bg labels AND valid render
                eval_mask = valid_mask & (gt_seg > 0)
                if eval_mask.sum() > 100:
                    miou_val, class_ious = compute_miou(sem_pred[eval_mask], gt_seg[eval_mask])
                    miou_list.append(miou_val)
                    for k, v in class_ious.items():
                        if v == v:  # not NaN
                            per_class_ious_accum.setdefault(k, []).append(v)

    if psnr_list:
        results['psnr_mean'] = float(np.mean(psnr_list))
        results['psnr_std'] = float(np.std(psnr_list))
        results['psnr_per_view'] = psnr_list

    if depth_mae_list:
        results['depth_mae_mean'] = float(np.mean(depth_mae_list))
        results['depth_mae_std'] = float(np.std(depth_mae_list))
        results['depth_mae_per_view'] = depth_mae_list

    if normal_cos_list:
        results['normal_cos_mean'] = float(np.mean(normal_cos_list))
        results['normal_cos_std'] = float(np.std(normal_cos_list))
        results['normal_cos_per_view'] = normal_cos_list

    if miou_list:
        results['semantic_miou_mean'] = float(np.mean(miou_list))
        results['semantic_miou_std'] = float(np.std(miou_list))
        results['semantic_miou_per_view'] = miou_list
        for k, v_list in per_class_ious_accum.items():
            results[f'iou_{k}_mean'] = float(np.mean(v_list))

    return results


def compare_results(current, compare_path):
    """Print a comparison table between current results and a previous JSON."""
    with open(compare_path) as f:
        prev = json.load(f)

    print("\n=== Comparison ===")
    print(f"{'Metric':<20} {'Previous':>12} {'Current':>12} {'Delta':>12}")
    print("-" * 58)

    for key in ['psnr_mean', 'depth_mae_mean', 'normal_cos_mean', 'semantic_miou_mean', 'plane_count']:
        if key in current and key in prev:
            p = prev[key]
            c = current[key]
            d = c - p
            sign = '+' if d >= 0 else ''
            if isinstance(c, int) and isinstance(p, int):
                print(f"{key:<20} {p:>12d} {c:>12d} {sign}{d:>11d}")
            else:
                print(f"{key:<20} {p:>12.4f} {c:>12.4f} {sign}{d:>11.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate PlanarSplatting checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pth')
    parser.add_argument('--metrics', nargs='+', default=['psnr', 'depth_mae', 'normal_cos'],
                        choices=['psnr', 'depth_mae', 'normal_cos', 'semantic_miou'],
                        help='Metrics to compute')
    parser.add_argument('--output', default='', help='Output JSON path')
    parser.add_argument('--compare_with', default='', help='Previous results JSON to compare')
    args = parser.parse_args()

    results = evaluate_checkpoint(args.checkpoint, args.metrics)

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Checkpoint: {results['checkpoint']}")
    print(f"Iteration:  {results['iter']}")
    print(f"Planes:     {results['plane_count']}")
    print(f"Views:      {results['n_views']}")
    if 'psnr_mean' in results:
        print(f"PSNR:       {results['psnr_mean']:.2f} +/- {results['psnr_std']:.2f} dB")
    if 'depth_mae_mean' in results:
        print(f"Depth MAE:  {results['depth_mae_mean']:.4f} +/- {results['depth_mae_std']:.4f}")
    if 'normal_cos_mean' in results:
        print(f"Normal cos: {results['normal_cos_mean']:.4f} +/- {results['normal_cos_std']:.4f}")
    if 'semantic_miou_mean' in results:
        print(f"Sem. mIoU:  {results['semantic_miou_mean']:.4f} +/- {results['semantic_miou_std']:.4f}")
        for cls in ['roof', 'wall', 'ground']:
            key = f'iou_{cls}_mean'
            if key in results:
                print(f"  IoU {cls}:  {results[key]:.4f}")

    # Save JSON
    if args.output:
        out_path = args.output
    else:
        exp_dir = find_experiment_dir(args.checkpoint)
        if exp_dir:
            out_path = str(exp_dir / 'eval_results.json')
        else:
            out_path = 'eval_results.json'

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {out_path}")

    # Compare if requested
    if args.compare_with and os.path.exists(args.compare_with):
        compare_results(results, args.compare_with)


if __name__ == '__main__':
    main()
