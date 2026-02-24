#!/usr/bin/env python3
"""Cross-evaluate FD and MVS models against both GT sources.

Produces a 2x2 comparison matrix:
  {FD model, MVS model} x {FD GT, MVS GT} -> normal_cos + depth_mae

Usage (inside Docker):
    python scripts/cross_eval_normals.py \
        --fd_ckpt  planarSplat_ExpRes/seongsu_normal_test_fd/exp_example/.../latest.pth \
        --mvs_ckpt planarSplat_ExpRes/seongsu_normal_test_mvs/exp_example/.../latest.pth \
        --fd_data  planarSplat_ExpRes/seongsu_normal_test_fd/input_data.pth \
        --mvs_data planarSplat_ExpRes/seongsu_normal_test_mvs/input_data.pth
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'planarsplat'))

from loguru import logger


def find_run_dir(checkpoint_path):
    current = Path(checkpoint_path).resolve().parent
    for _ in range(6):
        if list(current.glob('run_conf_*.conf')):
            return current
        current = current.parent
    return None


def load_model(checkpoint_path):
    """Load model only (no data)."""
    run_dir = find_run_dir(checkpoint_path)
    if run_dir is None:
        # Try walking up from checkpoint to find conf
        for search_dir in [Path(checkpoint_path).resolve().parent]:
            for _ in range(6):
                conf_files = list(search_dir.glob('run_conf_*.conf'))
                if conf_files:
                    run_dir = search_dir
                    break
                search_dir = search_dir.parent

    conf_files = list(run_dir.glob('run_conf_*.conf'))
    conf_path = max(conf_files, key=lambda f: f.stat().st_mtime)

    from pyhocon import ConfigFactory
    conf = ConfigFactory.parse_file(str(conf_path))

    from run.net_wrapper import PlanarRecWrapper
    net = PlanarRecWrapper(conf, '')
    net = net.cuda()

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    plane_num = ckpt['model_state_dict']['planarSplat._plane_center'].shape[0]
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
    logger.info(f"Loaded model: {checkpoint_path}, planes={plane_num}")
    return net, conf


def load_data(data_path, conf):
    """Load dataset from a specific input_data.pth."""
    data = torch.load(str(data_path), map_location='cpu')
    from utils.misc_util import get_class
    dataset_class = get_class(conf.get_string('train.dataset_class'))
    dataset = dataset_class(data, **conf.get_config('dataset'))
    logger.info(f"Loaded data: {data_path}, {dataset.n_images} views")
    return dataset


def evaluate_model_on_data(net, dataset, label=""):
    """Evaluate a model on a dataset, return depth_mae and normal_cos."""
    depth_mae_list = []
    normal_cos_list = []

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
            valid_depth_mask = view_info.mono_depth.abs() > 0
            valid_normal_mask = view_info.mono_normal_global.abs().sum(dim=-1) > 0
            valid_mask = valid_mask & valid_depth_mask & valid_normal_mask

            # Depth MAE
            gt_depth = view_info.mono_depth
            if valid_mask.sum() > 0:
                mae = torch.abs(depth[valid_mask] - gt_depth[valid_mask]).mean().item()
                depth_mae_list.append(mae)

            # Normal cosine similarity
            gt_normal = view_info.mono_normal_global
            normal_pred = F.normalize(normal_global, dim=-1)
            normal_gt = F.normalize(gt_normal, dim=-1)
            valid_normal = valid_mask & (gt_normal.abs().sum(dim=-1) > 0)
            if valid_normal.sum() > 0:
                cos_sim = (normal_pred * normal_gt).sum(dim=-1)[valid_normal].mean().item()
                normal_cos_list.append(cos_sim)

    depth_mae = float(np.mean(depth_mae_list)) if depth_mae_list else float('nan')
    normal_cos = float(np.mean(normal_cos_list)) if normal_cos_list else float('nan')
    return depth_mae, normal_cos


def run_eval(ckpt_path, data_path, label):
    """Load model + data, evaluate, then free GPU memory."""
    import gc
    torch.manual_seed(42)
    net, conf = load_model(ckpt_path)
    dataset = load_data(data_path, conf)
    d, n = evaluate_model_on_data(net, dataset, label)
    # Free GPU memory
    del net, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return d, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd_ckpt', required=True)
    parser.add_argument('--mvs_ckpt', required=True)
    parser.add_argument('--fd_data', required=True)
    parser.add_argument('--mvs_data', required=True)
    args = parser.parse_args()

    # 2x2 evaluation matrix (sequential to avoid OOM)
    print("\n" + "=" * 70)
    print("Cross-evaluation: {Model} x {GT Source}")
    print("=" * 70)

    results = {}

    print("\n[1/4] FD model vs FD GT...")
    d, n = run_eval(args.fd_ckpt, args.fd_data, "FD_model+FD_GT")
    results['fd_model_fd_gt'] = (d, n)

    print("[2/4] FD model vs MVS GT...")
    d, n = run_eval(args.fd_ckpt, args.mvs_data, "FD_model+MVS_GT")
    results['fd_model_mvs_gt'] = (d, n)

    print("[3/4] MVS model vs FD GT...")
    d, n = run_eval(args.mvs_ckpt, args.fd_data, "MVS_model+FD_GT")
    results['mvs_model_fd_gt'] = (d, n)

    print("[4/4] MVS model vs MVS GT...")
    d, n = run_eval(args.mvs_ckpt, args.mvs_data, "MVS_model+MVS_GT")
    results['mvs_model_mvs_gt'] = (d, n)

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS: Normal Cosine Similarity")
    print("=" * 70)
    print(f"{'':20s} {'FD GT':>15s} {'MVS GT':>15s}")
    print("-" * 52)
    print(f"{'FD model':20s} {results['fd_model_fd_gt'][1]:>15.4f} {results['fd_model_mvs_gt'][1]:>15.4f}")
    print(f"{'MVS model':20s} {results['mvs_model_fd_gt'][1]:>15.4f} {results['mvs_model_mvs_gt'][1]:>15.4f}")

    print(f"\n{'':20s} {'FD GT':>15s} {'MVS GT':>15s}")
    print("-" * 52)
    # Highlight: which model is better per GT
    for gt_name, fd_key, mvs_key in [
        ('FD GT', 'fd_model_fd_gt', 'mvs_model_fd_gt'),
        ('MVS GT', 'fd_model_mvs_gt', 'mvs_model_mvs_gt'),
    ]:
        fd_val = results[fd_key][1]
        mvs_val = results[mvs_key][1]
        diff = mvs_val - fd_val
        winner = "MVS" if diff > 0 else "FD"
        print(f"  Against {gt_name}: {winner} model wins by {abs(diff):.4f}")

    print("\n" + "=" * 70)
    print("RESULTS: Depth MAE (same MVS depth GT for both)")
    print("=" * 70)
    print(f"{'':20s} {'FD GT':>15s} {'MVS GT':>15s}")
    print("-" * 52)
    print(f"{'FD model':20s} {results['fd_model_fd_gt'][0]:>15.4f} {results['fd_model_mvs_gt'][0]:>15.4f}")
    print(f"{'MVS model':20s} {results['mvs_model_fd_gt'][0]:>15.4f} {results['mvs_model_mvs_gt'][0]:>15.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    # Same-GT comparison (diagonal) is self-serving
    # Cross-GT comparison (off-diagonal) tests generalization
    fd_cross = results['fd_model_mvs_gt'][1]  # FD model evaluated against MVS GT
    mvs_cross = results['mvs_model_fd_gt'][1]  # MVS model evaluated against FD GT
    print(f"Self-evaluation  (trained GT):  FD={results['fd_model_fd_gt'][1]:.4f}  MVS={results['mvs_model_mvs_gt'][1]:.4f}")
    print(f"Cross-evaluation (other GT):    FD→MVS={fd_cross:.4f}  MVS→FD={mvs_cross:.4f}")
    print(f"Cross-eval shows which model generalizes better to the other GT source.")


if __name__ == '__main__':
    main()
