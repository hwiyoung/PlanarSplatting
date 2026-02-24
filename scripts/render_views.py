#!/usr/bin/env python3
"""Render and save RGB, Depth, Normal images from a trained checkpoint.

Usage:
    python scripts/render_views.py \
        --checkpoint path/to/latest.pth \
        --output_dir results/phase0/images \
        --num_views 3
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

_project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'planarsplat'))

from pathlib import Path
from loguru import logger


def _walk_up(start_path, target_name, max_levels=6):
    current = Path(start_path).resolve().parent
    for _ in range(max_levels):
        if (current / target_name).exists():
            return current
        current = current.parent
    return None


def _find_run_dir(checkpoint_path):
    current = Path(checkpoint_path).resolve().parent
    for _ in range(6):
        if list(current.glob('run_conf_*.conf')):
            return current
        current = current.parent
    return None


def render_and_save(checkpoint_path, output_dir, num_views=3, view_indices=None):
    os.makedirs(output_dir, exist_ok=True)

    run_dir = _find_run_dir(checkpoint_path)
    data_dir = _walk_up(checkpoint_path, 'input_data.pth')

    # Find config
    conf_path = None
    for search_dir in [run_dir, data_dir]:
        if search_dir is not None:
            conf_files = list(search_dir.glob('run_conf_*.conf'))
            if conf_files:
                conf_path = max(conf_files, key=lambda f: f.stat().st_mtime)
                break
    if conf_path is None:
        raise FileNotFoundError(f"No run_conf_*.conf found for {checkpoint_path}")

    from pyhocon import ConfigFactory
    conf = ConfigFactory.parse_file(str(conf_path))

    # Find data
    data_path = None
    for search_dir in [data_dir, run_dir]:
        if search_dir is not None and (search_dir / 'input_data.pth').exists():
            data_path = search_dir / 'input_data.pth'
            break
    if data_path is None:
        raise FileNotFoundError(f"No input_data.pth found for {checkpoint_path}")
    data = torch.load(str(data_path), map_location='cpu', weights_only=False)
    logger.info(f"Loaded data: {len(data['color'])} views")

    from utils.misc_util import get_class
    dataset_class = get_class(conf.get_string('train.dataset_class'))
    dataset = dataset_class(data, **conf.get_config('dataset'))

    # Build model and load checkpoint
    net_class = get_class(conf.get_string('plane_model.plane_model_class'))
    net = net_class(conf, plot_dir='')
    net.initialize_as_zero(1)

    ckpt = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    plane_num = ckpt['model_state_dict']['planarSplat._plane_center'].shape[0]
    net.initialize_as_zero(plane_num)

    from run.net_wrapper import PlanarRecWrapper
    wrapper = PlanarRecWrapper(conf, '')
    wrapper.planarSplat.initialize_as_zero(plane_num)
    wrapper.load_state_dict(ckpt['model_state_dict'])
    wrapper = wrapper.cuda()
    wrapper.eval()

    # Select views
    n_total = len(dataset.view_info_list)
    if view_indices is not None:
        indices = view_indices
    else:
        step = max(1, n_total // num_views)
        indices = list(range(0, n_total, step))[:num_views]

    H = conf.dataset.img_res[0]
    W = conf.dataset.img_res[1]

    for i, idx in enumerate(indices):
        view = dataset.view_info_list[idx]
        with torch.no_grad():
            allmap = wrapper.planarSplat(view, iter=-1)

        # Extract channels
        depth = allmap[0:1].squeeze().cpu().numpy()
        alpha = allmap[1:2].squeeze().cpu().numpy()
        normal_local = allmap[2:5]  # (3, H, W)

        # Normal: local -> global
        raster_cam_w2c = view.raster_cam_w2c
        normal_global = (normal_local.permute(1, 2, 0) @ raster_cam_w2c[:3, :3].T).cpu().numpy()
        normal_vis = (normal_global + 1) / 2  # [-1,1] -> [0,1]

        # GT images
        gt_rgb = view.rgb.cpu().numpy().reshape(H, W, 3)
        gt_depth = view.mono_depth.cpu().numpy().reshape(H, W)
        gt_normal = ((view.mono_normal_global.cpu().numpy().reshape(H, W, 3)) + 1) / 2

        # Rendered RGB (random color, just for reference)
        render_rgb_dummy = np.random.rand(H, W, 3) * alpha.reshape(H, W, 1)

        # Save depth with viridis colormap
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        vmax = np.percentile(gt_depth[gt_depth > 0], 95) if (gt_depth > 0).any() else 1.0
        axes[0].imshow(gt_depth.reshape(H, W), cmap='viridis', vmin=0, vmax=vmax)
        axes[0].set_title('GT Depth (Mono)')
        axes[0].axis('off')
        axes[1].imshow(depth.reshape(H, W), cmap='viridis', vmin=0, vmax=vmax)
        axes[1].set_title('Rendered Depth')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'depth_view{i:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Save normal
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].imshow(np.clip(gt_normal, 0, 1))
        axes[0].set_title('GT Normal (Mono)')
        axes[0].axis('off')
        axes[1].imshow(np.clip(normal_vis.reshape(H, W, 3), 0, 1))
        axes[1].set_title('Rendered Normal')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'normal_view{i:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Save RGB (GT only since rendered color is random)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.imshow(gt_rgb)
        ax.set_title(f'GT RGB (View {idx})')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'rgb_view{i:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved view {i} (dataset idx={idx})")

    logger.info(f"All images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Render and save result images')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_views', type=int, default=3)
    parser.add_argument('--views', type=int, nargs='+', default=None,
                        help='Specific view indices to render (overrides --num_views)')
    args = parser.parse_args()
    render_and_save(args.checkpoint, args.output_dir, args.num_views,
                    view_indices=args.views)


if __name__ == '__main__':
    main()
