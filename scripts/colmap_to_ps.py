#!/usr/bin/env python3
"""Convert COLMAP output to PlanarSplatting input format and optionally run training.

Usage:
    # Convert only (saves input_data.pth)
    python scripts/colmap_to_ps.py \
        --colmap_path /path/to/colmap/output \
        --output_path planarSplat_ExpRes/seongsu

    # Convert + train
    python scripts/colmap_to_ps.py \
        --colmap_path /path/to/colmap/output \
        --output_path planarSplat_ExpRes/seongsu \
        --run_training

    # Use precomputed data (skip Metric3D)
    python scripts/colmap_to_ps.py \
        --colmap_path /path/to/colmap/output \
        --output_path planarSplat_ExpRes/seongsu \
        --use_precomputed --run_training

Expected COLMAP directory structure:
    colmap_path/
    ├── images/               # original images
    ├── sparse/0/             # mapper output
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.bin
    └── dense/                # (optional) undistorter output
        ├── images/           # undistorted images (preferred)
        └── sparse/           # undistorted camera params (preferred)
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

# Add project root and planarsplat to path (planarsplat/utils imports need this)
_project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'planarsplat'))

from utils_demo.read_write_model import read_model, qvec2rotmat


def find_colmap_model(colmap_path):
    """Find COLMAP sparse model directory. Prefer undistorted (dense/sparse/)."""
    candidates = [
        os.path.join(colmap_path, 'dense', 'sparse'),
        os.path.join(colmap_path, 'sparse', '0'),
        os.path.join(colmap_path, 'sparse'),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, 'cameras.bin')):
            print(f"[INFO] Using COLMAP model from: {path}")
            return path
    raise FileNotFoundError(
        f"No COLMAP model found. Checked: {candidates}\n"
        "Expected cameras.bin, images.bin, points3D.bin in one of these directories."
    )


def find_image_dir(colmap_path):
    """Find image directory. Prefer undistorted images."""
    candidates = [
        os.path.join(colmap_path, 'dense', 'images'),
        os.path.join(colmap_path, 'images'),
    ]
    for path in candidates:
        if os.path.isdir(path) and len(os.listdir(path)) > 0:
            print(f"[INFO] Using images from: {path}")
            return path
    raise FileNotFoundError(f"No image directory found. Checked: {candidates}")


def load_colmap_data(colmap_path, max_images=-1, target_res=None):
    """Load COLMAP cameras, images, and points3D.

    If target_res is given as (H, W), all images are resized to that resolution
    and intrinsics are scaled accordingly. This handles varying per-camera sizes
    from COLMAP undistortion.
    """
    model_path = find_colmap_model(colmap_path)
    image_dir = find_image_dir(colmap_path)

    cameras, images, points3D = read_model(model_path, ext=".bin")
    print(f"[INFO] Loaded {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points")

    # Sort images by name for reproducibility
    sorted_image_ids = sorted(images.keys(), key=lambda k: images[k].name)
    if max_images > 0:
        sorted_image_ids = sorted_image_ids[:max_images]
        print(f"[INFO] Using first {max_images} images")

    color_images_list = []
    image_paths_list = []
    c2ws_list = []
    intrinsics_list = []
    img_id_list = []
    skipped = 0

    for img_id in tqdm(sorted_image_ids, desc="Loading images"):
        img_meta = images[img_id]
        frame_path = os.path.join(image_dir, img_meta.name)

        if not os.path.exists(frame_path):
            print(f"[WARN] Image not found: {frame_path}, skipping")
            skipped += 1
            continue

        cam = cameras[img_meta.camera_id]
        params = cam.params
        if cam.model == 'PINHOLE':
            fx, fy, cx, cy = params[:4]
        elif cam.model == 'SIMPLE_PINHOLE':
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif cam.model in ('SIMPLE_RADIAL', 'RADIAL'):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif cam.model == 'OPENCV':
            fx, fy, cx, cy = params[:4]
        else:
            fx, fy, cx, cy = params[:4]
            print(f"[WARN] Camera model '{cam.model}' - using first 4 params as fx,fy,cx,cy")

        rgb = np.array(Image.open(frame_path))
        orig_h, orig_w = rgb.shape[:2]

        # Resize to target resolution if specified
        if target_res is not None:
            tgt_h, tgt_w = target_res
            scale_x = tgt_w / orig_w
            scale_y = tgt_h / orig_h
            rgb = cv2.resize(rgb, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
            fx *= scale_x
            fy *= scale_y
            cx *= scale_x
            cy *= scale_y

        intrinsic = np.array([
            [fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.0]
        ], dtype=np.float32)

        R = qvec2rotmat(img_meta.qvec)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = img_meta.tvec
        c2w = np.linalg.inv(w2c).astype(np.float32)

        color_images_list.append(rgb)
        image_paths_list.append(frame_path)
        c2ws_list.append(c2w)
        intrinsics_list.append(intrinsic)
        img_id_list.append(img_id)

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} images (not found)")
    h, w = color_images_list[0].shape[:2]
    print(f"[INFO] Loaded {len(color_images_list)} images, resolution: {h} x {w}")

    return {
        'color_images': color_images_list,
        'image_paths': image_paths_list,
        'c2ws': c2ws_list,
        'intrinsics': intrinsics_list,
        'img_ids': img_id_list,
        'cameras': cameras,
        'images': images,
        'points3D': points3D,
    }


def compute_target_resolution(colmap_path):
    """Compute a uniform target resolution from undistorted camera sizes."""
    model_path = find_colmap_model(colmap_path)
    cameras, _, _ = read_model(model_path, ext=".bin")

    widths = set()
    heights = set()
    for cam in cameras.values():
        widths.add(cam.width)
        heights.add(cam.height)

    if len(widths) == 1 and len(heights) == 1:
        return None  # already uniform

    # Use minimum dimensions to avoid padding/extrapolation
    min_w = min(widths)
    min_h = min(heights)
    print(f"[INFO] Varying camera sizes detected (w: {min(widths)}-{max(widths)}, h: {min(heights)}-{max(heights)})")
    print(f"[INFO] Using uniform target resolution: {min_h} x {min_w}")
    return (min_h, min_w)


def scale_align_depth(colmap_data, depth_maps_list):
    """Scale-align mono depth maps using COLMAP sparse 3D points."""
    images = colmap_data['images']
    cameras = colmap_data['cameras']
    points3D = colmap_data['points3D']
    img_ids = colmap_data['img_ids']

    if len(points3D) == 0:
        print("[WARN] No 3D points for scale alignment, using raw mono depth")
        return depth_maps_list

    pts_indices = np.array([points3D[key].id for key in points3D])
    pts_xyzs = np.array([points3D[key].xyz for key in points3D])
    max_idx = pts_indices.max()
    points3d_ordered = np.zeros([max_idx + 1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    aligned_depths = []
    n_aligned = 0

    for i, (img_id, mono_depth) in enumerate(zip(img_ids, depth_maps_list)):
        img_meta = images[img_id]
        cam = cameras[img_meta.camera_id]

        pts_idx = img_meta.point3D_ids
        mask = (pts_idx >= 0) & (pts_idx <= max_idx)
        pts_idx_valid = pts_idx[mask]
        valid_xys = img_meta.xys[mask]

        if len(pts_idx_valid) < 10:
            aligned_depths.append(mono_depth)
            continue

        pts = points3d_ordered[pts_idx_valid]
        R = qvec2rotmat(img_meta.qvec)
        pts_cam = pts @ R.T + img_meta.tvec
        colmap_depth = pts_cam[:, 2]

        inv_colmap_depth = 1.0 / colmap_depth
        inv_mono_depth = 1.0 / mono_depth

        if inv_mono_depth.ndim != 2:
            inv_mono_depth = inv_mono_depth[..., 0]
        inv_mono_depth = inv_mono_depth.astype(np.float32)

        # Scale 2D points to mono depth resolution
        s = inv_mono_depth.shape[0] / cam.height
        maps = (valid_xys * s).astype(np.float32)
        valid = (
            (maps[:, 0] >= 0) &
            (maps[:, 1] >= 0) &
            (maps[:, 0] < cam.width * s) &
            (maps[:, 1] < cam.height * s) &
            (inv_colmap_depth > 0)
        )

        if valid.sum() > 10 and (inv_colmap_depth[valid].max() - inv_colmap_depth[valid].min()) > 1e-3:
            maps_v = maps[valid]
            inv_colmap_v = inv_colmap_depth[valid]
            inv_mono_v = cv2.remap(
                inv_mono_depth,
                maps_v[:, 0], maps_v[:, 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )[..., 0]

            t_colmap = np.median(inv_colmap_v)
            s_colmap = np.mean(np.abs(inv_colmap_v - t_colmap))
            t_mono = np.median(inv_mono_v)
            s_mono = np.mean(np.abs(inv_mono_v - t_mono))

            if s_mono > 1e-8:
                scale = s_colmap / s_mono
                offset = t_colmap - t_mono * scale
                inv_aligned = inv_mono_depth * scale + offset
                inv_aligned = np.clip(inv_aligned, 1e-6, None)
                aligned = 1.0 / inv_aligned
                aligned_depths.append(aligned)
                n_aligned += 1
            else:
                aligned_depths.append(mono_depth)
        else:
            aligned_depths.append(mono_depth)

    print(f"[INFO] Scale-aligned {n_aligned}/{len(depth_maps_list)} depth maps")
    return aligned_depths


def export_sparse_pointcloud(points3D, output_path):
    """Export COLMAP sparse points as PLY."""
    if len(points3D) == 0:
        print("[WARN] No 3D points to export")
        return None

    ply_path = os.path.join(output_path, 'colmap_sparse.ply')
    pts = np.array([p.xyz for p in points3D.values()])
    rgb = np.array([p.rgb for p in points3D.values()], dtype=np.uint8)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(ply_path, 'w') as f:
        f.write(header)
        for p, c in zip(pts, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

    print(f"[INFO] Exported {len(pts)} sparse points to {ply_path}")
    return ply_path


def convert(args):
    """Main conversion: COLMAP -> PlanarSplatting data dict."""
    os.makedirs(args.output_path, exist_ok=True)
    data_path = os.path.join(args.output_path, 'input_data.pth')

    if args.use_precomputed and os.path.exists(data_path):
        print(f"[INFO] Loading precomputed data from {data_path}")
        data = torch.load(data_path, weights_only=False)
        print(f"[INFO] Loaded {len(data['color'])} views")
        return data

    # Determine target resolution for uniform image sizes
    target_res = compute_target_resolution(args.colmap_path)

    # Step 1: Load COLMAP data
    colmap_data = load_colmap_data(args.colmap_path, max_images=args.max_images, target_res=target_res)

    # Step 2: Export sparse point cloud
    pcd_path = export_sparse_pointcloud(colmap_data['points3D'], args.output_path)

    # Step 3: Run Metric3D for mono depth/normal
    print(f"[INFO] Running Metric3D on {len(colmap_data['color_images'])} images...")
    from utils_demo.run_metric3d import extract_mono_geo_demo
    depth_maps_list, normal_maps_list = extract_mono_geo_demo(
        colmap_data['color_images'],
        colmap_data['intrinsics'],
    )
    print(f"[INFO] Metric3D done. Depth shape: {depth_maps_list[0].shape}, Normal shape: {normal_maps_list[0].shape}")

    # Step 4: Scale-align mono depth with COLMAP sparse points
    depth_maps_aligned = scale_align_depth(colmap_data, depth_maps_list)

    # Step 5: Build data dict in PlanarSplatting format
    data = {
        'color': colmap_data['color_images'],       # list of (H,W,3) uint8
        'depth': depth_maps_aligned,                 # list of (H,W) float32
        'normal': normal_maps_list,                  # list of (3,H,W) float32 [0,1]
        'image_paths': colmap_data['image_paths'],   # list of str
        'extrinsics': colmap_data['c2ws'],           # list of (4,4) float32 c2w
        'intrinsics': colmap_data['intrinsics'],     # list of (3,3) float32
        'out_path': args.output_path,
        'init_method': args.init_method,
    }

    if pcd_path:
        data['colmap_pcd_path'] = pcd_path

    # Step 6: Save
    torch.save(data, data_path)
    print(f"[INFO] Saved input data to {data_path} ({len(data['color'])} views)")

    # Print summary
    h, w = data['color'][0].shape[:2]
    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}")
    print(f"  COLMAP path:     {args.colmap_path}")
    print(f"  Output path:     {args.output_path}")
    print(f"  Init method:     {args.init_method}")
    print(f"  Num views:       {len(data['color'])}")
    print(f"  Image resolution: {h} x {w}")
    print(f"  Sparse points:   {len(colmap_data['points3D'])}")
    print(f"  Data saved:      {data_path}")
    print(f"{'='*60}\n")

    return data


def run_training(data, args):
    """Run PlanarSplatting training."""
    from pyhocon import ConfigFactory, ConfigTree
    from utils_demo.run_planarSplatting import run_planarSplatting

    base_conf = ConfigFactory.parse_file('planarsplat/confs/base_conf_planarSplatCuda.conf')
    demo_conf = ConfigFactory.parse_file(args.conf_path)
    conf = ConfigTree.merge_configs(base_conf, demo_conf)

    conf.put('train.exps_folder_name', args.output_path)
    img_res = [data['color'][0].shape[0], data['color'][0].shape[1]]
    conf.put('dataset.img_res', img_res)
    conf.put('dataset.pre_align', True)
    conf.put('dataset.voxel_length', 0.1)
    conf.put('dataset.sdf_trunc', 0.2)
    conf.put('plane_model.init_plane_num', args.init_plane_num)
    conf.put('train.max_total_iters', args.max_iters)
    conf.put('train.expname', args.exp_name)

    print(f"\n[INFO] Starting training: {args.max_iters} iters, {args.init_plane_num} initial planes")
    print(f"[INFO] Image resolution: {img_res[0]} x {img_res[1]}")
    print(f"[INFO] Experiment name: {args.exp_name}")

    run_planarSplatting(data=data, conf=conf)
    print("[INFO] Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Convert COLMAP output to PlanarSplatting format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Conversion args
    parser.add_argument('--colmap_path', required=True,
                        help='Path to COLMAP output directory')
    parser.add_argument('--output_path', required=True,
                        help='Output directory for PlanarSplatting data')
    parser.add_argument('--init_method', choices=['colmap', 'vggt'], default='colmap',
                        help='Initialization method (colmap: TSDF mesh from Metric3D + COLMAP scale alignment)')
    parser.add_argument('--max_images', type=int, default=-1,
                        help='Max number of images to use (-1 for all)')
    parser.add_argument('--use_precomputed', action='store_true',
                        help='Use precomputed input_data.pth if it exists')

    # Training args
    parser.add_argument('--run_training', action='store_true',
                        help='Run training after conversion')
    parser.add_argument('--conf_path', type=str, default='utils_demo/demo.conf',
                        help='Config file path')
    parser.add_argument('--max_iters', type=int, default=5000,
                        help='Maximum training iterations')
    parser.add_argument('--init_plane_num', type=int, default=3000,
                        help='Initial number of plane primitives')
    parser.add_argument('--exp_name', type=str, default='colmap_exp',
                        help='Experiment name')

    args = parser.parse_args()

    # Step 1: Convert COLMAP -> PlanarSplatting format
    data = convert(args)

    # Step 2: Optionally run training
    if args.run_training:
        run_training(data, args)


if __name__ == '__main__':
    main()
