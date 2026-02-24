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

    # MVS depth+normal source (Phase 1)
    # Loads depth from depth_maps/*.geometric.bin
    # Loads normal from normal_maps/*.geometric.bin (COLMAP PatchMatch native)
    # Falls back to finite-diff normal if normal_maps/ not available
    python scripts/colmap_to_ps.py \
        --colmap_path /path/to/colmap/output \
        --output_path planarSplat_ExpRes/seongsu_phase1 \
        --depth_source mvs --run_training

Expected COLMAP directory structure:
    colmap_path/
    ├── images/               # original images
    ├── sparse/0/             # mapper output
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.bin
    └── dense/                # (optional) undistorter output
        ├── images/           # undistorted images (preferred)
        ├── sparse/           # undistorted camera params (preferred)
        └── stereo/           # MVS output (for --depth_source mvs)
            ├── depth_maps/   # *.geometric.bin
            └── normal_maps/  # *.geometric.bin
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


def read_colmap_array(path):
    """Read COLMAP MVS binary file (depth or normal map).

    Format: text header 'width&height&channels&' followed by float32 data.
    No null terminator between header and data.

    COLMAP's Mat class stores data in PLANAR layout:
        data[slice * W * H + row * W + col]
    where slice=channel index. For c=1 (depth), this is equivalent to row-major.
    For c=3 (normals), all nx values come first, then ny, then nz.
    """
    with open(path, 'rb') as f:
        header = b''
        amp_count = 0
        while amp_count < 3:
            b = f.read(1)
            if b == b'&':
                amp_count += 1
            header += b
        parts = header.decode('ascii').split('&')
        w, h, c = int(parts[0]), int(parts[1]), int(parts[2])
        data = np.frombuffer(f.read(), dtype=np.float32)[:w * h * c]
        if c == 1:
            return data.reshape(h, w)
        # Planar layout: (c, h, w) → transpose to (h, w, c)
        return data.reshape(c, h, w).transpose(1, 2, 0)


def depth_to_normal_cam(depth, intrinsic):
    """Compute camera-space normals from depth map via finite differences.

    Args:
        depth: (H, W) float32, metric depth. Invalid pixels = 0.
        intrinsic: (3, 3) float32, camera intrinsic matrix.

    Returns:
        normal: (3, H, W) float32 in [0, 1] range (matching Metric3D storage format).
               Invalid pixels have value 0.5 (which maps to 0 in [-1,1] after *2-1).
    """
    H, W = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    valid = depth > 0

    # Backproject to 3D in camera frame
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = (u - cx) / fx * depth
    Y = (v - cy) / fy * depth
    Z = depth.copy()

    # Forward finite differences
    # du direction (horizontal)
    dXdu = np.zeros_like(X)
    dYdu = np.zeros_like(Y)
    dZdu = np.zeros_like(Z)
    dXdu[:, :-1] = X[:, 1:] - X[:, :-1]
    dYdu[:, :-1] = Y[:, 1:] - Y[:, :-1]
    dZdu[:, :-1] = Z[:, 1:] - Z[:, :-1]

    # dv direction (vertical)
    dXdv = np.zeros_like(X)
    dYdv = np.zeros_like(Y)
    dZdv = np.zeros_like(Z)
    dXdv[:-1] = X[1:] - X[:-1]
    dYdv[:-1] = Y[1:] - Y[:-1]
    dZdv[:-1] = Z[1:] - Z[:-1]

    # Cross product: du x dv
    nx = dYdu * dZdv - dZdu * dYdv
    ny = dZdu * dXdv - dXdu * dZdv
    nz = dXdu * dYdv - dYdu * dXdv

    # Normalize
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-10
    nx /= norm
    ny /= norm
    nz /= norm

    # Ensure normals point toward camera (z < 0 in camera space)
    flip = nz > 0
    nx[flip] *= -1
    ny[flip] *= -1
    nz[flip] *= -1

    # Invalidate: current pixel invalid, or any neighbor used in diff is invalid
    neighbor_valid = np.ones_like(valid)
    neighbor_valid[:, :-1] &= valid[:, 1:]   # right neighbor for du
    neighbor_valid[:-1] &= valid[1:]          # bottom neighbor for dv
    all_valid = valid & neighbor_valid

    # Set invalid normals to 0 in [-1,1] space → 0.5 in [0,1] storage
    nx[~all_valid] = 0.0
    ny[~all_valid] = 0.0
    nz[~all_valid] = 0.0

    # Convert [-1,1] → [0,1] to match Metric3D format
    normal = np.stack([nx, ny, nz], axis=0)  # (3, H, W)
    normal = (normal + 1.0) / 2.0
    return normal.astype(np.float32)


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

    depth_source = getattr(args, 'depth_source', 'mono')

    if depth_source == 'mvs':
        # ===== MVS path: load COLMAP geometric depth AND normal maps =====
        stereo_dir = os.path.join(args.colmap_path, 'dense', 'stereo')
        depth_map_dir = os.path.join(stereo_dir, 'depth_maps')
        normal_map_dir = os.path.join(stereo_dir, 'normal_maps')
        if not os.path.isdir(depth_map_dir):
            raise FileNotFoundError(
                f"MVS depth maps not found at {depth_map_dir}. "
                "Run COLMAP patch_match_stereo first.")
        has_normal_maps = os.path.isdir(normal_map_dir)
        if not has_normal_maps:
            print("[WARN] MVS normal_maps dir not found, falling back to finite-diff normals")

        depth_maps_list = []
        normal_maps_list = []
        n_loaded = 0
        n_missing = 0
        n_normal_native = 0
        n_normal_fallback = 0

        for i, (img_path, intrinsic) in enumerate(tqdm(
                zip(colmap_data['image_paths'], colmap_data['intrinsics']),
                total=len(colmap_data['image_paths']),
                desc="Loading MVS depth/normal")):
            image_name = os.path.basename(img_path)
            geo_depth_path = os.path.join(depth_map_dir, f'{image_name}.geometric.bin')

            tgt_h, tgt_w = colmap_data['color_images'][i].shape[:2]

            if not os.path.exists(geo_depth_path):
                print(f"[WARN] MVS depth not found for {image_name}, using zeros")
                depth_maps_list.append(np.zeros((tgt_h, tgt_w), dtype=np.float32))
                normal_maps_list.append(np.full((3, tgt_h, tgt_w), 0.5, dtype=np.float32))
                n_missing += 1
                continue

            depth = read_colmap_array(geo_depth_path)  # (H_mvs, W_mvs)

            # Resize depth to target resolution if needed
            if depth.shape != (tgt_h, tgt_w):
                depth = cv2.resize(depth, (tgt_w, tgt_h),
                                   interpolation=cv2.INTER_NEAREST)

            # Load COLMAP native normal map (preferred) or fall back to finite-diff
            geo_normal_path = os.path.join(normal_map_dir, f'{image_name}.geometric.bin')
            if has_normal_maps and os.path.exists(geo_normal_path):
                mvs_normal = read_colmap_array(geo_normal_path)  # (H_mvs, W_mvs, 3) [-1,1] not unit
                # Resize to target resolution if needed
                if mvs_normal.shape[:2] != (tgt_h, tgt_w):
                    mvs_normal = cv2.resize(mvs_normal, (tgt_w, tgt_h),
                                            interpolation=cv2.INTER_NEAREST)
                # Normalize to unit vectors
                mvs_norm = np.linalg.norm(mvs_normal, axis=-1, keepdims=True)
                mvs_valid = (mvs_norm.squeeze() > 0.01) & (depth > 0)
                mvs_normal = np.where(mvs_norm > 0.01, mvs_normal / mvs_norm, 0.0)
                # Ensure normals point toward camera (z < 0), matching finite-diff convention
                flip_mask = mvs_normal[:, :, 2] > 0
                mvs_normal[flip_mask] *= -1
                # Mask invalid pixels
                mvs_normal[~mvs_valid] = 0.0
                # Convert to PlanarSplatting format: (3, H, W) in [0, 1]
                normal = mvs_normal.transpose(2, 0, 1).astype(np.float32)  # (3, H, W) [-1,1]
                normal = (normal + 1.0) / 2.0  # [-1,1] → [0,1]
                n_normal_native += 1
            else:
                # Fallback: derive normal from depth via finite differences
                normal = depth_to_normal_cam(depth, intrinsic)  # (3, H, W) in [0,1]
                n_normal_fallback += 1

            depth_maps_list.append(depth)
            normal_maps_list.append(normal)
            n_loaded += 1

        print(f"[INFO] MVS loaded: {n_loaded} views, {n_missing} missing")
        print(f"[INFO] Normals: {n_normal_native} native MVS, {n_normal_fallback} finite-diff fallback")
        valid_coverages = []
        normal_coverages = []
        for d, n in zip(depth_maps_list, normal_maps_list):
            cov = 100 * (d > 0).sum() / d.size
            valid_coverages.append(cov)
            # Normal valid = not [0.5, 0.5, 0.5] (which maps to [0,0,0] in [-1,1])
            n_valid = np.abs(n - 0.5).sum(axis=0) > 0.01
            normal_coverages.append(100 * n_valid.sum() / n_valid.size)
        print(f"[INFO] MVS depth coverage: mean={np.mean(valid_coverages):.1f}%, "
              f"min={np.min(valid_coverages):.1f}%, max={np.max(valid_coverages):.1f}%")
        print(f"[INFO] MVS normal coverage: mean={np.mean(normal_coverages):.1f}%, "
              f"min={np.min(normal_coverages):.1f}%, max={np.max(normal_coverages):.1f}%")

        # No scale alignment needed (MVS depth is absolute)
        depth_maps_aligned = depth_maps_list

    else:
        # ===== Mono depth path: Metric3D + scale alignment (original) =====
        print(f"[INFO] Running Metric3D on {len(colmap_data['color_images'])} images...")
        from utils_demo.run_metric3d import extract_mono_geo_demo
        depth_maps_list, normal_maps_list = extract_mono_geo_demo(
            colmap_data['color_images'],
            colmap_data['intrinsics'],
        )
        print(f"[INFO] Metric3D done. Depth shape: {depth_maps_list[0].shape}, "
              f"Normal shape: {normal_maps_list[0].shape}")

        # Scale-align mono depth with COLMAP sparse points
        depth_maps_aligned = scale_align_depth(colmap_data, depth_maps_list)

    # Build data dict in PlanarSplatting format
    data = {
        'color': colmap_data['color_images'],       # list of (H,W,3) uint8
        'depth': depth_maps_aligned,                 # list of (H,W) float32
        'normal': normal_maps_list,                  # list of (3,H,W) float32 [0,1]
        'image_paths': colmap_data['image_paths'],   # list of str
        'extrinsics': colmap_data['c2ws'],           # list of (4,4) float32 c2w
        'intrinsics': colmap_data['intrinsics'],     # list of (3,3) float32
        'out_path': args.output_path,
        'init_method': args.init_method,
        'depth_source': depth_source,
    }

    if pcd_path:
        data['colmap_pcd_path'] = pcd_path

    # Save
    torch.save(data, data_path)
    print(f"[INFO] Saved input data to {data_path} ({len(data['color'])} views)")

    # Print summary
    h, w = data['color'][0].shape[:2]
    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}")
    print(f"  COLMAP path:     {args.colmap_path}")
    print(f"  Output path:     {args.output_path}")
    print(f"  Depth source:    {depth_source}")
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

    depth_source = data.get('depth_source', 'mono')
    if depth_source == 'mvs':
        conf.put('dataset.pre_align', False)  # MVS depth is absolute scale
    else:
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
    parser.add_argument('--depth_source', choices=['mono', 'mvs'], default='mono',
                        help='Depth supervision source: mono (Metric3D) or mvs (COLMAP geometric depth)')
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
