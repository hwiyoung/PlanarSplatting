#!/usr/bin/env python3
import argparse
import hashlib
import http.server
import json
import os
import random
import re
import shutil
import socketserver
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _slugify(text: str, max_len: int = 64) -> str:
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))
    s = s.strip("._-")
    if not s:
        s = "run"
    return s[:max_len]


def _run_uid(run_dir: Path) -> str:
    return hashlib.sha1(str(run_dir).encode("utf-8")).hexdigest()[:12]


def _find_latest_run(out_root: Path) -> Optional[Path]:
    run_dirs = [p.parent for p in out_root.rglob("train.log")]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _find_recent_runs(out_root: Path, limit: int) -> List[Path]:
    run_dirs = [p.parent for p in out_root.rglob("train.log")]
    if not run_dirs:
        return []
    run_dirs_sorted = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs_sorted[: max(1, int(limit))]


def _infer_output_root_from_run(run_dir: Path) -> Optional[Path]:
    parts = run_dir.parts
    if len(parts) < 3:
        return None
    # expected: .../<out_root>/<expname>/<timestamp>
    return run_dir.parent.parent


def _load_data_pth(path: Path) -> Optional[Dict]:
    if torch is None:
        return None
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        return None
    return data


def _resolve_data_pth(args_data_pth: str, run_dir: Path, output_root: Path) -> Path:
    if args_data_pth:
        return Path(args_data_pth).resolve()
    run_local = run_dir / "input_data.pth"
    if run_local.exists():
        return run_local
    return output_root / "data.pth"


def _resolve_pcd_path(args_pcd_path: str, run_dir: Path, output_root: Path) -> Path:
    if args_pcd_path:
        return Path(args_pcd_path).resolve()
    run_local = run_dir / "input_pointcloud.ply"
    if run_local.exists():
        return run_local
    return output_root / "pcd_vggt_downsampled.ply"


def _sample_indices(total: int, max_keep: int, rng: random.Random) -> np.ndarray:
    if total <= max_keep:
        return np.arange(total, dtype=np.int64)
    idx = list(range(total))
    rng.shuffle(idx)
    return np.asarray(idx[:max_keep], dtype=np.int64)


def _load_pointcloud(
    pcd_path: Path,
    max_points: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    if o3d is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.size == 0:
        colors = np.full((points.shape[0], 3), 0.75, dtype=np.float32)

    keep = _sample_indices(points.shape[0], max_points, rng)
    points = points[keep]
    colors = colors[keep]
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return points, colors_u8


def _build_pointcloud_from_data(
    data: Dict,
    stride: int,
    max_points: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    if not all(k in data for k in ("depth", "intrinsics", "extrinsics")):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []
    color_list = data.get("color", None)

    for i, (depth_i, intr_i, c2w_i) in enumerate(zip(data["depth"], data["intrinsics"], data["extrinsics"])):
        depth = np.asarray(depth_i, dtype=np.float32)
        if depth.ndim != 2:
            continue
        h, w = depth.shape
        ys, xs = np.indices((h, w))
        valid = depth > 0.0
        if stride > 1:
            valid = valid & ((ys % stride) == 0) & ((xs % stride) == 0)
        if not np.any(valid):
            continue

        K = np.asarray(intr_i, dtype=np.float32)
        c2w = np.asarray(c2w_i, dtype=np.float32)
        fx = max(float(K[0, 0]), 1e-6)
        fy = max(float(K[1, 1]), 1e-6)
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        z = depth[valid]
        x = (xs[valid].astype(np.float32) - cx) / fx * z
        y = (ys[valid].astype(np.float32) - cy) / fy * z
        pts_cam = np.stack([x, y, z], axis=-1)

        R = c2w[:3, :3]
        t = c2w[:3, 3]
        pts_world = pts_cam @ R.T + t[None, :]
        points_all.append(pts_world.astype(np.float32))

        if color_list is not None and i < len(color_list):
            rgb = np.asarray(color_list[i])
            if rgb.ndim == 3 and rgb.shape[0] == h and rgb.shape[1] == w:
                c = rgb[valid]
                if c.dtype == np.uint8:
                    colors_all.append(c.astype(np.uint8))
                else:
                    colors_all.append(np.clip(c * 255.0, 0, 255).astype(np.uint8))

    if not points_all:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(points_all, axis=0)
    if colors_all:
        colors = np.concatenate(colors_all, axis=0)
        if colors.shape[0] != points.shape[0]:
            colors = np.full((points.shape[0], 3), 192, dtype=np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 192, dtype=np.uint8)

    keep = _sample_indices(points.shape[0], max_points, rng)
    return points[keep], colors[keep]


def _find_final_mesh_path(run_dir: Path, data: Optional[Dict]) -> Optional[Path]:
    final_mesh_candidates = list(run_dir.glob("*/*_planar_mesh.ply"))
    if final_mesh_candidates:
        return sorted(final_mesh_candidates)[-1]

    plot_mesh_candidates = sorted((run_dir / "plane_plots").glob("planarSplat_*_colorPrim.ply"))
    if plot_mesh_candidates:
        return plot_mesh_candidates[-1]

    if data and "image_paths" in data and len(data["image_paths"]) > 0:
        first_img = Path(str(data["image_paths"][0])).resolve()
        maybe_mono = first_img.parent.parent / "mono_mesh.ply"
        if maybe_mono.exists():
            return maybe_mono
    return None


def _find_raw_mesh_path(run_dir: Path, data: Optional[Dict]) -> Optional[Path]:
    plane_plot_dir = run_dir / "plane_plots"
    if plane_plot_dir.exists():
        preferred = [
            plane_plot_dir / "planarSplat_initial-mesh_colorPrim.ply",
            plane_plot_dir / "planarSplat_initial-mesh_colorNormal.ply",
            plane_plot_dir / "planarSplat_000__colorPrim.ply",
            plane_plot_dir / "planarSplat_000__colorNormal.ply",
        ]
        for p in preferred:
            if p.exists():
                return p

        # Fallback to the earliest intermediate mesh snapshot.
        all_plot_mesh = sorted(plane_plot_dir.glob("planarSplat_*_colorPrim.ply"))
        if all_plot_mesh:
            return all_plot_mesh[0]

    if data and "image_paths" in data and len(data["image_paths"]) > 0:
        first_img = Path(str(data["image_paths"][0])).resolve()
        maybe_mono = first_img.parent.parent / "mono_mesh.ply"
        if maybe_mono.exists():
            return maybe_mono

    return None


def _load_mesh(mesh_path: Path, max_faces: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if o3d is None:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
        )
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
        )

    if len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(max_faces)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if colors.size == 0:
        colors = np.full((vertices.shape[0], 3), 0.75, dtype=np.float32)
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return vertices, faces, colors_u8


def _camera_centers(data: Optional[Dict]) -> np.ndarray:
    if data is None or "extrinsics" not in data:
        return np.zeros((0, 3), dtype=np.float32)
    centers = []
    for c2w in data["extrinsics"]:
        c2w_np = np.asarray(c2w, dtype=np.float32)
        if c2w_np.shape == (4, 4):
            centers.append(c2w_np[:3, 3])
    if not centers:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32)


def _compute_scene_bounds(points: np.ndarray, vertices: np.ndarray, cam_centers: np.ndarray) -> Tuple[np.ndarray, float]:
    all_xyz = []
    if points.size > 0:
        all_xyz.append(points)
    if vertices.size > 0:
        all_xyz.append(vertices)
    if cam_centers.size > 0:
        all_xyz.append(cam_centers)
    if not all_xyz:
        return np.zeros(3, dtype=np.float32), 1.0

    xyz = np.concatenate(all_xyz, axis=0)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    radius = float(np.linalg.norm(xyz_max - xyz_min) * 0.5)
    radius = max(radius, 1.0)
    return center.astype(np.float32), radius


def _build_frustum_lines(
    data: Optional[Dict],
    img_h: int,
    img_w: int,
    frustum_depth: float,
    max_cameras: int,
) -> np.ndarray:
    if data is None or "intrinsics" not in data or "extrinsics" not in data:
        return np.zeros((0, 3), dtype=np.float32)

    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]
    n = min(len(intrinsics), len(extrinsics))
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    cam_indices = np.linspace(0, n - 1, num=min(max_cameras, n), dtype=np.int64)
    segments = []
    for idx in cam_indices:
        K = np.asarray(intrinsics[int(idx)], dtype=np.float32)
        c2w = np.asarray(extrinsics[int(idx)], dtype=np.float32)
        if K.shape[0] < 3 or K.shape[1] < 3 or c2w.shape != (4, 4):
            continue

        fx = max(float(K[0, 0]), 1e-6)
        fy = max(float(K[1, 1]), 1e-6)
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        corners_cam = np.array(
            [
                [(0.0 - cx) / fx * frustum_depth, (0.0 - cy) / fy * frustum_depth, frustum_depth],
                [((img_w - 1.0) - cx) / fx * frustum_depth, (0.0 - cy) / fy * frustum_depth, frustum_depth],
                [((img_w - 1.0) - cx) / fx * frustum_depth, ((img_h - 1.0) - cy) / fy * frustum_depth, frustum_depth],
                [(0.0 - cx) / fx * frustum_depth, ((img_h - 1.0) - cy) / fy * frustum_depth, frustum_depth],
            ],
            dtype=np.float32,
        )

        R = c2w[:3, :3]
        t = c2w[:3, 3]
        corners_world = corners_cam @ R.T + t[None, :]
        origin = t.astype(np.float32)

        # pyramid edges
        edges = [
            (origin, corners_world[0]),
            (origin, corners_world[1]),
            (origin, corners_world[2]),
            (origin, corners_world[3]),
            (corners_world[0], corners_world[1]),
            (corners_world[1], corners_world[2]),
            (corners_world[2], corners_world[3]),
            (corners_world[3], corners_world[0]),
        ]
        for a, b in edges:
            segments.append(a)
            segments.append(b)

    if not segments:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32)


def _default_image_size(data: Optional[Dict]) -> Tuple[int, int]:
    if data is None or "color" not in data or not data["color"]:
        return 480, 640
    rgb0 = np.asarray(data["color"][0])
    if rgb0.ndim == 3:
        return int(rgb0.shape[0]), int(rgb0.shape[1])
    return 480, 640


def _safe_mtime_ns(path: Optional[Path]) -> int:
    if path is None:
        return 0
    try:
        if path.exists():
            return int(path.stat().st_mtime_ns)
    except Exception:
        return 0
    return 0


def _count_images_in_directory(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    count = 0
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1
    return count


def _infer_dataset_image_count(data: Optional[Dict]) -> int:
    if data is None or "image_paths" not in data:
        return 0
    image_paths = data.get("image_paths", [])
    if not image_paths:
        return 0
    first_parent = Path(str(image_paths[0])).resolve().parent
    return _count_images_in_directory(first_parent)


def _infer_image_dir(data: Optional[Dict]) -> str:
    if data is None or "image_paths" not in data:
        return ""
    image_paths = data.get("image_paths", [])
    if not image_paths:
        return ""
    return str(Path(str(image_paths[0])).resolve().parent)


def _alignment_counts(data: Optional[Dict]) -> Tuple[int, int]:
    if data is None:
        return 0, 0
    intrinsics = data.get("intrinsics", [])
    extrinsics = data.get("extrinsics", [])
    n_pairs = min(len(intrinsics), len(extrinsics))
    aligned = 0
    for i in range(n_pairs):
        K = np.asarray(intrinsics[i], dtype=np.float32)
        c2w = np.asarray(extrinsics[i], dtype=np.float32)
        if K.shape[0] < 3 or K.shape[1] < 3 or c2w.shape != (4, 4):
            continue
        if not np.isfinite(K).all() or not np.isfinite(c2w).all():
            continue
        aligned += 1
    used = len(data.get("image_paths", []))
    if used <= 0:
        used = n_pairs
    return aligned, used


def _run_label(run_dir: Path, output_root: Path) -> str:
    try:
        rel = run_dir.relative_to(output_root)
        return str(rel)
    except Exception:
        return str(run_dir)


def _compute_scene_minmax(points: np.ndarray, vertices: np.ndarray, cam_centers: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    all_xyz = []
    if points.size > 0:
        all_xyz.append(points)
    if vertices.size > 0:
        all_xyz.append(vertices)
    if cam_centers.size > 0:
        all_xyz.append(cam_centers)
    if not all_xyz:
        return None, None
    xyz = np.concatenate(all_xyz, axis=0)
    return xyz.min(axis=0).astype(np.float32), xyz.max(axis=0).astype(np.float32)


def _prepare_static_assets(target_dir: Path) -> None:
    src_dir = Path(__file__).resolve().parent / "sync_viewer_assets"
    if not src_dir.exists():
        raise FileNotFoundError(f"sync viewer assets dir not found: {src_dir}")

    dst_dir = target_dir / "_viewer_static"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("three.module.js", "OrbitControls.js", "TrackballControls.js"):
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing asset file: {src}")
        dst = dst_dir / name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)


def _build_single_run_payload(
    *,
    run_dir: Path,
    output_root: Path,
    data_pth_arg: str,
    pcd_path_arg: str,
    mesh_path_arg: str,
    prefer_saved_pcd: bool,
    point_stride: int,
    max_points: int,
    max_faces: int,
    max_cameras: int,
    frustum_ratio: float,
    rng: random.Random,
) -> Optional[Dict]:
    inferred_output_root = _infer_output_root_from_run(run_dir) or output_root

    data_pth = _resolve_data_pth(data_pth_arg, run_dir, inferred_output_root)
    data = _load_data_pth(data_pth)

    pcd_path = _resolve_pcd_path(pcd_path_arg, run_dir, inferred_output_root)
    points_reprojection = np.zeros((0, 3), dtype=np.float32)
    colors_reprojection = np.zeros((0, 3), dtype=np.uint8)
    points_saved = np.zeros((0, 3), dtype=np.float32)
    colors_saved = np.zeros((0, 3), dtype=np.uint8)

    if data is not None:
        points_reprojection, colors_reprojection = _build_pointcloud_from_data(data, point_stride, max_points, rng)
    if pcd_path.exists():
        points_saved, colors_saved = _load_pointcloud(pcd_path, max_points, rng)

    has_reprojection = points_reprojection.shape[0] > 0
    has_saved = points_saved.shape[0] > 0
    point_source = "none"
    points = np.zeros((0, 3), dtype=np.float32)
    pcd_colors = np.zeros((0, 3), dtype=np.uint8)

    if prefer_saved_pcd and has_saved:
        points, pcd_colors = points_saved, colors_saved
        point_source = "saved_pcd"
    elif has_reprojection:
        points, pcd_colors = points_reprojection, colors_reprojection
        point_source = "data_pth_reprojection"
        # If confidence masking made the reprojection too sparse, prefer saved VGGT pcd for visualization.
        sparse_threshold = max(4000, min(20000, max_points // 12))
        if points.shape[0] < sparse_threshold and has_saved and points_saved.shape[0] > points.shape[0]:
            points, pcd_colors = points_saved, colors_saved
            point_source = "saved_pcd_auto_sparse_fallback"
    elif has_saved:
        points, pcd_colors = points_saved, colors_saved
        point_source = "saved_pcd_fallback"

    if mesh_path_arg:
        mesh_final_path = Path(mesh_path_arg).resolve()
    else:
        mesh_final_path = _find_final_mesh_path(run_dir, data)
    mesh_raw_path = _find_raw_mesh_path(run_dir, data)

    if mesh_final_path is not None and mesh_raw_path is not None:
        if mesh_final_path.resolve() == mesh_raw_path.resolve():
            mesh_raw_path = None

    has_final = mesh_final_path is not None and mesh_final_path.exists()
    has_raw = mesh_raw_path is not None and mesh_raw_path.exists()
    if not has_final and not has_raw:
        print(f"[sync-viewer][WARN] skipping run (mesh missing): {run_dir}")
        return None

    if has_final:
        final_vertices, final_faces, final_colors = _load_mesh(mesh_final_path, max_faces)
    else:
        final_vertices = np.zeros((0, 3), dtype=np.float32)
        final_faces = np.zeros((0, 3), dtype=np.int32)
        final_colors = np.zeros((0, 3), dtype=np.uint8)

    if has_raw:
        raw_vertices, raw_faces, raw_colors = _load_mesh(mesh_raw_path, max_faces)
    else:
        raw_vertices = np.zeros((0, 3), dtype=np.float32)
        raw_faces = np.zeros((0, 3), dtype=np.int32)
        raw_colors = np.zeros((0, 3), dtype=np.uint8)

    mesh_stage = "none"
    try:
        parts = set((mesh_final_path or Path("")).parts)
        if "plane_plots" in parts:
            mesh_stage = "intermediate_plot"
        elif (mesh_final_path and mesh_final_path.name.endswith("_planar_mesh.ply")):
            mesh_stage = "final_planar_mesh"
        elif (mesh_final_path and mesh_final_path.name == "mono_mesh.ply"):
            mesh_stage = "mono_input_mesh"
    except Exception:
        mesh_stage = "unknown"

    raw_mesh_stage = "none"
    try:
        parts = set((mesh_raw_path or Path("")).parts)
        if "plane_plots" in parts:
            raw_mesh_stage = "intermediate_plot"
        elif (mesh_raw_path and mesh_raw_path.name.endswith("_planar_mesh.ply")):
            raw_mesh_stage = "final_planar_mesh"
        elif (mesh_raw_path and mesh_raw_path.name == "mono_mesh.ply"):
            raw_mesh_stage = "mono_input_mesh"
    except Exception:
        raw_mesh_stage = "unknown"
    cam_centers = _camera_centers(data)
    aligned_count, used_count = _alignment_counts(data)
    dataset_count = _infer_dataset_image_count(data)
    image_dir = _infer_image_dir(data)

    bounds_points = points
    if has_reprojection and has_saved:
        bounds_points = np.concatenate([points_reprojection, points_saved], axis=0)
    elif has_reprojection:
        bounds_points = points_reprojection
    elif has_saved:
        bounds_points = points_saved

    bounds_vertices = final_vertices if final_vertices.size > 0 else raw_vertices
    center, radius = _compute_scene_bounds(bounds_points, bounds_vertices, cam_centers)
    xyz_min, xyz_max = _compute_scene_minmax(bounds_points, bounds_vertices, cam_centers)
    if xyz_min is None or xyz_max is None:
        xyz_min = (center - radius).astype(np.float32)
        xyz_max = (center + radius).astype(np.float32)

    img_h, img_w = _default_image_size(data)
    frustum_depth = max(radius * frustum_ratio, 0.02)
    frustum_segments = _build_frustum_lines(
        data=data,
        img_h=img_h,
        img_w=img_w,
        frustum_depth=frustum_depth,
        max_cameras=max_cameras,
    )

    run_update_token = "-".join(
        [
            str(_safe_mtime_ns(run_dir)),
            str(_safe_mtime_ns(mesh_final_path)),
            str(_safe_mtime_ns(mesh_raw_path)),
            str(_safe_mtime_ns(data_pth)),
            str(_safe_mtime_ns(pcd_path)),
            str(int(points.shape[0])),
            str(int(points_reprojection.shape[0])),
            str(int(points_saved.shape[0])),
            str(int(final_vertices.shape[0])),
            str(int(raw_vertices.shape[0])),
        ]
    )
    point_size = max(0.004 * radius, 0.002)
    point_sources_available = []
    if has_reprojection:
        point_sources_available.append("reprojection")
    if has_saved:
        point_sources_available.append("saved_pcd")
    point_source_manifest = {
        "reprojection": {
            "point_count": int(points_reprojection.shape[0]),
            "url": "",
            "token": f"repr-{int(points_reprojection.shape[0])}-{str(_safe_mtime_ns(data_pth))}",
        },
        "saved_pcd": {
            "point_count": int(points_saved.shape[0]),
            "url": "",
            "token": f"saved-{int(points_saved.shape[0])}-{str(_safe_mtime_ns(pcd_path))}",
        },
    }
    point_source_blobs = {
        "reprojection": {
            "positions": points_reprojection,
            "colors": colors_reprojection,
            "point_size": point_size,
        },
        "saved_pcd": {
            "positions": points_saved,
            "colors": colors_saved,
            "point_size": point_size,
        },
    }

    run_payload = {
        "id": str(run_dir),
        "label": _run_label(run_dir, output_root),
        "pointcloud": {
            "positions": points.tolist(),
            "colors": pcd_colors.tolist(),
            "point_size": point_size,
        },
        # Source data is stored in external sidecar json files to keep the main payload light.
        "pointcloud_sources": point_source_manifest,
        "mesh_final": {
            "vertices": final_vertices.tolist(),
            "faces": final_faces.tolist(),
            "colors": final_colors.tolist(),
        },
        "mesh_raw": {
            "vertices": raw_vertices.tolist(),
            "faces": raw_faces.tolist(),
            "colors": raw_colors.tolist(),
        },
        "frustums": {
            "positions": frustum_segments.tolist(),
        },
        "meta": {
            "run_dir": str(run_dir),
            "output_root": str(inferred_output_root),
            "mesh_path": str(mesh_final_path) if has_final else "",
            "mesh_stage": mesh_stage,
            "mesh_raw_path": str(mesh_raw_path) if has_raw else "",
            "mesh_raw_stage": raw_mesh_stage,
            "data_pth": str(data_pth) if data_pth.exists() else "",
            "pcd_path": str(pcd_path) if pcd_path.exists() else "",
            "point_source": point_source,
            "point_count": int(points.shape[0]),
            "point_count_reprojection": int(points_reprojection.shape[0]),
            "point_count_saved_pcd": int(points_saved.shape[0]),
            "point_sources_available": point_sources_available,
            "mesh_vertex_count": int(final_vertices.shape[0]),
            "mesh_face_count": int(final_faces.shape[0]),
            "raw_mesh_vertex_count": int(raw_vertices.shape[0]),
            "raw_mesh_face_count": int(raw_faces.shape[0]),
            "frustum_segment_count": int(frustum_segments.shape[0] // 2),
            "aligned_image_count": int(aligned_count),
            "used_image_count": int(used_count),
            "dataset_image_count": int(dataset_count),
            "image_dir": image_dir,
            "run_update_token": run_update_token,
        },
    }
    # Backward-compat alias
    run_payload["mesh"] = run_payload["mesh_final"]
    return {
        "run": run_payload,
        "xyz_min": xyz_min,
        "xyz_max": xyz_max,
        "center": center,
        "radius": float(radius),
        "token": run_update_token,
        "point_source_blobs": point_source_blobs,
    }


def _write_point_source_sidecars(
    html_path: Path,
    payload: Dict,
    point_source_blobs_by_run: Dict[str, Dict[str, Dict[str, np.ndarray]]],
) -> None:
    runs = payload.get("runs", [])
    if not runs:
        return
    sidecar_dir = html_path.parent / "_viewer_point_sources"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    for idx, run in enumerate(runs):
        run_id = str(run.get("id", f"run-{idx}"))
        run_label = str(run.get("label", run_id))
        run_sources = run.get("pointcloud_sources", {})
        blobs = point_source_blobs_by_run.get(run_id, {})
        if not isinstance(run_sources, dict):
            continue

        for source_name in ("reprojection", "saved_pcd"):
            entry = run_sources.get(source_name, {})
            if not isinstance(entry, dict):
                continue
            blob = blobs.get(source_name)
            if not isinstance(blob, dict):
                continue
            positions = blob.get("positions")
            colors = blob.get("colors")
            if not isinstance(positions, np.ndarray) or positions.shape[0] <= 0:
                continue
            if not isinstance(colors, np.ndarray) or colors.shape[0] != positions.shape[0]:
                colors = np.full((positions.shape[0], 3), 192, dtype=np.uint8)

            token = str(entry.get("token", ""))
            file_stem = f"{idx:02d}_{_slugify(run_label, 40)}_{source_name}_{token[:16] or _run_uid(Path(run_id))}"
            sidecar_path = sidecar_dir / f"{file_stem}.json"
            sidecar_payload = {
                "positions": positions.tolist(),
                "colors": colors.tolist(),
                "point_size": float(blob.get("point_size", 0.002)),
                "point_count": int(positions.shape[0]),
                "source_name": source_name,
                "run_id": run_id,
                "token": token,
            }
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_payload, f, separators=(",", ":"))

            rel_url = f"./_viewer_point_sources/{sidecar_path.name}"
            entry["url"] = rel_url
            entry["bytes"] = int(sidecar_path.stat().st_size)


def _write_viewer_files(
    html_path: Path,
    payload: Dict,
    point_source_blobs_by_run: Optional[Dict[str, Dict[str, Dict[str, np.ndarray]]]] = None,
) -> Path:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_static_assets(html_path.parent)
    if point_source_blobs_by_run:
        _write_point_source_sidecars(html_path, payload, point_source_blobs_by_run)
    json_path = html_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PlanarSplatting Sync Dual Viewer</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #0f1115;
      color: #e5e7eb;
      font-family: "IBM Plex Sans", "Noto Sans KR", sans-serif;
    }}
    #app {{
      display: grid;
      grid-template-rows: auto 1fr;
      width: 100%;
      height: 100%;
    }}
    #header {{
      display: flex;
      gap: 18px;
      align-items: center;
      padding: 10px 14px;
      border-bottom: 1px solid #1f2430;
      background: #121722;
      font-size: 14px;
    }}
    #views {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      width: 100%;
      height: 100%;
      min-height: 0;
    }}
    .panel {{
      position: relative;
      min-height: 0;
      border-right: 1px solid #1f2430;
    }}
    .panel:last-child {{
      border-right: none;
    }}
    .label {{
      position: absolute;
      left: 10px;
      top: 8px;
      z-index: 10;
      padding: 4px 7px;
      border-radius: 6px;
      background: rgba(8, 10, 14, 0.7);
      border: 1px solid rgba(85, 101, 125, 0.45);
      font-size: 12px;
      letter-spacing: 0.01em;
      pointer-events: none;
    }}
    .hint {{
      opacity: 0.85;
      font-size: 12px;
    }}
    #data-info {{
      margin-left: auto;
      padding-left: 8px;
    }}
    #data-info > summary {{
      cursor: pointer;
      font-size: 12px;
      color: #c7d2e3;
      user-select: none;
      list-style: none;
      outline: none;
    }}
    #data-info > summary::-webkit-details-marker {{
      display: none;
    }}
    #data-info > summary::before {{
      content: "▸ ";
    }}
    #data-info[open] > summary::before {{
      content: "▾ ";
    }}
    #data-info-text {{
      margin: 6px 0 0;
      padding: 8px 10px;
      border: 1px solid #2a3447;
      border-radius: 8px;
      background: rgba(6, 9, 14, 0.92);
      color: #c8d4e8;
      font-size: 11px;
      line-height: 1.35;
      max-width: min(56vw, 760px);
      max-height: 34vh;
      overflow: auto;
      white-space: pre-wrap;
      font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
    }}
    #layers-panel {{
      padding-left: 8px;
    }}
    #layers-panel > summary {{
      cursor: pointer;
      font-size: 12px;
      color: #c7d2e3;
      user-select: none;
      list-style: none;
      outline: none;
    }}
    #layers-panel > summary::-webkit-details-marker {{
      display: none;
    }}
    #layers-panel > summary::before {{
      content: "▸ ";
    }}
    #layers-panel[open] > summary::before {{
      content: "▾ ";
    }}
    .layer-list {{
      margin: 6px 0 0;
      padding: 8px 10px;
      border: 1px solid #2a3447;
      border-radius: 8px;
      background: rgba(6, 9, 14, 0.92);
      min-width: 240px;
      max-width: 320px;
      font-size: 12px;
      line-height: 1.35;
    }}
    .layer-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
      user-select: none;
      cursor: pointer;
    }}
    .layer-row-static {{
      cursor: default;
    }}
    .layer-row input[type="checkbox"] {{
      margin: 0;
      accent-color: #f5a524;
    }}
    .layer-select {{
      margin-left: auto;
      border: 1px solid #38465e;
      background: #0f1725;
      color: #d5def1;
      border-radius: 6px;
      padding: 2px 6px;
      font-size: 11px;
      min-width: 120px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.3);
      flex: 0 0 auto;
    }}
    .sw-point {{ background: #1fa3c7; }}
    .sw-mesh {{ background: #de3c93; }}
    .sw-mesh-raw {{ background: #6aa8ff; }}
    .sw-frustum {{ background: #ffbf3f; }}
    .sw-grid {{ background: #6f7f96; }}
    .sw-axes {{ background: #7ecf87; }}
    .sw-sync {{ background: #f5a524; }}
    #runs-panel {{
      padding-left: 8px;
    }}
    #runs-panel > summary {{
      cursor: pointer;
      font-size: 12px;
      color: #c7d2e3;
      user-select: none;
      list-style: none;
      outline: none;
    }}
    #runs-panel > summary::-webkit-details-marker {{
      display: none;
    }}
    #runs-panel > summary::before {{
      content: "▸ ";
    }}
    #runs-panel[open] > summary::before {{
      content: "▾ ";
    }}
    .run-list {{
      margin: 6px 0 0;
      padding: 8px 10px;
      border: 1px solid #2a3447;
      border-radius: 8px;
      background: rgba(6, 9, 14, 0.92);
      min-width: 320px;
      max-width: min(42vw, 560px);
      max-height: 36vh;
      overflow: auto;
      font-size: 12px;
      line-height: 1.35;
    }}
    .run-actions {{
      display: flex;
      gap: 8px;
      margin-bottom: 6px;
    }}
    .run-btn {{
      border: 1px solid #3a4760;
      background: #141b28;
      color: #d2d9e7;
      font-size: 11px;
      border-radius: 6px;
      padding: 2px 7px;
      cursor: pointer;
    }}
    .run-btn:hover {{
      border-color: #4c5d7a;
      background: #1a2434;
    }}
    .run-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
      user-select: none;
      cursor: pointer;
    }}
    .run-row input[type="checkbox"] {{
      margin: 0;
      accent-color: #f5a524;
    }}
    .run-swatch {{
      width: 11px;
      height: 11px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.35);
      flex: 0 0 auto;
    }}
    .run-label {{
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
    }}
  </style>
</head>
<body>
  <div id="app">
    <div id="header">
      <strong>Sync Dual Viewer</strong>
      <span class="hint">Left: camera + point cloud</span>
      <span class="hint">Right: camera + mesh</span>
      <span class="hint">Rotate/Pan/Zoom one side -> the other side syncs</span>
      <span id="img-align" class="hint"></span>
      <details id="runs-panel" open>
        <summary>Runs</summary>
        <div id="run-list" class="run-list"></div>
      </details>
      <details id="data-info">
        <summary>Data Info</summary>
        <pre id="data-info-text"></pre>
      </details>
      <details id="layers-panel" open>
        <summary>Layers</summary>
        <div class="layer-list">
          <label class="layer-row"><input id="layer-pointcloud" type="checkbox" checked /><span class="swatch sw-point"></span><span>Point Cloud (Left)</span></label>
          <div class="layer-row layer-row-static"><span class="swatch sw-point"></span><span>Point Source</span><select id="point-source-mode" class="layer-select"><option value="auto" selected>Auto</option><option value="reprojection">Reprojection</option><option value="saved_pcd">Saved PCD</option></select></div>
          <label class="layer-row"><input id="layer-mesh" type="checkbox" checked /><span class="swatch sw-mesh"></span><span>Final Mesh (Right)</span></label>
          <label class="layer-row"><input id="layer-mesh-raw" type="checkbox" /><span class="swatch sw-mesh-raw"></span><span>Raw Mesh (Right)</span></label>
          <label class="layer-row"><input id="layer-frustum" type="checkbox" checked /><span class="swatch sw-frustum"></span><span>Camera Frustums</span></label>
          <label class="layer-row"><input id="layer-grid" type="checkbox" checked /><span class="swatch sw-grid"></span><span>Grid</span></label>
          <label class="layer-row"><input id="layer-axes" type="checkbox" checked /><span class="swatch sw-axes"></span><span>Axes</span></label>
          <label class="layer-row"><input id="layer-sync" type="checkbox" checked /><span class="swatch sw-sync"></span><span>Sync Camera Motion</span></label>
        </div>
      </details>
    </div>
    <div id="views">
      <div id="left" class="panel"><div class="label">Camera + Point Cloud</div></div>
      <div id="right" class="panel"><div class="label">Camera + Mesh</div></div>
    </div>
  </div>

  <script type="module">
    import * as THREE from "./_viewer_static/three.module.js";
    import {{ TrackballControls }} from "./_viewer_static/TrackballControls.js";

    try {{
      async function loadPayload(cacheBust = false) {{
        const suffix = cacheBust ? `?t=${{Date.now()}}` : "";
        const response = await fetch(`./{json_path.name}${{suffix}}`, {{ cache: "no-store" }});
        if (!response.ok) {{
          throw new Error(`failed to load JSON: ${{response.status}} ${{response.statusText}}`);
        }}
        return await response.json();
      }}

      let payload = await loadPayload(false);
      const runState = new Map();
      const pointSourceCache = new Map();
      const pointSourceLoadState = new Map();
      let pointSourceRequestSeq = 0;

      function extractRuns(data) {{
        if (data && Array.isArray(data.runs) && data.runs.length > 0) {{
          return data.runs;
        }}
        // Backward-compat for old json schema
        const fallbackMeta = data.meta || {{}};
        const fallbackPointcloud = data.pointcloud || {{ positions: [], colors: [], point_size: 0.002 }};
        const fallbackSourceName = String(fallbackMeta.point_source || "reprojection");
        return [{{
          id: (data.meta && data.meta.run_dir) || "run-0",
          label: (data.meta && data.meta.run_dir) || "run-0",
          color: [31, 163, 199],
          pointcloud: fallbackPointcloud,
          pointcloud_sources: {{
            reprojection: {{
              point_count: fallbackSourceName === "reprojection" ? Number((fallbackPointcloud.positions || []).length) : 0,
              url: "",
              token: "legacy",
            }},
            saved_pcd: {{
              point_count: fallbackSourceName.startsWith("saved_pcd") ? Number((fallbackPointcloud.positions || []).length) : 0,
              url: "",
              token: "legacy",
            }},
          }},
          mesh_final: data.mesh || {{ vertices: [], faces: [], colors: [] }},
          mesh_raw: {{ vertices: [], faces: [], colors: [] }},
          mesh: data.mesh || {{ vertices: [], faces: [], colors: [] }},
          frustums: data.frustums || {{ positions: [] }},
          meta: fallbackMeta,
        }}];
      }}

      let runs = extractRuns(payload);
      let pointSourceMode = "auto";

      function runKey(run, idx) {{
        if (run && run.id !== undefined && run.id !== null) return String(run.id);
        return `run-${{idx}}`;
      }}

      function selectedRuns() {{
        const out = [];
        for (let i = 0; i < runs.length; i++) {{
          const run = runs[i];
          const key = runKey(run, i);
          if (runState.get(key) !== false) out.push(run);
        }}
        return out;
      }}

      function syncRunState(defaultEnabled = true) {{
        const next = new Map();
        for (let i = 0; i < runs.length; i++) {{
          const key = runKey(runs[i], i);
          if (runState.has(key)) {{
            next.set(key, !!runState.get(key));
          }} else {{
            next.set(key, defaultEnabled);
          }}
        }}
        runState.clear();
        for (const [k, v] of next.entries()) runState.set(k, v);
      }}

      function runColorCss(run) {{
        const c = (run && Array.isArray(run.color) && run.color.length === 3) ? run.color : [180, 196, 219];
        const r = Math.max(0, Math.min(255, Number(c[0] || 0)));
        const g = Math.max(0, Math.min(255, Number(c[1] || 0)));
        const b = Math.max(0, Math.min(255, Number(c[2] || 0)));
        return `rgb(${{r}}, ${{g}}, ${{b}})`;
      }}

      function runColorHex(run) {{
        const c = (run && Array.isArray(run.color) && run.color.length === 3) ? run.color : [255, 201, 75];
        const r = Math.max(0, Math.min(255, Number(c[0] || 0))) << 16;
        const g = Math.max(0, Math.min(255, Number(c[1] || 0))) << 8;
        const b = Math.max(0, Math.min(255, Number(c[2] || 0)));
        return (r | g | b);
      }}

      function updateHeaderMetrics() {{
        const alignEl = document.getElementById("img-align");
        if (!alignEl) return;
        const selected = selectedRuns();
        if (selected.length === 0) {{
          alignEl.textContent = `Runs: 0 / ${{runs.length}} (select at least 1)`;
          return;
        }}
        if (selected.length === 1) {{
          const meta = selected[0].meta || {{}};
          const aligned = Number(meta.aligned_image_count || 0);
          const used = Number(meta.used_image_count || 0);
          const total = Number(meta.dataset_image_count || 0);
          if (total > 0) {{
            const pct = Math.max(0, Math.min(100, (aligned / Math.max(total, 1)) * 100));
            alignEl.textContent = `Aligned images: ${{aligned}} / ${{total}} (${{pct.toFixed(1)}}%)`;
          }} else if (used > 0) {{
            alignEl.textContent = `Aligned images: ${{aligned}} / ${{used}} (used)`;
          }} else {{
            alignEl.textContent = "Aligned images: unavailable";
          }}
          return;
        }}
        let alignedSum = 0;
        let denomSum = 0;
        for (const run of selected) {{
          const meta = run.meta || {{}};
          const aligned = Number(meta.aligned_image_count || 0);
          const used = Number(meta.used_image_count || 0);
          const total = Number(meta.dataset_image_count || 0);
          alignedSum += aligned;
          denomSum += (total > 0 ? total : used);
        }}
        if (denomSum > 0) {{
          const pct = Math.max(0, Math.min(100, (alignedSum / Math.max(denomSum, 1)) * 100));
          alignEl.textContent = `Runs: ${{selected.length}} / ${{runs.length}}, aligned(sum): ${{alignedSum}} / ${{denomSum}} (${{pct.toFixed(1)}}%)`;
        }} else {{
          alignEl.textContent = `Runs: ${{selected.length}} / ${{runs.length}}`;
        }}
      }}

      function basename(pathStr) {{
        if (!pathStr) return "-";
        const s = String(pathStr).replace(/\\\\/g, "/");
        const idx = s.lastIndexOf("/");
        return idx >= 0 ? s.slice(idx + 1) : s;
      }}

      function formatInt(n) {{
        return Number(n || 0).toLocaleString("en-US");
      }}

      function summarizeAlignment(meta) {{
        const aligned = Number(meta.aligned_image_count || 0);
        const used = Number(meta.used_image_count || 0);
        const total = Number(meta.dataset_image_count || 0);
        const denom = total > 0 ? total : used;
        if (denom > 0) {{
          return total > 0 ? `${{aligned}} / ${{denom}} (dataset)` : `${{aligned}} / ${{denom}} (used)`;
        }}
        return `${{aligned}}`;
      }}

      function pointCloudPointCount(pointcloudData) {{
        if (!pointcloudData || !Array.isArray(pointcloudData.positions)) return 0;
        return pointcloudData.positions.length;
      }}

      function runIndex(run) {{
        return runs.indexOf(run);
      }}

      function runKeyByObject(run) {{
        const idx = runIndex(run);
        return runKey(run, idx >= 0 ? idx : 0);
      }}

      function sourceCacheKey(runId, sourceName) {{
        return `${{runId}}::${{sourceName}}`;
      }}

      function sourceEntry(run, sourceName) {{
        const sources = run && run.pointcloud_sources ? run.pointcloud_sources : {{}};
        const entry = sources[sourceName];
        if (!entry || typeof entry !== "object") return null;
        return entry;
      }}

      function sourceEntryPointCount(entry) {{
        if (!entry || typeof entry !== "object") return 0;
        if (Array.isArray(entry.positions)) return entry.positions.length;
        return Number(entry.point_count || 0);
      }}

      function normalizePointCloudBlob(blob) {{
        if (!blob || !Array.isArray(blob.positions)) {{
          throw new Error("invalid point source payload: positions missing");
        }}
        const positions = blob.positions;
        const colors = Array.isArray(blob.colors) ? blob.colors : [];
        const pointSize = Math.max(Number(blob.point_size || 0.002), 0.001);
        return {{
          positions,
          colors,
          point_size: pointSize,
        }};
      }}

      function availablePointSources(run) {{
        const meta = run && run.meta ? run.meta : {{}};
        const listed = meta.point_sources_available;
        if (Array.isArray(listed) && listed.length > 0) {{
          return listed.map((x) => String(x));
        }}
        const out = [];
        const repr = sourceEntry(run, "reprojection");
        const saved = sourceEntry(run, "saved_pcd");
        if (sourceEntryPointCount(repr) > 0) out.push("reprojection");
        if (sourceEntryPointCount(saved) > 0) out.push("saved_pcd");
        return out;
      }}

      function selectedPointCloudData(run) {{
        const fallback = run && run.pointcloud ? run.pointcloud : {{ positions: [], colors: [], point_size: 0.002 }};
        if (!run) return fallback;
        const rk = runKeyByObject(run);
        const state = pointSourceLoadState.get(rk);
        if (state && state.pointcloud) {{
          return state.pointcloud;
        }}
        if (pointSourceMode === "auto") {{
          return fallback;
        }}
        const entry = sourceEntry(run, pointSourceMode);
        if (entry && Array.isArray(entry.positions) && entry.positions.length > 0) {{
          return normalizePointCloudBlob(entry);
        }}
        const cacheKey = sourceCacheKey(rk, pointSourceMode);
        if (pointSourceCache.has(cacheKey)) {{
          return pointSourceCache.get(cacheKey);
        }}
        return fallback;
      }}

      async function fetchPointSourceForRun(run, idx, requestSeq) {{
        const rk = runKey(run, idx);
        const fallback = run && run.pointcloud ? run.pointcloud : {{ positions: [], colors: [], point_size: 0.002 }};
        const defaultName = String((run && run.meta && run.meta.point_source) || "none");
        if (pointSourceMode === "auto") {{
          return {{
            runKey: rk,
            pointcloud: fallback,
            activeSource: defaultName,
            requestedSource: "auto",
            status: "default",
            url: "",
          }};
        }}

        const requestedSource = pointSourceMode;
        const entry = sourceEntry(run, requestedSource);
        if (!entry || sourceEntryPointCount(entry) <= 0) {{
          return {{
            runKey: rk,
            pointcloud: fallback,
            activeSource: `${{defaultName}} (fallback)`,
            requestedSource,
            status: "unavailable",
            url: "",
          }};
        }}

        if (Array.isArray(entry.positions) && entry.positions.length > 0) {{
          const embedded = normalizePointCloudBlob(entry);
          return {{
            runKey: rk,
            pointcloud: embedded,
            activeSource: requestedSource,
            requestedSource,
            status: "embedded",
            url: "",
          }};
        }}

        const sourceUrl = String(entry.url || "");
        if (!sourceUrl) {{
          return {{
            runKey: rk,
            pointcloud: fallback,
            activeSource: `${{defaultName}} (fallback)`,
            requestedSource,
            status: "missing_url",
            url: "",
          }};
        }}

        const ckey = sourceCacheKey(rk, requestedSource);
        if (pointSourceCache.has(ckey)) {{
          return {{
            runKey: rk,
            pointcloud: pointSourceCache.get(ckey),
            activeSource: requestedSource,
            requestedSource,
            status: "cached",
            url: sourceUrl,
          }};
        }}

        try {{
          const response = await fetch(sourceUrl, {{ cache: "no-store" }});
          if (!response.ok) {{
            throw new Error(`HTTP ${{response.status}}`);
          }}
          const raw = await response.json();
          if (requestSeq !== pointSourceRequestSeq) {{
            return null;
          }}
          const blob = normalizePointCloudBlob(raw);
          pointSourceCache.set(ckey, blob);
          return {{
            runKey: rk,
            pointcloud: blob,
            activeSource: requestedSource,
            requestedSource,
            status: "fetched",
            url: sourceUrl,
          }};
        }} catch (err) {{
          console.warn("[sync-viewer] point source fetch failed:", requestedSource, sourceUrl, err);
          return {{
            runKey: rk,
            pointcloud: fallback,
            activeSource: `${{defaultName}} (fallback)`,
            requestedSource,
            status: "fetch_failed",
            url: sourceUrl,
          }};
        }}
      }}

      function activePointSourceState(run) {{
        const rk = runKeyByObject(run);
        return pointSourceLoadState.get(rk) || null;
      }}

      function selectedPointSourceName(run) {{
        const state = activePointSourceState(run);
        if (state && state.activeSource) {{
          return String(state.activeSource);
        }}
        const meta = run && run.meta ? run.meta : {{}};
        const fallbackName = String(meta.point_source || "none");
        if (pointSourceMode === "reprojection") {{
          const entry = sourceEntry(run, "reprojection");
          return sourceEntryPointCount(entry) > 0 ? "reprojection (pending)" : `${{fallbackName}} (fallback)`;
        }}
        if (pointSourceMode === "saved_pcd") {{
          const entry = sourceEntry(run, "saved_pcd");
          return sourceEntryPointCount(entry) > 0 ? "saved_pcd (pending)" : `${{fallbackName}} (fallback)`;
        }}
        return fallbackName;
      }}

      function activePointCountForRun(run) {{
        const state = activePointSourceState(run);
        if (state && Number.isFinite(state.pointCount)) {{
          return Number(state.pointCount);
        }}
        return pointCloudPointCount(selectedPointCloudData(run));
      }}

      function updateDataInfoPanel() {{
        const selected = selectedRuns();
        const lines = [];
        lines.push(`selected_runs: ${{selected.length}} / ${{runs.length}}`);
        const reqCompare = Number((payload.meta && payload.meta.compare_runs_requested) || (payload.meta && payload.meta.compare_runs) || runs.length);
        const effCompare = Number((payload.meta && payload.meta.compare_runs) || runs.length);
        lines.push(`compare_runs: requested=${{reqCompare}}, effective=${{effCompare}}`);
        lines.push(`point_source_mode: ${{pointSourceMode}}`);
        if (selected.length === 0) {{
          lines.push("no run selected");
        }} else if (selected.length === 1) {{
          const run = selected[0];
          const meta = run.meta || {{}};
          const sourceState = activePointSourceState(run);
          const sourceList = availablePointSources(run);
          lines.push(`run_label: ${{run.label || "-"}}`);
          lines.push(`run_dir: ${{meta.run_dir || "-"}}`);
          lines.push(`image_dir: ${{meta.image_dir || "-"}}`);
          lines.push(`pointcloud_source_default: ${{meta.point_source || "-"}}`);
          lines.push(`pointcloud_source_active: ${{selectedPointSourceName(run)}}`);
          if (sourceState) {{
            lines.push(`pointcloud_source_status: ${{sourceState.status || "-"}}`);
            lines.push(`pointcloud_source_url: ${{sourceState.url || "-"}}`);
          }}
          lines.push(`pointcloud_source_available: ${{sourceList.length > 0 ? sourceList.join(",") : "-"}}`);
          lines.push(`mesh_final_file: ${{meta.mesh_path || "-"}}`);
          lines.push(`mesh_final_stage: ${{meta.mesh_stage || "-"}}`);
          lines.push(`mesh_raw_file: ${{meta.mesh_raw_path || "-"}}`);
          lines.push(`mesh_raw_stage: ${{meta.mesh_raw_stage || "-"}}`);
          lines.push(`data_pth: ${{meta.data_pth || "-"}}`);
          lines.push(`pcd_path: ${{meta.pcd_path || "-"}}`);
          lines.push(`points(active): ${{formatInt(activePointCountForRun(run))}}`);
          lines.push(`points(default): ${{formatInt(meta.point_count)}}`);
          lines.push(`points(reprojection): ${{formatInt(meta.point_count_reprojection)}}`);
          lines.push(`points(saved_pcd): ${{formatInt(meta.point_count_saved_pcd)}}`);
          lines.push(`final_mesh: v=${{formatInt(meta.mesh_vertex_count)}}, f=${{formatInt(meta.mesh_face_count)}}`);
          lines.push(`raw_mesh: v=${{formatInt(meta.raw_mesh_vertex_count)}}, f=${{formatInt(meta.raw_mesh_face_count)}}`);
          lines.push(`frustum_lines: ${{formatInt(meta.frustum_segment_count)}}`);
          lines.push(`aligned_images: ${{summarizeAlignment(meta)}}`);
          lines.push(`mesh_name: ${{basename(meta.mesh_path)}}`);
        }} else {{
          let pointSum = 0;
          let meshVSum = 0;
          let meshFSum = 0;
          let rawMeshVSum = 0;
          let rawMeshFSum = 0;
          let alignedSum = 0;
          let denomSum = 0;
          for (const run of selected) {{
            const meta = run.meta || {{}};
            pointSum += Number(activePointCountForRun(run));
            meshVSum += Number(meta.mesh_vertex_count || 0);
            meshFSum += Number(meta.mesh_face_count || 0);
            rawMeshVSum += Number(meta.raw_mesh_vertex_count || 0);
            rawMeshFSum += Number(meta.raw_mesh_face_count || 0);
            alignedSum += Number(meta.aligned_image_count || 0);
            const used = Number(meta.used_image_count || 0);
            const total = Number(meta.dataset_image_count || 0);
            denomSum += (total > 0 ? total : used);
          }}
          lines.push(`sum_points(active): ${{formatInt(pointSum)}}`);
          lines.push(`sum_final_mesh: v=${{formatInt(meshVSum)}}, f=${{formatInt(meshFSum)}}`);
          lines.push(`sum_raw_mesh: v=${{formatInt(rawMeshVSum)}}, f=${{formatInt(rawMeshFSum)}}`);
          lines.push("note: when multiple runs are selected, absolute pose/scale can differ between runs.");
          if (denomSum > 0) {{
            lines.push(`sum_aligned_images: ${{alignedSum}} / ${{denomSum}}`);
          }} else {{
            lines.push(`sum_aligned_images: ${{alignedSum}}`);
          }}
          lines.push("selected_labels:");
          for (const run of selected) {{
            lines.push(`- ${{run.label || run.id || "-"}}`);
          }}
        }}
        const infoEl = document.getElementById("data-info-text");
        if (infoEl) infoEl.textContent = lines.join("\\n");
      }}

      function buildRunPanel() {{
        const listEl = document.getElementById("run-list");
        if (!listEl) return;
        listEl.innerHTML = "";

        const actions = document.createElement("div");
        actions.className = "run-actions";
        const btnAll = document.createElement("button");
        btnAll.type = "button";
        btnAll.className = "run-btn";
        btnAll.textContent = "All";
        btnAll.addEventListener("click", () => {{
          for (let i = 0; i < runs.length; i++) runState.set(runKey(runs[i], i), true);
          buildRunPanel();
          applyLayerVisibility();
          updateHeaderMetrics();
          updateDataInfoPanel();
        }});
        const btnNone = document.createElement("button");
        btnNone.type = "button";
        btnNone.className = "run-btn";
        btnNone.textContent = "None";
        btnNone.addEventListener("click", () => {{
          for (let i = 0; i < runs.length; i++) runState.set(runKey(runs[i], i), false);
          buildRunPanel();
          applyLayerVisibility();
          updateHeaderMetrics();
          updateDataInfoPanel();
        }});
        actions.appendChild(btnAll);
        actions.appendChild(btnNone);
        listEl.appendChild(actions);

        for (let i = 0; i < runs.length; i++) {{
          const run = runs[i];
          const key = runKey(run, i);

          const row = document.createElement("label");
          row.className = "run-row";

          const input = document.createElement("input");
          input.type = "checkbox";
          input.checked = runState.get(key) !== false;
          input.addEventListener("change", () => {{
            runState.set(key, !!input.checked);
            applyLayerVisibility();
            updateHeaderMetrics();
            updateDataInfoPanel();
          }});

          const sw = document.createElement("span");
          sw.className = "run-swatch";
          sw.style.background = runColorCss(run);

          const label = document.createElement("span");
          label.className = "run-label";
          label.textContent = run.label || key;

          row.appendChild(input);
          row.appendChild(sw);
          row.appendChild(label);
          listEl.appendChild(row);
        }}
      }}

      syncRunState(false);
      if (runs.length > 0 && selectedRuns().length === 0) {{
        runState.set(runKey(runs[0], 0), true);
      }}
      buildRunPanel();
      updateHeaderMetrics();
      updateDataInfoPanel();

      function flattenVec3(list) {{
        const out = new Float32Array(list.length * 3);
        for (let i = 0; i < list.length; i++) {{
          const v = list[i];
          out[i * 3 + 0] = v[0];
          out[i * 3 + 1] = v[1];
          out[i * 3 + 2] = v[2];
        }}
        return out;
      }}

      function flattenIndices(list) {{
        const out = new Uint32Array(list.length * 3);
        for (let i = 0; i < list.length; i++) {{
          const f = list[i];
          out[i * 3 + 0] = f[0];
          out[i * 3 + 1] = f[1];
          out[i * 3 + 2] = f[2];
        }}
        return out;
      }}

    function makeRenderer(container) {{
      const renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      container.appendChild(renderer.domElement);
      return renderer;
    }}

    function makeBaseScene(container, initCam, controlElement) {{
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0f1115);
      const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, 10000.0);
      camera.position.fromArray(initCam.position);
      camera.up.fromArray(initCam.up);

      const controls = new TrackballControls(camera, controlElement);
      controls.target.fromArray(initCam.target);
      controls.rotateSpeed = 4.0;
      controls.zoomSpeed = 1.5;
      controls.panSpeed = 1.0;
      controls.dynamicDampingFactor = 0.12;
      controls.noRoll = false;
      controls.update();

      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const dir = new THREE.DirectionalLight(0xffffff, 0.75);
      dir.position.set(1.0, 1.2, 1.6);
      scene.add(dir);
      const grid = new THREE.GridHelper(payload.scene.grid_size, 18, 0x3c465c, 0x273042);
      const axes = new THREE.AxesHelper(payload.scene.axes_size);
      scene.add(grid);
      scene.add(axes);
      return {{ scene, camera, controls, grid, axes }};
    }}

    function buildPointCloudObject(pointcloudData) {{
      const pos = pointcloudData && pointcloudData.positions;
      const col = pointcloudData && pointcloudData.colors;
      if (!pos || pos.length === 0) return null;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(flattenVec3(pos), 3));
      const colors = new Float32Array(pos.length * 3);
      const hasValidColor = Array.isArray(col) && col.length === pos.length;
      for (let i = 0; i < pos.length; i++) {{
        const c = hasValidColor ? col[i] : [192, 192, 192];
        colors[i * 3 + 0] = Number(c[0] || 192) / 255.0;
        colors[i * 3 + 1] = Number(c[1] || 192) / 255.0;
        colors[i * 3 + 2] = Number(c[2] || 192) / 255.0;
      }}
      g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      const pointSize = Math.max(Number((pointcloudData && pointcloudData.point_size) || 0.002), 0.001);
      const m = new THREE.PointsMaterial({{ size: pointSize, vertexColors: true }});
      return new THREE.Points(g, m);
    }}

    function buildMeshObject(meshData, style = "final") {{
      const v = meshData && meshData.vertices;
      const f = meshData && meshData.faces;
      if (!v || v.length === 0 || !f || f.length === 0) return null;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(flattenVec3(v), 3));
      g.setIndex(new THREE.BufferAttribute(flattenIndices(f), 1));
      const col = meshData && meshData.colors;
      if (col && col.length === v.length) {{
        const colors = new Float32Array(col.length * 3);
        for (let i = 0; i < col.length; i++) {{
          colors[i * 3 + 0] = col[i][0] / 255.0;
          colors[i * 3 + 1] = col[i][1] / 255.0;
          colors[i * 3 + 2] = col[i][2] / 255.0;
        }}
        g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      }}
      g.computeVertexNormals();
      const isRaw = (style === "raw");
      const m = new THREE.MeshStandardMaterial({{
        color: isRaw ? 0x6aa8ff : 0xc9d2e0,
        side: THREE.DoubleSide,
        roughness: isRaw ? 1.0 : 0.92,
        metalness: 0.02,
        vertexColors: !isRaw && !!g.getAttribute("color"),
        transparent: isRaw,
        opacity: isRaw ? 0.35 : 1.0,
        wireframe: isRaw,
      }});
      return new THREE.Mesh(g, m);
    }}

    function buildFrustumObject(data, colorHex = 0xffc94b) {{
      const seg = data.frustums && data.frustums.positions;
      if (!seg || seg.length === 0) return null;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(flattenVec3(seg), 3));
      const m = new THREE.LineBasicMaterial({{ color: colorHex, linewidth: 1 }});
      return new THREE.LineSegments(g, m);
    }}

    function disposeObject3D(obj) {{
      if (!obj) return;
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {{
        if (Array.isArray(obj.material)) {{
          for (const mat of obj.material) {{
            if (mat && mat.dispose) mat.dispose();
          }}
        }} else if (obj.material.dispose) {{
          obj.material.dispose();
        }}
      }}
    }}

    function replaceObject(scene, prevObj, nextObj) {{
      if (prevObj) {{
        scene.remove(prevObj);
        disposeObject3D(prevObj);
      }}
      if (nextObj) scene.add(nextObj);
      return nextObj;
    }}

    function setupPanel(containerId) {{
      const container = document.getElementById(containerId);
      const renderer = makeRenderer(container);
      const {{ scene, camera, controls, grid, axes }} = makeBaseScene(container, payload.default_camera, renderer.domElement);
      return {{ container, renderer, scene, camera, controls, grid, axes }};
    }}

    const left = setupPanel("left");
    const right = setupPanel("right");
    const runObjects = new Map();
    const layerState = {{
      pointcloud: true,
      mesh: true,
      meshRaw: false,
      frustums: true,
      grid: true,
      axes: true,
      sync: true,
    }};

    function checkboxValue(id, fallback = true) {{
      const el = document.getElementById(id);
      if (!el) return fallback;
      return !!el.checked;
    }}

    function readLayerState() {{
      layerState.pointcloud = checkboxValue("layer-pointcloud", true);
      layerState.mesh = checkboxValue("layer-mesh", true);
      layerState.meshRaw = checkboxValue("layer-mesh-raw", false);
      layerState.frustums = checkboxValue("layer-frustum", true);
      layerState.grid = checkboxValue("layer-grid", true);
      layerState.axes = checkboxValue("layer-axes", true);
      const sourceEl = document.getElementById("point-source-mode");
      if (sourceEl && typeof sourceEl.value === "string") {{
        const mode = sourceEl.value;
        pointSourceMode = (mode === "reprojection" || mode === "saved_pcd") ? mode : "auto";
      }} else {{
        pointSourceMode = "auto";
      }}
      const prevSync = layerState.sync;
      layerState.sync = checkboxValue("layer-sync", true);
      if (prevSync && !layerState.sync) {{
        clearActivePanel();
      }}
    }}

    function applyLayerVisibility() {{
      for (let i = 0; i < runs.length; i++) {{
        const key = runKey(runs[i], i);
        const visibleByRun = runState.get(key) !== false;
        const objs = runObjects.get(key);
        if (!objs) continue;
        if (objs.leftPointObj) objs.leftPointObj.visible = layerState.pointcloud && visibleByRun;
        if (objs.rightFinalMeshObj) objs.rightFinalMeshObj.visible = layerState.mesh && visibleByRun;
        if (objs.rightRawMeshObj) objs.rightRawMeshObj.visible = layerState.meshRaw && visibleByRun;
        if (objs.leftFrustumObj) objs.leftFrustumObj.visible = layerState.frustums && visibleByRun;
        if (objs.rightFrustumObj) objs.rightFrustumObj.visible = layerState.frustums && visibleByRun;
      }}
      if (left.grid) left.grid.visible = layerState.grid;
      if (right.grid) right.grid.visible = layerState.grid;
      if (left.axes) left.axes.visible = layerState.axes;
      if (right.axes) right.axes.visible = layerState.axes;
    }}

    function bindLayerControls() {{
      for (const id of ["layer-pointcloud", "layer-mesh", "layer-mesh-raw", "layer-frustum", "layer-grid", "layer-axes", "layer-sync"]) {{
        const el = document.getElementById(id);
        if (!el) continue;
        el.addEventListener("change", () => {{
          readLayerState();
          applyLayerVisibility();
        }});
      }}
      const sourceEl = document.getElementById("point-source-mode");
      if (sourceEl) {{
        sourceEl.addEventListener("change", () => {{
          readLayerState();
          updateDataInfoPanel();
          void refreshPointCloudObjects();
        }});
      }}
      readLayerState();
      applyLayerVisibility();
    }}

    async function refreshPointCloudObjects() {{
      const requestSeq = ++pointSourceRequestSeq;
      const tasks = [];
      for (let i = 0; i < runs.length; i++) {{
        tasks.push(fetchPointSourceForRun(runs[i], i, requestSeq));
      }}
      const results = await Promise.all(tasks);
      if (requestSeq !== pointSourceRequestSeq) {{
        return;
      }}
      for (let i = 0; i < results.length; i++) {{
        const result = results[i];
        if (!result) continue;
        const run = runs[i];
        const key = result.runKey || runKey(run, i);
        const pointcloudData = result.pointcloud || (run && run.pointcloud) || {{ positions: [], colors: [], point_size: 0.002 }};
        pointSourceLoadState.set(key, {{
          requestedSource: String(result.requestedSource || "auto"),
          activeSource: String(result.activeSource || "none"),
          status: String(result.status || "unknown"),
          url: String(result.url || ""),
          pointCount: Number(pointCloudPointCount(pointcloudData)),
          pointcloud: pointcloudData,
        }});

        const objs = runObjects.get(key);
        if (!objs) continue;
        const nextPointObj = buildPointCloudObject(pointcloudData);
        objs.leftPointObj = replaceObject(left.scene, objs.leftPointObj, nextPointObj);
      }}
      applyLayerVisibility();
      updateDataInfoPanel();
    }}

    function refreshSceneObjects() {{
      pointSourceLoadState.clear();
      for (const objs of runObjects.values()) {{
        if (objs.leftPointObj) {{
          left.scene.remove(objs.leftPointObj);
          disposeObject3D(objs.leftPointObj);
        }}
        if (objs.rightFinalMeshObj) {{
          right.scene.remove(objs.rightFinalMeshObj);
          disposeObject3D(objs.rightFinalMeshObj);
        }}
        if (objs.rightRawMeshObj) {{
          right.scene.remove(objs.rightRawMeshObj);
          disposeObject3D(objs.rightRawMeshObj);
        }}
        if (objs.leftFrustumObj) {{
          left.scene.remove(objs.leftFrustumObj);
          disposeObject3D(objs.leftFrustumObj);
        }}
        if (objs.rightFrustumObj) {{
          right.scene.remove(objs.rightFrustumObj);
          disposeObject3D(objs.rightFrustumObj);
        }}
      }}
      runObjects.clear();

      for (let i = 0; i < runs.length; i++) {{
        const run = runs[i];
        const key = runKey(run, i);
        const leftPointObj = buildPointCloudObject(run.pointcloud || {{ positions: [], colors: [], point_size: 0.002 }});
        const finalMeshData = run.mesh_final || run.mesh || null;
        const rawMeshData = run.mesh_raw || null;
        const rightFinalMeshObj = buildMeshObject(finalMeshData, "final");
        const rightRawMeshObj = buildMeshObject(rawMeshData, "raw");
        const leftFrustumObj = buildFrustumObject(run, runColorHex(run));
        const rightFrustumObj = buildFrustumObject(run, runColorHex(run));
        if (leftPointObj) left.scene.add(leftPointObj);
        if (rightFinalMeshObj) right.scene.add(rightFinalMeshObj);
        if (rightRawMeshObj) right.scene.add(rightRawMeshObj);
        if (leftFrustumObj) left.scene.add(leftFrustumObj);
        if (rightFrustumObj) right.scene.add(rightFrustumObj);
        runObjects.set(key, {{
          leftPointObj,
          rightFinalMeshObj,
          rightRawMeshObj,
          leftFrustumObj,
          rightFrustumObj,
        }});
      }}
      applyLayerVisibility();
      void refreshPointCloudObjects();
    }}
    refreshSceneObjects();
    bindLayerControls();

    let isSyncing = false;
    let activePanel = null; // "left" | "right" | null
    let masterPanel = "left";

    function setActivePanel(name) {{
      activePanel = name;
      masterPanel = name;
      left.controls.enabled = (name === "left");
      right.controls.enabled = (name === "right");
    }}

    function clearActivePanel() {{
      activePanel = null;
      left.controls.enabled = true;
      right.controls.enabled = true;
    }}

    function copyPose(src, dst) {{
      isSyncing = true;
      dst.camera.position.copy(src.camera.position);
      dst.camera.quaternion.copy(src.camera.quaternion);
      dst.camera.up.copy(src.camera.up);
      dst.controls.target.copy(src.controls.target);
      dst.camera.updateMatrixWorld(true);
      isSyncing = false;
    }}

    left.controls.addEventListener("start", () => {{
      if (isSyncing || !layerState.sync) return;
      setActivePanel("left");
    }});
    right.controls.addEventListener("start", () => {{
      if (isSyncing || !layerState.sync) return;
      setActivePanel("right");
    }});
    left.controls.addEventListener("end", () => {{
      if (isSyncing || !layerState.sync) return;
      clearActivePanel();
      copyPose(left, right);
    }});
    right.controls.addEventListener("end", () => {{
      if (isSyncing || !layerState.sync) return;
      clearActivePanel();
      copyPose(right, left);
    }});
    left.controls.addEventListener("change", () => {{
      if (!isSyncing && layerState.sync) masterPanel = "left";
    }});
    right.controls.addEventListener("change", () => {{
      if (!isSyncing && layerState.sync) masterPanel = "right";
    }});

    function bindInputAsMaster(panel, name) {{
      const el = panel.renderer.domElement;
      const setActive = () => {{
        if (!layerState.sync) return;
        if (!isSyncing) setActivePanel(name);
      }};
      el.addEventListener("pointerdown", setActive, {{ passive: true }});
      el.addEventListener("mousedown", setActive, {{ passive: true }});
      el.addEventListener("touchstart", setActive, {{ passive: true }});
      el.addEventListener("wheel", setActive, {{ passive: true }});
    }}

    bindInputAsMaster(left, "left");
    bindInputAsMaster(right, "right");
    window.addEventListener("pointerup", () => clearActivePanel(), {{ passive: true }});
    window.addEventListener("mouseup", () => clearActivePanel(), {{ passive: true }});
    window.addEventListener("touchend", () => clearActivePanel(), {{ passive: true }});

    function resizePanel(panel) {{
      const w = panel.container.clientWidth;
      const h = panel.container.clientHeight;
      panel.camera.aspect = w / Math.max(h, 1);
      panel.camera.updateProjectionMatrix();
      panel.renderer.setSize(w, h);
      if (panel.controls.handleResize) panel.controls.handleResize();
    }}
    window.addEventListener("resize", () => {{
      resizePanel(left);
      resizePanel(right);
    }});

    let pollingBusy = false;
    async function pollViewerPayload() {{
      if (pollingBusy) return;
      pollingBusy = true;
      try {{
        const next = await loadPayload(true);
        const prevToken = (payload.meta && payload.meta.update_token) ? String(payload.meta.update_token) : "";
        const nextToken = (next.meta && next.meta.update_token) ? String(next.meta.update_token) : "";
        if (prevToken && nextToken && prevToken === nextToken) return;
        payload = next;
        pointSourceRequestSeq += 1;
        pointSourceCache.clear();
        pointSourceLoadState.clear();
        runs = extractRuns(payload);
        syncRunState(false);
        if (runs.length > 0 && selectedRuns().length === 0) {{
          runState.set(runKey(runs[0], 0), true);
        }}
        buildRunPanel();
        updateHeaderMetrics();
        updateDataInfoPanel();
        refreshSceneObjects();
      }} catch (err) {{
        console.warn("[sync-viewer] polling failed:", err);
      }} finally {{
        pollingBusy = false;
      }}
    }}
    const livePollSec = Number((payload.meta && payload.meta.live_poll_sec) || 0);
    if (livePollSec > 0) {{
      const pollMs = Math.max(1000, Math.floor(livePollSec * 1000));
      window.setInterval(pollViewerPayload, pollMs);
    }}

    function animate() {{
      requestAnimationFrame(animate);
      if (layerState.sync) {{
        if (activePanel === "left") {{
          left.controls.update();
          copyPose(left, right);
        }} else if (activePanel === "right") {{
          right.controls.update();
          copyPose(right, left);
        }} else {{
          if (masterPanel === "left") {{
            left.controls.update();
            copyPose(left, right);
          }} else {{
            right.controls.update();
            copyPose(right, left);
          }}
        }}
      }} else {{
        left.controls.update();
        right.controls.update();
      }}
      left.renderer.render(left.scene, left.camera);
      right.renderer.render(right.scene, right.camera);
    }}
    animate();
    }} catch (err) {{
      console.error("[sync-viewer] initialization failed:", err);
      const header = document.getElementById("header");
      if (header) {{
        const span = document.createElement("span");
        span.style.color = "#ffb4b4";
        span.textContent = "[sync-viewer] init failed: " + String(err);
        header.appendChild(span);
      }}
    }}
  </script>
  <script nomodule>
    const header = document.getElementById("header");
    if (header) {{
      const span = document.createElement("span");
      span.style.color = "#ffb4b4";
      span.textContent = "[sync-viewer] ES module unsupported browser";
      header.appendChild(span);
    }}
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return json_path


def _serve_directory(root_dir: Path, host: str, port: int) -> None:
    class _TCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    os.chdir(root_dir)
    handler = http.server.SimpleHTTPRequestHandler
    with _TCPServer((host, port), handler) as httpd:
        print(f"[sync-viewer] Serving {root_dir}")
        print(f"[sync-viewer] URL: http://localhost:{port}")
        httpd.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build synced dual 3D viewer (pointcloud vs mesh) and optionally serve it.")
    parser.add_argument("--run_dir", type=str, default="", help="Run directory (.../<exp>/<timestamp>). If omitted, latest run is used.")
    parser.add_argument("--output_root", type=str, default="planarSplat_ExpRes/demo", help="Output root used by run_demo.py")
    parser.add_argument("--data_pth", type=str, default="", help="Explicit data.pth path")
    parser.add_argument("--pcd_path", type=str, default="", help="Explicit point cloud path (.ply)")
    parser.add_argument("--mesh_path", type=str, default="", help="Explicit mesh path (.ply)")
    parser.add_argument(
        "--prefer_saved_pcd",
        action="store_true",
        help="Use saved .ply point cloud first. Default is rebuilding point cloud from data.pth for coordinate consistency.",
    )
    parser.add_argument("--html_path", type=str, default="", help="Target html path (default: <run_dir>/sync_dual_viewer.html)")
    parser.add_argument("--point_stride", type=int, default=8, help="Depth-to-point stride if building from data.pth")
    parser.add_argument("--max_points", type=int, default=350000, help="Maximum point count to visualize")
    parser.add_argument("--max_faces", type=int, default=500000, help="Maximum mesh face count (decimated if larger)")
    parser.add_argument("--max_cameras", type=int, default=96, help="Maximum number of camera frustums")
    parser.add_argument("--frustum_ratio", type=float, default=0.04, help="Frustum depth ratio to scene radius")
    parser.add_argument("--compare_runs", type=int, default=4, help="Number of recent runs to include in Run checklist (1 = current run only)")
    parser.add_argument("--live_poll_sec", type=float, default=0.0, help="Browser-side JSON polling interval in seconds (0 disables live polling)")
    parser.add_argument("--serve", action="store_true", help="Serve html/json with a local HTTP server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if o3d is None:
        print("[sync-viewer][ERR] open3d is not installed. Run this inside the Docker container or install open3d.")
        return 3

    rng = random.Random(args.seed)

    output_root = Path(args.output_root).resolve()
    compare_runs = max(int(args.compare_runs), 1)
    compare_runs_requested = compare_runs
    explicit_geometry_paths = bool(args.data_pth or args.pcd_path or args.mesh_path)
    if explicit_geometry_paths and compare_runs > 1:
        print("[sync-viewer][WARN] compare_runs>1 with explicit --data_pth/--pcd_path/--mesh_path. Forcing compare_runs=1.")
        compare_runs = 1

    if args.run_dir:
        primary_run = Path(args.run_dir).resolve()
        if not primary_run.exists():
            print(f"[sync-viewer][ERR] run_dir does not exist: {primary_run}")
            return 1
        scan_root = _infer_output_root_from_run(primary_run) or output_root
        recent = _find_recent_runs(scan_root, max(compare_runs * 4, compare_runs))
        run_dirs: List[Path] = [primary_run]
        for run_dir in recent:
            if run_dir == primary_run:
                continue
            run_dirs.append(run_dir)
            if len(run_dirs) >= compare_runs:
                break
    else:
        scan_root = output_root
        run_dirs = _find_recent_runs(scan_root, compare_runs)
        if not run_dirs:
            print(f"[sync-viewer][ERR] No run found under: {scan_root}")
            return 1
        primary_run = run_dirs[0]

    if not run_dirs:
        print("[sync-viewer][ERR] No valid run selected.")
        return 1

    per_run_max_points = args.max_points
    per_run_max_faces = args.max_faces
    if len(run_dirs) > 1:
        per_run_max_points = max(60000, args.max_points // len(run_dirs))
        per_run_max_faces = max(120000, args.max_faces // len(run_dirs))

    palette = [
        [61, 117, 254],
        [35, 176, 159],
        [246, 122, 66],
        [208, 61, 131],
        [245, 165, 36],
        [114, 140, 167],
        [122, 204, 98],
        [95, 92, 255],
    ]

    runs_payload: List[Dict] = []
    point_source_blobs_by_run: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    token_parts: List[str] = []
    global_min: Optional[np.ndarray] = None
    global_max: Optional[np.ndarray] = None

    for idx, run_dir in enumerate(run_dirs):
        use_explicit_paths = idx == 0 and explicit_geometry_paths
        built = _build_single_run_payload(
            run_dir=run_dir,
            output_root=scan_root,
            data_pth_arg=args.data_pth if use_explicit_paths else "",
            pcd_path_arg=args.pcd_path if use_explicit_paths else "",
            mesh_path_arg=args.mesh_path if use_explicit_paths else "",
            prefer_saved_pcd=args.prefer_saved_pcd,
            point_stride=args.point_stride,
            max_points=per_run_max_points,
            max_faces=per_run_max_faces,
            max_cameras=args.max_cameras,
            frustum_ratio=args.frustum_ratio,
            rng=rng,
        )
        if built is None:
            continue
        run_item = built["run"]
        run_item["color"] = palette[idx % len(palette)]
        runs_payload.append(run_item)
        point_source_blobs_by_run[str(run_item.get("id", f"run-{idx}"))] = built.get("point_source_blobs", {})
        token_parts.append(str(built["token"]))

        xyz_min = np.asarray(built["xyz_min"], dtype=np.float32)
        xyz_max = np.asarray(built["xyz_max"], dtype=np.float32)
        if global_min is None:
            global_min = xyz_min.copy()
            global_max = xyz_max.copy()
        else:
            global_min = np.minimum(global_min, xyz_min)
            global_max = np.maximum(global_max, xyz_max)

    if not runs_payload:
        print("[sync-viewer][ERR] No run has valid mesh/geometry. Check output directory and rerun after training.")
        return 2

    if global_min is None or global_max is None:
        center = np.zeros(3, dtype=np.float32)
        radius = 1.0
    else:
        center = 0.5 * (global_min + global_max)
        radius = float(np.linalg.norm(global_max - global_min) * 0.5)
        radius = max(radius, 1.0)

    default_cam_pos = (center + np.array([radius * 1.8, radius * 1.2, radius * 1.8], dtype=np.float32)).tolist()
    update_token = "-".join(token_parts + [str(len(runs_payload)), str(_safe_mtime_ns(primary_run))])

    payload = {
        "scene": {
            "grid_size": float(radius * 4.0),
            "axes_size": float(radius * 0.4),
        },
        "default_camera": {
            "position": default_cam_pos,
            "target": center.tolist(),
            "up": [0.0, 1.0, 0.0],
        },
        "runs": runs_payload,
        "meta": {
            "output_root": str(scan_root),
            "run_count": int(len(runs_payload)),
            "selected_run_id": str(runs_payload[0]["id"]),
            "update_token": update_token,
            "live_poll_sec": max(float(args.live_poll_sec), 0.0),
            "compare_runs": int(compare_runs),
            "compare_runs_requested": int(compare_runs_requested),
            "point_source_format": "external_manifest_v1",
        },
    }

    # Backward-compat fields for older viewer parsers.
    payload["pointcloud"] = runs_payload[0].get("pointcloud", {"positions": [], "colors": [], "point_size": 0.002})
    payload["mesh"] = runs_payload[0].get("mesh", {"vertices": [], "faces": [], "colors": []})
    payload["frustums"] = runs_payload[0].get("frustums", {"positions": []})

    html_path = Path(args.html_path).resolve() if args.html_path else (primary_run / "sync_dual_viewer.html")
    json_path = _write_viewer_files(html_path, payload, point_source_blobs_by_run=point_source_blobs_by_run)

    print(f"[sync-viewer] primary run: {primary_run}")
    print(f"[sync-viewer] output root: {scan_root}")
    print(f"[sync-viewer] compare runs: {len(runs_payload)} (requested={compare_runs_requested}, effective={compare_runs})")
    print(f"[sync-viewer] per-run limits: points={per_run_max_points}, faces={per_run_max_faces}")
    print(f"[sync-viewer] html: {html_path}")
    print(f"[sync-viewer] json: {json_path}")
    for run in runs_payload:
        meta = run.get("meta", {})
        denom = int(meta.get("dataset_image_count", 0) or meta.get("used_image_count", 0))
        print(
            "[sync-viewer] run:",
            f"label={run.get('label', '-')},",
            f"point_source={meta.get('point_source', '-')},",
            f"points={meta.get('point_count', 0)},",
            f"final_mesh_v={meta.get('mesh_vertex_count', 0)},",
            f"final_mesh_f={meta.get('mesh_face_count', 0)},",
            f"raw_mesh_v={meta.get('raw_mesh_vertex_count', 0)},",
            f"raw_mesh_f={meta.get('raw_mesh_face_count', 0)},",
            f"aligned={meta.get('aligned_image_count', 0)}/{denom}",
        )

    if args.serve:
        _serve_directory(html_path.parent, args.host, args.port)
    else:
        print("[sync-viewer] To view in browser, run with --serve or host the html/json folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
