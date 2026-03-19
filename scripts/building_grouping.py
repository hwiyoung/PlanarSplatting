#!/usr/bin/env python3
"""Phase 4 Part 2: Group primitives into building instances.

1. Load primitives_data.npz from Part 1
2. Roof centers → XZ projection → distance-based connected components → building_id
3. Wall → assign to nearest roof cluster
4. Export per-building PLY

Usage (inside Docker):
    python scripts/building_grouping.py \
        --input results/phase4/part1/primitives_data.npz \
        --output_dir results/phase4/part2 \
        --distance_threshold 0.3
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'planarsplat'))

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


CLASS_NAMES = ['bg', 'roof', 'wall', 'ground']
# Distinct colors for buildings
BUILDING_COLORS = [
    [1.0, 0.0, 0.0],   # red
    [0.0, 0.8, 0.0],   # green
    [0.0, 0.0, 1.0],   # blue
    [1.0, 0.6, 0.0],   # orange
    [0.8, 0.0, 0.8],   # purple
    [0.0, 0.8, 0.8],   # cyan
    [1.0, 1.0, 0.0],   # yellow
    [0.6, 0.3, 0.0],   # brown
    [1.0, 0.4, 0.7],   # pink
    [0.4, 0.6, 0.2],   # olive
]


def build_rect_mesh_single(center, radii_p, radii_n, rot_q, color):
    """Build a single rectangle mesh from primitive params."""
    from utils.model_util import quat_to_rot
    q = F.normalize(torch.from_numpy(rot_q).float().unsqueeze(0).cuda(), dim=-1)
    R = quat_to_rot(q).cpu().squeeze(0).numpy()  # (3,3)

    rx_p, ry_p = radii_p[0], radii_p[1]
    rx_n, ry_n = radii_n[0], radii_n[1]

    verts_local = np.array([
        [rx_p, ry_p, 0],
        [-rx_n, ry_p, 0],
        [-rx_n, -ry_n, 0],
        [rx_p, -ry_n, 0],
    ])
    verts_world = (verts_local @ R.T) + center

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2],[0,2,3]]))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([color]*4))
    return mesh


def group_roofs_by_xz(centers, distance_threshold):
    """Group roof primitives by XZ proximity using connected components."""
    if len(centers) == 0:
        return np.array([], dtype=int)
    if len(centers) == 1:
        return np.array([0])

    # Project to XZ plane (X=0, Z=2 in COLMAP frame)
    xz = centers[:, [0, 2]]

    # Pairwise distance
    dists = squareform(pdist(xz))

    # Adjacency: connected if within threshold
    adj = (dists < distance_threshold).astype(int)
    np.fill_diagonal(adj, 0)

    n_components, labels = connected_components(csr_matrix(adj), directed=False)
    return labels


def assign_walls_to_buildings(wall_centers, roof_centers, roof_building_ids):
    """Assign each wall to the nearest roof cluster (by 3D distance to cluster centroid)."""
    if len(wall_centers) == 0 or len(roof_centers) == 0:
        return np.full(len(wall_centers), -1, dtype=int)

    unique_bids = np.unique(roof_building_ids)
    centroids = np.array([roof_centers[roof_building_ids == bid].mean(axis=0) for bid in unique_bids])

    # For each wall, find nearest centroid
    wall_bids = np.zeros(len(wall_centers), dtype=int)
    for i, wc in enumerate(wall_centers):
        dists = np.linalg.norm(centroids - wc, axis=1)
        wall_bids[i] = unique_bids[np.argmin(dists)]

    return wall_bids


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Part 2: Building grouping')
    parser.add_argument('--input', default='results/phase4/part1/primitives_data.npz')
    parser.add_argument('--output_dir', default='results/phase4/part2')
    parser.add_argument('--distance_threshold', type=float, default=0.3,
                        help='XZ distance threshold for roof grouping')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading primitive data...")
    data = np.load(args.input, allow_pickle=True)
    centers = data['centers']
    normals = data['normals']
    radii_p = data['radii_p']
    radii_n = data['radii_n']
    rot_q = data['rot_q']
    class_pred = data['class_pred']
    areas = data['areas']
    N = len(centers)

    roof_mask = (class_pred == 1)
    wall_mask = (class_pred == 2)
    n_roof = roof_mask.sum()
    n_wall = wall_mask.sum()
    print(f"  Total={N}, Roof={n_roof}, Wall={n_wall}")

    # Step 1: Group roofs by XZ proximity
    roof_centers = centers[roof_mask]
    print(f"\nGrouping roofs (XZ distance threshold={args.distance_threshold})...")
    roof_building_ids = group_roofs_by_xz(roof_centers, args.distance_threshold)
    n_buildings = len(np.unique(roof_building_ids))
    print(f"  Found {n_buildings} building clusters from roofs")

    # Step 2: Assign walls to buildings
    wall_centers = centers[wall_mask]
    print("Assigning walls to nearest building...")
    wall_building_ids = assign_walls_to_buildings(wall_centers, roof_centers, roof_building_ids)

    # Build full building_id array (-1 = not a building)
    building_ids = np.full(N, -1, dtype=int)
    building_ids[roof_mask] = roof_building_ids
    building_ids[wall_mask] = wall_building_ids

    # Stats per building
    print(f"\n{'='*60}")
    print(f"Building Instance Statistics")
    print(f"{'='*60}")
    building_stats = {}
    for bid in sorted(np.unique(building_ids)):
        if bid < 0:
            continue
        bmask = (building_ids == bid)
        b_classes = class_pred[bmask]
        b_roof = (b_classes == 1).sum()
        b_wall = (b_classes == 2).sum()
        b_centers = centers[bmask]
        b_areas = areas[bmask]

        stats = {
            'building_id': int(bid),
            'total': int(bmask.sum()),
            'roof': int(b_roof),
            'wall': int(b_wall),
            'center_mean': b_centers.mean(axis=0).tolist(),
            'area_total': float(b_areas.sum()),
            'height_y_range': [float(b_centers[:, 1].min()), float(b_centers[:, 1].max())],
        }
        building_stats[str(bid)] = stats
        print(f"  Building {bid}: roof={b_roof}, wall={b_wall}, "
              f"total_area={b_areas.sum():.3f}, "
              f"Y=[{b_centers[:,1].min():.3f}, {b_centers[:,1].max():.3f}]")

    # Step 3: Export per-building PLY
    print("\nExporting PLY files...")

    # Combined building PLY with distinct colors per building
    all_meshes = []
    for bid in sorted(np.unique(building_ids)):
        if bid < 0:
            continue
        bmask = (building_ids == bid)
        color = BUILDING_COLORS[bid % len(BUILDING_COLORS)]

        bid_indices = np.where(bmask)[0]
        for idx in bid_indices:
            m = build_rect_mesh_single(
                centers[idx], radii_p[idx], radii_n[idx], rot_q[idx], color
            )
            all_meshes.append(m)

    if all_meshes:
        combined = all_meshes[0]
        for m in all_meshes[1:]:
            combined += m
        combined.compute_vertex_normals()
        ply_path = os.path.join(args.output_dir, 'buildings_colored.ply')
        o3d.io.write_triangle_mesh(ply_path, combined)
        print(f"  Combined building PLY: {ply_path}")

    # Per-building PLY (roof=lighter, wall=darker shade)
    for bid in sorted(np.unique(building_ids)):
        if bid < 0:
            continue
        bmask = (building_ids == bid)
        bid_indices = np.where(bmask)[0]
        meshes = []
        for idx in bid_indices:
            cls = class_pred[idx]
            base_color = np.array(BUILDING_COLORS[bid % len(BUILDING_COLORS)])
            # Roof: brighter, Wall: darker
            color = base_color * (0.8 if cls == 2 else 1.0)
            m = build_rect_mesh_single(
                centers[idx], radii_p[idx], radii_n[idx], rot_q[idx], color.tolist()
            )
            meshes.append(m)
        if meshes:
            combined_b = meshes[0]
            for m in meshes[1:]:
                combined_b += m
            combined_b.compute_vertex_normals()
            ply_path = os.path.join(args.output_dir, f'building_{bid:03d}.ply')
            o3d.io.write_triangle_mesh(ply_path, combined_b)

    print(f"  Per-building PLYs: {n_buildings} files")

    # Save grouping results
    result = {
        'n_buildings': n_buildings,
        'distance_threshold': args.distance_threshold,
        'buildings': building_stats,
    }
    json_path = os.path.join(args.output_dir, 'building_grouping.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved grouping: {json_path}")

    # Save building_ids for Part 3/4
    np.savez(
        os.path.join(args.output_dir, 'building_data.npz'),
        building_ids=building_ids,
        class_pred=class_pred,
        centers=centers,
        normals=normals,
        radii_p=radii_p,
        radii_n=radii_n,
        rot_q=rot_q,
        areas=areas,
    )
    print(f"Saved building data for Part 3/4")


if __name__ == '__main__':
    main()
