#!/usr/bin/env python3
"""Phase 4 Part 1: Analyze primitives from checkpoint.

Extracts center, normal, radii, semantic class from a checkpoint.
Prints statistics, filters building primitives (roof+wall), exports class PLYs.

Usage (inside Docker):
    python scripts/analyze_primitives.py \
        --checkpoint planarSplat_ExpRes/.../latest.pth \
        --output_dir results/phase4/part1
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'planarsplat'))

import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d


CLASS_NAMES = ['bg', 'roof', 'wall', 'ground']
CLASS_COLORS = np.array([
    [0.3, 0.3, 0.3],   # bg: dark gray
    [1.0, 0.0, 0.0],   # roof: red
    [0.0, 0.0, 1.0],   # wall: blue
    [0.7, 0.7, 0.7],   # ground: light gray
])


def load_primitives(checkpoint_path):
    """Load primitive parameters from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model_state_dict']
    params = {
        'center': state['planarSplat._plane_center'],
        'radii_p': state['planarSplat._plane_radii_xy_p'],
        'radii_n': state['planarSplat._plane_radii_xy_n'],
        'rot_q_normal_wxy': state['planarSplat._plane_rot_q_normal_wxy'],
        'rot_q_xyAxis_w': state['planarSplat._plane_rot_q_xyAxis_w'],
        'rot_q_xyAxis_z': state['planarSplat._plane_rot_q_xyAxis_z'],
    }
    sem_key = 'planarSplat._plane_semantic_features'
    if sem_key in state:
        params['semantic_features'] = state[sem_key]
    else:
        raise ValueError("Checkpoint has no semantic features")
    return params, ckpt.get('iter', -1)


def compute_normals(params):
    """Compute plane normals via quaternion rotation (requires CUDA)."""
    N = params['center'].shape[0]
    q_normal = torch.cat([params['rot_q_normal_wxy'], torch.zeros(N, 1)], dim=-1)
    q_normal = F.normalize(q_normal, dim=-1)
    q_xyAxis = torch.cat([params['rot_q_xyAxis_w'], torch.zeros(N, 2), params['rot_q_xyAxis_z']], dim=-1)
    q_xyAxis = F.normalize(q_xyAxis, dim=-1)
    from utils.model_util import quaternion_mult, quat_to_rot
    q = F.normalize(quaternion_mult(q_normal, q_xyAxis), dim=-1)
    R = quat_to_rot(q.cuda())  # (N,3,3)
    z_axis = torch.tensor([0., 0., 1.], device='cuda').reshape(1, 3, 1).expand(N, 3, 1)
    normals = torch.bmm(R, z_axis).squeeze(-1).cpu()  # (N,3)
    return normals, q


def compute_areas(params):
    """Compute approximate area of each primitive."""
    rx = params['radii_p'][:, 0] + params['radii_n'][:, 0]
    ry = params['radii_p'][:, 1] + params['radii_n'][:, 1]
    return (rx * ry).numpy()


def build_class_mesh(centers, normals, radii_p, radii_n, rot_q, mask, color):
    """Build rectangle mesh for a subset of primitives."""
    from utils.model_util import quat_to_rot
    idx = mask.nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return None
    c = centers[idx]
    n = normals[idx]
    rp = radii_p[idx]
    rn = radii_n[idx]
    rq = F.normalize(rot_q[idx].cuda(), dim=-1)
    R = quat_to_rot(rq).cpu()
    N = len(idx)

    rx_p, ry_p = rp[:, 0], rp[:, 1]
    rx_n, ry_n = rn[:, 0], rn[:, 1]
    zero = torch.zeros_like(rx_p)

    v1 = torch.stack([rx_p, ry_p, zero], dim=-1)
    v2 = torch.stack([-rx_n, ry_p, zero], dim=-1)
    v3 = torch.stack([-rx_n, -ry_n, zero], dim=-1)
    v4 = torch.stack([rx_p, -ry_n, zero], dim=-1)
    verts_local = torch.stack([v1, v2, v3, v4], dim=1)  # (N,4,3)

    verts_world = torch.bmm(
        verts_local.reshape(N*4, 1, 3),
        R.unsqueeze(1).expand(N, 4, 3, 3).reshape(N*4, 3, 3).transpose(1, 2)
    ).reshape(N, 4, 3) + c.unsqueeze(1)

    all_verts = verts_world.reshape(-1, 3).numpy()
    faces = []
    vert_colors = []
    for i in range(N):
        base = i * 4
        faces.append([base, base+1, base+2])
        faces.append([base, base+2, base+3])
        for _ in range(4):
            vert_colors.append(color)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vert_colors))
    mesh.compute_vertex_normals()
    return mesh


def print_statistics(params, normals, class_pred, areas):
    """Print per-class statistics."""
    centers = params['center'].numpy()
    normals_np = normals.numpy()
    N = len(class_pred)

    # COLMAP gravity direction: -Y
    e_gravity = np.array([0., -1., 0.])

    print(f"\n{'='*60}")
    print(f"Total primitives: {N}")
    print(f"{'='*60}")

    stats = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = (class_pred == cls_id)
        count = mask.sum()
        if count == 0:
            print(f"\n[{cls_name}] count=0")
            stats[cls_name] = {'count': 0}
            continue

        c = centers[mask]
        n = normals_np[mask]
        a = areas[mask]
        dot_gravity = np.abs(n @ e_gravity)  # |n · e_gravity|

        cls_stats = {
            'count': int(count),
            'pct': float(count / N * 100),
            'area_mean': float(a.mean()),
            'area_median': float(np.median(a)),
            'area_total': float(a.sum()),
            'height_y_mean': float(c[:, 1].mean()),
            'height_y_std': float(c[:, 1].std()),
            'height_y_min': float(c[:, 1].min()),
            'height_y_max': float(c[:, 1].max()),
            'normal_gravity_dot_mean': float(dot_gravity.mean()),
            'normal_gravity_dot_std': float(dot_gravity.std()),
        }
        stats[cls_name] = cls_stats

        print(f"\n[{cls_name}] count={count} ({cls_stats['pct']:.1f}%)")
        print(f"  Area: mean={a.mean():.5f}, median={np.median(a):.5f}, total={a.sum():.3f}")
        print(f"  Height(Y): mean={c[:,1].mean():.3f}, std={c[:,1].std():.3f}, "
              f"range=[{c[:,1].min():.3f}, {c[:,1].max():.3f}]")
        print(f"  |n·gravity|: mean={dot_gravity.mean():.3f}, std={dot_gravity.std():.3f}")
        if cls_name == 'roof':
            nearly_horiz = (dot_gravity > 0.85).sum()
            print(f"  Nearly horizontal (|dot|>0.85): {nearly_horiz}/{count} ({nearly_horiz/count*100:.1f}%)")
        elif cls_name == 'wall':
            nearly_vert = (dot_gravity < 0.15).sum()
            print(f"  Nearly vertical (|dot|<0.15): {nearly_vert}/{count} ({nearly_vert/count*100:.1f}%)")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Part 1: Analyze primitives')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='results/phase4/part1')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading checkpoint...")
    params, iter_num = load_primitives(args.checkpoint)
    N = params['center'].shape[0]
    print(f"  iter={iter_num}, N={N}")

    print("Computing normals...")
    normals, rot_q = compute_normals(params)
    areas = compute_areas(params)

    # Semantic class prediction
    probs = F.softmax(params['semantic_features'], dim=-1)
    class_pred = probs.argmax(dim=-1).numpy()
    confidence = probs.max(dim=-1).values.numpy()

    print(f"Semantic confidence: mean={confidence.mean():.3f}, "
          f"min={confidence.min():.3f}, >0.9={( confidence > 0.9).sum()}/{N}")

    stats = print_statistics(params, normals, class_pred, areas)

    # Building primitives (roof + wall)
    building_mask = (class_pred == 1) | (class_pred == 2)  # roof=1, wall=2
    n_building = building_mask.sum()
    print(f"\n{'='*60}")
    print(f"Building primitives (roof+wall): {n_building}/{N} ({n_building/N*100:.1f}%)")
    print(f"{'='*60}")

    # Save statistics JSON
    stats_path = os.path.join(args.output_dir, 'primitive_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'iter': iter_num,
            'total_primitives': N,
            'building_primitives': int(n_building),
            'confidence_mean': float(confidence.mean()),
            'per_class': stats,
        }, f, indent=2)
    print(f"\nSaved stats: {stats_path}")

    # Export class PLYs
    print("\nExporting PLY files...")
    class_pred_t = torch.from_numpy(class_pred)
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = (class_pred_t == cls_id)
        if mask.sum() == 0:
            continue
        mesh = build_class_mesh(
            params['center'], normals,
            params['radii_p'], params['radii_n'],
            rot_q, mask, CLASS_COLORS[cls_id]
        )
        if mesh is not None:
            ply_path = os.path.join(args.output_dir, f'primitives_{cls_name}.ply')
            o3d.io.write_triangle_mesh(ply_path, mesh)
            print(f"  {cls_name}: {mask.sum()} primitives → {ply_path}")

    # Combined building PLY (roof=red, wall=blue)
    all_meshes = []
    for cls_id, cls_name in [(1, 'roof'), (2, 'wall')]:
        mask = (class_pred_t == cls_id)
        if mask.sum() > 0:
            mesh = build_class_mesh(
                params['center'], normals,
                params['radii_p'], params['radii_n'],
                rot_q, mask, CLASS_COLORS[cls_id]
            )
            if mesh is not None:
                all_meshes.append(mesh)
    if all_meshes:
        combined = all_meshes[0]
        for m in all_meshes[1:]:
            combined += m
        ply_path = os.path.join(args.output_dir, 'primitives_building.ply')
        o3d.io.write_triangle_mesh(ply_path, combined)
        print(f"  building (combined): {n_building} primitives → {ply_path}")

    # Save building indices for Part 2
    building_data = {
        'building_mask': building_mask,
        'class_pred': class_pred,
        'centers': params['center'].numpy(),
        'normals': normals.numpy(),
        'radii_p': params['radii_p'].numpy(),
        'radii_n': params['radii_n'].numpy(),
        'rot_q': rot_q.numpy(),
        'areas': areas,
        'confidence': confidence,
        'semantic_probs': probs.numpy(),
    }
    data_path = os.path.join(args.output_dir, 'primitives_data.npz')
    np.savez(data_path, **building_data)
    print(f"\nSaved primitive data: {data_path}")


if __name__ == '__main__':
    main()
