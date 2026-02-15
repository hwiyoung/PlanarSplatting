#!/usr/bin/env python3
"""Export planar primitives from a checkpoint as PLY.

Usage (inside Docker container):
    python scripts/visualize_primitives.py \
        --checkpoint planarSplat_ExpRes/demo/exp_example/2026_.../checkpoints/Parameters/latest.pth \
        --color_by normal \
        --export_ply output.ply

Options:
    --color_by   normal | rgb (random) | depth (z-based colormap)
    --export_ply Output PLY path (default: <checkpoint_dir>/primitives_<color_by>.ply)
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'planarsplat'))

import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d


def load_checkpoint_params(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("Error: CUDA is required (quat_to_rot uses CUDA). Run inside the Docker container.")
        sys.exit(1)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' not in ckpt:
        print(f"Error: checkpoint missing 'model_state_dict' key. Keys found: {list(ckpt.keys())}")
        sys.exit(1)
    state = ckpt['model_state_dict']
    params = {
        'center': state['planarSplat._plane_center'],
        'radii_p': state['planarSplat._plane_radii_xy_p'],
        'radii_n': state['planarSplat._plane_radii_xy_n'],
        'rot_q_normal_wxy': state['planarSplat._plane_rot_q_normal_wxy'],
        'rot_q_xyAxis_w': state['planarSplat._plane_rot_q_xyAxis_w'],
        'rot_q_xyAxis_z': state['planarSplat._plane_rot_q_xyAxis_z'],
    }
    return params, ckpt.get('iter', -1)


def reconstruct_rotation(params):
    """Reconstruct full rotation quaternion from factored representation."""
    N = params['center'].shape[0]
    # q_normal = [w, x, y, 0]
    q_normal = torch.cat([
        params['rot_q_normal_wxy'],
        torch.zeros(N, 1)
    ], dim=-1)
    q_normal = F.normalize(q_normal, dim=-1)

    # q_xyAxis = [w, 0, 0, z]
    q_xyAxis = torch.cat([
        params['rot_q_xyAxis_w'],
        torch.zeros(N, 2),
        params['rot_q_xyAxis_z']
    ], dim=-1)
    q_xyAxis = F.normalize(q_xyAxis, dim=-1)

    # q = q_normal * q_xyAxis
    from utils.model_util import quaternion_mult, quat_to_rot
    q = quaternion_mult(q_normal, q_xyAxis)
    q = F.normalize(q, dim=-1)
    return q


def get_normals(params):
    """Compute plane normals from rotation parameters."""
    q = reconstruct_rotation(params).cuda()
    from utils.model_util import quat_to_rot
    R = quat_to_rot(q)  # (N, 3, 3)
    z_axis = torch.tensor([0., 0., 1.], device='cuda').reshape(1, 3, 1).expand(R.shape[0], 3, 1)
    normals = torch.bmm(R, z_axis).squeeze(-1)  # (N, 3)
    return normals.cpu(), q.cpu()


def build_rectangle_mesh(centers, normals, radii_p, radii_n, rot_q, colors):
    """Build an Open3D mesh of rectangles from primitive parameters."""
    from utils.model_util import quat_to_rot
    N = centers.shape[0]
    # Compute rotation on CUDA (required by quat_to_rot)
    rot_q_cuda = F.normalize(rot_q.cuda(), dim=-1)
    R = quat_to_rot(rot_q_cuda).cpu()  # (N, 3, 3)

    # 4 corner vertices per rectangle
    rx_p, ry_p = radii_p[:, 0], radii_p[:, 1]
    rx_n, ry_n = radii_n[:, 0], radii_n[:, 1]
    zero = torch.zeros_like(rx_p)

    v1 = torch.stack([rx_p, ry_p, zero], dim=-1)   # (+x, +y)
    v2 = torch.stack([-rx_n, ry_p, zero], dim=-1)  # (-x, +y)
    v3 = torch.stack([-rx_n, -ry_n, zero], dim=-1) # (-x, -y)
    v4 = torch.stack([rx_p, -ry_n, zero], dim=-1)  # (+x, -y)

    verts_local = torch.stack([v1, v2, v3, v4], dim=1)  # (N, 4, 3)

    # Rotate to world frame (on CPU now)
    verts_world = torch.bmm(
        verts_local.reshape(N * 4, 1, 3),
        R.unsqueeze(1).expand(N, 4, 3, 3).reshape(N * 4, 3, 3).transpose(1, 2)
    ).reshape(N, 4, 3)
    verts_world = verts_world + centers.unsqueeze(1)

    # Build mesh arrays
    all_verts = verts_world.reshape(-1, 3).numpy()
    faces = []
    vert_colors = []
    for i in range(N):
        base = i * 4
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
        c = colors[i].numpy()
        for _ in range(4):
            vert_colors.append(c)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vert_colors))
    mesh.compute_vertex_normals()
    return mesh


def color_by_normal(normals):
    """Map normals to RGB: (n + 1) / 2."""
    return (normals + 1.0) / 2.0


def color_by_depth(centers):
    """Map z-coordinate to colormap."""
    z = centers[:, 2]
    z_min, z_max = z.min(), z.max()
    if z_max - z_min < 1e-6:
        t = torch.zeros_like(z)
    else:
        t = (z - z_min) / (z_max - z_min)
    import matplotlib.cm as cm
    colors_np = cm.viridis(t.numpy())[:, :3]
    return torch.from_numpy(colors_np).float()


def main():
    parser = argparse.ArgumentParser(description='Export planar primitives to PLY')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pth')
    parser.add_argument('--color_by', default='normal', choices=['normal', 'rgb', 'depth'],
                        help='Coloring method')
    parser.add_argument('--export_ply', default='', help='Output PLY path')
    args = parser.parse_args()

    params, iter_num = load_checkpoint_params(args.checkpoint)
    N = params['center'].shape[0]
    print(f"Loaded checkpoint: iter={iter_num}, primitives={N}")

    if N == 0:
        print("Warning: checkpoint contains 0 primitives. Nothing to export.")
        sys.exit(0)

    normals, rot_q = get_normals(params)

    if args.color_by == 'normal':
        colors = color_by_normal(normals)
    elif args.color_by == 'rgb':
        colors = torch.rand(N, 3)
    elif args.color_by == 'depth':
        colors = color_by_depth(params['center'])

    mesh = build_rectangle_mesh(
        params['center'], normals,
        params['radii_p'], params['radii_n'],
        rot_q, colors
    )

    if args.export_ply:
        out_path = args.export_ply
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        out_path = os.path.join(ckpt_dir, f'primitives_{args.color_by}.ply')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Exported PLY: {out_path}")
    print(f"  Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")


if __name__ == '__main__':
    main()
