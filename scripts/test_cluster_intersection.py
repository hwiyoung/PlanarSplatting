#!/usr/bin/env python3
"""
Cluster-intersection based CityGML algorithm (synthetic test).

Core idea (user's approach):
  1. Cluster primitives by class + normal → surface groups
  2. Each cluster → one merged rectangle (plane eq + bounded extent)
  3. Adjacent cluster rectangles intersect → building edges
  4. Edges form polygon boundaries → LOD2 surfaces

Outputs intermediate PLY files at each step for visual inspection.
"""

import json
import os
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull
from itertools import combinations

# =============================================================================
# Step 0: Synthetic Building (simple box for clarity)
# =============================================================================

def create_box_building(width=10.0, depth=8.0, height=6.0):
    """
    Simple flat-roof box building. 6 surfaces: 4 walls + 1 roof + 1 ground.
    Y-up coordinate system.
    """
    hw, hd, h = width / 2, depth / 2, height

    vertices = np.array([
        [-hw, 0, -hd],  # 0: bottom-left-front
        [ hw, 0, -hd],  # 1: bottom-right-front
        [ hw, 0,  hd],  # 2: bottom-right-back
        [-hw, 0,  hd],  # 3: bottom-left-back
        [-hw, h, -hd],  # 4: top-left-front
        [ hw, h, -hd],  # 5: top-right-front
        [ hw, h,  hd],  # 6: top-right-back
        [-hw, h,  hd],  # 7: top-left-back
    ], dtype=np.float64)

    # faces: ordered vertex indices, outward normals
    faces = [
        [0, 3, 2, 1],  # ground (y=0, normal -Y)
        [4, 5, 6, 7],  # roof (y=h, normal +Y)  -- actually -Y in COLMAP
        [0, 1, 5, 4],  # front wall (z=-hd, normal -Z)
        [2, 3, 7, 6],  # back wall (z=+hd, normal +Z)
        [0, 4, 7, 3],  # left wall (x=-hw, normal -X)
        [1, 2, 6, 5],  # right wall (x=+hw, normal +X)
    ]

    # 1=roof, 2=wall, 3=ground
    labels = [3, 1, 2, 2, 2, 2]
    names = ["ground", "roof", "front_wall", "back_wall", "left_wall", "right_wall"]

    # Outward normals
    normals = np.array([
        [0, -1, 0],  # ground
        [0,  1, 0],  # roof  (we'll flip to -Y for COLMAP later if needed)
        [0, 0, -1],  # front
        [0, 0,  1],  # back
        [-1, 0, 0],  # left
        [1,  0, 0],  # right
    ], dtype=np.float64)

    return vertices, faces, labels, names, normals


def create_gable_building(width=10.0, depth=8.0, wall_height=6.0, pitch_deg=30.0):
    """
    Gable-roof building (LOD2). Y-up coordinate system.
    7 surfaces: 4 walls (2 rectangular + 2 pentagonal gable), 2 roof slopes, 1 ground.
    """
    hw = width / 2.0
    hd = depth / 2.0
    h = wall_height
    ridge_rise = hw * np.tan(np.radians(pitch_deg))
    ridge_y = h + ridge_rise

    vertices = np.array([
        [-hw, 0, -hd],  # 0
        [ hw, 0, -hd],  # 1
        [ hw, 0,  hd],  # 2
        [-hw, 0,  hd],  # 3
        [-hw, h, -hd],  # 4
        [ hw, h, -hd],  # 5
        [ hw, h,  hd],  # 6
        [-hw, h,  hd],  # 7
        [  0, ridge_y, -hd],  # 8 ridge-front
        [  0, ridge_y,  hd],  # 9 ridge-back
    ], dtype=np.float64)

    faces = [
        [0, 3, 2, 1],        # ground
        [0, 1, 5, 8, 4],     # front wall (pentagon)
        [2, 3, 7, 9, 6],     # back wall (pentagon)
        [0, 4, 7, 3],        # left wall
        [1, 2, 6, 5],        # right wall
        [4, 8, 9, 7],        # left roof slope
        [5, 6, 9, 8],        # right roof slope
    ]

    labels = [3, 2, 2, 2, 2, 1, 1]
    names = ["ground", "front_wall(gable)", "back_wall(gable)",
             "left_wall", "right_wall", "left_roof", "right_roof"]

    # Compute normals
    roof_normal_angle = np.radians(90 - pitch_deg)  # from vertical
    normals_list = [
        np.array([0, -1, 0]),   # ground
        np.array([0, 0, -1]),   # front
        np.array([0, 0,  1]),   # back
        np.array([-1, 0, 0]),   # left
        np.array([1,  0, 0]),   # right
        np.array([-np.sin(np.radians(pitch_deg)), np.cos(np.radians(pitch_deg)), 0]),  # left roof
        np.array([ np.sin(np.radians(pitch_deg)), np.cos(np.radians(pitch_deg)), 0]),  # right roof
    ]
    gt_normals = np.array(normals_list, dtype=np.float64)

    return vertices, faces, labels, names, gt_normals


def generate_primitives_on_face(vertices, face_indices, normal, label,
                                 n_prims=6, noise_std=0.0, rng=None):
    """Generate rectangular primitives on a face."""
    if rng is None:
        rng = np.random.RandomState(0)

    pts = vertices[face_indices]
    centers = []
    normals_out = []

    for _ in range(n_prims):
        # Random barycentric on triangulated face
        tri_idx = rng.randint(0, max(1, len(face_indices) - 2))
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        w = 1 - u - v
        point = w * pts[0] + u * pts[tri_idx + 1] + v * pts[tri_idx + 2]
        point += rng.randn(3) * noise_std

        n = normal + rng.randn(3) * 0.02 * (noise_std > 0)
        n = n / np.linalg.norm(n)

        centers.append(point)
        normals_out.append(n)

    return np.array(centers), np.array(normals_out)


# =============================================================================
# Step 1: Clustering → surface groups with plane equations
# =============================================================================

def cluster_primitives(centers, normals, classes, cos_thresh=0.85):
    """
    Cluster primitives by semantic class + normal similarity.
    Returns list of groups, each with: plane_normal, plane_d, member indices, class.
    """
    groups = []
    unique_classes = np.unique(classes)

    for cls in unique_classes:
        if cls == 0:  # skip background
            continue
        mask = classes == cls
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        cls_normals = normals[idx]
        cls_centers = centers[idx]

        if len(idx) == 1:
            n = cls_normals[0]
            c = cls_centers[0]
            groups.append({
                'class': int(cls),
                'indices': idx,
                'normal': n,
                'center': c,
                'd': np.dot(n, c),
            })
            continue

        # Hierarchical clustering on normal directions
        # Use cosine distance = 1 - cos(angle) (NOT abs — distinguish opposite directions)
        cos_dist = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                cos_sim = np.dot(cls_normals[i], cls_normals[j])
                cos_dist.append(max(0.0, 1.0 - cos_sim))  # clamp to avoid negative
        cos_dist = np.array(cos_dist)

        if len(cos_dist) == 0:
            continue

        Z = linkage(cos_dist.reshape(-1, 1) if len(cos_dist) == 1 else cos_dist,
                    method='complete')
        labels = fcluster(Z, t=1 - cos_thresh, criterion='distance')

        for lbl in np.unique(labels):
            sub_mask = labels == lbl
            sub_idx = idx[sub_mask]
            sub_normals = normals[sub_idx]
            sub_centers = centers[sub_idx]

            # Area-weighted mean normal (uniform weight for now)
            mean_normal = sub_normals.mean(axis=0)
            mean_normal /= np.linalg.norm(mean_normal)

            mean_center = sub_centers.mean(axis=0)
            d = np.dot(mean_normal, mean_center)

            groups.append({
                'class': int(cls),
                'indices': sub_idx,
                'normal': mean_normal,
                'center': mean_center,
                'd': d,
            })

    return groups


def refine_clusters(groups, centers, normals, min_wall_size=10,
                    min_roof_size=3, spatial_outlier_std=2.0):
    """
    Post-process clusters:
    1. Remove small clusters (wall < min_wall_size, roof < min_roof_size)
    2. Remove spatial outliers within each cluster (> spatial_outlier_std * std from center)
    3. Recompute plane equations after outlier removal
    4. Remove clusters that become too small after outlier removal

    Returns refined list of groups.
    """
    refined = []

    for g in groups:
        idx = g['indices']
        min_size = min_wall_size if g['class'] == 2 else min_roof_size

        # 1. Skip small clusters
        if len(idx) < min_size:
            continue

        # 2. Spatial outlier removal within cluster
        member_centers = centers[idx]
        cluster_center = member_centers.mean(axis=0)
        dists = np.linalg.norm(member_centers - cluster_center, axis=1)
        dist_mean = dists.mean()
        dist_std = dists.std()

        if dist_std > 1e-6:
            keep_mask = dists < dist_mean + spatial_outlier_std * dist_std
        else:
            keep_mask = np.ones(len(idx), dtype=bool)

        kept_idx = idx[keep_mask]

        # Check size again after outlier removal
        if len(kept_idx) < min_size:
            continue

        # 3. Recompute plane equation
        kept_normals = normals[kept_idx]
        kept_centers = centers[kept_idx]

        mean_normal = kept_normals.mean(axis=0)
        mean_normal /= np.linalg.norm(mean_normal)
        mean_center = kept_centers.mean(axis=0)
        d = np.dot(mean_normal, mean_center)

        refined.append({
            'class': g['class'],
            'indices': kept_idx,
            'normal': mean_normal,
            'center': mean_center,
            'd': d,
        })

    return refined


def synthesize_missing_faces(groups, centers, gravity=None):
    """
    Synthesize missing wall faces and ground face to close the building solid.

    Strategy:
    1. For each observed wall, check if an opposite wall exists.
       If not, create one at the far side of the building.
    2. Check wall coverage around the building. If fewer than 4 wall
       directions exist, synthesize walls to fill gaps.
    3. Always synthesize a ground face (horizontal plane at building bottom).

    Args:
        groups: refined cluster groups (roof + wall)
        centers: all primitive centers for this building
        gravity: gravity direction vector (default: computed)

    Returns:
        groups with synthesized faces appended (class=2 for wall, class=3 for ground)
    """
    if gravity is None:
        gravity = GRAVITY

    up = -gravity  # up direction

    # Collect existing wall normals (project to horizontal plane)
    wall_groups = [g for g in groups if g['class'] == 2]
    roof_groups = [g for g in groups if g['class'] == 1]

    if len(wall_groups) == 0 or len(roof_groups) == 0:
        return groups

    # Building center and extent from all primitive centers in groups
    all_idx = np.concatenate([g['indices'] for g in groups])
    bld_centers = centers[all_idx]
    bld_center = bld_centers.mean(axis=0)

    # Project centers onto gravity axis for height
    heights = bld_centers @ (-gravity)
    wall_idx = np.concatenate([g['indices'] for g in wall_groups])
    wall_heights = centers[wall_idx] @ (-gravity)

    # Building bottom = lowest wall primitive height
    bld_bottom_h = wall_heights.min()
    # Building top = highest roof primitive height
    roof_idx = np.concatenate([g['indices'] for g in roof_groups])
    roof_heights = centers[roof_idx] @ (-gravity)
    bld_top_h = roof_heights.max()

    # Horizontal plane: project normals onto plane perpendicular to gravity
    def horizontal_component(n):
        """Project normal onto horizontal plane (perpendicular to gravity)."""
        h = n - np.dot(n, gravity) * gravity
        norm = np.linalg.norm(h)
        if norm < 1e-6:
            return None
        return h / norm

    # Get horizontal directions of existing walls
    wall_horiz_dirs = []
    for g in wall_groups:
        h = horizontal_component(g['normal'])
        if h is not None:
            wall_horiz_dirs.append(h)

    # Project building centers onto horizontal plane for extent
    horiz_centers = bld_centers - np.outer(bld_centers @ gravity, gravity)

    synthesized = list(groups)  # start with existing groups

    # --- Synthesize opposite walls ---
    for g in wall_groups:
        h_dir = horizontal_component(g['normal'])
        if h_dir is None:
            continue

        # Check if opposite wall exists (direction within 30° of -h_dir)
        opposite_dir = -h_dir
        has_opposite = False
        for h2 in wall_horiz_dirs:
            if np.dot(opposite_dir, h2) > 0.85:  # within ~30°
                has_opposite = True
                break

        if not has_opposite:
            # Synthesize opposite wall
            # Normal = opposite horizontal direction (keep vertical = perpendicular to gravity)
            synth_normal = opposite_dir - np.dot(opposite_dir, gravity) * gravity
            synth_normal /= np.linalg.norm(synth_normal)

            # Position: project building centers onto this normal direction
            # Place at the far side of the building
            projections = bld_centers @ synth_normal
            # The wall should be at the extreme position in the normal direction
            # (opposite side from the observed wall)
            synth_d = projections.min()  # farthest in -normal direction
            synth_center = bld_center + (synth_d - np.dot(bld_center, synth_normal)) * synth_normal

            synthesized.append({
                'class': 2,
                'indices': np.array([], dtype=int),  # no real primitives
                'normal': synth_normal,
                'center': synth_center,
                'd': synth_d,
                'synthesized': True,
            })
            print(f"    Synthesized opposite wall: n=[{synth_normal[0]:.3f},{synth_normal[1]:.3f},{synth_normal[2]:.3f}]")

    # --- Synthesize ground face ---
    # Ground = horizontal plane at building bottom
    ground_normal = gravity.copy()  # points downward (outward for ground)
    ground_center = bld_center - (np.dot(bld_center, -gravity) - bld_bottom_h) * (-gravity)
    ground_d = np.dot(ground_normal, ground_center)

    synthesized.append({
        'class': 3,  # ground
        'indices': np.array([], dtype=int),
        'normal': ground_normal,
        'center': ground_center,
        'd': ground_d,
        'synthesized': True,
    })
    print(f"    Synthesized ground: n=[{ground_normal[0]:.3f},{ground_normal[1]:.3f},{ground_normal[2]:.3f}], "
          f"d={ground_d:.3f}")

    n_synth = len(synthesized) - len(groups)
    print(f"    Total: {len(groups)} observed + {n_synth} synthesized = {len(synthesized)} groups")

    return synthesized


# =============================================================================
# Primitive rectangle corners (from checkpoint)
# =============================================================================

_PRIM_CORNERS_CACHE = None


def _quaternion_mult_np(q1, q2):
    """Numpy batched quaternion multiply [w,x,y,z]."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _quat_to_rot_np(q):
    """Numpy quaternion to rotation matrix. (N,4) -> (N,3,3)"""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    qr, qi, qj, qk = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2*(qj**2 + qk**2)
    R[:, 0, 1] = 2*(qj*qi - qk*qr)
    R[:, 0, 2] = 2*(qi*qk + qr*qj)
    R[:, 1, 0] = 2*(qj*qi + qk*qr)
    R[:, 1, 1] = 1 - 2*(qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2*(qk*qi - qj*qr)
    R[:, 2, 1] = 2*(qj*qk + qi*qr)
    R[:, 2, 2] = 1 - 2*(qi**2 + qj**2)
    return R


def load_primitive_corners(ckpt_path=None):
    """Load checkpoint and compute 4 corners per primitive. Cached."""
    global _PRIM_CORNERS_CACHE
    if _PRIM_CORNERS_CACHE is not None:
        return _PRIM_CORNERS_CACHE

    if ckpt_path is None:
        ckpt_path = ('planarSplat_ExpRes/phase3b_prime/'
                     'cprime_independent_photo_example/'
                     '2026_03_11_14_06_52/checkpoints/Parameters/latest.pth')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model_state_dict']

    N = state['planarSplat._plane_center'].shape[0]
    centers = state['planarSplat._plane_center'].numpy()
    radii_p = state['planarSplat._plane_radii_xy_p'].numpy()  # real-space
    radii_n = state['planarSplat._plane_radii_xy_n'].numpy()

    rot_q_nwxy = state['planarSplat._plane_rot_q_normal_wxy'].numpy()
    rot_q_xyw = state['planarSplat._plane_rot_q_xyAxis_w'].numpy()
    rot_q_xyz = state['planarSplat._plane_rot_q_xyAxis_z'].numpy()

    q_normal = np.column_stack([rot_q_nwxy, np.zeros(N)])
    q_normal /= np.linalg.norm(q_normal, axis=-1, keepdims=True)
    q_xy = np.column_stack([rot_q_xyw, np.zeros((N, 2)), rot_q_xyz])
    q_xy /= np.linalg.norm(q_xy, axis=-1, keepdims=True)

    q = _quaternion_mult_np(q_normal, q_xy)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    R = _quat_to_rot_np(q)  # (N, 3, 3)

    # Compute 4 corners per primitive: v_local @ R^T + center
    all_corners = np.zeros((N, 4, 3))
    for i in range(N):
        rp = radii_p[i]
        rn = radii_n[i]
        local = np.array([
            [rp[0], rp[1], 0],
            [-rn[0], rp[1], 0],
            [-rn[0], -rn[1], 0],
            [rp[0], -rn[1], 0],
        ])
        all_corners[i] = local @ R[i].T + centers[i]

    _PRIM_CORNERS_CACHE = all_corners
    print(f"  Loaded {N} primitive corners from checkpoint")
    return all_corners


# =============================================================================
# Step 2: Compute merged extent for each cluster
# =============================================================================

def compute_cluster_extent(group, centers):
    """
    Compute the extent of a cluster on its plane using actual primitive corners.

    If primitive corners are available (real data), uses convex hull of all
    primitive corners projected onto the cluster plane.
    Otherwise falls back to bounding rectangle of centers.
    """
    n = group['normal']
    member_centers = centers[group['indices']]

    # Build local 2D coordinate system on the plane
    if abs(n[1]) < 0.9:
        up = np.array([0, 1, 0], dtype=np.float64)
    else:
        up = np.array([1, 0, 0], dtype=np.float64)

    axis_u = np.cross(n, up)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(n, axis_u)
    axis_v /= np.linalg.norm(axis_v)

    # Try to use actual primitive corners
    prim_corners = _PRIM_CORNERS_CACHE
    if prim_corners is not None and 'global_indices' in group:
        # Collect all 4 corners from each primitive in this cluster
        global_idx = group['global_indices']
        all_pts = prim_corners[global_idx].reshape(-1, 3)  # (N*4, 3)

        # Project all corners onto the cluster plane
        rel = all_pts - group['center']
        u_coords = rel @ axis_u
        v_coords = rel @ axis_v
    else:
        # Fallback: use primitive centers only
        rel = member_centers - group['center']
        u_coords = rel @ axis_u
        v_coords = rel @ axis_v

    # Bounding rectangle in local coords
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()

    rect_center_3d = group['center'] + \
        ((u_min + u_max) / 2) * axis_u + ((v_min + v_max) / 2) * axis_v
    half_u = (u_max - u_min) / 2
    half_v = (v_max - v_min) / 2

    # 4 corners in 3D
    corners = np.array([
        rect_center_3d + half_u * axis_u + half_v * axis_v,
        rect_center_3d - half_u * axis_u + half_v * axis_v,
        rect_center_3d - half_u * axis_u - half_v * axis_v,
        rect_center_3d + half_u * axis_u - half_v * axis_v,
    ])

    return {
        'center': rect_center_3d,
        'axes': np.array([axis_u, axis_v]),
        'half_extents': np.array([half_u, half_v]),
        'corners': corners,
    }


# =============================================================================
# Step 3: Plane-plane intersection → edge (line segment)
# =============================================================================

def intersect_two_planes(g1, g2):
    """
    Compute intersection line of two planes.

    Plane i: n_i · x = d_i

    Returns:
        line_dir: direction vector of intersection line
        line_point: a point on the intersection line
        None if planes are parallel
    """
    n1, d1 = g1['normal'], g1['d']
    n2, d2 = g2['normal'], g2['d']

    line_dir = np.cross(n1, n2)
    dir_norm = np.linalg.norm(line_dir)

    if dir_norm < 1e-10:
        return None  # parallel planes

    line_dir /= dir_norm

    # Find a point on the intersection line
    # Solve: n1 · p = d1, n2 · p = d2
    # Choose the component with largest |line_dir| as free parameter = 0
    abs_dir = np.abs(line_dir)
    free_idx = np.argmax(abs_dir)

    # Set free component to 0, solve 2x2
    other = [i for i in range(3) if i != free_idx]
    A = np.array([[n1[other[0]], n1[other[1]]],
                  [n2[other[0]], n2[other[1]]]])
    b = np.array([d1, d2])

    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        return None

    sol = np.linalg.solve(A, b)
    point = np.zeros(3)
    point[other[0]] = sol[0]
    point[other[1]] = sol[1]
    point[free_idx] = 0.0

    return {'dir': line_dir, 'point': point}


def project_extent_onto_line(extent, line):
    """
    Project a cluster's bounding rectangle corners onto the intersection line.
    Returns (t_min, t_max) — the range on line parameterized by t.

    t = (corner - line_point) · line_dir
    """
    corners = extent['corners']
    t_values = [(c - line['point']) @ line['dir'] for c in corners]
    return min(t_values), max(t_values)


def compute_edge_segment(g1, g2, ext1, ext2):
    """
    Compute the edge segment between two adjacent cluster rectangles.

    1. Find intersection line L of the two planes
    2. Project both rectangles' corners onto L → two intervals
    3. The edge = overlap of the two intervals (or gap analysis)

    Returns:
        dict with: p1, p2 (3D endpoints), overlap info, gap info
        None if planes are parallel
    """
    line = intersect_two_planes(g1, g2)
    if line is None:
        return None

    t1_min, t1_max = project_extent_onto_line(ext1, line)
    t2_min, t2_max = project_extent_onto_line(ext2, line)

    # Overlap analysis
    overlap_min = max(t1_min, t2_min)
    overlap_max = min(t1_max, t2_max)

    if overlap_min <= overlap_max:
        # Case A or C: overlapping intervals
        status = 'overlap'
        edge_min = overlap_min
        edge_max = overlap_max
        gap = 0.0
    else:
        # Case B: gap
        gap = overlap_min - overlap_max
        status = 'gap'
        # Use the full span from one end to the other
        edge_min = min(t1_min, t2_min)
        edge_max = max(t1_max, t2_max)

    p1 = line['point'] + edge_min * line['dir']
    p2 = line['point'] + edge_max * line['dir']

    return {
        'p1': p1, 'p2': p2,
        'line': line,
        'status': status,
        'gap': gap,
        't1_range': (t1_min, t1_max),
        't2_range': (t2_min, t2_max),
        'edge_range': (edge_min, edge_max),
    }


# =============================================================================
# Step 3b: Triple plane intersection → corners
# =============================================================================

def compute_triple_intersection(g1, g2, g3):
    """
    Compute the point where three planes meet.
    Plane i: n_i · x = d_i
    Returns 3D point or None if degenerate.
    """
    N = np.array([g1['normal'], g2['normal'], g3['normal']])
    d = np.array([g1['d'], g2['d'], g3['d']])

    det = np.linalg.det(N)
    if abs(det) < 1e-10:
        return None

    return np.linalg.solve(N, d)


def compute_corners(groups, extents, adjacency, all_centers):
    """
    Find building corners = points where 3 mutually adjacent planes meet.

    For each triple (i, j, k) of mutually adjacent groups:
      1. Compute triple intersection point
      2. Check if point is within building bounding box (with margin)

    Returns:
        corners: list of {'point': (3,), 'groups': (i,j,k)}
    """
    adj_set = set(adjacency.keys())
    group_indices = list(range(len(groups)))
    corners = []

    # Building bounding box from ALL primitive centers
    bbox_min = all_centers.min(axis=0)
    bbox_max = all_centers.max(axis=0)
    bbox_size = bbox_max - bbox_min
    # Margin: 50% of bbox size (corners may be slightly outside observed prims)
    margin = bbox_size * 0.5
    bbox_min_m = bbox_min - margin
    bbox_max_m = bbox_max + margin

    for i, j, k in combinations(group_indices, 3):
        # Check all three pairs are adjacent
        pairs = [(min(i,j), max(i,j)), (min(i,k), max(i,k)), (min(j,k), max(j,k))]
        if not all(p in adj_set for p in pairs):
            continue

        pt = compute_triple_intersection(groups[i], groups[j], groups[k])
        if pt is None:
            continue

        # Check if point is within building bbox (with margin)
        if np.all(pt >= bbox_min_m) and np.all(pt <= bbox_max_m):
            corners.append({'point': pt, 'groups': (i, j, k)})
        else:
            print(f"    [rejected] triple ({i},{j},{k}): "
                  f"[{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}] outside bbox")

    return corners


# =============================================================================
# Step 4: Corners → Polygons → LOD2
# =============================================================================

def build_polygons_from_corners(groups, extents, corners, adjacency):
    """
    For each group, collect corners that involve this group,
    then order them by angle to form a polygon.

    Uses shared-edge topology: two adjacent groups share exactly
    the corners that belong to both. The polygon's edge sequence
    follows this shared structure.
    """
    polygons = {}

    for gi, group in enumerate(groups):
        # Collect corners for this group
        group_corners = []
        corner_indices = []
        for ci, corner in enumerate(corners):
            if gi in corner['groups']:
                group_corners.append(corner['point'].copy())
                corner_indices.append(ci)

        if len(group_corners) < 3:
            polygons[gi] = None
            continue

        verts = np.array(group_corners)

        # Deduplicate nearby vertices
        unique_verts = [verts[0]]
        unique_idx = [corner_indices[0]]
        for v, idx in zip(verts[1:], corner_indices[1:]):
            if all(np.linalg.norm(v - u) > 0.01 for u in unique_verts):
                unique_verts.append(v)
                unique_idx.append(idx)
        verts = np.array(unique_verts)

        if len(verts) < 3:
            polygons[gi] = None
            continue

        # Project onto plane, sort by angle
        ext = extents[gi]
        axis_u, axis_v = ext['axes']
        centroid = verts.mean(axis=0)

        rel = verts - centroid
        u_coords = rel @ axis_u
        v_coords = rel @ axis_v
        angles = np.arctan2(v_coords, u_coords)
        order = np.argsort(angles)

        polygons[gi] = verts[order]

    return polygons


# =============================================================================
# Step 5: Topology guarantee
# =============================================================================

def verify_and_fix_topology(groups, polygons, corners, adjacency):
    """
    Verify and report topology properties:
    1. Shared edges: adjacent groups share exactly 2 corners → 1 edge
    2. Closed solid: every edge belongs to exactly 2 faces
    3. Orientation consistency: adjacent faces traverse shared edge in opposite directions

    Returns:
        topology_report: dict with stats and issues
    """
    report = {
        'shared_edges': [],
        'boundary_edges': [],
        'non_manifold_edges': [],
        'orientation_issues': [],
        'valid': True,
    }

    # Build corner→groups mapping
    corner_groups = {}
    for ci, corner in enumerate(corners):
        corner_groups[ci] = set(corner['groups'])

    # Build edge usage: edge = (corner_i, corner_j) sorted
    # For each polygon, list its edges as corner index pairs
    edge_faces = {}  # edge → list of (group_idx, direction)

    # Map polygon vertices back to corner indices
    poly_corner_map = {}  # gi → list of corner indices in polygon order
    for gi, poly in polygons.items():
        if poly is None:
            continue

        # Find which corner each polygon vertex corresponds to
        ci_list = []
        for v in poly:
            best_ci = None
            best_dist = float('inf')
            for ci, corner in enumerate(corners):
                if gi in corner['groups']:
                    d = np.linalg.norm(v - corner['point'])
                    if d < best_dist:
                        best_dist = d
                        best_ci = ci
            ci_list.append(best_ci)

        poly_corner_map[gi] = ci_list

        # Extract edges
        n = len(ci_list)
        for ei in range(n):
            c1 = ci_list[ei]
            c2 = ci_list[(ei + 1) % n]
            edge_key = (min(c1, c2), max(c1, c2))
            direction = 1 if c1 < c2 else -1

            if edge_key not in edge_faces:
                edge_faces[edge_key] = []
            edge_faces[edge_key].append((gi, direction))

    # Analyze edges
    for edge_key, faces_list in edge_faces.items():
        c1, c2 = edge_key
        p1 = corners[c1]['point']
        p2 = corners[c2]['point']
        length = np.linalg.norm(p2 - p1)

        edge_info = {
            'corners': edge_key,
            'faces': [f[0] for f in faces_list],
            'length': length,
        }

        if len(faces_list) == 2:
            # Check orientation: adjacent faces should traverse in opposite directions
            if faces_list[0][1] == faces_list[1][1]:
                report['orientation_issues'].append(edge_info)
                edge_info['issue'] = 'same_direction'
            report['shared_edges'].append(edge_info)
        elif len(faces_list) == 1:
            report['boundary_edges'].append(edge_info)
            report['valid'] = False
        else:
            report['non_manifold_edges'].append(edge_info)
            report['valid'] = False

    # Check: do adjacent groups share exactly one edge?
    adj_set = set(adjacency.keys())
    for (i, j) in adj_set:
        if polygons.get(i) is None or polygons.get(j) is None:
            continue

        # Count shared corners between groups i and j
        shared = []
        for ci, corner in enumerate(corners):
            if i in corner['groups'] and j in corner['groups']:
                shared.append(ci)

        if len(shared) < 2:
            report['valid'] = False

    return report


# =============================================================================
# Visualization Helpers
# =============================================================================

CLASS_COLORS = {
    0: [128, 128, 128],  # background - gray
    1: [255, 0, 0],      # roof - red
    2: [0, 0, 255],      # wall - blue
    3: [0, 255, 0],      # ground - green
}


def save_primitives_ply(path, centers, normals, classes):
    """Save primitives as colored point cloud."""
    n = len(centers)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            c = centers[i]
            nm = normals[i]
            rgb = CLASS_COLORS.get(int(classes[i]), [128, 128, 128])
            f.write(f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f} "
                    f"{nm[0]:.4f} {nm[1]:.4f} {nm[2]:.4f} "
                    f"{rgb[0]} {rgb[1]} {rgb[2]}\n")


def save_rectangles_ply(path, groups, extents):
    """Save cluster bounding rectangles as colored quads (triangle pairs)."""
    all_verts = []
    all_faces = []
    all_colors = []

    for gi, (group, ext) in enumerate(zip(groups, extents)):
        corners = ext['corners']  # (4, 3)
        base = len(all_verts)
        for c in corners:
            all_verts.append(c)
            all_colors.append(CLASS_COLORS.get(group['class'], [128, 128, 128]))

        # Two triangles for the quad
        all_faces.append([base, base + 1, base + 2])
        all_faces.append([base, base + 2, base + 3])

    nv = len(all_verts)
    nf = len(all_faces)

    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(all_verts, all_colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for face in all_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def save_edges_ply(path, edges_dict, groups):
    """Save edge segments as a line-segment PLY (using degenerate triangles)."""
    all_verts = []
    all_colors = []
    all_edges = []

    for (a, b), edge in edges_dict.items():
        if edge is None:
            continue
        base = len(all_verts)
        p1, p2 = edge['p1'], edge['p2']
        # Midpoint for a thin triangle
        mid = (p1 + p2) / 2 + np.array([0, 0.01, 0])

        # Color by status
        if edge['status'] == 'overlap':
            color = [0, 255, 0]  # green = good overlap
        else:
            color = [255, 255, 0]  # yellow = gap

        all_verts.extend([p1, p2, mid])
        all_colors.extend([color, color, color])
        all_edges.append([base, base + 1, base + 2])

    nv = len(all_verts)
    nf = len(all_edges)

    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(all_verts, all_colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for face in all_edges:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def save_intersection_line_ply(path, line, t_range=(-15, 15)):
    """Save a full intersection line for visualization."""
    p1 = line['point'] + t_range[0] * line['dir']
    p2 = line['point'] + t_range[1] * line['dir']
    mid = (p1 + p2) / 2 + np.array([0, 0.02, 0])

    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex 3\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("element face 1\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        f.write(f"{p1[0]:.4f} {p1[1]:.4f} {p1[2]:.4f} 255 0 255\n")
        f.write(f"{p2[0]:.4f} {p2[1]:.4f} {p2[2]:.4f} 255 0 255\n")
        f.write(f"{mid[0]:.4f} {mid[1]:.4f} {mid[2]:.4f} 255 0 255\n")
        f.write("3 0 1 2\n")


def save_projection_ply(path, line, t1_range, t2_range):
    """Save the projected intervals on line L as colored segments."""
    all_verts = []
    all_colors = []
    all_faces = []

    # Interval 1 (red)
    p1a = line['point'] + t1_range[0] * line['dir']
    p1b = line['point'] + t1_range[1] * line['dir']
    offset1 = np.array([0, 0.05, 0])
    mid1 = (p1a + p1b) / 2 + offset1
    base = 0
    all_verts.extend([p1a + offset1, p1b + offset1, mid1 + offset1])
    all_colors.extend([[255, 100, 100]] * 3)
    all_faces.append([base, base + 1, base + 2])

    # Interval 2 (blue)
    p2a = line['point'] + t2_range[0] * line['dir']
    p2b = line['point'] + t2_range[1] * line['dir']
    offset2 = np.array([0, 0.10, 0])
    mid2 = (p2a + p2b) / 2 + offset2
    base = 3
    all_verts.extend([p2a + offset2, p2b + offset2, mid2 + offset2])
    all_colors.extend([[100, 100, 255]] * 3)
    all_faces.append([base, base + 1, base + 2])

    nv = len(all_verts)
    nf = len(all_faces)

    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(all_verts, all_colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for face in all_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def save_polygons_ply(path, polygons, groups):
    """Save final LOD2 polygons as colored mesh."""
    all_verts = []
    all_faces = []
    all_colors = []

    for gi, poly in polygons.items():
        if poly is None:
            continue
        base = len(all_verts)
        color = CLASS_COLORS.get(groups[gi]['class'], [128, 128, 128])

        for v in poly:
            all_verts.append(v)
            all_colors.append(color)

        # Fan triangulation
        for i in range(1, len(poly) - 1):
            all_faces.append([base, base + i, base + i + 1])

    nv = len(all_verts)
    nf = len(all_faces)

    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(all_verts, all_colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for face in all_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def save_gt_building_ply(path, vertices, faces, labels):
    """Save ground truth building as colored mesh."""
    all_verts = []
    all_faces = []
    all_colors = []

    for fi, (face, label) in enumerate(zip(faces, labels)):
        base = len(all_verts)
        color = CLASS_COLORS.get(label, [128, 128, 128])

        for vi in face:
            all_verts.append(vertices[vi])
            all_colors.append(color)

        for i in range(1, len(face) - 1):
            all_faces.append([base, base + i, base + i + 1])

    nv = len(all_verts)
    nf = len(all_faces)

    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(all_verts, all_colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for face in all_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# =============================================================================
# Main: Step-by-step with intermediate output
# =============================================================================

def determine_adjacency(groups, all_centers):
    """
    Determine which groups should be adjacent.

    Two groups are adjacent if:
    1. Their planes are not parallel (normals not aligned)
    2. Their members are spatially close (min distance between centers)
    3. The intersection line passes near both groups' members
    """
    adj = {}
    n = len(groups)

    # Compute building scale for threshold
    bbox_size = all_centers.max(axis=0) - all_centers.min(axis=0)
    building_scale = np.linalg.norm(bbox_size)
    proximity_thresh = building_scale * 0.6  # groups must be within 60% of building size

    def point_line_dist(p, line):
        v = p - line['point']
        proj = np.dot(v, line['dir']) * line['dir']
        return np.linalg.norm(v - proj)

    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = groups[i], groups[j]

            # Skip parallel planes
            cos_angle = abs(np.dot(gi['normal'], gj['normal']))
            if cos_angle > 0.95:
                continue

            # Synthesized faces: adjacent if not parallel (skip spatial checks)
            is_synth = gi.get('synthesized', False) or gj.get('synthesized', False)

            if not is_synth:
                # Spatial proximity for observed groups
                center_dist = np.linalg.norm(gi['center'] - gj['center'])
                if center_dist > proximity_thresh:
                    continue

                line = intersect_two_planes(gi, gj)
                if line is None:
                    continue

                dist_i = point_line_dist(gi['center'], line)
                dist_j = point_line_dist(gj['center'], line)

                max_dist = building_scale * 0.5
                if dist_i > max_dist or dist_j > max_dist:
                    continue
            else:
                # For synthesized: just check intersection exists
                line = intersect_two_planes(gi, gj)
                if line is None:
                    continue

            adj[(i, j)] = True

    return adj


GRAVITY = np.array([0.1223, 0.3702, 0.9209])
GRAVITY = GRAVITY / np.linalg.norm(GRAVITY)


def load_real_building(building_id, prims_path='results/phase4/part1/primitives_data.npz',
                       faces_path='results/phase4_prime/step3_buildings/building_faces.npz',
                       apply_quality_filter=True):
    """
    Load real PlanarSplatting primitives for a specific building.

    Returns:
        centers, normals, classes (filtered to this building, ground/bg removed,
        optionally quality-filtered)
    """
    prims = np.load(prims_path)
    faces = np.load(faces_path, allow_pickle=True)

    # Get primitive indices for this building
    bld_mask = faces['building_ids'] == building_id
    face_prim_indices = faces['face_prim_indices'][bld_mask]
    unique_prims = np.unique(face_prim_indices)
    unique_prims = unique_prims[unique_prims >= 0]

    centers = prims['centers'][unique_prims]
    normals = prims['normals'][unique_prims]
    classes = prims['class_pred'][unique_prims]
    sprobs = prims['semantic_probs'][unique_prims]

    # Filter out ground (class=3) and background (class=0)
    building_mask = (classes == 1) | (classes == 2)
    global_prim_idx = unique_prims[building_mask]
    centers = centers[building_mask]
    normals = normals[building_mask]
    sprobs = sprobs[building_mask]
    classes = classes[building_mask]

    n_before = len(centers)

    if apply_quality_filter and len(centers) > 0:
        dots = np.abs(normals @ GRAVITY)
        keep = np.ones(len(classes), dtype=bool)
        for i in range(len(classes)):
            if classes[i] == 2:  # wall
                if dots[i] > 0.5 or sprobs[i, 2] < 0.5:
                    keep[i] = False
            if classes[i] == 1:  # roof
                if dots[i] < 0.3 or sprobs[i, 1] < 0.5:
                    keep[i] = False
        centers = centers[keep]
        normals = normals[keep]
        classes = classes[keep]
        global_prim_idx = global_prim_idx[keep]
        n_filtered = n_before - len(centers)
        print(f"  Building {building_id}: {n_before} prims (roof+wall), "
              f"filtered {n_filtered} -> {len(centers)} remaining")
    else:
        print(f"  Building {building_id}: {n_before} prims (roof+wall), no filter")

    print(f"  Classes: roof={np.sum(classes==1)}, wall={np.sum(classes==2)}")

    return centers, normals, classes, global_prim_idx


def run_real_building(building_id, out_dir, cos_thresh=0.70):
    """Run cluster-intersection pipeline on a real building."""
    bld_dir = os.path.join(out_dir, f'building_{building_id:03d}')
    os.makedirs(bld_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Building {building_id}")
    print(f"{'='*60}")

    # Load data
    centers, normals, classes, global_prim_idx = load_real_building(building_id)
    if len(centers) < 3:
        print(f"  SKIP: too few primitives ({len(centers)})")
        return None

    # Load primitive corners for extent computation
    load_primitive_corners()

    # Save input primitives
    save_primitives_ply(os.path.join(bld_dir, 'step0_primitives.ply'),
                        centers, normals, classes)

    # Step 1: Clustering
    print(f"\n  [Step 1a] Clustering (cos_thresh={cos_thresh})...")
    groups = cluster_primitives(centers, normals, classes, cos_thresh=cos_thresh)
    print(f"    {len(groups)} raw groups")

    # Step 1b: Refine clusters
    print(f"\n  [Step 1b] Refining clusters (min_wall=10, min_roof=3, outlier_std=2.0)...")
    groups = refine_clusters(groups, centers, normals,
                             min_wall_size=10, min_roof_size=3,
                             spatial_outlier_std=2.0)

    # Attach global primitive indices to each group for corner lookup
    for g in groups:
        if len(g['indices']) > 0:
            g['global_indices'] = global_prim_idx[g['indices']]
        else:
            g['global_indices'] = np.array([], dtype=int)

    print(f"    {len(groups)} refined groups:")
    for gi, g in enumerate(groups):
        cls_name = {1: 'roof', 2: 'wall', 3: 'ground'}.get(g['class'], '?')
        print(f"    G{gi}: {cls_name}, n={len(g['indices'])}, "
              f"normal=[{g['normal'][0]:.3f}, {g['normal'][1]:.3f}, {g['normal'][2]:.3f}], "
              f"d={g['d']:.3f}")

    # Step 1c: Synthesize missing faces
    print(f"\n  [Step 1c] Synthesizing missing faces...")
    groups = synthesize_missing_faces(groups, centers)

    # Re-attach global indices for synthesized groups
    for g in groups:
        if 'global_indices' not in g:
            g['global_indices'] = np.array([], dtype=int)

    print(f"    Final {len(groups)} groups:")
    for gi, g in enumerate(groups):
        cls_name = {1: 'roof', 2: 'wall', 3: 'ground'}.get(g['class'], '?')
        synth = " [SYNTH]" if g.get('synthesized') else ""
        print(f"    G{gi}: {cls_name}, n={len(g['indices'])}, "
              f"normal=[{g['normal'][0]:.3f}, {g['normal'][1]:.3f}, {g['normal'][2]:.3f}], "
              f"d={g['d']:.3f}{synth}")

    # Step 2: Extents (skip synthesized groups with no primitives)
    print(f"\n  [Step 2] Cluster extents...")
    extents = []
    for g in groups:
        if len(g['indices']) > 0:
            extents.append(compute_cluster_extent(g, centers))
        else:
            # Synthesized: no extent, use placeholder
            extents.append({
                'center': g['center'],
                'axes': np.eye(3)[:2],
                'half_extents': np.array([0.5, 0.5]),
                'corners': np.array([g['center']] * 4),
            })
    # Only save observed groups in PLY
    obs_groups = [g for g in groups if len(g['indices']) > 0]
    obs_extents = [e for g, e in zip(groups, extents) if len(g['indices']) > 0]
    if obs_groups:
        save_rectangles_ply(os.path.join(bld_dir, 'step2_rectangles.ply'), obs_groups, obs_extents)

    # Step 3: Adjacency + edges
    print(f"\n  [Step 3] Adjacency + edges...")
    adjacency = determine_adjacency(groups, centers)
    print(f"    Adjacency pairs: {len(adjacency)}")

    edges_dict = {}
    for (i, j) in adjacency:
        edge = compute_edge_segment(groups[i], groups[j], extents[i], extents[j])
        edges_dict[(i, j)] = edge
    save_edges_ply(os.path.join(bld_dir, 'step3_edges.ply'), edges_dict, groups)

    # Step 3b: Corners
    print(f"\n  [Step 3b] Corners...")
    corners = compute_corners(groups, extents, adjacency, centers)
    print(f"    {len(corners)} corners found")
    for ci, corner in enumerate(corners):
        pt = corner['point']
        grp_names = [f"G{g}" for g in corner['groups']]
        print(f"    C{ci}: [{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}] ← {','.join(grp_names)}")

    if len(corners) < 4:
        print(f"  FAIL: too few corners ({len(corners)}) for a solid")
        return {'building_id': building_id, 'status': 'fail_corners',
                'n_groups': len(groups), 'n_corners': len(corners)}

    # Step 4: Polygons
    print(f"\n  [Step 4] Polygons...")
    polygons = build_polygons_from_corners(groups, extents, corners, adjacency)
    valid_polys = {k: v for k, v in polygons.items() if v is not None}
    print(f"    {len(valid_polys)}/{len(groups)} groups → valid polygons")
    for gi, poly in polygons.items():
        if poly is not None:
            cls_name = {1: 'roof', 2: 'wall', 3: 'ground'}.get(groups[gi]['class'], '?')
            print(f"    G{gi} ({cls_name}): {len(poly)} vertices")
    save_polygons_ply(os.path.join(bld_dir, 'step4_polygons.ply'), polygons, groups)

    # Step 5: Topology
    print(f"\n  [Step 5] Topology...")
    topo = verify_and_fix_topology(groups, polygons, corners, adjacency)
    print(f"    Shared: {len(topo['shared_edges'])}, "
          f"Boundary: {len(topo['boundary_edges'])}, "
          f"Non-manifold: {len(topo['non_manifold_edges'])}, "
          f"Orientation: {len(topo['orientation_issues'])}")
    print(f"    TOPOLOGY: {'✓' if topo['valid'] else '✗'}")

    return {
        'building_id': building_id,
        'status': 'ok',
        'n_prims': len(centers),
        'n_groups': len(groups),
        'n_corners': len(corners),
        'n_valid_polys': len(valid_polys),
        'shared_edges': len(topo['shared_edges']),
        'boundary_edges': len(topo['boundary_edges']),
        'non_manifold_edges': len(topo['non_manifold_edges']),
        'topology_valid': topo['valid'],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='results/phase4_prime/cluster_intersection_test')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Positional noise std for primitives')
    parser.add_argument('--n_prims', type=int, default=6,
                        help='Number of primitives per face')
    parser.add_argument('--building', default='box', choices=['box', 'gable'],
                        help='Building type: box (LOD1) or gable (LOD2)')
    # Real data mode
    parser.add_argument('--real', action='store_true',
                        help='Run on real PlanarSplatting data')
    parser.add_argument('--building_ids', type=str, default='14',
                        help='Comma-separated building IDs (or "all")')
    parser.add_argument('--cos_thresh', type=float, default=0.70,
                        help='Cosine threshold for clustering')
    args = parser.parse_args()

    if args.real:
        out_dir = os.path.join(args.out_dir, 'real')
        os.makedirs(out_dir, exist_ok=True)

        if args.building_ids == 'all':
            faces = np.load('results/phase4_prime/step3_buildings/building_faces.npz',
                           allow_pickle=True)
            bids = [b for b in np.unique(faces['building_ids']) if b >= 0]
        else:
            bids = [int(x) for x in args.building_ids.split(',')]

        results = []
        for bid in bids:
            r = run_real_building(bid, out_dir, cos_thresh=args.cos_thresh)
            if r:
                results.append(r)

        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(results)} buildings processed")
        print(f"{'='*60}")
        ok = [r for r in results if r['status'] == 'ok']
        topo_valid = [r for r in ok if r['topology_valid']]
        print(f"  OK: {len(ok)}/{len(results)}")
        print(f"  Topology valid: {len(topo_valid)}/{len(ok)}")
        if ok:
            avg_groups = np.mean([r['n_groups'] for r in ok])
            avg_corners = np.mean([r['n_corners'] for r in ok])
            avg_shared = np.mean([r['shared_edges'] for r in ok])
            avg_boundary = np.mean([r['boundary_edges'] for r in ok])
            print(f"  Avg groups: {avg_groups:.1f}, corners: {avg_corners:.1f}")
            print(f"  Avg shared edges: {avg_shared:.1f}, boundary: {avg_boundary:.1f}")

        return

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    bldg_type = args.building.upper()
    print("=" * 60)
    print(f"Cluster-Intersection CityGML Test (Synthetic {bldg_type} Building)")
    print("=" * 60)

    # ── Step 0: Create building and primitives ──
    print(f"\n[Step 0] Creating synthetic {bldg_type} building...")
    if args.building == 'gable':
        vertices, faces, labels, names, gt_normals = create_gable_building()
    else:
        vertices, faces, labels, names, gt_normals = create_box_building()
    print(f"  Building: 10m x 8m x 6m box, {len(faces)} faces")
    for i, name in enumerate(names):
        print(f"  Face {i}: {name}, class={labels[i]}, normal={gt_normals[i]}")

    # Save GT building
    save_gt_building_ply(os.path.join(out_dir, 'step0_gt_building.ply'),
                         vertices, faces, labels)

    # Generate primitives
    rng = np.random.RandomState(42)
    all_centers, all_normals, all_classes = [], [], []

    for fi, (face, label) in enumerate(zip(faces, labels)):
        c, n = generate_primitives_on_face(
            vertices, face, gt_normals[fi], label,
            n_prims=args.n_prims, noise_std=args.noise, rng=rng
        )
        all_centers.append(c)
        all_normals.append(n)
        all_classes.extend([label] * len(c))

    all_centers = np.vstack(all_centers)
    all_normals = np.vstack(all_normals)
    all_classes = np.array(all_classes)
    print(f"  Generated {len(all_centers)} primitives (noise_std={args.noise})")

    save_primitives_ply(os.path.join(out_dir, 'step0_primitives.ply'),
                        all_centers, all_normals, all_classes)
    print(f"  → Saved: step0_gt_building.ply, step0_primitives.ply")

    # ── Step 1: Clustering ──
    print("\n[Step 1] Clustering primitives → surface groups...")
    groups = cluster_primitives(all_centers, all_normals, all_classes)
    print(f"  Found {len(groups)} groups:")
    for gi, g in enumerate(groups):
        cls_name = {1: 'roof', 2: 'wall', 3: 'ground'}[g['class']]
        print(f"  Group {gi}: class={cls_name}, n_prims={len(g['indices'])}, "
              f"normal=[{g['normal'][0]:.3f}, {g['normal'][1]:.3f}, {g['normal'][2]:.3f}], "
              f"d={g['d']:.3f}")

    # ── Step 2: Compute merged extents ──
    print("\n[Step 2] Computing cluster bounding rectangles...")
    extents = [compute_cluster_extent(g, all_centers) for g in groups]
    for gi, (g, ext) in enumerate(zip(groups, extents)):
        cls_name = {1: 'roof', 2: 'wall', 3: 'ground'}[g['class']]
        print(f"  Group {gi} ({cls_name}): half_extents=[{ext['half_extents'][0]:.2f}, "
              f"{ext['half_extents'][1]:.2f}]")

    save_rectangles_ply(os.path.join(out_dir, 'step2_cluster_rectangles.ply'),
                        groups, extents)
    print(f"  → Saved: step2_cluster_rectangles.ply")

    # ── Step 3: Plane intersection → edges ──
    print("\n[Step 3] Computing plane intersections → edges...")
    adjacency = determine_adjacency(groups, all_centers)
    print(f"  Adjacency pairs: {len(adjacency)}")

    edges_dict = {}
    for (i, j) in adjacency:
        edge = compute_edge_segment(groups[i], groups[j], extents[i], extents[j])
        edges_dict[(i, j)] = edge

        if edge:
            gi_name = {1: 'roof', 2: 'wall', 3: 'ground'}[groups[i]['class']]
            gj_name = {1: 'roof', 2: 'wall', 3: 'ground'}[groups[j]['class']]
            length = np.linalg.norm(edge['p2'] - edge['p1'])
            print(f"  Edge ({gi_name} {i}, {gj_name} {j}): "
                  f"status={edge['status']}, gap={edge['gap']:.3f}, "
                  f"length={length:.2f}")
            print(f"    L range: t1=[{edge['t1_range'][0]:.2f}, {edge['t1_range'][1]:.2f}], "
                  f"t2=[{edge['t2_range'][0]:.2f}, {edge['t2_range'][1]:.2f}], "
                  f"edge=[{edge['edge_range'][0]:.2f}, {edge['edge_range'][1]:.2f}]")

            # Save individual intersection line + projections for first pair
            pair_dir = os.path.join(out_dir, f'step3_pair_{i}_{j}')
            os.makedirs(pair_dir, exist_ok=True)
            save_intersection_line_ply(
                os.path.join(pair_dir, 'intersection_line_L.ply'),
                edge['line'])
            save_projection_ply(
                os.path.join(pair_dir, 'projected_intervals.ply'),
                edge['line'], edge['t1_range'], edge['t2_range'])

    save_edges_ply(os.path.join(out_dir, 'step3_edges.ply'), edges_dict, groups)
    print(f"  → Saved: step3_edges.ply + per-pair details")

    # ── Step 3b: Triple intersection → corners ──
    print("\n[Step 3b] Computing triple plane intersections → corners...")
    corners = compute_corners(groups, extents, adjacency, all_centers)
    print(f"  Found {len(corners)} corners:")
    for ci, corner in enumerate(corners):
        pt = corner['point']
        grps = corner['groups']
        grp_names = []
        for g in grps:
            grp_names.append(f"{g}({['','roof','wall','ground'][groups[g]['class']]})")
        print(f"  Corner {ci}: [{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}] "
              f"← groups {', '.join(grp_names)}")

    # ── Step 4: Corners → Polygons → LOD2 ──
    print("\n[Step 4] Building polygons from corners → LOD2...")
    polygons = build_polygons_from_corners(groups, extents, corners, adjacency)

    for gi, poly in polygons.items():
        if poly is None:
            print(f"  Group {gi}: FAILED (too few vertices)")
        else:
            cls_name = {1: 'roof', 2: 'wall', 3: 'ground'}[groups[gi]['class']]
            print(f"  Group {gi} ({cls_name}): {len(poly)} vertices")
            for vi, v in enumerate(poly):
                print(f"    v{vi}: [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]")

    save_polygons_ply(os.path.join(out_dir, 'step4_lod2_polygons.ply'),
                      polygons, groups)
    print(f"  → Saved: step4_lod2_polygons.ply")

    # ── Step 5: Topology verification ──
    print("\n[Step 5] Topology verification...")
    topo = verify_and_fix_topology(groups, polygons, corners, adjacency)
    print(f"  Shared edges: {len(topo['shared_edges'])}")
    for e in topo['shared_edges']:
        faces_str = ', '.join(str(f) for f in e['faces'])
        cls_names = [['','roof','wall','ground'][groups[f]['class']] for f in e['faces']]
        print(f"    corners ({e['corners'][0]},{e['corners'][1]}): "
              f"length={e['length']:.2f}, faces=[{', '.join(cls_names)}]"
              f"{' ⚠ ORIENTATION' if e.get('issue') else ''}")
    print(f"  Boundary edges (should be 0): {len(topo['boundary_edges'])}")
    for e in topo['boundary_edges']:
        print(f"    corners ({e['corners'][0]},{e['corners'][1]}): "
              f"length={e['length']:.2f}, face={e['faces']}")
    print(f"  Non-manifold edges (should be 0): {len(topo['non_manifold_edges'])}")
    print(f"  Orientation issues: {len(topo['orientation_issues'])}")
    print(f"  TOPOLOGY VALID: {'✓' if topo['valid'] else '✗'}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("OUTPUT FILES (load in MeshLab/CloudCompare):")
    print(f"  {out_dir}/step0_gt_building.ply     — Ground truth building")
    print(f"  {out_dir}/step0_primitives.ply       — Input primitives (colored by class)")
    print(f"  {out_dir}/step2_cluster_rectangles.ply — Merged cluster rectangles")
    print(f"  {out_dir}/step3_edges.ply            — Edge segments (green=overlap, yellow=gap)")
    print(f"  {out_dir}/step3_pair_*/               — Per-pair: intersection line + projections")
    print(f"  {out_dir}/step4_lod2_polygons.ply    — Final LOD2 polygons")
    print("=" * 60)

    # Compare GT vs result
    print("\n[Comparison] GT vertices vs LOD2 polygon vertices:")
    print(f"  GT building has {len(vertices)} vertices:")
    for vi, v in enumerate(vertices):
        print(f"    GT v{vi}: [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]")


if __name__ == '__main__':
    main()
