#!/usr/bin/env python3
"""Phase 4' Step 3: Extract individual buildings from labeled mesh.

Building separation strategy: Roof-seed approach.
1. Find roof connected components on mesh → each = one building seed
2. BFS from each roof seed through wall faces → assign wall to nearest roof
3. Wall faces equidistant or unreachable → assigned by mesh proximity

This avoids two failure modes:
- Primitive XZ clustering: merged adjacent buildings (only 11 found)
- Full building CC: wall faces connect adjacent buildings on mesh

Key insight: buildings don't share roofs. Roof connectivity on mesh
naturally separates buildings even when walls are adjacent.

Usage (inside Docker):
    python scripts/extract_buildings.py \
        --mesh results/phase4/poisson/poisson_trimmed.ply \
        --labels results/phase4_prime/step2_labeling/face_labels.npz \
        --output_dir results/phase4_prime/step3_buildings
"""
import os
import json
import argparse
import time
from collections import defaultdict, deque

import numpy as np
import open3d as o3d


CLASS_NAMES = ['bg', 'roof', 'wall', 'ground']


def build_face_adjacency(triangles):
    """Build face adjacency from triangle mesh edges."""
    edge_faces = defaultdict(list)
    for fi in range(len(triangles)):
        t = triangles[fi]
        for e in [(min(t[0], t[1]), max(t[0], t[1])),
                  (min(t[1], t[2]), max(t[1], t[2])),
                  (min(t[0], t[2]), max(t[0], t[2]))]:
            edge_faces[e].append(fi)

    face_neighbors = defaultdict(set)
    for edge, faces in edge_faces.items():
        if len(faces) == 2:
            face_neighbors[faces[0]].add(faces[1])
            face_neighbors[faces[1]].add(faces[0])

    return face_neighbors


def find_roof_components(face_labels, face_neighbors, min_roof_faces=100):
    """Find connected components of roof faces (label=1).

    Returns:
        list of (face_indices_set, component_size) sorted by size descending
    """
    roof_faces = set(np.where(face_labels == 1)[0])
    visited = set()
    components = []

    for f in roof_faces:
        if f in visited:
            continue
        comp = set()
        queue = deque([f])
        while queue:
            cf = queue.popleft()
            if cf in visited or cf not in roof_faces:
                continue
            visited.add(cf)
            comp.add(cf)
            for nf in face_neighbors[cf]:
                if nf not in visited and nf in roof_faces:
                    queue.append(nf)
        components.append(comp)

    # Filter by minimum size and sort descending
    components = [c for c in components if len(c) >= min_roof_faces]
    components.sort(key=len, reverse=True)
    return components


def assign_walls_to_roofs(face_labels, face_neighbors, roof_components,
                          face_normals):
    """Assign wall faces to nearest roof component via BFS on mesh.

    Multi-source BFS: all roof components expand simultaneously.
    Wall faces are assigned to the roof component that reaches them first.
    BFS only traverses through building faces (roof=1 or wall=2).

    Args:
        face_labels: (M,) face labels
        face_neighbors: dict of face adjacency
        roof_components: list of sets of face indices (roof seeds)
        face_normals: (M, 3) face normals (for potential normal-based splitting)

    Returns:
        building_ids: (M,) array, -1 for non-building faces
    """
    M = len(face_labels)
    building_ids = np.full(M, -1, dtype=np.int64)
    distances = np.full(M, np.inf)

    # Initialize: assign roof faces to their component
    queue = deque()
    for bid, roof_faces in enumerate(roof_components):
        for f in roof_faces:
            building_ids[f] = bid
            distances[f] = 0
            queue.append((f, bid, 0))

    # BFS: expand through wall faces only
    wall_faces = set(np.where(face_labels == 2)[0])

    while queue:
        f, bid, dist = queue.popleft()

        for nf in face_neighbors.get(f, set()):
            new_dist = dist + 1
            if nf in wall_faces and new_dist < distances[nf]:
                building_ids[nf] = bid
                distances[nf] = new_dist
                queue.append((nf, bid, new_dist))

    return building_ids


def extract_submesh(mesh, face_indices):
    """Extract a submesh containing only the specified faces."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    sub_tris = triangles[face_indices]
    unique_verts = np.unique(sub_tris.flatten())
    vert_map = {old: new for new, old in enumerate(unique_verts)}

    new_verts = vertices[unique_verts]
    new_tris = np.array([[vert_map[v] for v in tri] for tri in sub_tris])

    sub_mesh = o3d.geometry.TriangleMesh()
    sub_mesh.vertices = o3d.utility.Vector3dVector(new_verts)
    sub_mesh.triangles = o3d.utility.Vector3iVector(new_tris)
    sub_mesh.compute_vertex_normals()

    return sub_mesh


def color_submesh_by_labels(sub_mesh, face_labels_sub):
    """Color submesh faces by semantic labels."""
    CLASS_COLORS = np.array([
        [0.5, 0.5, 0.5],  # bg
        [0.9, 0.2, 0.2],  # roof
        [0.2, 0.2, 0.9],  # wall
        [0.2, 0.8, 0.2],  # ground
    ])

    n_faces = len(sub_mesh.triangles)
    face_colors = CLASS_COLORS[face_labels_sub]

    vertices = np.asarray(sub_mesh.vertices)
    triangles = np.asarray(sub_mesh.triangles)
    vertex_colors = np.zeros((len(vertices), 3))
    vertex_counts = np.zeros(len(vertices))

    for i in range(n_faces):
        for vi in triangles[i]:
            vertex_colors[vi] += face_colors[i]
            vertex_counts[vi] += 1

    vertex_counts = np.maximum(vertex_counts, 1)
    vertex_colors /= vertex_counts[:, None]

    sub_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return sub_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', default='results/phase4/poisson/poisson_trimmed.ply')
    parser.add_argument('--labels', default='results/phase4_prime/step2_labeling/face_labels.npz')
    parser.add_argument('--output_dir', default='results/phase4_prime/step3_buildings')
    parser.add_argument('--min_roof_faces', type=int, default=100,
                        help='Minimum roof faces for a building seed')
    parser.add_argument('--min_building_faces', type=int, default=500,
                        help='Minimum total faces (roof+wall) for a building to be kept')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load mesh and labels ──
    print(f"Loading mesh: {args.mesh}")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    triangles = np.asarray(mesh.triangles)
    face_normals = np.asarray(mesh.triangle_normals)
    n_faces = len(triangles)
    print(f"  Faces: {n_faces:,}")

    print(f"Loading labels: {args.labels}")
    data = np.load(args.labels)
    face_labels = data['face_labels']
    face_prim_indices = data['face_prim_indices']

    n_roof = (face_labels == 1).sum()
    n_wall = (face_labels == 2).sum()
    print(f"  Roof faces: {n_roof:,}, Wall faces: {n_wall:,}")

    # ── 2. Build face adjacency ──
    print(f"\nBuilding face adjacency...")
    t0 = time.time()
    face_neighbors = build_face_adjacency(triangles)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ── 3. Find roof connected components (building seeds) ──
    print(f"\nFinding roof connected components (min_roof_faces={args.min_roof_faces})...")
    roof_components = find_roof_components(face_labels, face_neighbors,
                                           min_roof_faces=args.min_roof_faces)
    print(f"  Roof components found: {len(roof_components)}")
    for i, rc in enumerate(roof_components):
        print(f"    Roof {i}: {len(rc)} faces")

    # ── 4. Assign wall faces to nearest roof via BFS ──
    print(f"\nAssigning wall faces to nearest roof component (BFS)...")
    t0 = time.time()
    building_ids = assign_walls_to_roofs(
        face_labels, face_neighbors, roof_components, face_normals
    )
    t_assign = time.time() - t0

    n_wall_assigned = ((face_labels == 2) & (building_ids >= 0)).sum()
    n_wall_unassigned = ((face_labels == 2) & (building_ids < 0)).sum()
    print(f"  Wall assigned: {n_wall_assigned:,}")
    print(f"  Wall unassigned: {n_wall_unassigned:,} (no roof reachable)")
    print(f"  Time: {t_assign:.1f}s")

    # ── 5. Extract and analyze buildings ──
    print(f"\n{'='*65}")
    print(f"  BUILDING EXTRACTION (Roof-seed)")
    print(f"{'='*65}")
    print(f"  {'ID':<4} {'Total':>8} {'Roof':>8} {'Wall':>8} {'Roof%':>6} "
          f"{'Verts':>8} {'WT':>4} {'EM':>4}")
    print(f"  {'-'*60}")

    building_results = []

    for bid in range(len(roof_components)):
        bmask = building_ids == bid
        if bmask.sum() < args.min_building_faces:
            continue

        face_indices = np.where(bmask)[0]
        sub_labels = face_labels[face_indices]
        b_n_roof = (sub_labels == 1).sum()
        b_n_wall = (sub_labels == 2).sum()
        b_total = len(face_indices)

        # Extract submesh
        sub_mesh = extract_submesh(mesh, face_indices)
        n_verts = len(sub_mesh.vertices)

        # Topology
        is_wt = sub_mesh.is_watertight()
        is_em = sub_mesh.is_edge_manifold()

        # Bbox
        bbox = sub_mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        center = bbox.get_center()

        roof_pct = 100 * b_n_roof / b_total

        print(f"  {bid:<4} {b_total:>8,} {b_n_roof:>8,} {b_n_wall:>8,} "
              f"{roof_pct:>5.1f}% {n_verts:>8,} "
              f"{'Y' if is_wt else 'n':>4} {'Y' if is_em else 'n':>4}")

        # Color and save
        sub_mesh = color_submesh_by_labels(sub_mesh, sub_labels)
        ply_path = os.path.join(args.output_dir, f'building_{bid:03d}.ply')
        o3d.io.write_triangle_mesh(ply_path, sub_mesh, write_ascii=True)

        building_results.append({
            'building_id': bid,
            'n_faces': b_total,
            'n_roof': int(b_n_roof),
            'n_wall': int(b_n_wall),
            'roof_ratio': round(roof_pct, 1),
            'n_vertices': n_verts,
            'is_watertight': is_wt,
            'is_edge_manifold': is_em,
            'bbox_center': center.tolist(),
            'bbox_extent': extent.tolist(),
        })

    # ── 6. Save data ──
    summary = {
        'method': 'roof-seed BFS',
        'mesh': args.mesh,
        'n_total_faces': n_faces,
        'n_roof_components': len(roof_components),
        'min_roof_faces': args.min_roof_faces,
        'min_building_faces': args.min_building_faces,
        'n_buildings': len(building_results),
        'n_wall_assigned': int(n_wall_assigned),
        'n_wall_unassigned': int(n_wall_unassigned),
        'buildings': building_results,
    }
    json_path = os.path.join(args.output_dir, 'buildings_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {json_path}")

    # Save face-level data
    npz_path = os.path.join(args.output_dir, 'building_faces.npz')
    np.savez(npz_path,
             face_labels=face_labels,
             building_ids=building_ids,
             face_prim_indices=face_prim_indices)
    print(f"  Data: {npz_path}")

    # ── 7. Summary ──
    print(f"\n{'='*60}")
    print(f"  Buildings: {len(building_results)}")
    total_b_faces = sum(r['n_faces'] for r in building_results)
    print(f"  Total building faces: {total_b_faces:,}")
    n_wt = sum(1 for r in building_results if r['is_watertight'])
    print(f"  Watertight: {n_wt}/{len(building_results)}")


if __name__ == '__main__':
    main()
