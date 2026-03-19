#!/usr/bin/env python3
"""
Phase 4' v4: Convex Polytope CityGML from PlanarSplatting Planes

Building = convex polytope bounded by surface-group half-spaces.
Each surface group defines a plane (normal + d) + semantic class.

Algorithm:
  1. Cluster primitives → surface groups (by normal similarity within class)
  2. Orient normals outward, add virtual ground plane
  3. Enumerate all 3-plane intersection vertices, keep those inside ALL half-spaces
  4. ConvexHull → manifold solid (each edge shared by exactly 2 faces)
  5. Map hull faces to surface groups → merge coplanar triangles → CityJSON

Key insight: ConvexHull is manifold by construction → val3dity VALID guaranteed.
No mesh adjacency needed — only PlanarSplatting plane equations + building assignment.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Primitive Clustering → Surface Groups
# ─────────────────────────────────────────────────────────────────────────────

def cluster_primitives(centers, normals, areas, labels, cos_thresh=0.85):
    """
    Cluster primitives within same semantic class by normal similarity.
    Same wall/roof plane → same surface group, regardless of spatial distance.
    """
    groups = []
    for cls in [1, 2]:  # roof=1, wall=2
        cls_mask = labels == cls
        cls_ids = np.where(cls_mask)[0]
        if len(cls_ids) == 0:
            continue

        cls_normals = normals[cls_ids]
        cls_centers = centers[cls_ids]
        cls_areas = areas[cls_ids]

        if len(cls_ids) == 1:
            n = cls_normals[0] / (np.linalg.norm(cls_normals[0]) + 1e-12)
            groups.append({
                'plane_normal': n,
                'plane_d': float(np.dot(n, cls_centers[0])),
                'class': cls,
                'prim_ids': cls_ids.tolist(),
                'center': cls_centers[0].copy(),
                'area': float(cls_areas[0]),
            })
            continue

        # Normal-only distance: 1 - |cos(angle)|
        n_hat = cls_normals / (np.linalg.norm(cls_normals, axis=1, keepdims=True) + 1e-12)
        cos_sim = np.clip(np.abs(n_hat @ n_hat.T), 0, 1)
        condensed = (1.0 - cos_sim)[np.triu_indices(len(cls_ids), k=1)]

        Z = linkage(condensed, method='average')
        cluster_ids = fcluster(Z, t=1.0 - cos_thresh, criterion='distance')

        for cid in np.unique(cluster_ids):
            cmask = cluster_ids == cid
            pids = cls_ids[cmask]
            cn = cls_normals[cmask]
            cc = cls_centers[cmask]
            ca = cls_areas[cmask]

            w = ca / (ca.sum() + 1e-12)
            mean_n = (cn * w[:, None]).sum(0)
            mean_n /= np.linalg.norm(mean_n) + 1e-12
            mean_c = (cc * w[:, None]).sum(0)

            groups.append({
                'plane_normal': mean_n,
                'plane_d': float(np.dot(mean_n, mean_c)),
                'class': cls,
                'prim_ids': pids.tolist(),
                'center': mean_c.copy(),
                'area': float(ca.sum()),
            })
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Plane Orientation + Ground Surface
# ─────────────────────────────────────────────────────────────────────────────

def orient_normals_outward(groups, building_center):
    """Flip normals to point away from building center."""
    for g in groups:
        if np.dot(g['plane_normal'], g['center'] - building_center) < 0:
            g['plane_normal'] = -g['plane_normal']
            g['plane_d'] = -g['plane_d']


def add_ground_surface(groups, wall_centers, building_center):
    """Add virtual GroundSurface at wall base. COLMAP: -Y up, down = +Y outward."""
    y_base = float(np.percentile(wall_centers[:, 1], 95))
    groups.append({
        'plane_normal': np.array([0.0, 1.0, 0.0]),  # outward = down (+Y)
        'plane_d': float(y_base),
        'class': -1,
        'prim_ids': [],
        'center': np.array([building_center[0], y_base, building_center[2]]),
        'area': 0.0,
        'is_ground': True,
    })


def add_bbox_planes(groups, prim_centers, margin=0.05):
    """
    Add axis-aligned bounding box planes for unobserved building faces.

    Aerial oblique imagery typically observes only certain faces. Missing
    direction coverage causes degenerate (coplanar) polytopes. Bbox planes
    fill in gaps, ensuring a bounded 3D solid.

    Only adds planes for directions not already covered by real surface groups.
    """
    normals_existing = np.array([g['plane_normal'] for g in groups])
    bbox_min = prim_centers.min(axis=0) - margin
    bbox_max = prim_centers.max(axis=0) + margin
    center = prim_centers.mean(axis=0)

    # 5 directions (ground/+Y already handled by add_ground_surface)
    candidates = [
        (np.array([1.0, 0, 0]), bbox_max[0], 2,
         np.array([bbox_max[0], center[1], center[2]])),
        (np.array([-1.0, 0, 0]), -bbox_min[0], 2,
         np.array([bbox_min[0], center[1], center[2]])),
        (np.array([0, -1.0, 0]), -bbox_min[1], 1,  # upward = roof-like
         np.array([center[0], bbox_min[1], center[2]])),
        (np.array([0, 0, 1.0]), bbox_max[2], 2,
         np.array([center[0], center[1], bbox_max[2]])),
        (np.array([0, 0, -1.0]), -bbox_min[2], 2,
         np.array([center[0], center[1], bbox_min[2]])),
    ]

    n_added = 0
    for normal, d, cls, c in candidates:
        # Check if any existing surface already covers this direction
        cos_sims = normals_existing @ normal
        if np.any(cos_sims > 0.7):
            continue  # Already covered

        groups.append({
            'plane_normal': normal,
            'plane_d': float(d),
            'class': cls,
            'prim_ids': [],
            'center': c,
            'area': 0.0,
            'is_bbox': True,
        })
        n_added += 1

    return n_added


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Convex Polytope from Half-Space Intersection
# ─────────────────────────────────────────────────────────────────────────────

def intersect_three_planes(n1, d1, n2, d2, n3, d3):
    """Solve n1·x=d1, n2·x=d2, n3·x=d3 for x."""
    A = np.array([n1, n2, n3])
    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        return None
    return np.linalg.solve(A, np.array([d1, d2, d3]))


def build_convex_polytope(groups, prim_centers, hs_tol=0.05, plane_tol=0.1,
                          bbox_margin=1.0):
    """
    Build building polyhedron as a convex polytope.

    Each surface group defines a half-space: n_i · x ≤ d_i (outward normal).
    Valid vertices = 3-plane intersections satisfying ALL half-spaces.
    ConvexHull of valid vertices → manifold solid.

    Returns: {group_idx: polygon_vertices_ndarray} or None
    """
    N = len(groups)
    if N < 4:
        return None

    normals = np.array([g['plane_normal'] for g in groups])
    ds = np.array([g['plane_d'] for g in groups])

    # Bounding box to reject extreme vertices from ill-conditioned intersections
    bbox_min = prim_centers.min(axis=0) - bbox_margin
    bbox_max = prim_centers.max(axis=0) + bbox_margin

    # Enumerate all 3-plane intersection vertices
    valid_verts = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                pt = intersect_three_planes(
                    normals[i], ds[i], normals[j], ds[j], normals[k], ds[k])
                if pt is None:
                    continue
                # Half-space check: n_l · pt ≤ d_l + tol for ALL planes
                if np.any(normals @ pt - ds > hs_tol):
                    continue
                # Bounding box check
                if np.any(pt < bbox_min) or np.any(pt > bbox_max):
                    continue
                valid_verts.append(pt)

    if len(valid_verts) < 4:
        return None

    valid_verts = np.array(valid_verts)

    # Deduplicate close vertices
    unique = [0]
    for i in range(1, len(valid_verts)):
        if all(np.linalg.norm(valid_verts[i] - valid_verts[j]) > 0.001
               for j in unique):
            unique.append(i)
    valid_verts = valid_verts[unique]

    if len(valid_verts) < 4:
        return None

    try:
        hull = ConvexHull(valid_verts)
    except Exception as e:
        print(f"    ConvexHull failed: {e}")
        return None

    # Map hull triangles to surface groups
    # For each hull face, find the group whose plane has the smallest max residual
    group_tris = defaultdict(list)
    unmatched = 0

    for fi, simplex in enumerate(hull.simplices):
        face_verts = valid_verts[simplex]

        # Find best-matching group by minimum max-residual
        best_gi, best_res = -1, float('inf')
        for gi in range(N):
            max_res = float(np.abs(normals[gi] @ face_verts.T - ds[gi]).max())
            if max_res < best_res:
                best_res = max_res
                best_gi = gi

        if best_res < plane_tol:
            group_tris[best_gi].append(simplex.tolist())
        else:
            # Fallback: match by hull face normal → closest surface group normal
            eq = hull.equations[fi][:3]
            eq_len = np.linalg.norm(eq)
            if eq_len > 1e-10:
                fn = eq / eq_len
                cos_sims = normals @ fn
                best = int(np.argmax(cos_sims))
                if cos_sims[best] > 0.3:
                    group_tris[best].append(simplex.tolist())
                else:
                    unmatched += 1
            else:
                unmatched += 1

    if unmatched:
        print(f"    WARNING: {unmatched} hull faces unmatched")

    # Merge coplanar triangles into polygons
    polygons = {}
    for gi, tris in group_tris.items():
        pts = _merge_coplanar_triangles(valid_verts, tris, groups[gi]['plane_normal'])
        if pts is not None and len(pts) >= 3:
            polygons[gi] = pts

    print(f"    Polytope: {len(valid_verts)} verts, {len(hull.simplices)} hull tris → "
          f"{len(polygons)}/{N} groups used")

    return polygons


def _merge_coplanar_triangles(vertices, tris, group_normal):
    """
    Merge coplanar triangles from ConvexHull into a single polygon.
    Returns vertices in winding order consistent with group_normal (outward).

    Uses undirected boundary walk (not directed edges from triangles) because
    scipy ConvexHull doesn't guarantee consistent winding within coplanar faces.
    """
    if len(tris) == 1:
        pts = vertices[tris[0]]
        e1, e2 = pts[1] - pts[0], pts[2] - pts[0]
        if np.dot(np.cross(e1, e2), group_normal) < 0:
            pts = pts[::-1]
        return pts

    # Count undirected edge occurrences
    edge_count = defaultdict(int)
    for tri in tris:
        for a in range(3):
            e = tuple(sorted([tri[a], tri[(a + 1) % 3]]))
            edge_count[e] += 1

    # Boundary = undirected edges appearing exactly once
    boundary = [e for e, c in edge_count.items() if c == 1]

    if len(boundary) < 3:
        # All edges internal → single triangle equivalent
        if len(tris) == 1:
            return vertices[tris[0]]
        return None

    # Build undirected adjacency from boundary edges
    adj = defaultdict(set)
    for a, b in boundary:
        adj[a].add(b)
        adj[b].add(a)

    # Walk boundary (each vertex has degree 2 for a simple closed boundary)
    start = boundary[0][0]
    polygon = [start]
    visited = {start}
    current = start
    for _ in range(len(adj) + 1):
        nbs = adj[current] - visited
        if not nbs:
            break
        nxt = next(iter(nbs))
        polygon.append(nxt)
        visited.add(nxt)
        current = nxt

    if len(polygon) < 3:
        return None

    pts = vertices[polygon]

    # Fix winding to match outward normal
    center = pts.mean(axis=0)
    area_normal = sum(
        np.cross(pts[i] - center, pts[(i + 1) % len(pts)] - center)
        for i in range(len(pts)))
    if np.dot(area_normal, group_normal) < 0:
        pts = pts[::-1]

    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: CityJSON Output
# ─────────────────────────────────────────────────────────────────────────────

def compute_signed_volume(surfaces):
    """Signed volume of a closed solid from its face polygons."""
    volume = 0.0
    for verts in surfaces:
        if len(verts) < 3:
            continue
        v0 = verts[0]
        for i in range(1, len(verts) - 1):
            volume += np.dot(v0, np.cross(verts[i], verts[i + 1]))
    return volume / 6.0


def build_cityjson(building_id, groups, polygons, out_dir):
    """Build CityJSON 2.0 Solid with shared vertices and proper topology."""
    scale = 0.0001
    vert_map = {}
    all_verts = []

    def add_vertex(pt):
        ix = round(pt[0] / scale)
        iy = round(pt[1] / scale)
        iz = round(pt[2] / scale)
        key = (ix, iy, iz)
        if key not in vert_map:
            vert_map[key] = len(vert_map)
            all_verts.append([ix, iy, iz])
        return vert_map[key]

    surface_data = []
    for gi in sorted(polygons.keys()):
        verts = polygons[gi]
        indices = [add_vertex(v) for v in verts]

        # Remove consecutive duplicates
        cleaned = [indices[0]]
        for idx in indices[1:]:
            if idx != cleaned[-1]:
                cleaned.append(idx)
        if len(cleaned) > 1 and cleaned[-1] == cleaned[0]:
            cleaned = cleaned[:-1]
        if len(cleaned) < 3:
            continue

        cls = groups[gi]['class']
        stype = {1: "RoofSurface", 2: "WallSurface"}.get(cls, "GroundSurface")

        surface_data.append({
            'group_idx': gi,
            'indices': cleaned,
            'type': stype,
            'normal': groups[gi]['plane_normal'].copy(),
        })

    if len(surface_data) < 4:
        print(f"  WARNING: Building {building_id}: only {len(surface_data)} surfaces, skip")
        return None

    # Check signed volume, flip all windings if negative
    surf_verts = [np.array([np.array(all_verts[i]) * scale for i in sd['indices']])
                  for sd in surface_data]
    vol = compute_signed_volume(surf_verts)
    if vol < 0:
        for sd in surface_data:
            sd['indices'] = sd['indices'][::-1]
        vol = -vol

    # CityJSON structure
    translate = [min(v[j] for v in all_verts) * scale for j in range(3)]
    t_ijk = [round(translate[j] / scale) for j in range(3)]
    adjusted_verts = [[v[j] - t_ijk[j] for j in range(3)] for v in all_verts]

    boundaries = []
    sem_surfaces = []
    sem_values = []
    for i, sd in enumerate(surface_data):
        boundaries.append([sd['indices']])
        sem_surfaces.append({"type": sd['type']})
        sem_values.append(i)

    building_name = f"building_{building_id:03d}"
    cityjson = {
        "type": "CityJSON",
        "version": "2.0",
        "transform": {"scale": [scale] * 3, "translate": translate},
        "CityObjects": {
            building_name: {
                "type": "Building",
                "attributes": {
                    "building_id": int(building_id),
                    "n_surfaces": len(surface_data),
                    "signed_volume": float(vol),
                },
                "geometry": [{
                    "type": "Solid",
                    "lod": "2",
                    "boundaries": [boundaries],
                    "semantics": {
                        "surfaces": sem_surfaces,
                        "values": [sem_values],
                    },
                }],
            }
        },
        "vertices": adjusted_verts,
    }

    bdir = os.path.join(out_dir, f"building_{building_id:03d}")
    os.makedirs(bdir, exist_ok=True)
    cj_path = os.path.join(bdir, "building.city.json")
    with open(cj_path, 'w') as f:
        json.dump(cityjson, f, indent=2)

    save_lod2_ply(os.path.join(bdir, "lod2.ply"), surface_data, all_verts,
                  scale, translate)

    # Edge sharing diagnostics
    edges = {}
    for i, sd in enumerate(surface_data):
        ring = sd['indices']
        for j in range(len(ring)):
            v1, v2 = ring[j], ring[(j + 1) % len(ring)]
            edge = (min(v1, v2), max(v1, v2))
            edges.setdefault(edge, []).append(i)

    n_shared = sum(1 for faces in edges.values() if len(faces) == 2)
    n_boundary = sum(1 for faces in edges.values() if len(faces) == 1)
    n_nonmanifold = sum(1 for faces in edges.values() if len(faces) > 2)

    return {
        'building_id': int(building_id),
        'n_surfaces': len(surface_data),
        'n_vertices': len(all_verts),
        'signed_volume': float(vol),
        'n_edges_shared': n_shared,
        'n_edges_boundary': n_boundary,
        'n_edges_nonmanifold': n_nonmanifold,
        'surface_types': {
            'RoofSurface': sum(1 for s in surface_data if s['type'] == 'RoofSurface'),
            'WallSurface': sum(1 for s in surface_data if s['type'] == 'WallSurface'),
            'GroundSurface': sum(1 for s in surface_data if s['type'] == 'GroundSurface'),
        },
        'cityjson_path': cj_path,
    }


def save_lod2_ply(path, surface_data, all_verts, scale, translate):
    """Save LOD2 model as colored PLY (triangulated)."""
    type_colors = {
        'RoofSurface': [255, 0, 0],
        'WallSurface': [0, 0, 255],
        'GroundSurface': [128, 128, 128],
    }
    tris, colors = [], []
    for sd in surface_data:
        c = type_colors.get(sd['type'], [200, 200, 200])
        idx = sd['indices']
        for i in range(1, len(idx) - 1):
            tris.append([idx[0], idx[i], idx[i + 1]])
            colors.append(c)
    if not tris:
        return
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(all_verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(tris)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v in all_verts:
            f.write(f"{v[0]*scale+translate[0]:.6f} {v[1]*scale+translate[1]:.6f} "
                    f"{v[2]*scale+translate[2]:.6f}\n")
        for tri, c in zip(tris, colors):
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]} {c[0]} {c[1]} {c[2]}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_building(building_id, prim_ids, primitives, out_dir,
                     cos_thresh=0.85, hs_tol=0.05):
    """Process one building: cluster → orient → ground → polytope → CityJSON."""
    centers = primitives['centers'][prim_ids]
    normals = primitives['normals'][prim_ids]
    areas = primitives['areas'][prim_ids]
    labels = primitives['semantic_probs'][prim_ids].argmax(axis=1)

    n_roof, n_wall = int((labels == 1).sum()), int((labels == 2).sum())
    print(f"\n=== Building {building_id}: {len(prim_ids)} prims "
          f"(roof={n_roof}, wall={n_wall}) ===")

    if n_roof == 0 or n_wall == 0:
        print(f"  Skip: need both roof and wall")
        return None

    # Step 1: Cluster primitives → surface groups
    groups = cluster_primitives(centers, normals, areas, labels,
                                cos_thresh=cos_thresh)
    # Remap local prim_ids to global
    for g in groups:
        g['prim_ids'] = [prim_ids[lid] for lid in g['prim_ids']]

    # Orient normals outward from building center
    building_center = centers.mean(axis=0)
    orient_normals_outward(groups, building_center)

    # Add virtual ground surface at wall base
    wall_centers = centers[labels == 2]
    add_ground_surface(groups, wall_centers, building_center)

    # Add bbox planes for unobserved directions
    n_bbox = add_bbox_planes(groups, centers)

    n_r = sum(1 for g in groups if g['class'] == 1)
    n_w = sum(1 for g in groups if g['class'] == 2)
    n_real = sum(1 for g in groups if not g.get('is_bbox') and not g.get('is_ground'))
    print(f"  {len(groups)} surfaces (roof={n_r}, wall={n_w}, ground=1, bbox={n_bbox})")

    # Step 2: Build convex polytope
    polygons = build_convex_polytope(groups, centers, hs_tol=hs_tol)
    if polygons is None or len(polygons) < 4:
        print(f"  Polytope failed or <4 faces")
        return None

    for gi in sorted(polygons.keys()):
        cls = {1: 'roof', 2: 'wall', -1: 'ground'}.get(groups[gi]['class'], '?')
        gnd = ' [G]' if groups[gi].get('is_ground') else ''
        print(f"    S{gi}({cls}{gnd}): {len(polygons[gi])}v")

    # Step 3: CityJSON output
    result = build_cityjson(building_id, groups, polygons, out_dir)
    if result:
        print(f"  → {result['n_surfaces']}s {result['n_vertices']}v "
              f"vol={result['signed_volume']:.4f} "
              f"edges={result['n_edges_shared']}sh/"
              f"{result['n_edges_boundary']}bd/"
              f"{result['n_edges_nonmanifold']}nm")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Phase 4\' v4: Convex Polytope CityGML')
    parser.add_argument('--primitives',
                        default='results/phase4/part1/primitives_data.npz')
    parser.add_argument('--building_faces',
                        default='results/phase4_prime/step3_buildings/building_faces.npz')
    parser.add_argument('--out_dir',
                        default='results/phase4_prime/step456_v4')
    parser.add_argument('--cos_thresh', type=float, default=0.85,
                        help='Normal clustering threshold')
    parser.add_argument('--hs_tol', type=float, default=0.05,
                        help='Half-space tolerance for vertex validation')
    parser.add_argument('--building_ids', type=int, nargs='*', default=None,
                        help='Process specific buildings (default: all)')
    args = parser.parse_args()

    print("Loading data...")
    primitives = dict(np.load(args.primitives))
    bf = dict(np.load(args.building_faces))
    building_ids_arr = bf['building_ids']
    face_prim_indices = bf['face_prim_indices']

    print(f"  Primitives: {primitives['centers'].shape[0]}")
    print(f"  Face→prim: {len(face_prim_indices)} faces")

    unique_bids = np.unique(building_ids_arr)
    unique_bids = unique_bids[unique_bids >= 0]
    if args.building_ids is not None:
        unique_bids = [b for b in args.building_ids if b in unique_bids]

    print(f"\nProcessing {len(unique_bids)} buildings...")
    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    for bid in sorted(unique_bids):
        bmask = building_ids_arr == bid
        prim_ids = np.unique(face_prim_indices[bmask])
        prim_ids = prim_ids[prim_ids >= 0]
        if len(prim_ids) < 3:
            continue
        result = process_building(
            bid, prim_ids, primitives, args.out_dir,
            cos_thresh=args.cos_thresh, hs_tol=args.hs_tol)
        if result:
            results.append(result)

    # Summary
    summary = {
        'version': 'v4-convex',
        'params': {
            'cos_thresh': args.cos_thresh,
            'hs_tol': args.hs_tol,
        },
        'n_buildings_processed': len(results),
        'n_buildings_total': len(unique_bids),
        'buildings': results,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: int(x) if isinstance(x, np.integer)
                  else float(x) if isinstance(x, np.floating) else x)

    print(f"\n{'=' * 60}")
    print(f"Summary: {len(results)}/{len(unique_bids)} buildings processed")
    if results:
        total_sh = sum(r['n_edges_shared'] for r in results)
        total_bd = sum(r['n_edges_boundary'] for r in results)
        total_nm = sum(r['n_edges_nonmanifold'] for r in results)
        print(f"Edges total: {total_sh} shared, {total_bd} boundary, {total_nm} non-manifold")
    for r in results:
        print(f"  B{r['building_id']:03d}: {r['n_surfaces']}s {r['n_vertices']}v "
              f"e={r['n_edges_shared']}sh/{r['n_edges_boundary']}bd/"
              f"{r['n_edges_nonmanifold']}nm "
              f"({r['surface_types']})")


if __name__ == '__main__':
    main()
