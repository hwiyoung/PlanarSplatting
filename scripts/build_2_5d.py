#!/usr/bin/env python3
"""
2.5D Building Solid Construction from Planar Primitives.

Standard CityGML LOD2 generation approach:
  Step 1: Footprint from wall planes (handles non-convex)
  Step 2: Ridge lines from roof-roof plane intersections
  Step 3: Roof polygons from ridge + footprint boundary
  Step 4: Wall quads from footprint edges (ground → roof)
  Step 5: Ground face = footprint at ground height
"""

import numpy as np
from shapely.geometry import Polygon as ShapelyPoly


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Footprint extraction from wall planes
# ─────────────────────────────────────────────────────────────────────────────

def extract_footprint(wall_groups, centers):
    """Extract 2D footprint polygon from wall plane groups.

    Algorithm:
      1. Each wall group → line in XZ plane (nx*x + nz*z = d)
      2. Each wall's active range = primitive extent + proportional margin
      3. Pairwise line intersections, filtered by both walls' active range
      4. Boundary tracing via wall adjacency graph

    Args:
        wall_groups: list of dicts with 'plane_normal', 'plane_d', 'prim_ids'
        centers: (N, 3) all primitive centers

    Returns:
        footprint_vertices: (M, 2) ndarray of XZ coordinates, or None
        wall_edges: list of (wall_group_index, vertex_i, vertex_j) for each footprint edge
    """
    # Extract wall lines with tangent-based active range.
    # Project wall normal to XZ plane (ignore Y component) to handle
    # non-vertical walls (e.g., dormers, overhangs labeled as WallSurface).
    # Tangent direction = [-nz, nx] (perpendicular to projected normal in XZ).
    # Active range = projection of primitive centers onto tangent direction.
    walls = []
    for gi, g in enumerate(wall_groups):
        nx_raw, ny_raw, nz_raw = g['plane_normal']
        # Project normal to XZ plane
        nx, nz = nx_raw, nz_raw
        nxz_len = np.sqrt(nx**2 + nz**2)
        if nxz_len < 0.1:
            continue  # nearly horizontal normal = not usable as wall
        # Normalize the XZ projection
        nx /= nxz_len
        nz /= nxz_len
        # Recompute plane_d using projected normal and primitive centers at ground level
        pids = g['prim_ids']
        if not pids:
            continue
        pc = centers[pids]
        # d = mean(nx*x + nz*z) — ignoring Y for footprint
        d = float(np.mean(pc[:, 0] * nx + pc[:, 2] * nz))
        pids = g['prim_ids']
        if not pids:
            continue
        pc_xz = centers[pids][:, [0, 2]]

        # Tangent direction in XZ plane (perpendicular to wall normal)
        tangent = np.array([-nz, nx])
        tn = np.linalg.norm(tangent)
        if tn < 1e-6:
            continue
        tangent /= tn

        # Project primitive centers onto tangent direction
        projections = pc_xz @ tangent
        t_min, t_max = float(projections.min()), float(projections.max())

        # Proportional margin: 30% of wall length, min 1m
        wall_len = t_max - t_min
        margin = max(1.0, wall_len * 0.3)

        walls.append({
            'gi': gi, 'nx': float(nx), 'nz': float(nz), 'd': float(d),
            'tangent': tangent,
            't_min': t_min - margin, 't_max': t_max + margin,
        })

    if len(walls) < 3:
        return None, None

    # Pairwise intersections.
    # Two-pass filtering:
    #   Pass 1: Collect ALL non-parallel intersections
    #   Pass 2: For each wall, find its valid range from its actual intersections
    #           with other walls, then filter corners by proximity to primitives.

    # Pass 1: All intersections
    all_intersections = []
    for a in range(len(walls)):
        for b in range(a + 1, len(walls)):
            wa, wb = walls[a], walls[b]
            det = wa['nx'] * wb['nz'] - wb['nx'] * wa['nz']
            if abs(det) < 0.1:
                continue  # parallel
            x = (wa['d'] * wb['nz'] - wb['d'] * wa['nz']) / det
            z = (wa['nx'] * wb['d'] - wb['nx'] * wa['d']) / det
            all_intersections.append({
                'pt': (x, z),
                'walls': (a, b),
            })

    if len(all_intersections) < 3:
        return None, None

    # Pass 2: Filter corners by checking if intersection is near primitives
    # of BOTH walls. "Near" = within the wall's primitive extent + generous margin.
    # The margin is based on the building's overall size, not individual wall length.
    all_prim_xz = centers[:, [0, 2]]
    building_size = max(
        all_prim_xz[:, 0].max() - all_prim_xz[:, 0].min(),
        all_prim_xz[:, 1].max() - all_prim_xz[:, 1].min(),
    )
    global_margin = max(2.0, building_size * 0.3)

    corners = []
    for isec in all_intersections:
        pt_xz = np.array(isec['pt'])
        a, b = isec['walls']
        wa, wb = walls[a], walls[b]

        proj_a = float(pt_xz @ wa['tangent'])
        proj_b = float(pt_xz @ wb['tangent'])

        # Use per-wall range + global margin
        ok_a = (wa['t_min'] - global_margin) <= proj_a <= (wa['t_max'] + global_margin)
        ok_b = (wb['t_min'] - global_margin) <= proj_b <= (wb['t_max'] + global_margin)

        if ok_a and ok_b:
            corners.append({
                'pt': (round(isec['pt'][0], 4), round(isec['pt'][1], 4)),
                'walls': (a, b),
            })

    if len(corners) < 3:
        return None, None

    # Build adjacency graph: corners connected via shared walls
    wall_to_corners = {}
    for ci, c in enumerate(corners):
        for wi in c['walls']:
            wall_to_corners.setdefault(wi, []).append(ci)

    adj = {ci: [] for ci in range(len(corners))}
    for wi, cis in wall_to_corners.items():
        wa = walls[wi]
        # Sort corners along wall's tangent direction
        cis_sorted = sorted(cis, key=lambda ci: np.array(corners[ci]['pt']) @ wa['tangent'])
        for k in range(len(cis_sorted) - 1):
            adj[cis_sorted[k]].append((wi, cis_sorted[k + 1]))
            adj[cis_sorted[k + 1]].append((wi, cis_sorted[k]))

    # Trace boundary
    path = [0]
    visited_edges = set()
    cur = 0
    for _ in range(50):
        found = False
        for wi, nxt in adj[cur]:
            edge = (min(cur, nxt), max(cur, nxt))
            if edge not in visited_edges:
                visited_edges.add(edge)
                path.append(nxt)
                cur = nxt
                found = True
                break
        if not found or cur == 0:
            break

    if len(path) < 4 or path[-1] != 0:
        # Fallback: use convex hull of all corner points
        if len(corners) >= 3:
            from scipy.spatial import ConvexHull
            pts_arr = np.array([c['pt'] for c in corners])
            try:
                hull = ConvexHull(pts_arr)
                hull_pts = pts_arr[hull.vertices]
                fp_poly = ShapelyPoly(hull_pts)
                if fp_poly.is_valid and fp_poly.area > 0.1:
                    fp_pts = hull_pts
                else:
                    return None, None
            except Exception:
                return None, None
        else:
            return None, None
    else:
        # Build output from traced path
        fp_pts = np.array([corners[ci]['pt'] for ci in path[:-1]])
        fp_poly = ShapelyPoly(fp_pts)

    # Fix self-intersecting polygons
    if not fp_poly.is_valid:
        from shapely.validation import make_valid
        fixed = make_valid(fp_poly)
        # Extract the largest polygon from the result
        if fixed.geom_type == 'MultiPolygon':
            fp_poly = max(fixed.geoms, key=lambda g: g.area)
        elif fixed.geom_type == 'Polygon':
            fp_poly = fixed
        else:
            return None, None
        fp_pts = np.array(fp_poly.exterior.coords[:-1])  # remove closing vertex

    if fp_poly.area < 0.1:
        return None, None

    # Wall edge mapping: match each footprint edge to the closest wall group
    # by comparing edge normal direction with wall normal directions.
    wall_edges = []
    n_fp = len(fp_pts)
    for k in range(n_fp):
        p1 = fp_pts[k]
        p2 = fp_pts[(k + 1) % n_fp]
        # Edge normal (outward, perpendicular to edge direction)
        edge_dir = p2 - p1
        edge_normal = np.array([edge_dir[1], -edge_dir[0]])
        en_len = np.linalg.norm(edge_normal)
        if en_len > 1e-12:
            edge_normal /= en_len
        # Find wall with most similar normal
        best_wi = 0
        best_cos = -1
        for wi, w in enumerate(walls):
            w_normal_xz = np.array([w['nx'], w['nz']])
            cos_sim = abs(np.dot(edge_normal, w_normal_xz))
            if cos_sim > best_cos:
                best_cos = cos_sim
                best_wi = wi
        shared_wi = best_wi
        wall_gi = walls[shared_wi]['gi'] if shared_wi is not None else -1
        wall_edges.append((wall_gi, k, (k + 1) % n_fp))

    return fp_pts, wall_edges


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Ridge lines from roof plane intersections
# ─────────────────────────────────────────────────────────────────────────────

def compute_ridges(roof_groups, footprint_xz, centers=None):
    """Compute ridge lines between adjacent roof planes.

    For each pair of roof groups, compute their plane intersection line,
    then clip it to the footprint boundary.

    Args:
        roof_groups: list of dicts with 'plane_normal', 'plane_d', 'prim_ids', 'gi'
        footprint_xz: (M, 2) footprint vertices in XZ

    Returns:
        ridges: list of {'line': (p1_3d, p2_3d), 'roofs': (gi_a, gi_b)}
    """
    if len(roof_groups) < 2:
        return []

    fp_poly = ShapelyPoly(footprint_xz)
    ridges = []

    for a in range(len(roof_groups)):
        for b in range(a + 1, len(roof_groups)):
            # Adjacency check: only compute ridge for spatially adjacent roof pairs
            if centers is not None and roof_groups[a]['prim_ids'] and roof_groups[b]['prim_ids']:
                from scipy.spatial.distance import cdist
                ca = centers[roof_groups[a]['prim_ids']][:, [0, 2]]
                cb = centers[roof_groups[b]['prim_ids']][:, [0, 2]]
                min_dist = float(cdist(ca, cb).min())
                if min_dist > 3.0:
                    continue  # non-adjacent (e.g., opposite hip slopes)
            ra, rb = roof_groups[a], roof_groups[b]
            na, da = ra['plane_normal'], ra['plane_d']
            nb, db = rb['plane_normal'], rb['plane_d']

            # Ridge direction = cross(na, nb)
            ridge_dir = np.cross(na, nb)
            rd_len = np.linalg.norm(ridge_dir)
            if rd_len < 1e-6:
                continue  # parallel planes
            ridge_dir /= rd_len

            # Find a point on the ridge line: solve na·p=da, nb·p=db
            # Add third equation: ridge_dir·p = 0 (arbitrary, just to solve)
            A = np.array([na, nb, ridge_dir])
            det = np.linalg.det(A)
            if abs(det) < 1e-10:
                continue
            ridge_pt = np.linalg.solve(A, np.array([da, db, 0]))

            # Project ridge line to XZ and clip to footprint
            # Ridge in XZ: point + t * direction_xz
            dir_xz = np.array([ridge_dir[0], ridge_dir[2]])
            dir_xz_len = np.linalg.norm(dir_xz)
            if dir_xz_len < 1e-6:
                # Ridge is vertical in XZ — skip (unusual for buildings)
                continue

            pt_xz = np.array([ridge_pt[0], ridge_pt[2]])

            # Clip to footprint bounding box (extended)
            from shapely.geometry import LineString
            far = 100.0
            line_xz = LineString([
                (pt_xz[0] - dir_xz[0] * far, pt_xz[1] - dir_xz[1] * far),
                (pt_xz[0] + dir_xz[0] * far, pt_xz[1] + dir_xz[1] * far),
            ])

            clipped = fp_poly.intersection(line_xz)
            if clipped.is_empty:
                continue

            if clipped.geom_type == 'LineString':
                coords = list(clipped.coords)
            elif clipped.geom_type == 'MultiLineString':
                # Take longest segment
                longest = max(clipped.geoms, key=lambda g: g.length)
                coords = list(longest.coords)
            else:
                continue

            if len(coords) < 2:
                continue

            # Convert back to 3D: for each XZ point, compute Y from one of the roof planes
            p1_xz, p2_xz = np.array(coords[0]), np.array(coords[-1])

            def xz_to_3d(xz, n, d):
                """Given XZ point and plane n·x=d, solve for Y."""
                # n[0]*x + n[1]*y + n[2]*z = d → y = (d - n[0]*x - n[2]*z) / n[1]
                if abs(n[1]) < 1e-6:
                    return 0.0
                return (d - n[0] * xz[0] - n[2] * xz[1]) / n[1]

            y1 = xz_to_3d(p1_xz, na, da)
            y2 = xz_to_3d(p2_xz, na, da)

            p1_3d = np.array([p1_xz[0], y1, p1_xz[1]])
            p2_3d = np.array([p2_xz[0], y2, p2_xz[1]])

            ridges.append({
                'line': (p1_3d, p2_3d),
                'roofs': (ra['gi'], rb['gi']),
            })

    return ridges


# ─────────────────────────────────────────────────────────────────────────────
# Step 3-5: Build complete 2.5D solid
# ─────────────────────────────────────────────────────────────────────────────

def _plane_y_at_xz(normal, d, x, z):
    """Compute Y coordinate on plane n·p=d at given XZ. Snapped to 1mm."""
    if abs(normal[1]) < 1e-6:
        return 0.0
    y = (d - normal[0] * x - normal[2] * z) / normal[1]
    return round(y * 1000) / 1000  # 1mm snap


def _is_flat_roof(roof_groups):
    """Check if all roof groups form a single flat (horizontal) roof.
    A roof is flat if ALL normals are nearly vertical (|n_y| > 0.95)
    AND all normals are nearly parallel to each other."""
    normals = [g['plane_normal'] for g in roof_groups]
    # All must be nearly vertical
    if not all(abs(n[1]) > 0.95 for n in normals):
        return False
    # All must be nearly parallel
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            if abs(np.dot(normals[i], normals[j])) < 0.95:
                return False
    return True


def _assign_footprint_edges_to_roofs(footprint_xz, roof_groups, centers):
    """For pitched roofs, determine which roof group covers each footprint edge.

    Uses the projected position: for each footprint edge midpoint,
    find which roof group's primitives are closest.
    """
    n_edges = len(footprint_xz)
    assignments = []

    for ei in range(n_edges):
        ej = (ei + 1) % n_edges
        mid_xz = (footprint_xz[ei] + footprint_xz[ej]) / 2

        best_gi = 0
        best_dist = float('inf')
        for ri, rg in enumerate(roof_groups):
            pids = rg['prim_ids']
            if not pids:
                continue
            roof_xz = centers[pids][:, [0, 2]]
            dists = np.sqrt(((roof_xz - mid_xz) ** 2).sum(axis=1))
            min_d = float(dists.min())
            if min_d < best_dist:
                best_dist = min_d
                best_gi = ri
        assignments.append(best_gi)

    return assignments


def _build_hip_skeleton(footprint_xz, roof_groups, wall_groups, wall_edges,
                        eaves_y, ground_y, groups, centers):
    """Build hip roof solid using straight skeleton of rectangular footprint."""
    n_fp = len(footprint_xz)

    # Skeleton: inset from shorter edges by half the shorter edge length
    edge_lens = [np.linalg.norm(footprint_xz[(i+1)%4] - footprint_xz[i]) for i in range(4)]
    min_len = min(edge_lens)
    half_min = min_len / 2.0

    # Straight skeleton for rectangle: ridge endpoints are at the intersection
    # of bisectors from adjacent edges. For a rectangle, these are at:
    # inset by half_min from ALL edges simultaneously.
    # Result: 2 points on the longer edge's midline, offset by half_min from shorter edges.

    # Find the longer edge direction
    longer_edges = [(ei, edge_lens[ei]) for ei in range(4) if edge_lens[ei] > min_len * 1.1]
    if not longer_edges:
        # Square: single ridge point at center
        skeleton_pts = [footprint_xz.mean(0)]
    else:
        # Rectangle: 2 skeleton points along the longer edge direction
        ei0 = longer_edges[0][0]
        ej0 = (ei0 + 1) % 4
        long_dir = footprint_xz[ej0] - footprint_xz[ei0]
        long_dir /= np.linalg.norm(long_dir)

        # Center of footprint
        center = footprint_xz.mean(0)

        # Skeleton points: center ± (long_length/2 - half_min) along long_dir
        long_len = longer_edges[0][1]
        offset = long_len / 2 - half_min
        skeleton_pts = [
            center - long_dir * offset,
            center + long_dir * offset,
        ]

    if len(skeleton_pts) < 2:
        return None

    # Ridge height from main roof plane
    main_rg = max(roof_groups, key=lambda g: g.get('area', 0))
    rn, rd = main_rg['plane_normal'], main_rg['plane_d']

    def y_at(x, z):
        if abs(rn[1]) < 1e-6:
            return eaves_y
        return (rd - rn[0] * x - rn[2] * z) / rn[1]

    edge_to_roof = _assign_footprint_edges_to_roofs(footprint_xz, roof_groups, centers)

    def _gi(g):
        return next(i for i, gg in enumerate(groups) if gg is g)

    faces = []

    # Roof faces: each footprint edge → triangle or trapezoid
    for ei in range(4):
        ej = (ei + 1) % 4
        ci, cj = footprint_xz[ei], footprint_xz[ej]

        di = [np.linalg.norm(ci - sp) for sp in skeleton_pts]
        dj = [np.linalg.norm(cj - sp) for sp in skeleton_pts]
        si, sj = int(np.argmin(di)), int(np.argmin(dj))

        ri = edge_to_roof[ei]
        rg = roof_groups[ri]

        def skel_3d(idx):
            sp = skeleton_pts[idx]
            return np.array([sp[0], y_at(sp[0], sp[1]), sp[1]])

        if si == sj:
            pts = np.array([[ci[0], eaves_y, ci[1]],
                            [cj[0], eaves_y, cj[1]],
                            skel_3d(si)])
        else:
            pts = np.array([[ci[0], eaves_y, ci[1]],
                            [cj[0], eaves_y, cj[1]],
                            skel_3d(sj), skel_3d(si)])

        # Winding: roof outward = -Y
        e1, e2 = pts[1] - pts[0], pts[2] - pts[0]
        if np.cross(e1, e2)[1] > 0:
            pts = pts[::-1]

        faces.append({'vertices': pts, 'type': 'RoofSurface', 'group_idx': _gi(rg)})

    # Hip wall faces: always simple quads (no peaks — peaks are in roof faces)
    for ei in range(4):
        ej = (ei + 1) % 4
        x_i, z_i = footprint_xz[ei]
        x_j, z_j = footprint_xz[ej]
        wall_verts = np.array([
            [x_i, eaves_y, z_i],
            [x_i, ground_y, z_i],
            [x_j, ground_y, z_j],
            [x_j, eaves_y, z_j],
        ])
        wgi = wall_edges[ei][0] if wall_edges else -1
        faces.append({'vertices': wall_verts, 'type': 'WallSurface',
                      'group_idx': wgi if wgi >= 0 else -1})

    # Ground
    ground_verts = np.array([[fp[0], ground_y, fp[1]] for fp in footprint_xz[::-1]])
    faces.append({'vertices': ground_verts, 'type': 'GroundSurface',
                  'group_idx': next(i for i, g in enumerate(groups) if g.get('is_ground'))})

    return faces


def _compute_eaves_from_planes(roof_groups, wall_groups, footprint_xz, centers, ground_y):
    """Compute eaves height from roof-wall plane intersection.

    For each roof plane and each footprint corner, compute the Y where
    the roof plane passes through that corner's XZ position.
    The eaves height = the Y closest to ground among these values.
    For flat roofs, returns the roof height directly.
    """
    if not roof_groups:
        return ground_y

    eaves_candidates = []
    for rg in roof_groups:
        rn = rg['plane_normal']
        rd = rg['plane_d']
        if abs(rn[1]) < 0.1:
            continue  # nearly horizontal normal component too small
        # For each footprint corner, compute Y on this roof plane
        for pt in footprint_xz:
            # rn[0]*x + rn[1]*y + rn[2]*z = rd
            # y = (rd - rn[0]*x - rn[2]*z) / rn[1]
            y = (rd - rn[0] * pt[0] - rn[2] * pt[1]) / rn[1]
            eaves_candidates.append(float(y))

    if not eaves_candidates:
        # Fallback: median of per-wall min Y
        wall_min_ys = []
        for wg in wall_groups:
            if wg['prim_ids']:
                wall_min_ys.append(float(centers[wg['prim_ids'], 1].min()))
        return float(np.median(wall_min_ys)) if wall_min_ys else ground_y

    # Eaves = the Y value closest to ground (highest Y in COLMAP = closest to 0)
    # Among all roof-footprint intersections, the one nearest to ground is the eaves
    return float(max(eaves_candidates))


def build_2_5d_solid(groups, centers):
    """Build complete 2.5D building solid.

    Args:
        groups: list of surface groups from cluster_primitives + orient + ground
        centers: (N, 3) all primitive centers

    Returns:
        faces: list of {'vertices': (K, 3) ndarray, 'type': str, 'group_idx': int}
        or None if construction fails
    """
    def _gi(g):
        """Find index of group g in groups list (by identity)."""
        return next(i for i, gg in enumerate(groups) if gg is g)

    # Separate groups by class
    wall_groups = [g for g in groups if g['class'] == 2 and not g.get('is_bbox')]
    roof_groups = [g for g in groups if g['class'] == 1 and not g.get('is_bbox')]
    ground_groups = [g for g in groups if g.get('is_ground')]

    if not wall_groups or not roof_groups or not ground_groups:
        return None

    # Step 1: Footprint
    wall_gs_for_fp = [{'plane_normal': g['plane_normal'], 'plane_d': g['plane_d'],
                        'prim_ids': g['prim_ids']} for g in wall_groups]
    footprint_xz, wall_edges = extract_footprint(wall_gs_for_fp, centers)
    if footprint_xz is None:
        return None

    n_fp = len(footprint_xz)

    # Ground height
    gn = ground_groups[0]['plane_normal']
    gd = ground_groups[0]['plane_d']
    ground_y = gd / gn[1] if abs(gn[1]) > 0.1 else ground_groups[0]['center'][1]

    # Eaves height: compute from roof-wall plane intersection (exact).
    # For pitched roofs, eaves = where roof plane meets side wall.
    # For flat roofs, eaves = roof height.
    # This is exact and independent of primitive count/distribution.
    all_wall_pids = []
    for wg in wall_groups:
        all_wall_pids.extend(wg['prim_ids'])
    eaves_y = _compute_eaves_from_planes(roof_groups, wall_groups, footprint_xz, centers, ground_y)
    # Round to CityJSON quantization scale to ensure vertex merge
    SNAP = 0.001  # 1mm resolution
    eaves_y = round(eaves_y / SNAP) * SNAP
    ground_y = round(ground_y / SNAP) * SNAP

    # ── Hip roof on rectangular footprint: use straight skeleton ──
    if n_fp == 4 and len(roof_groups) >= 4 and not _is_flat_roof(roof_groups):
        return _build_hip_skeleton(footprint_xz, roof_groups, wall_groups,
                                    wall_edges, eaves_y, ground_y, groups, centers)

    faces = []
    flat = _is_flat_roof(roof_groups)

    if flat:
        # ── Flat roof: simple extrusion ──
        # Roof height from the main roof group
        rg = max(roof_groups, key=lambda g: g.get('area', 0))
        rn, rd = rg['plane_normal'], rg['plane_d']
        roof_y = rd / rn[1] if abs(rn[1]) > 0.1 else rg['center'][1]

        # Roof face = footprint at roof_y
        # Roof: footprint order (not reversed) → outward with wall convention
        roof_verts = np.array([[pt[0], roof_y, pt[1]] for pt in footprint_xz])
        faces.append({
            'vertices': roof_verts,
            'type': 'RoofSurface',
            'group_idx': _gi(rg),
        })

        # Wall faces = footprint edges extruded from ground_y to roof_y
        # Winding: (top_i, bottom_i, bottom_j, top_j) for outward normal
        # when footprint is CW in XZ
        for ei in range(n_fp):
            ej = (ei + 1) % n_fp
            quad = np.array([
                [footprint_xz[ei][0], roof_y, footprint_xz[ei][1]],
                [footprint_xz[ei][0], ground_y, footprint_xz[ei][1]],
                [footprint_xz[ej][0], ground_y, footprint_xz[ej][1]],
                [footprint_xz[ej][0], roof_y, footprint_xz[ej][1]],
            ])
            # Match to wall group
            wgi = wall_edges[ei][0] if wall_edges else -1
            faces.append({
                'vertices': quad,
                'type': 'WallSurface',
                'group_idx': wgi if wgi >= 0 else -1,
            })

    else:
        # ── Pitched roof: gable/shed (hip already handled above) ──
        roof_gs_for_ridge = []
        for g in roof_groups:
            gi = next(i for i, gg in enumerate(groups) if gg is g)
            roof_gs_for_ridge.append({
                'plane_normal': g['plane_normal'], 'plane_d': g['plane_d'],
                'prim_ids': g['prim_ids'], 'gi': gi,
            })
        ridges = compute_ridges(roof_gs_for_ridge, footprint_xz, centers)

        edge_to_roof = _assign_footprint_edges_to_roofs(footprint_xz, roof_groups, centers)

        if False:  # placeholder (hip case handled by _build_hip_skeleton above)
            edge_to_roof = _assign_footprint_edges_to_roofs(footprint_xz, roof_groups, centers)

            # Each footprint edge → one roof face
            for ei in range(4):
                ej = (ei + 1) % 4
                ci_xz, cj_xz = footprint_xz[ei], footprint_xz[ej]

                # Nearest skeleton point to each corner
                di = [np.linalg.norm(ci_xz - sp) for sp in skeleton_pts]
                dj = [np.linalg.norm(cj_xz - sp) for sp in skeleton_pts]
                si, sj = np.argmin(di), np.argmin(dj)

                ri = edge_to_roof[ei]
                rg = roof_groups[ri]
                rn, rd = rg['plane_normal'], rg['plane_d']
                rgi = _gi(rg)

                def skel_3d(idx):
                    sp = skeleton_pts[idx]
                    return np.array([sp[0], _plane_y_at_xz(rn, rd, sp[0], sp[1]), sp[1]])

                if si == sj:
                    pts = np.array([
                        [ci_xz[0], eaves_y, ci_xz[1]],
                        [cj_xz[0], eaves_y, cj_xz[1]],
                        skel_3d(si),
                    ])
                else:
                    pts = np.array([
                        [ci_xz[0], eaves_y, ci_xz[1]],
                        [cj_xz[0], eaves_y, cj_xz[1]],
                        skel_3d(sj),
                        skel_3d(si),
                    ])

                # Winding: roof outward = -Y
                e1, e2 = pts[1] - pts[0], pts[2] - pts[0]
                if np.cross(e1, e2)[1] > 0:
                    pts = pts[::-1]

                faces.append({'vertices': pts, 'type': 'RoofSurface', 'group_idx': rgi})

            # Wall faces for hip
            for ei in range(4):
                ej = (ei + 1) % 4
                x_i, z_i = footprint_xz[ei]
                x_j, z_j = footprint_xz[ej]

                # Check for skeleton points on this edge
                hip_peaks = []
                for si_idx, sp in enumerate(skeleton_pts):
                    edge_vec = footprint_xz[ej] - footprint_xz[ei]
                    el = np.linalg.norm(edge_vec)
                    if el < 1e-6: continue
                    ed = edge_vec / el
                    proj = np.dot(sp - footprint_xz[ei], ed)
                    perp = abs(np.dot(sp - footprint_xz[ei], np.array([-ed[1], ed[0]])))
                    if perp < 0.5 and -0.5 < proj < el + 0.5:
                        ri = edge_to_roof[ei]
                        rg = roof_groups[ri]
                        y_peak = _plane_y_at_xz(rg['plane_normal'], rg['plane_d'], sp[0], sp[1])
                        hip_peaks.append(np.array([sp[0], y_peak, sp[1]]))

                if hip_peaks:
                    ed = (footprint_xz[ej] - footprint_xz[ei])
                    ed /= np.linalg.norm(ed) + 1e-12
                    peaks_sorted = sorted(hip_peaks,
                        key=lambda p: np.dot(np.array([p[0], p[2]]) - footprint_xz[ei], ed))
                    wall_verts = [
                        np.array([x_i, eaves_y, z_i]),
                        np.array([x_i, ground_y, z_i]),
                        np.array([x_j, ground_y, z_j]),
                        np.array([x_j, eaves_y, z_j]),
                    ]
                    wall_verts.extend(reversed(peaks_sorted))
                else:
                    wall_verts = [
                        np.array([x_i, eaves_y, z_i]),
                        np.array([x_i, ground_y, z_i]),
                        np.array([x_j, ground_y, z_j]),
                        np.array([x_j, eaves_y, z_j]),
                    ]

                wgi = wall_edges[ei][0] if wall_edges else -1
                faces.append({'vertices': np.array(wall_verts), 'type': 'WallSurface',
                              'group_idx': wgi if wgi >= 0 else -1})

            # Skip the generic pitched roof code below
            # (jump to ground face)

        # (generic gable/shed pitched roof code continues below)

        # For each roof group, collect:
        #   - Ridge endpoints that border this roof
        #   - Footprint edge corners that belong to this roof
        # Then form the roof polygon by connecting them in order.

        # For each roof group, determine which footprint corners belong to it
        # using the ridge line: corners on the same side of ridge = same roof
        for ri, rg in enumerate(roof_groups):
            rn, rd = rg['plane_normal'], rg['plane_d']
            rgi = _gi(rg)

            # Find ridges bordering this roof
            my_ridges = [r for r in ridges if rgi in r['roofs']]

            if not my_ridges:
                # No ridges → this roof covers the full footprint (shouldn't happen
                # for pitched roofs, but handle gracefully)
                roof_pts = []
                for fi in range(n_fp):
                    x, z = footprint_xz[fi]
                    y = _plane_y_at_xz(rn, rd, x, z)
                    roof_pts.append(np.array([x, y, z]))
                pts_sorted = np.array(roof_pts)
            else:
                # Use the main ridge to determine which side each corner is on
                main_ridge = max(my_ridges, key=lambda r: np.linalg.norm(
                    r['line'][1] - r['line'][0]))
                rp1, rp2 = main_ridge['line']
                ridge_dir_xz = np.array([rp2[0] - rp1[0], rp2[2] - rp1[2]])
                ridge_dir_xz /= np.linalg.norm(ridge_dir_xz) + 1e-12
                ridge_normal_xz = np.array([-ridge_dir_xz[1], ridge_dir_xz[0]])
                ridge_pt_xz = np.array([rp1[0], rp1[2]])

                # This roof's side: use the roof group's primitive centroid
                roof_prim_centroid_xz = centers[rg['prim_ids']][:, [0, 2]].mean(0)
                my_side = np.dot(roof_prim_centroid_xz - ridge_pt_xz, ridge_normal_xz)

                # Collect footprint corners on this side, at eaves height
                eaves_y_tmp = _compute_eaves_from_planes(
                    roof_groups, wall_groups, footprint_xz, centers, 0)

                my_corners = []
                for fi in range(n_fp):
                    corner_xz = footprint_xz[fi]
                    side = np.dot(corner_xz - ridge_pt_xz, ridge_normal_xz)
                    if side * my_side >= 0:  # same side as roof primitives
                        x, z = corner_xz
                        # Footprint corners are at eaves height, not projected onto roof plane
                        my_corners.append(np.array([x, eaves_y_tmp, z]))

                # Add ridge endpoints
                ridge_pts = []
                for r in my_ridges:
                    for rp in r['line']:
                        is_dup = any(np.linalg.norm(rp - p) < 0.01
                                     for p in my_corners + ridge_pts)
                        if not is_dup:
                            ridge_pts.append(rp)

                all_pts = my_corners + ridge_pts
                if len(all_pts) < 3:
                    continue

                # Sort by angle in XZ (safe for convex roof polygons)
                pts = np.array(all_pts)
                cen = pts.mean(0)
                angles = np.arctan2(pts[:, 2] - cen[2], pts[:, 0] - cen[0])
                order = np.argsort(angles)
                pts_sorted = pts[order]

            # Ensure outward winding: roof normal should point away from building center
            if len(pts_sorted) >= 3:
                e1 = pts_sorted[1] - pts_sorted[0]
                e2 = pts_sorted[2] - pts_sorted[0]
                face_n = np.cross(e1, e2)
                face_center = pts_sorted.mean(0)
                bldg_center = centers.mean(0)
                if np.dot(face_n, face_center - bldg_center) < 0:
                    pts_sorted = pts_sorted[::-1]  # flip winding

            faces.append({
                'vertices': pts_sorted,
                'type': 'RoofSurface',
                'group_idx': rgi,
            })

        eaves_y = _compute_eaves_from_planes(
            roof_groups, wall_groups, footprint_xz, centers, ground_y)

        # Wall faces: each footprint edge, extruded from ground to eaves/roof
        for ei in range(n_fp):
            ej = (ei + 1) % n_fp
            ri = edge_to_roof[ei]
            rg = roof_groups[ri]
            rn, rd = rg['plane_normal'], rg['plane_d']

            x_i, z_i = footprint_xz[ei]
            x_j, z_j = footprint_xz[ej]

            # Wall top: for shed/single-slope (no ridges), follow roof plane at each corner
            # For gable (with ridges), use eaves height
            if len(ridges) == 0 and len(roof_groups) == 1:
                # Shed: wall top follows roof slope
                rn_w, rd_w = roof_groups[0]['plane_normal'], roof_groups[0]['plane_d']
                y_top_i = _plane_y_at_xz(rn_w, rd_w, x_i, z_i)
                y_top_j = _plane_y_at_xz(rn_w, rd_w, x_j, z_j)
            else:
                y_top_i = eaves_y
                y_top_j = eaves_y

            # Check if this edge has a ridge endpoint on it
            # (for gable walls, there's a peak at the ridge)
            gable_pts = []
            for ridge in ridges:
                for rp in ridge['line']:
                    # Is this ridge point on this footprint edge?
                    rp_xz = np.array([rp[0], rp[2]])
                    edge_vec = footprint_xz[ej] - footprint_xz[ei]
                    edge_len = np.linalg.norm(edge_vec)
                    if edge_len < 1e-6:
                        continue
                    edge_dir = edge_vec / edge_len
                    proj = np.dot(rp_xz - footprint_xz[ei], edge_dir)
                    perp_dist = abs(np.dot(rp_xz - footprint_xz[ei],
                                           np.array([-edge_dir[1], edge_dir[0]])))
                    if perp_dist < 0.5 and -0.5 < proj < edge_len + 0.5:
                        gable_pts.append(rp)

            if gable_pts:
                # Gable wall: pentagon (top_i→bottom_i→bottom_j→top_j→peak)
                edge_dir = (footprint_xz[ej] - footprint_xz[ei])
                edge_dir /= np.linalg.norm(edge_dir) + 1e-12
                gable_sorted = sorted(gable_pts,
                    key=lambda p: np.dot(np.array([p[0], p[2]]) - footprint_xz[ei], edge_dir))
                wall_verts = [
                    np.array([x_i, y_top_i, z_i]),
                    np.array([x_i, ground_y, z_i]),
                    np.array([x_j, ground_y, z_j]),
                    np.array([x_j, y_top_j, z_j]),
                ]
                # Add gable peaks in reverse order (to maintain winding)
                wall_verts.extend(reversed(gable_sorted))
            else:
                # Regular wall: quad (top_i→bottom_i→bottom_j→top_j)
                wall_verts = [
                    np.array([x_i, y_top_i, z_i]),
                    np.array([x_i, ground_y, z_i]),
                    np.array([x_j, ground_y, z_j]),
                    np.array([x_j, y_top_j, z_j]),
                ]

            wgi = wall_edges[ei][0] if wall_edges else -1
            faces.append({
                'vertices': np.array(wall_verts),
                'type': 'WallSurface',
                'group_idx': wgi if wgi >= 0 else -1,
            })

    # Ground: reversed footprint → outward with wall convention
    ground_verts = np.array([[pt[0], ground_y, pt[1]] for pt in footprint_xz[::-1]])
    faces.append({
        'vertices': ground_verts,
        'type': 'GroundSurface',
        'group_idx': _gi(ground_groups[0]),
    })

    return faces


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: CityJSON export
# ─────────────────────────────────────────────────────────────────────────────

def faces_to_cityjson(faces, building_id, out_dir, scale=0.0001):
    """Convert face list to CityJSON file.

    Handles vertex merging (via integer quantization) and winding correction.
    """
    import json, os

    vert_map = {}
    all_verts = []

    def add_vert(pt):
        ix = round(pt[0] / scale)
        iy = round(pt[1] / scale)
        iz = round(pt[2] / scale)
        key = (ix, iy, iz)
        if key not in vert_map:
            vert_map[key] = len(vert_map)
            all_verts.append([ix, iy, iz])
        return vert_map[key]

    boundaries = []
    sem_surfaces = []
    sem_values = []

    for fi, face in enumerate(faces):
        indices = [add_vert(v) for v in face['vertices']]
        # Remove consecutive duplicates
        cleaned = [indices[0]]
        for idx in indices[1:]:
            if idx != cleaned[-1]:
                cleaned.append(idx)
        if len(cleaned) > 1 and cleaned[-1] == cleaned[0]:
            cleaned = cleaned[:-1]
        if len(cleaned) < 3:
            continue

        boundaries.append([cleaned])
        sem_surfaces.append({"type": face['type']})
        sem_values.append(len(sem_surfaces) - 1)

    if len(boundaries) < 4:
        return None

    # Signed volume check → flip winding if negative
    def _signed_vol(boundaries, verts, s):
        vol = 0
        for bound in boundaries:
            ring = bound[0]
            if len(ring) < 3:
                continue
            pts = [np.array(verts[i]) * s for i in ring]
            for i in range(1, len(pts) - 1):
                v0, v1, v2 = pts[0], pts[i], pts[i + 1]
                vol += np.dot(v0, np.cross(v1, v2))
        return vol / 6.0

    vol = _signed_vol(boundaries, all_verts, scale)
    if vol < 0:
        for b in boundaries:
            b[0] = b[0][::-1]
        vol = -vol

    # Build CityJSON
    translate = [min(v[j] for v in all_verts) * scale for j in range(3)]
    t_ijk = [round(translate[j] / scale) for j in range(3)]
    adj_verts = [[v[j] - t_ijk[j] for j in range(3)] for v in all_verts]

    bname = f"building_{building_id:03d}" if isinstance(building_id, int) else f"building_{building_id}"
    cityjson = {
        "type": "CityJSON",
        "version": "2.0",
        "transform": {"scale": [scale] * 3, "translate": translate},
        "CityObjects": {
            bname: {
                "type": "Building",
                "attributes": {"building_id": building_id, "signed_volume": float(vol)},
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
        "vertices": adj_verts,
    }

    os.makedirs(out_dir, exist_ok=True)
    cj_path = os.path.join(out_dir, "building.city.json")
    with open(cj_path, 'w') as f:
        json.dump(cityjson, f, indent=2)

    # Edge diagnostics
    edges = {}
    for b in boundaries:
        ring = b[0]
        for j in range(len(ring)):
            e = (min(ring[j], ring[(j + 1) % len(ring)]),
                 max(ring[j], ring[(j + 1) % len(ring)]))
            edges.setdefault(e, []).append(1)
    n_shared = sum(1 for v in edges.values() if len(v) == 2)
    n_boundary = sum(1 for v in edges.values() if len(v) == 1)
    n_nonmanifold = sum(1 for v in edges.values() if len(v) > 2)

    return {
        'cityjson_path': cj_path,
        'n_surfaces': len(boundaries),
        'n_vertices': len(all_verts),
        'signed_volume': float(vol),
        'n_edges_shared': n_shared,
        'n_edges_boundary': n_boundary,
        'n_edges_nonmanifold': n_nonmanifold,
        'surface_types': {
            'RoofSurface': sum(1 for s in sem_surfaces if s['type'] == 'RoofSurface'),
            'WallSurface': sum(1 for s in sem_surfaces if s['type'] == 'WallSurface'),
            'GroundSurface': sum(1 for s in sem_surfaces if s['type'] == 'GroundSurface'),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'scripts')
    from stage3_synthetic.buildings import create_all_buildings
    from stage3_synthetic.primitives import generate_primitives_for_building
    from building_to_citygml_v4 import cluster_primitives, orient_normals_outward

    buildings = create_all_buildings()
    print("Step 1 Verification: Footprint Extraction")
    print("=" * 60)

    all_ok = True
    for bid in range(20):
        b = buildings[bid]
        prims = generate_primitives_for_building(b, 30, seed=42 + bid)
        c = prims['centers']
        lab = prims['semantic_probs'].argmax(axis=1)
        groups = cluster_primitives(c, prims['normals'], prims['areas'], lab)
        orient_normals_outward(groups, c.mean(0))

        wall_groups = [(i, g) for i, g in enumerate(groups) if g['class'] == 2]
        wall_gs = [{'plane_normal': g['plane_normal'], 'plane_d': g['plane_d'],
                     'prim_ids': g['prim_ids']} for _, g in wall_groups]

        fp, edges = extract_footprint(wall_gs, c)
        if fp is not None:
            area = float(ShapelyPoly(fp).area)
            print(f"  B{bid:02d} {b['name']:18s} {len(fp):2d}v area={area:6.1f} edges={len(edges)}")
        else:
            print(f"  B{bid:02d} {b['name']:18s} FAIL")
            all_ok = False

    print(f"\nResult: {'ALL PASSED' if all_ok else 'SOME FAILED'}")

    # Step 5-6 test: CityJSON + val3dity
    import subprocess, json, os
    from building_to_citygml_v4 import add_ground_surface
    print("\n\nStep 5-6: CityJSON Export + val3dity (20 buildings)")
    print("=" * 60)
    out_dir = '/tmp/test_2_5d'
    n_valid = 0
    for bid in range(20):
        b = buildings[bid]
        prims = generate_primitives_for_building(b, 30, seed=42 + bid)
        c = prims['centers']
        lab = prims['semantic_probs'].argmax(axis=1)
        groups = cluster_primitives(c, prims['normals'], prims['areas'], lab)
        orient_normals_outward(groups, c.mean(0))
        add_ground_surface(groups, c[lab == 2], c.mean(0))

        solid = build_2_5d_solid(groups, c)
        if solid is None:
            print(f"  B{bid:02d} {b['name']:18s} BUILD FAILED")
            continue

        result = faces_to_cityjson(solid, bid, out_dir)
        if result is None:
            print(f"  B{bid:02d} {b['name']:18s} CITYJSON FAILED")
            continue

        # val3dity
        cp = result['cityjson_path']
        rp = cp.replace('.city.json', '_v.json')
        subprocess.run(['val3dity', '--report', rp, cp],
                      capture_output=True, timeout=10)
        valid = False
        errs = ''
        if os.path.exists(rp):
            with open(rp) as f:
                vr = json.load(f)
            valid = vr.get('validity', False)
            if not valid:
                ecodes = set()
                for feat in vr.get('features', []):
                    for pr in feat.get('primitives', []):
                        for sh in pr.get('shells', []):
                            for e in sh.get('errors', []):
                                ecodes.add(str(e.get('code', '?')))
                errs = ' ' + str(ecodes) if ecodes else ''
            os.remove(rp)

        status = 'VALID' if valid else f'INVALID{errs}'
        if valid:
            n_valid += 1
        bnd = f"bnd={result['n_edges_boundary']}" if result['n_edges_boundary'] > 0 else ''
        print(f"  B{bid:02d} {b['name']:18s} {status:20s} "
              f"R:{result['surface_types']['RoofSurface']} "
              f"W:{result['surface_types']['WallSurface']} "
              f"G:{result['surface_types']['GroundSurface']} {bnd}")

    print(f"\nval3dity: {n_valid}/20 VALID")

    # Step 3-4 test: Full 2.5D solid
    print("\n\nStep 3-4 Verification: Full 2.5D Solid")
    print("=" * 60)
    from building_to_citygml_v4 import add_ground_surface, add_bbox_planes
    for bid in range(20):
        b = buildings[bid]
        prims = generate_primitives_for_building(b, 30, seed=42 + bid)
        c = prims['centers']
        lab = prims['semantic_probs'].argmax(axis=1)
        groups = cluster_primitives(c, prims['normals'], prims['areas'], lab)
        orient_normals_outward(groups, c.mean(0))
        add_ground_surface(groups, c[lab == 2], c.mean(0))

        solid_faces = build_2_5d_solid(groups, c)
        if solid_faces is None:
            print(f"  B{bid:02d} {b['name']:18s} FAILED")
            continue

        n_roof = sum(1 for f in solid_faces if f['type'] == 'RoofSurface')
        n_wall = sum(1 for f in solid_faces if f['type'] == 'WallSurface')
        n_ground = sum(1 for f in solid_faces if f['type'] == 'GroundSurface')
        total_verts = sum(len(f['vertices']) for f in solid_faces)

        # Check: all faces have >=3 vertices, no degenerate
        ok = all(len(f['vertices']) >= 3 for f in solid_faces)
        status = "OK" if ok else "DEGENERATE"

        print(f"  B{bid:02d} {b['name']:18s} R:{n_roof} W:{n_wall} G:{n_ground} "
              f"total_faces={len(solid_faces)} verts={total_verts} {status}")

    # Step 2 test: Ridge computation
    print("\n\nStep 2 Verification: Ridge Computation")
    print("=" * 60)
    for bid in range(20):
        b = buildings[bid]
        prims = generate_primitives_for_building(b, 30, seed=42 + bid)
        c = prims['centers']
        lab = prims['semantic_probs'].argmax(axis=1)
        groups = cluster_primitives(c, prims['normals'], prims['areas'], lab)
        orient_normals_outward(groups, c.mean(0))

        # Footprint
        wall_gs = [{'plane_normal': g['plane_normal'], 'plane_d': g['plane_d'],
                     'prim_ids': g['prim_ids']} for _, g in enumerate(groups) if g['class'] == 2]
        fp, _ = extract_footprint(wall_gs, c)
        if fp is None:
            continue

        # Roof groups
        roof_gs = [{'plane_normal': g['plane_normal'], 'plane_d': g['plane_d'],
                     'prim_ids': g['prim_ids'], 'gi': i}
                    for i, g in enumerate(groups) if g['class'] == 1]

        ridges = compute_ridges(roof_gs, fp)

        n_roof = len(roof_gs)
        roof_type = 'flat' if n_roof <= 1 else f'{n_roof} planes'
        if ridges:
            ridge_strs = []
            for r in ridges:
                p1, p2 = r['line']
                length = np.linalg.norm(p2 - p1)
                ridge_strs.append(f"L={length:.1f}m")
            print(f"  B{bid:02d} {b['name']:18s} roof={roof_type:10s} ridges: {', '.join(ridge_strs)}")
        else:
            print(f"  B{bid:02d} {b['name']:18s} roof={roof_type:10s} no ridges (flat roof)")
