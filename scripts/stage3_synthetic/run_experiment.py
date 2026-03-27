#!/usr/bin/env python3
"""
Stage 3-Synthetic A: Main experiment runner.

For each building × n_prims × noise condition:
  1. Generate primitives from GT faces
  2. Apply noise
  3. Run Stage 3 (convex polytope → CityJSON)
  4. Run val3dity
  5. Compute metrics (face IoU, semantic accuracy, Hausdorff)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildings import create_all_buildings, save_gt_buildings, building_to_cityjson

# Import Stage 3 algorithm directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from building_to_citygml_v4 import (
    cluster_primitives, orient_normals_outward, add_ground_surface,
    add_bbox_planes, build_convex_polytope, build_cityjson,
)
from build_2_5d import build_2_5d_solid, faces_to_cityjson
from stage3_synthetic.primitives import (
    generate_primitives_for_building,
    add_noise_normal, add_noise_position, add_noise_classification,
    add_noise_missing_faces, add_noise_outliers,
)


def _offset_boundaries(boundaries, offset):
    """Recursively add offset to vertex indices in CityJSON boundaries."""
    if isinstance(boundaries, list):
        for i, item in enumerate(boundaries):
            if isinstance(item, int):
                boundaries[i] = item + offset
            elif isinstance(item, list):
                _offset_boundaries(item, offset)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _detect_decomposition(groups, centers, height_gap=2.0, parallel_gap=1.0):
    """Detect if building needs convex decomposition.

    Returns list of primitive-index sets for each sub-building, or None.

    Detection methods:
    1. Multi-level: roof groups at different heights → Y-split
    2. Non-convex footprint: parallel walls (same normal, different d)
       → split along the internal wall boundary
    """
    all_pids = []
    for g in groups:
        all_pids.extend(g['prim_ids'])
    all_pids = np.array(all_pids)

    # --- Method 1: multi-level (Y-split) ---
    roof_groups = [(i, g) for i, g in enumerate(groups) if g['class'] == 1]
    if len(roof_groups) >= 2:
        heights = [(i, g['plane_d']) for i, g in roof_groups]
        heights.sort(key=lambda x: x[1])
        levels = [[heights[0]]]
        for k in range(1, len(heights)):
            if abs(heights[k][1] - heights[k - 1][1]) > height_gap:
                levels.append([heights[k]])
            else:
                levels[-1].append(heights[k])
        if len(levels) >= 2:
            return _split_by_levels(groups, levels, centers), None

    # --- Method 2: non-convex footprint (parallel wall split) ---
    wall_groups = [(i, g) for i, g in enumerate(groups) if g['class'] == 2]
    best_split = None
    best_gap = 0

    for a in range(len(wall_groups)):
        for b in range(a + 1, len(wall_groups)):
            gi, g1 = wall_groups[a]
            gj, g2 = wall_groups[b]
            cos = float(np.dot(g1['plane_normal'], g2['plane_normal']))
            if cos < 0.95:
                continue  # not parallel
            dd = abs(g1['plane_d'] - g2['plane_d'])
            if dd < parallel_gap:
                continue
            if dd > best_gap:
                best_gap = dd
                # Split plane halfway between the two parallel walls
                split_normal = g1['plane_normal'].copy()
                split_d = (g1['plane_d'] + g2['plane_d']) / 2
                best_split = (split_normal, split_d)

    if best_split is not None:
        sn, sd = best_split
        # Partition all primitives by which side of split plane they're on
        side = centers[all_pids] @ sn - sd
        part_a = all_pids[side <= 0]
        part_b = all_pids[side > 0]
        if len(part_a) >= 4 and len(part_b) >= 4:
            # Store the split plane so run_stage3 can add it as internal wall
            return [part_a, part_b], (sn, sd)

    return None


def _split_by_levels(groups, levels, centers):
    """Split groups into sub-buildings by roof height levels."""
    # Assign walls to nearest roof level by mean Y
    level_roof_ys = []
    for level in levels:
        ys = []
        for gid, _ in level:
            for pid in groups[gid]['prim_ids']:
                ys.append(centers[pid, 1])
        level_roof_ys.append(np.mean(ys))

    parts = [[] for _ in levels]
    for i, g in enumerate(groups):
        if g['class'] == 1:
            # Already assigned to a level
            for li, level in enumerate(levels):
                if any(gid == i for gid, _ in level):
                    parts[li].extend(g['prim_ids'])
                    break
        elif g['class'] == 2:
            # Assign wall to closest roof level
            wall_mean_y = np.mean([centers[pid, 1] for pid in g['prim_ids']])
            best_li = np.argmin([abs(wall_mean_y - ry) for ry in level_roof_ys])
            parts[best_li].extend(g['prim_ids'])
        # Ground prims: add to all parts (each needs its own ground)

    return [np.array(p) for p in parts if len(p) >= 4]


def run_stage3(prims, building_id, out_dir, cos_thresh=0.85, hs_tol=0.05):
    """Run Stage 3 on primitives. Returns result dict or None."""
    centers = prims['centers']
    normals = prims['normals']
    areas = prims['areas']
    labels = prims['semantic_probs'].argmax(axis=1)

    n_roof = int((labels == 1).sum())
    n_wall = int((labels == 2).sum())

    if n_roof == 0 or n_wall == 0:
        return None

    groups = cluster_primitives(centers, normals, areas, labels,
                                cos_thresh=cos_thresh)

    # Hybrid: try 2.5D first (handles non-convex), fall back to convex polytope
    building_center = centers.mean(axis=0)
    orient_normals_outward(groups, building_center)

    wall_centers = centers[labels == 2]
    add_ground_surface(groups, wall_centers, building_center)

    # Try 2.5D solid
    solid_faces = build_2_5d_solid(groups, centers)
    if solid_faces is not None:
        result = faces_to_cityjson(solid_faces, building_id, out_dir)
        if result is not None:
            # Quick val3dity check
            cj_path = result['cityjson_path']
            rp = cj_path.replace('.city.json', '_val_quick.json')
            proc = subprocess.run(['val3dity', '--report', rp, cj_path],
                                  capture_output=True, timeout=10)
            if os.path.exists(rp):
                with open(rp) as f:
                    vr = json.load(f)
                os.remove(rp)
                if vr.get('validity', False):
                    return result

    # Fallback: convex polytope
    add_bbox_planes(groups, centers)
    polygons = build_convex_polytope(groups, centers, hs_tol=hs_tol)
    if polygons is None or len(polygons) < 4:
        return None

    result = build_cityjson(building_id, groups, polygons, out_dir)
    return result


def run_val3dity(cityjson_path):
    """Run val3dity on a CityJSON file. Returns dict with pass/fail + errors."""
    try:
        report_path = cityjson_path.replace('.city.json', '_val3dity.json')
        cmd = ['val3dity', '--report', report_path, cityjson_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        result = {'valid': False, 'errors': [], 'error_codes': []}

        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            result['valid'] = report.get('validity', False)

            # Extract error codes
            for feat in report.get('features', []):
                for prim in feat.get('primitives', []):
                    for shell in prim.get('shells', []):
                        for err in shell.get('errors', []):
                            code = err.get('code', 'unknown')
                            result['error_codes'].append(str(code))
                            result['errors'].append(err)

            if not result['error_codes'] and result['valid']:
                result['error_codes'] = []

        return result
    except Exception as e:
        return {'valid': False, 'errors': [str(e)], 'error_codes': ['EXEC_FAIL']}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_hausdorff_approx(gt_cityjson_path, pred_cityjson_path):
    """
    Approximate Hausdorff distance by sampling surface points.
    Returns max(d(gt→pred), d(pred→gt)).
    """
    try:
        gt_pts = _sample_cityjson_surface(gt_cityjson_path, n_samples=500)
        pred_pts = _sample_cityjson_surface(pred_cityjson_path, n_samples=500)
        if gt_pts is None or pred_pts is None or len(gt_pts) == 0 or len(pred_pts) == 0:
            return float('inf')

        # One-directional Hausdorff
        def one_dir(a, b):
            dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
            return float(dists.min(axis=1).max())

        return max(one_dir(gt_pts, pred_pts), one_dir(pred_pts, gt_pts))
    except Exception:
        return float('inf')


def _sample_cityjson_surface(path, n_samples=500):
    """Sample points from CityJSON surface polygons."""
    try:
        with open(path) as f:
            cj = json.load(f)
        scale = cj['transform']['scale']
        translate = cj['transform']['translate']
        verts = np.array(cj['vertices'], dtype=np.float64)
        verts = verts * scale + translate

        points = []
        rng = np.random.RandomState(42)
        for obj in cj['CityObjects'].values():
            for geom in obj.get('geometry', []):
                for shell in geom.get('boundaries', []):
                    n_faces = max(1, len(shell))
                    samples_per_face = max(3, n_samples // n_faces)
                    for face in shell:
                        for ring in face:
                            if len(ring) >= 3:
                                pts = verts[ring]
                                for _ in range(samples_per_face):
                                    ti = rng.randint(0, max(1, len(pts) - 2))
                                    u, v = rng.random(), rng.random()
                                    if u + v > 1:
                                        u, v = 1 - u, 1 - v
                                    i1 = min(ti + 1, len(pts) - 1)
                                    i2 = min(ti + 2, len(pts) - 1)
                                    p = (1-u-v)*pts[0] + u*pts[i1] + v*pts[i2]
                                    points.append(p)

        return np.array(points) if points else None
    except Exception:
        return None


def compute_semantic_accuracy(gt_cityjson_path, pred_cityjson_path):
    """
    Compare semantic labels of matched faces between GT and prediction.
    Match faces by combined normal similarity + center proximity.
    """
    try:
        gt_faces = _extract_face_info(gt_cityjson_path)
        pred_faces = _extract_face_info(pred_cityjson_path)
        if not gt_faces or not pred_faces:
            return 0.0

        correct = 0
        for gf in gt_faces:
            # Skip ground faces (always virtual in prediction)
            if gf['type'] == 'GroundSurface':
                continue
            best_score = -float('inf')
            best_type = None
            for pf in pred_faces:
                # Combined score: normal similarity (dominant) + inverse center distance
                nsim = abs(np.dot(gf['normal'], pf['normal']))
                cdist = np.linalg.norm(gf['center'] - pf['center'])
                score = nsim - 0.1 * cdist
                if score > best_score:
                    best_score = score
                    best_type = pf['type']
            if best_type == gf['type']:
                correct += 1

        n_non_ground = sum(1 for f in gt_faces if f['type'] != 'GroundSurface')
        return correct / max(1, n_non_ground)
    except Exception as e:
        return 0.0


def _extract_face_info(cityjson_path):
    """Extract face normals and types from CityJSON."""
    try:
        with open(cityjson_path) as f:
            cj = json.load(f)
        scale = cj['transform']['scale']
        translate = cj['transform']['translate']
        verts = np.array(cj['vertices'], dtype=np.float64)
        verts = verts * scale + translate

        faces = []
        for obj in cj['CityObjects'].values():
            for geom in obj.get('geometry', []):
                sem = geom.get('semantics', {})
                surfaces = sem.get('surfaces', [])
                values = sem.get('values', [[]])[0]
                boundaries = geom.get('boundaries', [[]])[0]

                for fi, face in enumerate(boundaries):
                    for ring in face:
                        if len(ring) >= 3:
                            pts = verts[ring]
                            center = pts.mean(axis=0)
                            e1, e2 = pts[1] - pts[0], pts[2] - pts[0]
                            n = np.cross(e1, e2)
                            nrm = np.linalg.norm(n)
                            if nrm > 1e-12:
                                n /= nrm
                            stype = 'Unknown'
                            if fi < len(values) and values[fi] < len(surfaces):
                                stype = surfaces[values[fi]].get('type', 'Unknown')
                            faces.append({'normal': n, 'center': center, 'type': stype})

        return faces
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Noise conditions
# ─────────────────────────────────────────────────────────────────────────────

NOISE_CONDITIONS = {
    'clean': {'type': 'none'},
    # Normal noise
    'normal_2deg': {'type': 'normal', 'sigma_deg': 2},
    'normal_10deg': {'type': 'normal', 'sigma_deg': 10},
    'normal_20deg': {'type': 'normal', 'sigma_deg': 20},
    # Position noise (isotropic)
    'pos_iso_0.1m': {'type': 'position', 'sigma_m': 0.1, 'mode': 'isotropic'},
    'pos_iso_0.5m': {'type': 'position', 'sigma_m': 0.5, 'mode': 'isotropic'},
    'pos_iso_1.0m': {'type': 'position', 'sigma_m': 1.0, 'mode': 'isotropic'},
    # Position noise (vertical)
    'pos_vert_0.1m': {'type': 'position', 'sigma_m': 0.1, 'mode': 'vertical'},
    'pos_vert_0.5m': {'type': 'position', 'sigma_m': 0.5, 'mode': 'vertical'},
    'pos_vert_1.0m': {'type': 'position', 'sigma_m': 1.0, 'mode': 'vertical'},
    # Position noise (horizontal)
    'pos_horiz_0.1m': {'type': 'position', 'sigma_m': 0.1, 'mode': 'horizontal'},
    'pos_horiz_0.5m': {'type': 'position', 'sigma_m': 0.5, 'mode': 'horizontal'},
    'pos_horiz_1.0m': {'type': 'position', 'sigma_m': 1.0, 'mode': 'horizontal'},
    # Classification errors
    'cls_5pct': {'type': 'classification', 'error_rate': 0.05},
    'cls_15pct': {'type': 'classification', 'error_rate': 0.15},
    'cls_30pct': {'type': 'classification', 'error_rate': 0.30},
    # Missing faces
    'missing_10pct': {'type': 'missing', 'drop_rate': 0.10},
    'missing_30pct': {'type': 'missing', 'drop_rate': 0.30},
    'missing_50pct': {'type': 'missing', 'drop_rate': 0.50},
    # Outliers
    'outlier_1pct': {'type': 'outlier', 'outlier_rate': 0.01},
    'outlier_5pct': {'type': 'outlier', 'outlier_rate': 0.05},
    'outlier_10pct': {'type': 'outlier', 'outlier_rate': 0.10},
    # ── Combined noise (realistic Stage 2 output simulation) ──
    # Mild: 실제 Stage 2가 잘 수렴한 경우
    'combined_mild': {'type': 'combined', 'steps': [
        {'type': 'normal', 'sigma_deg': 2},
        {'type': 'position', 'sigma_m': 0.1, 'mode': 'isotropic'},
        {'type': 'classification', 'error_rate': 0.05},
    ]},
    # Moderate: 실제 Stage 2 전형적 출력
    'combined_moderate': {'type': 'combined', 'steps': [
        {'type': 'normal', 'sigma_deg': 10},
        {'type': 'position', 'sigma_m': 0.5, 'mode': 'isotropic'},
        {'type': 'classification', 'error_rate': 0.15},
        {'type': 'outlier', 'outlier_rate': 0.03},
    ]},
    # Severe: Stage 2가 잘 수렴하지 못한 경우
    'combined_severe': {'type': 'combined', 'steps': [
        {'type': 'normal', 'sigma_deg': 20},
        {'type': 'position', 'sigma_m': 1.0, 'mode': 'isotropic'},
        {'type': 'classification', 'error_rate': 0.30},
        {'type': 'outlier', 'outlier_rate': 0.10},
    ]},
    # Moderate + missing: 일부 면 미관측
    'combined_mod_missing': {'type': 'combined', 'steps': [
        {'type': 'normal', 'sigma_deg': 10},
        {'type': 'position', 'sigma_m': 0.5, 'mode': 'isotropic'},
        {'type': 'classification', 'error_rate': 0.15},
        {'type': 'missing', 'drop_rate': 0.10},
    ]},
    # Normal + classification: 법선과 분류가 동시에 나쁜 경우
    'combined_norm_cls': {'type': 'combined', 'steps': [
        {'type': 'normal', 'sigma_deg': 10},
        {'type': 'classification', 'error_rate': 0.15},
    ]},
}


def apply_noise(prims, noise_cfg, seed=123):
    """Apply a noise condition (single or combined) to primitives."""
    rng = np.random.RandomState(seed)
    ntype = noise_cfg['type']
    if ntype == 'none':
        return prims
    elif ntype == 'normal':
        return add_noise_normal(prims, noise_cfg['sigma_deg'], rng)
    elif ntype == 'position':
        return add_noise_position(prims, noise_cfg['sigma_m'], noise_cfg['mode'], rng)
    elif ntype == 'classification':
        return add_noise_classification(prims, noise_cfg['error_rate'], rng)
    elif ntype == 'missing':
        return add_noise_missing_faces(prims, noise_cfg['drop_rate'], rng)
    elif ntype == 'outlier':
        return add_noise_outliers(prims, noise_cfg['outlier_rate'], rng)
    elif ntype == 'combined':
        result = prims
        for i, step in enumerate(noise_cfg['steps']):
            result = apply_noise(result, step, seed=seed + i * 37)
        return result
    else:
        raise ValueError(f"Unknown noise type: {ntype}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_single(bldg, n_prims, noise_name, noise_cfg, base_out_dir):
    """Run one experiment: building × n_prims × noise condition."""
    bid = bldg['id']
    out_dir = os.path.join(base_out_dir, f"nprims_{n_prims}", noise_name,
                           f"building_{bid:03d}")
    os.makedirs(out_dir, exist_ok=True)

    # Generate primitives
    prims = generate_primitives_for_building(bldg, n_prims, seed=42 + bid)

    # Apply noise
    noisy_prims = apply_noise(prims, noise_cfg, seed=123 + bid)

    # Run Stage 3
    result = run_stage3(noisy_prims, bid, out_dir)

    entry = {
        'building_id': bid,
        'building_name': bldg['name'],
        'building_type': bldg['type'],
        'n_prims_per_face': n_prims,
        'noise': noise_name,
        'n_primitives': len(noisy_prims['centers']),
        'stage3_success': result is not None,
    }

    if result is None:
        entry['val3dity_valid'] = False
        entry['val3dity_errors'] = ['STAGE3_FAIL']
        entry['hausdorff'] = float('inf')
        entry['semantic_accuracy'] = 0.0
        return entry

    entry.update({
        'n_surfaces': result['n_surfaces'],
        'n_vertices': result['n_vertices'],
        'signed_volume': result['signed_volume'],
        'n_edges_shared': result['n_edges_shared'],
        'n_edges_boundary': result['n_edges_boundary'],
        'n_edges_nonmanifold': result['n_edges_nonmanifold'],
    })

    # val3dity
    cj_path = result['cityjson_path']
    val_result = run_val3dity(cj_path)
    entry['val3dity_valid'] = val_result['valid']
    entry['val3dity_errors'] = val_result['error_codes']

    # Hausdorff distance
    gt_cj_path = os.path.join(base_out_dir, 'gt_buildings',
                               f"building_{bid:03d}", 'gt.city.json')
    if os.path.exists(gt_cj_path):
        entry['hausdorff'] = compute_hausdorff_approx(gt_cj_path, cj_path)
    else:
        entry['hausdorff'] = float('inf')

    # Semantic accuracy
    entry['semantic_accuracy'] = compute_semantic_accuracy(gt_cj_path, cj_path)

    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='results/stage3_synthetic_a')
    parser.add_argument('--n_prims', type=int, nargs='+', default=[10, 30, 50])
    parser.add_argument('--noise_filter', nargs='*', default=None,
                        help='Run only specified noise conditions')
    parser.add_argument('--building_ids', type=int, nargs='*', default=None)
    args = parser.parse_args()

    # Part A: Create GT buildings
    print("=" * 70)
    print("STAGE 3-SYNTHETIC A: Creating GT buildings")
    print("=" * 70)
    buildings = create_all_buildings()
    gt_dir = os.path.join(args.out_dir, 'gt_buildings')
    catalog = save_gt_buildings(buildings, gt_dir)

    # Filter buildings if specified
    if args.building_ids is not None:
        buildings = [b for b in buildings if b['id'] in args.building_ids]

    # Filter noise conditions
    conditions = NOISE_CONDITIONS
    if args.noise_filter:
        conditions = {k: v for k, v in conditions.items() if k in args.noise_filter}

    # Run experiments
    total = len(buildings) * len(args.n_prims) * len(conditions)
    print(f"\n{'=' * 70}")
    print(f"Running {total} experiments: "
          f"{len(buildings)} buildings × {len(args.n_prims)} densities × "
          f"{len(conditions)} noise conditions")
    print(f"{'=' * 70}")

    all_results = []
    done = 0

    for n_prims in args.n_prims:
        for noise_name, noise_cfg in conditions.items():
            for bldg in buildings:
                done += 1
                sys.stdout.write(f"\r[{done}/{total}] n={n_prims} {noise_name} "
                                 f"B{bldg['id']:02d}({bldg['name']})... ")
                sys.stdout.flush()
                try:
                    entry = run_single(bldg, n_prims, noise_name, noise_cfg,
                                       args.out_dir)
                    all_results.append(entry)
                except Exception as e:
                    print(f"\n  ERROR: {e}")
                    traceback.print_exc()
                    all_results.append({
                        'building_id': bldg['id'],
                        'building_name': bldg['name'],
                        'n_prims_per_face': n_prims,
                        'noise': noise_name,
                        'stage3_success': False,
                        'val3dity_valid': False,
                        'val3dity_errors': [str(e)],
                        'error': str(e),
                    })

    print(f"\n\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")

    # Save full results
    results_path = os.path.join(args.out_dir, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: int(x) if isinstance(x, (np.integer,)) else
                  float(x) if isinstance(x, (np.floating,)) else
                  x.tolist() if isinstance(x, np.ndarray) else str(x))

    # Print summary
    for n_prims in args.n_prims:
        print(f"\n--- n_prims_per_face = {n_prims} ---")
        for noise_name in conditions:
            subset = [r for r in all_results
                      if r.get('n_prims_per_face') == n_prims and r.get('noise') == noise_name]
            n_total = len(subset)
            n_success = sum(1 for r in subset if r.get('stage3_success', False))
            n_valid = sum(1 for r in subset if r.get('val3dity_valid', False))
            avg_haus = np.mean([r.get('hausdorff', float('inf')) for r in subset
                                if r.get('hausdorff', float('inf')) < 1e6])
            avg_sem = np.mean([r.get('semantic_accuracy', 0) for r in subset])
            print(f"  {noise_name:20s}: S3={n_success}/{n_total} "
                  f"val3d={n_valid}/{n_total} "
                  f"Haus={avg_haus:.3f}m sem_acc={avg_sem:.3f}")

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
