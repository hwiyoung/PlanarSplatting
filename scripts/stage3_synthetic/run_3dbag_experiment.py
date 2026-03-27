#!/usr/bin/env python3
"""
Synthetic A: Run noise experiment on 3D BAG buildings.

Evaluates Stage 3 algorithm tolerance using real-world buildings from 3D BAG.
Measures val3dity (structural quality), Chamfer distance (shape accuracy),
and Semantic accuracy (label correctness).
"""

import json
import os
import subprocess
import sys
import time
import io
import contextlib
import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from buildings_3dbag import load_all_scenes
from primitives import (
    generate_primitives_for_building,
    add_noise_normal, add_noise_position, add_noise_classification,
    add_noise_missing_faces, add_noise_outliers, add_noise_area,
)
from run_experiment import run_stage3, compute_hausdorff_approx, compute_semantic_accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Noise conditions
# ─────────────────────────────────────────────────────────────────────────────

NOISE_CONDITIONS = {
    # Baseline
    'clean': {'type': 'none'},

    # Single factor: Normal (가장 민감한 요인)
    'normal_2deg': {'type': 'normal', 'sigma_deg': 2},
    'normal_10deg': {'type': 'normal', 'sigma_deg': 10},
    'normal_20deg': {'type': 'normal', 'sigma_deg': 20},

    # Single factor: Position
    'pos_iso_0.5m': {'type': 'position', 'sigma_m': 0.5, 'mode': 'isotropic'},
    'pos_iso_1.0m': {'type': 'position', 'sigma_m': 1.0, 'mode': 'isotropic'},

    # Single factor: Classification
    'cls_15pct': {'type': 'classification', 'error_rate': 0.15},
    'cls_30pct': {'type': 'classification', 'error_rate': 0.30},

    # Single factor: Missing primitives
    'missing_30pct': {'type': 'missing', 'drop_rate': 0.30},
    'missing_50pct': {'type': 'missing', 'drop_rate': 0.50},

    # Single factor: Outliers
    'outlier_5pct': {'type': 'outlier', 'outlier_rate': 0.05},
    'outlier_10pct': {'type': 'outlier', 'outlier_rate': 0.10},

    # Single factor: Area error
    'area_30pct': {'type': 'area', 'sigma_pct': 30},
    'area_50pct': {'type': 'area', 'sigma_pct': 50},
    'area_100pct': {'type': 'area', 'sigma_pct': 100},

    # Combined: "법선이 임계 수준일 때, 나머지가 전부 최악이면?"
    'N10_worst': {'type': 'combined', 'steps': [
        {'type': 'normal', 'sigma_deg': 10},
        {'type': 'position', 'sigma_m': 1.0, 'mode': 'isotropic'},
        {'type': 'classification', 'error_rate': 0.30},
        {'type': 'missing', 'drop_rate': 0.50},
        {'type': 'outlier', 'outlier_rate': 0.10},
        {'type': 'area', 'sigma_pct': 100},
    ]},
    'N2_worst': {'type': 'combined', 'steps': [
        {'type': 'normal', 'sigma_deg': 2},
        {'type': 'position', 'sigma_m': 1.0, 'mode': 'isotropic'},
        {'type': 'classification', 'error_rate': 0.30},
        {'type': 'missing', 'drop_rate': 0.50},
        {'type': 'outlier', 'outlier_rate': 0.10},
        {'type': 'area', 'sigma_pct': 100},
    ]},
}


# ─────────────────────────────────────────────────────────────────────────────
# Noise application
# ─────────────────────────────────────────────────────────────────────────────

def apply_noise(prims, cfg, seed):
    rng = np.random.RandomState(seed)
    ntype = cfg['type']
    if ntype == 'none':
        return prims
    elif ntype == 'normal':
        return add_noise_normal(prims, cfg['sigma_deg'], rng)
    elif ntype == 'position':
        return add_noise_position(prims, cfg['sigma_m'], cfg['mode'], rng)
    elif ntype == 'classification':
        return add_noise_classification(prims, cfg['error_rate'], rng)
    elif ntype == 'missing':
        return add_noise_missing_faces(prims, cfg['drop_rate'], rng)
    elif ntype == 'outlier':
        return add_noise_outliers(prims, cfg['outlier_rate'], rng)
    elif ntype == 'area':
        return add_noise_area(prims, cfg['sigma_pct'], rng)
    elif ntype == 'combined':
        result = prims
        for i, step in enumerate(cfg['steps']):
            result = apply_noise(result, step, seed + i * 37)
        return result
    return prims


# ─────────────────────────────────────────────────────────────────────────────
# Chamfer distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_chamfer(gt_path, pred_path, n_samples=500):
    """Bidirectional mean surface distance (Chamfer)."""
    try:
        gt_pts = _sample_surface(gt_path, n_samples)
        pred_pts = _sample_surface(pred_path, n_samples)
        if gt_pts is None or pred_pts is None:
            return float('inf')
        # A→B mean min distance
        def one_dir(a, b):
            dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
            return float(dists.min(axis=1).mean())
        return (one_dir(gt_pts, pred_pts) + one_dir(pred_pts, gt_pts)) / 2
    except Exception:
        return float('inf')


def _sample_surface(path, n_samples):
    """Sample points from CityJSON surfaces."""
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
                    spf = max(3, n_samples // n_faces)
                    for face in shell:
                        for ring in face:
                            if len(ring) >= 3:
                                pts = verts[ring]
                                for _ in range(spf):
                                    ti = rng.randint(0, max(1, len(pts) - 2))
                                    u, v = rng.random(), rng.random()
                                    if u + v > 1:
                                        u, v = 1 - u, 1 - v
                                    i1 = min(ti + 1, len(pts) - 1)
                                    i2 = min(ti + 2, len(pts) - 1)
                                    p = (1 - u - v) * pts[0] + u * pts[i1] + v * pts[i2]
                                    points.append(p)
        return np.array(points) if points else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def find_valid_buildings(scenes, n_per_scene=None):
    """Find all buildings that pass val3dity in clean n=30."""
    valid = []
    for sn, buildings in scenes.items():
        count = 0
        for b in buildings:
            prims = generate_primitives_for_building(b, 30, seed=42 + b['id'])
            out_dir = f'/tmp/3dbag_valid/{sn}/b{b["id"]:03d}'
            os.makedirs(out_dir, exist_ok=True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = run_stage3(prims, b['id'], out_dir)
            if result:
                cp = result.get('cityjson_path', '')
                if cp and os.path.exists(cp):
                    rp = cp.replace('.city.json', '_v.json')
                    subprocess.run(['val3dity', '--report', rp, cp],
                                   capture_output=True, timeout=10)
                    if os.path.exists(rp):
                        vr = json.load(open(rp))
                        if vr.get('validity', False):
                            valid.append((sn, b, cp))  # keep GT path
                            count += 1
                        os.remove(rp)
            if n_per_scene and count >= n_per_scene:
                break
        sys.stdout.write(f'  {sn}: {count} valid / {len(buildings)} total\n')
        sys.stdout.flush()
    return valid


def main():
    out_base = 'results/stage3_synthetic_a'
    log_path = '/tmp/3dbag_experiment.log'

    def log(msg):
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}'
        sys.stdout.write(line + '\n')
        sys.stdout.flush()
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log('=== Synthetic A: 3D BAG Experiment ===')
    log('Loading scenes...')
    scenes = load_all_scenes()
    total_buildings = sum(len(b) for b in scenes.values())
    log(f'Total buildings: {total_buildings}')

    # Stratified sampling by roof type
    # slanted: 200, horizontal: 200, multiple horizontal: all (~210)
    ROOF_TYPE_TARGETS = {
        'slanted': 200,
        'horizontal': 200,
        'multiple horizontal': 9999,  # take all
    }
    log(f'Sampling targets (by roof type): {ROOF_TYPE_TARGETS}')

    log('Finding valid buildings (clean n=30) — early stop per roof type...')
    # Find valid buildings, stopping when each roof type has enough
    valid_by_type = {}  # type → [(sn, b, gt)]
    n_checked = 0
    for sn, buildings in scenes.items():
        for b in buildings:
            # Check if this type already has enough
            rt = b['type']
            target = ROOF_TYPE_TARGETS.get(rt, 200)
            current = len(valid_by_type.get(rt, []))
            if current >= target:
                continue

            n_checked += 1
            prims = generate_primitives_for_building(b, 30, seed=42 + b['id'])
            out_dir = f'/tmp/3dbag_valid/{sn}/b{b["id"]:03d}'
            os.makedirs(out_dir, exist_ok=True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = run_stage3(prims, b['id'], out_dir)
            if result:
                cp = result.get('cityjson_path', '')
                if cp and os.path.exists(cp):
                    rp = cp.replace('.city.json', '_v.json')
                    subprocess.run(['val3dity', '--report', rp, cp],
                                   capture_output=True, timeout=10)
                    if os.path.exists(rp):
                        vr = json.load(open(rp))
                        if vr.get('validity', False):
                            valid_by_type.setdefault(rt, []).append((sn, b, cp))
                        os.remove(rp)

            # Progress
            if n_checked % 100 == 0:
                counts = {t: len(v) for t, v in valid_by_type.items()}
                log(f'  checked {n_checked}, valid so far: {counts}')

        # Check if all types have enough
        all_done = all(
            len(valid_by_type.get(rt, [])) >= target
            for rt, target in ROOF_TYPE_TARGETS.items()
            if target < 9999
        )
        if all_done and len(valid_by_type.get('multiple horizontal', [])) > 0:
            log(f'  All types reached target after {n_checked} checks')
            break

    sampled = []
    rng = np.random.RandomState(42)
    for rt, items in sorted(valid_by_type.items()):
        n_target = ROOF_TYPE_TARGETS.get(rt, 200)
        n_take = min(n_target, len(items))
        idx = rng.choice(len(items), n_take, replace=False)
        sampled.extend([items[i] for i in idx])
        log(f'  {rt}: {n_take}/{len(items)} sampled')
    log(f'Total sampled: {len(sampled)} buildings (checked {n_checked})')

    # Save sampled building list
    valid_list_path = os.path.join(out_base, '3dbag_sampled_buildings.json')
    with open(valid_list_path, 'w') as f:
        json.dump([{'scene': sn, 'id': b['id'], 'name': b['name'],
                     'type': b['type'], 'gt_path': gt}
                    for sn, b, gt in sampled], f, indent=2)
    log(f'Sampled list saved: {valid_list_path}')

    valid = sampled  # use sampled for experiment

    # Run experiments
    total_exp = len(valid) * len(NOISE_CONDITIONS)
    log(f'Running {total_exp} experiments ({len(valid)} buildings x {len(NOISE_CONDITIONS)} conditions)')

    all_results = []
    done = 0
    t0 = time.time()

    for noise_name, noise_cfg in NOISE_CONDITIONS.items():
        n_valid = 0
        n_total = 0
        chamfers = []
        sem_accs = []

        for sn, b, gt_path in valid:
            done += 1
            n_total += 1

            prims = generate_primitives_for_building(b, 30, seed=42 + b['id'])
            noisy = apply_noise(prims, noise_cfg, seed=123 + b['id'])

            out_dir = os.path.join(out_base, f'3dbag_results/{noise_name}/b{b["id"]:03d}')
            os.makedirs(out_dir, exist_ok=True)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = run_stage3(noisy, b['id'], out_dir)

            val_ok = False
            chamfer = float('inf')
            sem_acc = 0.0

            if result:
                cp = result.get('cityjson_path', '')
                if cp and os.path.exists(cp):
                    # val3dity
                    rp = cp.replace('.city.json', '_v.json')
                    subprocess.run(['val3dity', '--report', rp, cp],
                                   capture_output=True, timeout=10)
                    if os.path.exists(rp):
                        vr = json.load(open(rp))
                        val_ok = vr.get('validity', False)
                        os.remove(rp)

                    # Chamfer distance (vs clean GT)
                    chamfer = compute_chamfer(gt_path, cp)

                    # Semantic accuracy
                    sem_acc = compute_semantic_accuracy(gt_path, cp)

            if val_ok:
                n_valid += 1
            if chamfer < 1e6:
                chamfers.append(chamfer)
            sem_accs.append(sem_acc)

            all_results.append({
                'building_id': b['id'],
                'building_name': b['name'],
                'building_type': b['type'],
                'scene': sn,
                'noise': noise_name,
                'val3dity_valid': val_ok,
                'stage3_success': result is not None,
                'chamfer': chamfer if chamfer < 1e6 else None,
                'semantic_accuracy': sem_acc,
            })

        avg_chamfer = np.mean(chamfers) if chamfers else float('inf')
        avg_sem = np.mean(sem_accs) if sem_accs else 0

        elapsed = time.time() - t0
        eta = elapsed / done * (total_exp - done) if done > 0 else 0

        log(f'{noise_name:25s}: val3d={n_valid}/{n_total} ({n_valid/n_total*100:.0f}%) '
            f'chamfer={avg_chamfer:.2f}m sem={avg_sem:.2f} '
            f'[{done}/{total_exp}, ETA {eta/3600:.1f}h]')

        # Save intermediate results after each condition
        intermediate_path = os.path.join(out_base, '3dbag_results_partial.json')
        with open(intermediate_path, 'w') as f:
            json.dump(all_results, f, indent=2,
                      default=lambda x: int(x) if isinstance(x, np.integer) else
                      float(x) if isinstance(x, np.floating) else str(x))

    # Save final results
    res_path = os.path.join(out_base, '3dbag_results.json')
    with open(res_path, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: int(x) if isinstance(x, np.integer) else
                  float(x) if isinstance(x, np.floating) else str(x))
    log(f'Final results saved: {res_path}')
    log('Done.')


if __name__ == '__main__':
    main()
