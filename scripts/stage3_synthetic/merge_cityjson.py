#!/usr/bin/env python3
"""
Merge all GT and/or result CityJSON files into one file for CityJSON Ninja viewing.

Buildings are spatially arranged in a grid so all 20 are visible at once.
GT buildings go in one row, results in a parallel row for side-by-side comparison.

Usage:
  python scripts/stage3_synthetic/merge_cityjson.py
  python scripts/stage3_synthetic/merge_cityjson.py --noise combined_mild --n_prims 30
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildings import create_all_buildings


def load_cityjson(path):
    """Load a CityJSON file and return decoded vertices + objects."""
    with open(path) as f:
        cj = json.load(f)
    scale = cj['transform']['scale']
    translate = cj['transform']['translate']
    verts = np.array(cj['vertices'], dtype=np.float64)
    # Decode to world coordinates
    world_verts = verts * np.array(scale) + np.array(translate)
    return cj, world_verts


def merge_cityjson_files(file_list, offsets, output_path, row_labels=None):
    """
    Merge multiple CityJSON files into one, with spatial offsets.

    Args:
        file_list: list of CityJSON file paths
        offsets: list of (dx, dy, dz) offsets for each file
        output_path: where to save merged CityJSON
        row_labels: optional labels for each entry
    """
    scale = 0.0001
    all_verts = []
    vert_map = {}
    all_objects = {}
    all_sem_surfaces = []

    for fi, (path, offset) in enumerate(zip(file_list, offsets)):
        if not os.path.exists(path):
            continue

        cj, world_verts = load_cityjson(path)
        world_verts += np.array(offset)

        # Map vertices
        local_to_global = {}
        for vi, v in enumerate(world_verts):
            ix = round(v[0] / scale)
            iy = round(v[1] / scale)
            iz = round(v[2] / scale)
            key = (ix, iy, iz)
            if key not in vert_map:
                vert_map[key] = len(vert_map)
                all_verts.append([ix, iy, iz])
            local_to_global[vi] = vert_map[key]

        # Remap objects
        for obj_name, obj_data in cj.get('CityObjects', {}).items():
            label = row_labels[fi] if row_labels else ""
            new_name = f"{label}_{obj_name}" if label else obj_name
            new_obj = json.loads(json.dumps(obj_data))  # deep copy

            # Remap vertex indices in boundaries
            for geom in new_obj.get('geometry', []):
                _remap_boundaries(geom.get('boundaries', []), local_to_global)

            all_objects[new_name] = new_obj

    # Build merged CityJSON
    translate = [min(v[j] for v in all_verts) * scale for j in range(3)]
    t_ijk = [round(translate[j] / scale) for j in range(3)]
    adj_verts = [[v[j] - t_ijk[j] for j in range(3)] for v in all_verts]

    merged = {
        'type': 'CityJSON',
        'version': '2.0',
        'transform': {'scale': [scale] * 3, 'translate': translate},
        'CityObjects': all_objects,
        'vertices': adj_verts,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged, f)

    print(f"Merged {len(all_objects)} objects, {len(adj_verts)} vertices → {output_path}")
    return output_path


def _remap_boundaries(boundaries, local_to_global):
    """Recursively remap vertex indices in CityJSON boundaries."""
    if isinstance(boundaries, list):
        for i, item in enumerate(boundaries):
            if isinstance(item, int):
                boundaries[i] = local_to_global.get(item, item)
            elif isinstance(item, list):
                _remap_boundaries(item, local_to_global)


def get_grid_offset(idx, n_cols=5, spacing_x=25, spacing_z=25, row_offset_z=0):
    """Grid position for building idx."""
    col = idx % n_cols
    row = idx // n_cols
    return (col * spacing_x, 0, row * spacing_z + row_offset_z)


def main():
    parser = argparse.ArgumentParser(
        description='Merge CityJSON files for side-by-side viewing in CityJSON Ninja')
    parser.add_argument('--base_dir', default='results/stage3_synthetic_a')
    parser.add_argument('--noise', default='clean',
                        help='Noise condition for result row')
    parser.add_argument('--n_prims', type=int, default=30)
    parser.add_argument('--n_cols', type=int, default=5,
                        help='Grid columns')
    parser.add_argument('--spacing', type=float, default=25,
                        help='Spacing between buildings (meters)')
    args = parser.parse_args()

    # Auto-detect number of buildings from GT directory
    gt_base = os.path.join(args.base_dir, 'gt_buildings')
    if os.path.isdir(gt_base):
        n_buildings = len([d for d in os.listdir(gt_base) if d.startswith('building_')])
    else:
        n_buildings = 20

    # --- GT-only merged file ---
    gt_files = []
    gt_offsets = []
    gt_labels = []
    for bid in range(n_buildings):
        gt_path = os.path.join(args.base_dir, 'gt_buildings',
                               f'building_{bid:03d}', 'gt.city.json')
        gt_files.append(gt_path)
        gt_offsets.append(get_grid_offset(bid, args.n_cols, args.spacing, args.spacing))
        gt_labels.append(f'GT')

    gt_out = os.path.join(args.base_dir, 'merged_gt_all.city.json')
    print(f"=== GT (all {n_buildings} buildings) ===")
    merge_cityjson_files(gt_files, gt_offsets, gt_out, gt_labels)

    # --- Result-only merged file ---
    res_files = []
    res_offsets = []
    res_labels = []
    for bid in range(n_buildings):
        res_path = os.path.join(args.base_dir, f'nprims_{args.n_prims}',
                                args.noise, f'building_{bid:03d}', 'building.city.json')
        res_files.append(res_path)
        res_offsets.append(get_grid_offset(bid, args.n_cols, args.spacing, args.spacing))
        res_labels.append(f'RES')

    res_out = os.path.join(args.base_dir,
                           f'merged_result_{args.noise}_n{args.n_prims}.city.json')
    print(f"\n=== Results ({args.noise}, n={args.n_prims}) ===")
    merge_cityjson_files(res_files, res_offsets, res_out, res_labels)

    # --- Side-by-side: GT (top row) + Result (bottom row) ---
    all_files = []
    all_offsets = []
    all_labels = []
    row_gap = args.spacing * ((n_buildings - 1) // args.n_cols + 1) + args.spacing * 2

    for bid in range(n_buildings):
        # GT row
        gt_path = os.path.join(args.base_dir, 'gt_buildings',
                               f'building_{bid:03d}', 'gt.city.json')
        all_files.append(gt_path)
        all_offsets.append(get_grid_offset(bid, args.n_cols, args.spacing, args.spacing))
        all_labels.append(f'GT_B{bid:02d}')

        # Result row (shifted in Z)
        res_path = os.path.join(args.base_dir, f'nprims_{args.n_prims}',
                                args.noise, f'building_{bid:03d}', 'building.city.json')
        all_files.append(res_path)
        all_offsets.append(get_grid_offset(bid, args.n_cols, args.spacing, args.spacing,
                                           row_offset_z=row_gap))
        all_labels.append(f'RES_B{bid:02d}')

    sbs_out = os.path.join(args.base_dir,
                           f'merged_sidebyside_{args.noise}_n{args.n_prims}.city.json')
    print(f"\n=== Side-by-side (GT top, {args.noise} bottom) ===")
    merge_cityjson_files(all_files, all_offsets, sbs_out, all_labels)

    print(f"\n--- Output files ---")
    print(f"  GT only:      {gt_out}")
    print(f"  Results only:  {res_out}")
    print(f"  Side-by-side:  {sbs_out}")
    print(f"\nOpen in CityJSON Ninja: https://ninja.cityjson.org/")


if __name__ == '__main__':
    main()
