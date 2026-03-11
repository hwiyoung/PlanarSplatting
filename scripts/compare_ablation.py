#!/usr/bin/env python3
"""Compare multiple ablation evaluation results.

Reads eval_results.json files from each ablation condition, produces:
  - A terminal table showing key metrics + deltas vs (c) independent
  - A CSV file for further analysis

Usage:
    python scripts/compare_ablation.py \
        --results_dir results/phase3b/eval \
        --output_csv results/phase3b/ablation_comparison.csv

    # Or specify individual JSON files:
    python scripts/compare_ablation.py \
        --json_files \
            results/phase3b/eval/abl_a_geo_only.json \
            results/phase3b/eval/abl_b_sem_only.json \
            results/phase3b/eval/abl_c_independent.json \
            results/phase3b/eval/abl_d_joint.json \
            results/phase3b/eval/abl_e_sem2geo.json \
            results/phase3b/eval/abl_f_geo2sem.json \
            results/phase3b/eval/abl_g_no_warmup.json \
        --output_csv results/phase3b/ablation_comparison.csv
"""
import os
import sys
import json
import argparse
import csv
from pathlib import Path


# Ablation condition metadata
ABLATION_ORDER = [
    ('a', 'geo_only',     'L_d+L_n+L_geo'),
    ('b', 'sem_only',     'L_d+L_n+L_sem'),
    ('c', 'independent',  'L_d+L_n+L_geo+L_sem'),
    ('d', 'joint',        'L_d+L_n+L_geo+L_sem+L_mut(full)'),
    ('e', 'sem2geo',      'L_d+L_n+L_geo+L_sem+L_mut(s→g)'),
    ('f', 'geo2sem',      'L_d+L_n+L_geo+L_sem+L_mut(g→s)'),
    ('g', 'no_warmup',    'L_d+L_n+L_geo+L_sem+L_mut(no-wu)'),
]

METRICS = [
    ('depth_mae_mean',      'Depth MAE',  '{:.4f}', -1),   # lower is better
    ('normal_cos_mean',     'Normal cos', '{:.4f}', +1),   # higher is better
    ('semantic_miou_mean',  'mIoU',       '{:.4f}', +1),   # higher is better
    ('iou_roof_mean',       'IoU roof',   '{:.4f}', +1),
    ('iou_wall_mean',       'IoU wall',   '{:.4f}', +1),
    ('iou_ground_mean',     'IoU ground', '{:.4f}', +1),
    ('plane_count',         'Planes',     '{:d}',    0),   # informational
]


def load_results(results_dir=None, json_files=None):
    """Load ablation results from directory or explicit file list."""
    data = {}

    if json_files:
        for fp in json_files:
            fp = Path(fp)
            if not fp.exists():
                print(f"WARNING: {fp} not found, skipping")
                continue
            with open(fp) as f:
                result = json.load(f)
            # Infer ablation key from filename
            name = fp.stem
            for letter, key, _ in ABLATION_ORDER:
                if key in name or f'abl_{letter}' in name:
                    data[letter] = result
                    break
            else:
                # Try to match by position
                data[name] = result
    elif results_dir:
        rdir = Path(results_dir)
        for fp in sorted(rdir.glob('*.json')):
            with open(fp) as f:
                result = json.load(f)
            name = fp.stem
            for letter, key, _ in ABLATION_ORDER:
                if key in name or f'abl_{letter}' in name:
                    data[letter] = result
                    break
    return data


def format_delta(val, ref, fmt, direction):
    """Format a delta value with sign and direction indicator."""
    if val is None or ref is None:
        return '—'
    d = val - ref
    if direction == 0:
        return ''
    sign = '+' if d >= 0 else ''
    # For depth MAE, negative delta is good (lower); for others, positive is good
    if (direction < 0 and d < 0) or (direction > 0 and d > 0):
        indicator = ' ↑'  # improved
    elif d == 0:
        indicator = ' ='
    else:
        indicator = ' ↓'  # degraded
    if isinstance(val, int):
        return f'{sign}{d:d}{indicator}'
    return f'{sign}{d:.4f}{indicator}'


def print_table(data):
    """Print a formatted comparison table to terminal."""
    ref = data.get('c')  # (c) independent is the reference

    # Header
    print()
    print('=' * 110)
    print(f'Phase 3-B Ablation Comparison (reference: (c) Independent)')
    print('=' * 110)

    # Column widths
    cw_cond = 8
    cw_desc = 35
    cw_val = 12
    cw_delta = 16

    # Print metric headers
    header1 = f'{"Cond":<{cw_cond}} {"Description":<{cw_desc}}'
    header2 = f'{"":—<{cw_cond}} {"":—<{cw_desc}}'
    for metric_key, metric_name, fmt, direction in METRICS:
        header1 += f' {metric_name:>{cw_val}}'
        if direction != 0:
            header1 += f' {"Δ(c)":>{cw_delta}}'
        header2 += f' {"":—>{cw_val}}'
        if direction != 0:
            header2 += f' {"":—>{cw_delta}}'

    print(header1)
    print(header2.replace('—', '-'))

    # Print each condition
    for letter, key, desc in ABLATION_ORDER:
        if letter not in data:
            continue
        r = data[letter]
        row = f'({letter})    {desc:<{cw_desc}}'

        for metric_key, metric_name, fmt, direction in METRICS:
            val = r.get(metric_key)
            if val is not None:
                if isinstance(val, int):
                    row += f' {val:>{cw_val}d}'
                else:
                    row += f' {val:>{cw_val}.4f}'
            else:
                row += f' {"N/A":>{cw_val}}'

            if direction != 0:
                ref_val = ref.get(metric_key) if ref else None
                delta_str = format_delta(val, ref_val, fmt, direction)
                row += f' {delta_str:>{cw_delta}}'

        print(row)

    print('=' * 110)

    # Key comparisons
    if 'd' in data and 'c' in data:
        print('\n--- Key Comparisons ---')
        d_r = data['d']
        c_r = data['c']

        for metric_key, metric_name, fmt, direction in METRICS[:3]:
            d_val = d_r.get(metric_key)
            c_val = c_r.get(metric_key)
            if d_val is not None and c_val is not None:
                delta = d_val - c_val
                print(f'  (d)-(c) {metric_name}: {delta:+.4f}  '
                      f'[L_mutual total effect]')

    if 'd' in data and 'e' in data:
        d_r = data['d']
        e_r = data['e']
        nc_d = d_r.get('normal_cos_mean')
        nc_e = e_r.get('normal_cos_mean')
        if nc_d is not None and nc_e is not None:
            print(f'  (d)-(e) Normal cos: {nc_d - nc_e:+.4f}  '
                  f'[Sem→Geo direction contribution]')

    if 'd' in data and 'f' in data:
        d_r = data['d']
        f_r = data['f']
        miou_d = d_r.get('semantic_miou_mean')
        miou_f = f_r.get('semantic_miou_mean')
        if miou_d is not None and miou_f is not None:
            print(f'  (d)-(f) mIoU: {miou_d - miou_f:+.4f}  '
                  f'[Geo→Sem direction contribution]')

    # Synergy check: (d) > max((e), (f))
    if all(k in data for k in ['d', 'e', 'f']):
        d_r, e_r, f_r = data['d'], data['e'], data['f']
        nc_d = d_r.get('normal_cos_mean', 0)
        nc_e = e_r.get('normal_cos_mean', 0)
        nc_f = f_r.get('normal_cos_mean', 0)
        miou_d = d_r.get('semantic_miou_mean', 0)
        miou_e = e_r.get('semantic_miou_mean', 0)
        miou_f = f_r.get('semantic_miou_mean', 0)

        nc_synergy = nc_d > max(nc_e, nc_f)
        miou_synergy = miou_d > max(miou_e, miou_f)
        print(f'\n  Synergy check (d) > max((e),(f)):')
        print(f'    Normal cos: {nc_d:.4f} > max({nc_e:.4f}, {nc_f:.4f}) = '
              f'{max(nc_e, nc_f):.4f} → {"YES" if nc_synergy else "NO"}')
        print(f'    mIoU:       {miou_d:.4f} > max({miou_e:.4f}, {miou_f:.4f}) = '
              f'{max(miou_e, miou_f):.4f} → {"YES" if miou_synergy else "NO"}')

    if 'd' in data and 'g' in data:
        d_r = data['d']
        g_r = data['g']
        for metric_key, metric_name, fmt, direction in METRICS[:3]:
            d_val = d_r.get(metric_key)
            g_val = g_r.get(metric_key)
            if d_val is not None and g_val is not None:
                delta = g_val - d_val
                print(f'  (g)-(d) {metric_name}: {delta:+.4f}  '
                      f'[warmup necessity]')

    print()


def save_csv(data, csv_path):
    """Save comparison table as CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    ref = data.get('c')
    rows = []
    headers = ['Condition', 'Description']
    for metric_key, metric_name, fmt, direction in METRICS:
        headers.append(metric_name)
        if direction != 0 and ref:
            headers.append(f'Δ(c) {metric_name}')

    for letter, key, desc in ABLATION_ORDER:
        if letter not in data:
            continue
        r = data[letter]
        row = [f'({letter})', desc]

        for metric_key, metric_name, fmt, direction in METRICS:
            val = r.get(metric_key)
            if val is not None:
                if isinstance(val, int):
                    row.append(str(val))
                else:
                    row.append(f'{val:.4f}')
            else:
                row.append('N/A')

            if direction != 0 and ref:
                ref_val = ref.get(metric_key)
                if val is not None and ref_val is not None:
                    d = val - ref_val
                    row.append(f'{d:+.4f}')
                else:
                    row.append('—')

        rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f'CSV saved to: {csv_path}')


def main():
    parser = argparse.ArgumentParser(description='Compare ablation study results')
    parser.add_argument('--results_dir', type=str, default='',
                        help='Directory containing eval_results JSON files')
    parser.add_argument('--json_files', type=str, nargs='+', default=[],
                        help='Individual JSON result files')
    parser.add_argument('--output_csv', type=str, default='results/phase3b/ablation_comparison.csv',
                        help='Output CSV file path')
    args = parser.parse_args()

    if not args.results_dir and not args.json_files:
        parser.error('Provide either --results_dir or --json_files')

    data = load_results(args.results_dir, args.json_files)

    if not data:
        print('No results loaded!')
        sys.exit(1)

    print(f'Loaded {len(data)} ablation conditions: {sorted(data.keys())}')

    print_table(data)
    save_csv(data, args.output_csv)


if __name__ == '__main__':
    main()
