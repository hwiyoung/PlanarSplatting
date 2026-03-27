#!/usr/bin/env python3
"""
Stage 3-Synthetic A: Analysis and visualization.

Generates all required plots and tables from experiment results.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path):
    with open(results_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: val3dity pass rate by noise type
# ─────────────────────────────────────────────────────────────────────────────

def plot_val3dity_by_noise_type(results, out_dir, n_prims=30):
    """X=noise level, Y=val3dity pass rate, one line per noise type."""
    noise_groups = {
        'Normal (deg)': {
            'clean': 0, 'normal_2deg': 2, 'normal_10deg': 10, 'normal_20deg': 20,
        },
        'Position iso (m)': {
            'clean': 0, 'pos_iso_0.1m': 0.1, 'pos_iso_0.5m': 0.5, 'pos_iso_1.0m': 1.0,
        },
        'Position vert (m)': {
            'clean': 0, 'pos_vert_0.1m': 0.1, 'pos_vert_0.5m': 0.5, 'pos_vert_1.0m': 1.0,
        },
        'Position horiz (m)': {
            'clean': 0, 'pos_horiz_0.1m': 0.1, 'pos_horiz_0.5m': 0.5, 'pos_horiz_1.0m': 1.0,
        },
        'Classification (%)': {
            'clean': 0, 'cls_5pct': 5, 'cls_15pct': 15, 'cls_30pct': 30,
        },
        'Missing faces (%)': {
            'clean': 0, 'missing_10pct': 10, 'missing_30pct': 30, 'missing_50pct': 50,
        },
        'Outliers (%)': {
            'clean': 0, 'outlier_1pct': 1, 'outlier_5pct': 5, 'outlier_10pct': 10,
        },
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    subset = [r for r in results if r.get('n_prims_per_face') == n_prims]

    for ax_idx, (group_name, noise_map) in enumerate(noise_groups.items()):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        xs, ys_val, ys_s3 = [], [], []

        for noise_name, x_val in sorted(noise_map.items(), key=lambda kv: kv[1]):
            ns = [r for r in subset if r.get('noise') == noise_name]
            if not ns:
                continue
            xs.append(x_val)
            ys_val.append(sum(1 for r in ns if r.get('val3dity_valid', False)) / len(ns) * 100)
            ys_s3.append(sum(1 for r in ns if r.get('stage3_success', False)) / len(ns) * 100)

        ax.plot(xs, ys_val, 'b-o', label='val3dity pass', linewidth=2)
        ax.plot(xs, ys_s3, 'r--s', label='Stage 3 success', linewidth=1.5)
        ax.set_title(group_name, fontsize=11)
        ax.set_ylim(-5, 105)
        ax.set_ylabel('Rate (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=80, color='green', linestyle=':', alpha=0.5, label='80% threshold')

    # Hide unused axes
    for i in range(len(noise_groups), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'val3dity Pass Rate by Noise Type (n_prims={n_prims})', fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, 'val3dity_by_noise_type.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Multi-primitive count comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_nprims_comparison(results, out_dir):
    """Compare n_prims=10,30,50 across noise conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ('val3dity pass rate (%)', lambda r: r.get('val3dity_valid', False)),
        ('Hausdorff (m)', lambda r: r.get('hausdorff', float('inf'))),
        ('Semantic accuracy', lambda r: r.get('semantic_accuracy', 0)),
    ]

    n_prims_list = sorted(set(r.get('n_prims_per_face', 30) for r in results))

    for ax_idx, (metric_name, metric_fn) in enumerate(metrics):
        ax = axes[ax_idx]
        # Group by noise condition
        noise_names = sorted(set(r.get('noise', '') for r in results))
        x = np.arange(len(noise_names))
        width = 0.8 / len(n_prims_list)

        for ni, np_ in enumerate(n_prims_list):
            vals = []
            for nn in noise_names:
                sub = [r for r in results
                       if r.get('n_prims_per_face') == np_ and r.get('noise') == nn]
                if metric_name.startswith('val3dity'):
                    vals.append(sum(1 for r in sub if metric_fn(r)) / max(1, len(sub)) * 100)
                elif 'Hausdorff' in metric_name:
                    hvals = [metric_fn(r) for r in sub if metric_fn(r) < 1e6]
                    vals.append(np.mean(hvals) if hvals else 0)
                else:
                    vals.append(np.mean([metric_fn(r) for r in sub]) if sub else 0)

            ax.bar(x + ni * width, vals, width, label=f'n={np_}', alpha=0.8)

        ax.set_title(metric_name)
        ax.set_xticks(x + width * (len(n_prims_list) - 1) / 2)
        ax.set_xticklabels(noise_names, rotation=45, ha='right', fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Multi-Primitive Count Comparison', fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, 'nprims_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Sensitivity ranking
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity_ranking(results, out_dir, n_prims=30):
    """Bar chart ranking noise types by impact on val3dity."""
    subset = [r for r in results if r.get('n_prims_per_face') == n_prims]
    clean = [r for r in subset if r.get('noise') == 'clean']
    clean_rate = sum(1 for r in clean if r.get('val3dity_valid', False)) / max(1, len(clean))

    noise_impact = {}
    for noise_name in set(r.get('noise', '') for r in subset):
        if noise_name == 'clean':
            continue
        ns = [r for r in subset if r.get('noise') == noise_name]
        rate = sum(1 for r in ns if r.get('val3dity_valid', False)) / max(1, len(ns))
        noise_impact[noise_name] = clean_rate - rate

    # Sort by impact (descending)
    sorted_noise = sorted(noise_impact.items(), key=lambda kv: kv[1], reverse=True)
    names = [kv[0] for kv in sorted_noise]
    impacts = [kv[1] * 100 for kv in sorted_noise]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#d32f2f' if v > 20 else '#f57c00' if v > 10 else '#4caf50' for v in impacts]
    bars = ax.barh(range(len(names)), impacts, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('val3dity pass rate drop (pp)')
    ax.set_title(f'Sensitivity Ranking: Impact on val3dity (n_prims={n_prims})')
    ax.invert_yaxis()
    ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Critical (20pp)')
    ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='Warning (10pp)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    path = os.path.join(out_dir, 'sensitivity_ranking.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: val3dity error type distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_distribution(results, out_dir, n_prims=30):
    """Distribution of val3dity error codes across noise conditions."""
    subset = [r for r in results if r.get('n_prims_per_face') == n_prims]

    error_counts = defaultdict(lambda: defaultdict(int))
    for r in subset:
        noise = r.get('noise', 'unknown')
        for code in r.get('val3dity_errors', []):
            error_counts[noise][str(code)] += 1

    # Get all error codes
    all_codes = sorted(set(c for nc in error_counts.values() for c in nc.keys()))
    if not all_codes:
        print("  No errors to plot")
        return

    noise_names = sorted(error_counts.keys())

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(noise_names))
    width = 0.8 / max(1, len(all_codes))

    for ci, code in enumerate(all_codes):
        vals = [error_counts[nn].get(code, 0) for nn in noise_names]
        ax.bar(x + ci * width, vals, width, label=f'Error {code}', alpha=0.8)

    ax.set_xticks(x + width * (len(all_codes) - 1) / 2)
    ax.set_xticklabels(noise_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Count')
    ax.set_title(f'val3dity Error Distribution (n_prims={n_prims})')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, 'error_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Outlier threshold analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_outlier_analysis(results, out_dir):
    """Outlier rate vs metrics — identify cliff point."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    outlier_noises = ['clean', 'outlier_1pct', 'outlier_5pct', 'outlier_10pct']
    outlier_rates = [0, 1, 5, 10]

    for ni, n_prims in enumerate(sorted(set(r.get('n_prims_per_face', 30) for r in results))):
        val_rates, haus_vals, sem_vals = [], [], []
        for nn in outlier_noises:
            sub = [r for r in results
                   if r.get('n_prims_per_face') == n_prims and r.get('noise') == nn]
            val_rates.append(sum(1 for r in sub if r.get('val3dity_valid', False)) / max(1, len(sub)) * 100)
            hvals = [r.get('hausdorff', float('inf')) for r in sub if r.get('hausdorff', float('inf')) < 1e6]
            haus_vals.append(np.mean(hvals) if hvals else 0)
            sem_vals.append(np.mean([r.get('semantic_accuracy', 0) for r in sub]) if sub else 0)

        axes[0].plot(outlier_rates, val_rates, '-o', label=f'n={n_prims}')
        axes[1].plot(outlier_rates, haus_vals, '-o', label=f'n={n_prims}')
        axes[2].plot(outlier_rates, sem_vals, '-o', label=f'n={n_prims}')

    for ax, title in zip(axes, ['val3dity pass (%)', 'Hausdorff (m)', 'Semantic accuracy']):
        ax.set_xlabel('Outlier rate (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Outlier Rate Analysis')
    plt.tight_layout()
    path = os.path.join(out_dir, 'outlier_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary_table(results, out_dir, n_prims=30):
    """Generate markdown summary table."""
    subset = [r for r in results if r.get('n_prims_per_face') == n_prims]
    noise_names = sorted(set(r.get('noise', '') for r in subset))

    lines = [
        f"## Summary Table (n_prims_per_face={n_prims})\n",
        "| Noise Condition | Stage3 Success | val3dity Pass | Hausdorff (m) | Sem. Acc. |",
        "|---|---|---|---|---|",
    ]

    for nn in noise_names:
        sub = [r for r in subset if r.get('noise') == nn]
        n = len(sub)
        s3 = sum(1 for r in sub if r.get('stage3_success', False))
        val = sum(1 for r in sub if r.get('val3dity_valid', False))
        hvals = [r.get('hausdorff', float('inf')) for r in sub if r.get('hausdorff', float('inf')) < 1e6]
        haus = f"{np.mean(hvals):.3f}" if hvals else "N/A"
        sem = np.mean([r.get('semantic_accuracy', 0) for r in sub])
        lines.append(f"| {nn} | {s3}/{n} ({s3/n*100:.0f}%) | {val}/{n} ({val/n*100:.0f}%) "
                      f"| {haus} | {sem:.3f} |")

    # Tolerance thresholds (80% val3dity pass)
    lines.append("\n## Tolerance Thresholds (≥80% val3dity pass)\n")
    noise_groups = {
        'Normal': ['normal_2deg', 'normal_10deg', 'normal_20deg'],
        'Position (iso)': ['pos_iso_0.1m', 'pos_iso_0.5m', 'pos_iso_1.0m'],
        'Position (vert)': ['pos_vert_0.1m', 'pos_vert_0.5m', 'pos_vert_1.0m'],
        'Position (horiz)': ['pos_horiz_0.1m', 'pos_horiz_0.5m', 'pos_horiz_1.0m'],
        'Classification': ['cls_5pct', 'cls_15pct', 'cls_30pct'],
        'Missing faces': ['missing_10pct', 'missing_30pct', 'missing_50pct'],
        'Outliers': ['outlier_1pct', 'outlier_5pct', 'outlier_10pct'],
    }

    lines.append("| Noise Type | Max Tolerated Level | val3dity Pass at Max |")
    lines.append("|---|---|---|")

    for gname, noise_list in noise_groups.items():
        max_level = "None"
        max_rate = 0
        for nn in noise_list:
            sub = [r for r in subset if r.get('noise') == nn]
            if not sub:
                continue
            rate = sum(1 for r in sub if r.get('val3dity_valid', False)) / len(sub) * 100
            if rate >= 80:
                max_level = nn
                max_rate = rate
        lines.append(f"| {gname} | {max_level} | {max_rate:.0f}% |")

    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Building type breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_building_type_comparison(results, out_dir, n_prims=30):
    """Compare val3dity pass rate by building type."""
    subset = [r for r in results
              if r.get('n_prims_per_face') == n_prims and r.get('noise') == 'clean']

    by_type = defaultdict(list)
    for r in subset:
        by_type[r.get('building_type', 'unknown')].append(r)

    types = sorted(by_type.keys())
    rates = []
    for t in types:
        rs = by_type[t]
        rates.append(sum(1 for r in rs if r.get('val3dity_valid', False)) / max(1, len(rs)) * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(types, rates, color='steelblue', alpha=0.8)
    ax.set_ylabel('val3dity pass rate (%)')
    ax.set_title(f'val3dity Pass Rate by Building Type (clean, n_prims={n_prims})')
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, 'building_type_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_combined_noise(results, out_dir, n_prims=30):
    """Combined vs single noise comparison."""
    combined_names = ['combined_mild', 'combined_moderate', 'combined_severe',
                      'combined_mod_missing', 'combined_norm_cls']
    labels_map = {
        'clean': 'Clean',
        'combined_mild': 'Mild\n(N2°+P0.1+C5%)',
        'combined_moderate': 'Moderate\n(N10°+P0.5+C15%+O3%)',
        'combined_severe': 'Severe\n(N20°+P1.0+C30%+O10%)',
        'combined_mod_missing': 'Mod+Missing\n(N10°+P0.5+C15%+M10%)',
        'combined_norm_cls': 'Norm+Cls\n(N10°+C15%)',
    }

    subset = [r for r in results if r.get('n_prims_per_face') == n_prims]
    noise_order = ['clean'] + combined_names

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(noise_order))

    # Metric 1: val3dity pass rate
    val_rates = []
    for nn in noise_order:
        sub = [r for r in subset if r.get('noise') == nn]
        val_rates.append(sum(1 for r in sub if r.get('val3dity_valid', False)) / max(1, len(sub)) * 100)

    colors = ['#4caf50'] + ['#2196f3', '#ff9800', '#f44336', '#9c27b0', '#795548']
    axes[0].bar(x, val_rates, color=colors, alpha=0.85)
    axes[0].set_ylabel('val3dity pass rate (%)')
    axes[0].set_ylim(0, 105)
    axes[0].axhline(y=80, color='green', linestyle='--', alpha=0.5)
    axes[0].set_title('val3dity Pass Rate')

    # Metric 2: Hausdorff
    haus_vals = []
    for nn in noise_order:
        sub = [r for r in subset if r.get('noise') == nn]
        hvals = [r.get('hausdorff', float('inf')) for r in sub
                 if r.get('hausdorff', float('inf')) < 1e6]
        haus_vals.append(np.mean(hvals) if hvals else 0)

    axes[1].bar(x, haus_vals, color=colors, alpha=0.85)
    axes[1].set_ylabel('Hausdorff distance (m)')
    axes[1].set_title('Geometric Fidelity')

    # Metric 3: Semantic accuracy
    sem_vals = []
    for nn in noise_order:
        sub = [r for r in subset if r.get('noise') == nn]
        sem_vals.append(np.mean([r.get('semantic_accuracy', 0) for r in sub]) if sub else 0)

    axes[2].bar(x, sem_vals, color=colors, alpha=0.85)
    axes[2].set_ylabel('Semantic accuracy')
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title('Semantic Accuracy')

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([labels_map.get(nn, nn) for nn in noise_order],
                           fontsize=8, rotation=0)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Combined Noise: Single vs Multi-Factor (n_prims={n_prims})', fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, 'combined_noise_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_single_vs_combined(results, out_dir, n_prims=30):
    """
    Compare: sum of individual effects vs actual combined effect.
    Shows whether noise effects are additive, sub-additive, or super-additive.
    """
    subset = [r for r in results if r.get('n_prims_per_face') == n_prims]

    def pass_rate(noise_name):
        sub = [r for r in subset if r.get('noise') == noise_name]
        return sum(1 for r in sub if r.get('val3dity_valid', False)) / max(1, len(sub)) * 100

    clean_rate = pass_rate('clean')

    # For each combined condition, compare actual drop vs expected (sum of individual drops)
    combos = {
        'Mild': {
            'actual': 'combined_mild',
            'parts': ['normal_2deg', 'pos_iso_0.1m', 'cls_5pct'],
        },
        'Moderate': {
            'actual': 'combined_moderate',
            'parts': ['normal_10deg', 'pos_iso_0.5m', 'cls_15pct', 'outlier_1pct'],
        },
        'Severe': {
            'actual': 'combined_severe',
            'parts': ['normal_20deg', 'pos_iso_1.0m', 'cls_30pct', 'outlier_10pct'],
        },
        'Norm+Cls': {
            'actual': 'combined_norm_cls',
            'parts': ['normal_10deg', 'cls_15pct'],
        },
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(combos))
    width = 0.35

    actual_drops = []
    expected_drops = []
    names = []

    for name, cfg in combos.items():
        actual_rate = pass_rate(cfg['actual'])
        actual_drop = clean_rate - actual_rate

        expected_drop = sum(clean_rate - pass_rate(p) for p in cfg['parts'])

        actual_drops.append(actual_drop)
        expected_drops.append(expected_drop)
        names.append(name)

    ax.bar(x - width/2, expected_drops, width, label='Expected (sum of individuals)',
           color='#90caf9', alpha=0.9)
    ax.bar(x + width/2, actual_drops, width, label='Actual combined',
           color='#e53935', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('val3dity pass rate drop (pp)')
    ax.set_title(f'Noise Interaction: Additive vs Actual (n_prims={n_prims})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate super/sub-additive
    for i, (exp, act) in enumerate(zip(expected_drops, actual_drops)):
        if act > exp * 1.1:
            ax.text(i, max(exp, act) + 1, 'super-additive', ha='center', fontsize=8, color='red')
        elif act < exp * 0.9:
            ax.text(i, max(exp, act) + 1, 'sub-additive', ha='center', fontsize=8, color='blue')
        else:
            ax.text(i, max(exp, act) + 1, '~additive', ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    path = os.path.join(out_dir, 'single_vs_combined.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/stage3_synthetic_a/all_results.json')
    parser.add_argument('--out_dir', default='results/stage3_synthetic_a/images')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = load_results(args.results)
    print(f"Loaded {len(results)} experiment results")

    print("\nGenerating plots...")
    plot_val3dity_by_noise_type(results, args.out_dir, n_prims=30)
    plot_nprims_comparison(results, args.out_dir)
    plot_sensitivity_ranking(results, args.out_dir, n_prims=30)
    plot_error_distribution(results, args.out_dir, n_prims=30)
    plot_outlier_analysis(results, args.out_dir)
    plot_building_type_comparison(results, args.out_dir, n_prims=30)
    plot_combined_noise(results, args.out_dir, n_prims=30)
    plot_single_vs_combined(results, args.out_dir, n_prims=30)

    # Summary table
    table = generate_summary_table(results, args.out_dir, n_prims=30)
    table_path = os.path.join(args.out_dir, '..', 'summary_table.md')
    with open(table_path, 'w') as f:
        f.write(table)
    print(f"\n  Summary table: {table_path}")


if __name__ == '__main__':
    main()
