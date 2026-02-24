"""Compare 3 normal sources side by side for selected views.

Uses the ACTUAL functions from the codebase:
  1. finite-diff: colmap_to_ps.py depth_to_normal_cam()
  2. MVS native: stored in input_data.pth
  3. smoothed depth-derived: generate_segmentation.py compute_depth_normals()
"""

import argparse
import os
import sys
import numpy as np
import torch
import cv2

# Add project paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from colmap_to_ps import depth_to_normal_cam, read_colmap_array
from generate_segmentation import compute_depth_normals


def read_colmap_array_buggy(path):
    """Read COLMAP array with the OLD BUGGY interleaved interpretation.

    This was the original bug: treating planar layout as interleaved.
    COLMAP stores data as (c, h, w) but this reads it as (h, w, c).
    Result: channels are mixed, per-channel stats become nearly identical.
    """
    with open(path, 'rb') as f:
        header = b''
        amp_count = 0
        while amp_count < 3:
            b = f.read(1)
            if b == b'&':
                amp_count += 1
            header += b
        parts = header.decode('ascii').split('&')
        w, h, c = int(parts[0]), int(parts[1]), int(parts[2])
        data = np.frombuffer(f.read(), dtype=np.float32)[:w * h * c]
        return data.reshape(h, w, c)  # BUGGY: interleaved interpretation


def channel_stats_text(normal_hw3, valid, label):
    """Get per-channel mean/std text for annotation."""
    lines = [label]
    for ch, name in enumerate(['nx', 'ny', 'nz']):
        vals = normal_hw3[:, :, ch][valid]
        if len(vals) > 0:
            lines.append(f"  {name}: mean={vals.mean():.4f} std={vals.std():.4f}")
        else:
            lines.append(f"  {name}: N/A")
    return '\n'.join(lines)


def normal_to_color(normal_01_or_neg1, valid, storage='01'):
    """Convert normal map to RGB color image.
    storage='01': input in [0,1] (colmap_to_ps format)
    storage='neg1': input in [-1,1] (generate_segmentation format)
    """
    H, W = normal_01_or_neg1.shape[1], normal_01_or_neg1.shape[2]
    color = np.zeros((H, W, 3), dtype=np.uint8)
    if not valid.any():
        return color
    if storage == '01':
        n01 = normal_01_or_neg1
    else:
        n01 = (normal_01_or_neg1 + 1) / 2
    r = (n01[0] * 255).clip(0, 255).astype(np.uint8)
    g = (n01[1] * 255).clip(0, 255).astype(np.uint8)
    b = (n01[2] * 255).clip(0, 255).astype(np.uint8)
    color[valid, 0] = r[valid]
    color[valid, 1] = g[valid]
    color[valid, 2] = b[valid]
    return color


def to_neg1(normal_01):
    """Convert [0,1] normal to [-1,1]."""
    return normal_01 * 2 - 1


def angular_diff_map(n1, n2, valid1, valid2):
    """Per-pixel angular difference in degrees. Both inputs in [-1,1]."""
    both = valid1 & valid2
    H, W = n1.shape[1], n1.shape[2]
    diff = np.zeros((H, W), dtype=np.float32)
    if both.any():
        dot = np.clip(
            n1[0][both]*n2[0][both] + n1[1][both]*n2[1][both] + n1[2][both]*n2[2][both],
            -1, 1)
        diff[both] = np.degrees(np.arccos(dot))
    return diff, both


def angle_to_heatmap(angle_map, valid, vmax=45):
    """0°=blue → 22.5°=yellow → 45°+=red."""
    norm = (angle_map / vmax).clip(0, 1)
    heatmap = np.zeros((*angle_map.shape, 3), dtype=np.uint8)
    heatmap[valid, 2] = (255 * (1 - norm[valid])).astype(np.uint8)
    heatmap[valid, 1] = (255 * (1 - np.abs(norm[valid] - 0.5) * 2)).astype(np.uint8)
    heatmap[valid, 0] = (255 * norm[valid]).astype(np.uint8)
    return heatmap


def depth_to_color(depth):
    valid = depth > 0
    d = np.zeros_like(depth)
    if valid.any():
        d[valid] = (depth[valid] - depth[valid].min()) / (depth[valid].max() - depth[valid].min() + 1e-8)
    d_uint8 = (d * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d_uint8, cv2.COLORMAP_VIRIDIS)
    colored[~valid] = 0
    return colored


def add_text(img, text, pos=(10, 30), scale=0.7, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    x, y = pos
    cv2.rectangle(img, (x-2, y-th-6), (x+tw+2, y+4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', required=True)
    parser.add_argument('--views', type=int, nargs='+', default=[5, 15, 40, 60, 80])
    parser.add_argument('--output_dir', default='results/normal_comparison')
    parser.add_argument('--sigma', type=float, default=3.0)
    parser.add_argument('--colmap_path', default=None,
                        help='COLMAP dense dir (for raw normal maps). If set, reads normals from disk instead of input_data.pth')
    parser.add_argument('--show_bug', action='store_true',
                        help='Show planar reading bug: interleaved(buggy) vs planar(corrected) side by side. Requires --colmap_path.')
    args = parser.parse_args()

    if args.show_bug and not args.colmap_path:
        parser.error('--show_bug requires --colmap_path')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input_data}...")
    data = torch.load(args.input_data, map_location='cpu')

    stats_all = []

    for view_idx in args.views:
        img_name = data['image_paths'][view_idx].split('/')[-1]
        print(f"\n{'='*60}")
        print(f"View {view_idx}: {img_name}")
        print(f"{'='*60}")

        rgb = data['color'][view_idx]
        depth = data['depth'][view_idx].astype(np.float32)
        intrinsic = data['intrinsics'][view_idx].astype(np.float32)

        H, W = depth.shape
        total_px = H * W

        # === 1. finite-diff (colmap_to_ps.py actual function) ===
        n_fd_01 = depth_to_normal_cam(depth, intrinsic)   # (3,H,W) [0,1]
        n_fd = to_neg1(n_fd_01)
        # Valid: not the "zero normal" point (0.5,0.5,0.5 in [0,1] = 0,0,0 in [-1,1])
        valid_fd = np.abs(n_fd).sum(0) > 0.1

        # === 2. MVS native ===
        normal_map_dir = None
        geo_normal_path = None
        if args.colmap_path:
            normal_map_dir = os.path.join(args.colmap_path, 'dense', 'stereo', 'normal_maps')
            geo_normal_path = os.path.join(normal_map_dir, f'{img_name}.geometric.bin')
            # Read directly from raw COLMAP files (with fixed planar reading)
            mvs_raw = read_colmap_array(geo_normal_path)  # (H_mvs, W_mvs, 3)
            if mvs_raw.shape[:2] != (H, W):
                mvs_raw = cv2.resize(mvs_raw, (W, H), interpolation=cv2.INTER_NEAREST)
            mvs_norm = np.linalg.norm(mvs_raw, axis=-1, keepdims=True)
            mvs_valid_mask = (mvs_norm.squeeze() > 0.01) & (depth > 0)
            mvs_raw = np.where(mvs_norm > 0.01, mvs_raw / mvs_norm, 0.0)
            flip_mvs = mvs_raw[:, :, 2] > 0
            mvs_raw[flip_mvs] *= -1
            mvs_raw[~mvs_valid_mask] = 0.0
            # (H,W,3) [-1,1] → (3,H,W) [0,1]
            normal_mvs_01 = (mvs_raw.transpose(2, 0, 1).astype(np.float32) + 1.0) / 2.0
        else:
            normal_mvs_01 = data['normal'][view_idx]   # (3,H,W) [0,1]
        n_mvs = to_neg1(normal_mvs_01)
        valid_mvs = (depth > 0) & (np.abs(n_mvs).sum(0) > 0.1)

        # === 2b. MVS buggy reading (for --show_bug mode) ===
        if args.show_bug and geo_normal_path:
            buggy_raw = read_colmap_array_buggy(geo_normal_path)  # (H_mvs, W_mvs, 3) BUGGY
            if buggy_raw.shape[:2] != (H, W):
                buggy_raw = cv2.resize(buggy_raw, (W, H), interpolation=cv2.INTER_NEAREST)
            buggy_norm = np.linalg.norm(buggy_raw, axis=-1, keepdims=True)
            buggy_valid = (buggy_norm.squeeze() > 0.01) & (depth > 0)
            buggy_raw = np.where(buggy_norm > 0.01, buggy_raw / buggy_norm, 0.0)
            flip_buggy = buggy_raw[:, :, 2] > 0
            buggy_raw[flip_buggy] *= -1
            buggy_raw[~buggy_valid] = 0.0
            # (H,W,3) [-1,1] → (3,H,W) [0,1]
            normal_buggy_01 = (buggy_raw.transpose(2, 0, 1).astype(np.float32) + 1.0) / 2.0
            n_buggy = to_neg1(normal_buggy_01)
            valid_buggy = (depth > 0) & (np.abs(n_buggy).sum(0) > 0.1)

            # Print channel stats showing the bug evidence
            print("  --- Buggy (interleaved) channel stats ---")
            for ch, name in enumerate(['nx', 'ny', 'nz']):
                vals = buggy_raw[:, :, ch][buggy_valid]
                if len(vals) > 0:
                    print(f"    {name}: mean={vals.mean():.4f} std={vals.std():.4f}")
            print("  --- Corrected (planar) channel stats ---")
            for ch, name in enumerate(['nx', 'ny', 'nz']):
                vals = mvs_raw[:, :, ch][mvs_valid_mask]
                if len(vals) > 0:
                    print(f"    {name}: mean={vals.mean():.4f} std={vals.std():.4f}")

        # === 3. smoothed depth-derived (generate_segmentation.py actual function) ===
        n_sd, valid_sd, _ = compute_depth_normals(depth, intrinsic, sigma=args.sigma)
        # n_sd is in [-1,1], but may not be toward-camera. Flip to match convention (nz < 0).
        flip_sd = (n_sd[2] > 0) & valid_sd
        n_sd[:, flip_sd] = -n_sd[:, flip_sd]

        # Coverage
        cov_fd = valid_fd.sum() / total_px * 100
        cov_mvs = valid_mvs.sum() / total_px * 100
        cov_sd = valid_sd.sum() / total_px * 100
        cov_depth = (depth > 0).sum() / total_px * 100
        print(f"  Depth coverage:     {cov_depth:.1f}%")
        print(f"  1.finite-diff:      {cov_fd:.1f}%")
        print(f"  2.MVS native:       {cov_mvs:.1f}%")
        print(f"  3.smoothed(s={args.sigma}):  {cov_sd:.1f}%")

        # Angular differences (all in [-1,1])
        diff_mvs_fd, both_mvs_fd = angular_diff_map(n_mvs, n_fd, valid_mvs, valid_fd)
        diff_sd_fd, both_sd_fd = angular_diff_map(n_sd, n_fd, valid_sd, valid_fd)
        diff_mvs_sd, both_mvs_sd = angular_diff_map(n_mvs, n_sd, valid_mvs, valid_sd)

        mean_mvs_fd = diff_mvs_fd[both_mvs_fd].mean() if both_mvs_fd.any() else 0
        mean_sd_fd = diff_sd_fd[both_sd_fd].mean() if both_sd_fd.any() else 0
        mean_mvs_sd = diff_mvs_sd[both_mvs_sd].mean() if both_mvs_sd.any() else 0

        med_mvs_fd = np.median(diff_mvs_fd[both_mvs_fd]) if both_mvs_fd.any() else 0
        med_sd_fd = np.median(diff_sd_fd[both_sd_fd]) if both_sd_fd.any() else 0
        med_mvs_sd = np.median(diff_mvs_sd[both_mvs_sd]) if both_mvs_sd.any() else 0

        print(f"  MVS vs FD:  mean={mean_mvs_fd:.1f}° median={med_mvs_fd:.1f}° >30°: {(diff_mvs_fd[both_mvs_fd]>30).mean()*100:.1f}%" if both_mvs_fd.any() else "  MVS vs FD:  no overlap")
        print(f"  SD vs FD:   mean={mean_sd_fd:.1f}° median={med_sd_fd:.1f}° >30°: {(diff_sd_fd[both_sd_fd]>30).mean()*100:.1f}%" if both_sd_fd.any() else "  SD vs FD:   no overlap")
        print(f"  MVS vs SD:  mean={mean_mvs_sd:.1f}° median={med_mvs_sd:.1f}° >30°: {(diff_mvs_sd[both_mvs_sd]>30).mean()*100:.1f}%" if both_mvs_sd.any() else "  MVS vs SD:  no overlap")

        # --- Build comparison image ---
        scale_f = 0.5
        sH, sW = int(H * scale_f), int(W * scale_f)
        def resize(img):
            return cv2.resize(img, (sW, sH), interpolation=cv2.INTER_AREA)

        if args.show_bug and geo_normal_path:
            # ==== Bug comparison mode ====
            # Row 1: RGB | Depth
            rgb_bgr = resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            depth_c = resize(depth_to_color(depth))
            blank = np.zeros((sH, sW, 3), dtype=np.uint8)
            row1 = np.concatenate([
                add_text(rgb_bgr.copy(), f"RGB ({img_name})"),
                add_text(depth_c.copy(), f"Depth ({cov_depth:.0f}%)"),
                blank,
            ], axis=1)

            # Row 2: FD | MVS-buggy | MVS-corrected
            n_fd_color = resize(normal_to_color(n_fd_01, valid_fd, '01'))
            n_buggy_color = resize(normal_to_color(normal_buggy_01, valid_buggy, '01'))
            n_mvs_color = resize(normal_to_color(normal_mvs_01, valid_mvs, '01'))
            cov_buggy = valid_buggy.sum() / total_px * 100
            row2 = np.concatenate([
                add_text(n_fd_color.copy(), f"Finite-Diff ({cov_fd:.0f}%)"),
                add_text(n_buggy_color.copy(), f"MVS BUGGY interleaved ({cov_buggy:.0f}%)"),
                add_text(n_mvs_color.copy(), f"MVS CORRECTED planar ({cov_mvs:.0f}%)"),
            ], axis=1)

            # Row 3: Angle diffs (buggy vs FD, corrected vs FD, buggy vs corrected)
            diff_buggy_fd, both_buggy_fd = angular_diff_map(n_buggy, n_fd, valid_buggy, valid_fd)
            diff_buggy_mvs, both_buggy_mvs = angular_diff_map(n_buggy, n_mvs, valid_buggy, valid_mvs)
            mean_buggy_fd = diff_buggy_fd[both_buggy_fd].mean() if both_buggy_fd.any() else 0
            mean_buggy_mvs = diff_buggy_mvs[both_buggy_mvs].mean() if both_buggy_mvs.any() else 0

            hm1 = resize(angle_to_heatmap(diff_buggy_fd, both_buggy_fd))
            hm2 = resize(angle_to_heatmap(diff_mvs_fd, both_mvs_fd))
            hm3 = resize(angle_to_heatmap(diff_buggy_mvs, both_buggy_mvs))
            row3 = np.concatenate([
                add_text(hm1.copy(), f"buggy vs FD: {mean_buggy_fd:.1f}deg"),
                add_text(hm2.copy(), f"corrected vs FD: {mean_mvs_fd:.1f}deg"),
                add_text(hm3.copy(), f"buggy vs corrected: {mean_buggy_mvs:.1f}deg"),
            ], axis=1)

            # Row 4: Channel stats text panels
            def stats_panel(normal_hw3, valid_mask, title):
                panel = np.zeros((sH // 2, sW, 3), dtype=np.uint8)
                y = 15
                panel = add_text(panel, title, pos=(10, y), scale=0.5)
                for ch, name in enumerate(['nx', 'ny', 'nz']):
                    vals = normal_hw3[:, :, ch][valid_mask]
                    if len(vals) > 0:
                        y += 22
                        panel = add_text(panel, f"{name}: m={vals.mean():.4f} s={vals.std():.4f}",
                                         pos=(10, y), scale=0.45)
                return panel

            sp1 = stats_panel(np.stack([to_neg1(n_fd_01)[i] for i in range(3)], axis=-1),
                              valid_fd, "FD stats")
            sp2 = stats_panel(buggy_raw, buggy_valid, "BUGGY stats (channels mixed)")
            sp3 = stats_panel(mvs_raw, mvs_valid_mask, "CORRECTED stats")
            row4 = np.concatenate([sp1, sp2, sp3], axis=1)

            full = np.concatenate([row1, row2, row3, row4], axis=0)
        else:
            # ==== Normal 3-source comparison mode ====
            # Row 1: RGB | Depth | (blank)
            rgb_bgr = resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            depth_c = resize(depth_to_color(depth))
            blank = np.zeros((sH, sW, 3), dtype=np.uint8)
            row1 = np.concatenate([
                add_text(rgb_bgr.copy(), f"RGB ({img_name})"),
                add_text(depth_c.copy(), f"Depth ({cov_depth:.0f}%)"),
                blank
            ], axis=1)

            # Row 2: 3 normals
            n_fd_color = resize(normal_to_color(n_fd_01, valid_fd, '01'))
            n_mvs_color = resize(normal_to_color(normal_mvs_01, valid_mvs, '01'))
            n_sd_color = resize(normal_to_color(n_sd, valid_sd, 'neg1'))
            row2 = np.concatenate([
                add_text(n_fd_color.copy(), f"1.finite-diff ({cov_fd:.0f}%)"),
                add_text(n_mvs_color.copy(), f"2.MVS native ({cov_mvs:.0f}%)"),
                add_text(n_sd_color.copy(), f"3.smoothed-depth ({cov_sd:.0f}%)"),
            ], axis=1)

            # Row 3: Angle difference heatmaps
            hm1 = resize(angle_to_heatmap(diff_mvs_fd, both_mvs_fd))
            hm2 = resize(angle_to_heatmap(diff_sd_fd, both_sd_fd))
            hm3 = resize(angle_to_heatmap(diff_mvs_sd, both_mvs_sd))
            row3 = np.concatenate([
                add_text(hm1.copy(), f"2vs1: {mean_mvs_fd:.1f}deg mean"),
                add_text(hm2.copy(), f"3vs1: {mean_sd_fd:.1f}deg mean"),
                add_text(hm3.copy(), f"2vs3: {mean_mvs_sd:.1f}deg mean"),
            ], axis=1)

            full = np.concatenate([row1, row2, row3], axis=0)

        out_path = os.path.join(args.output_dir, f"normal_compare_view{view_idx:03d}.png")
        cv2.imwrite(out_path, full)
        print(f"  Saved: {out_path}")

        stats_all.append({
            'view': view_idx, 'name': img_name,
            'cov_fd': cov_fd, 'cov_mvs': cov_mvs, 'cov_sd': cov_sd,
            'mean_mvs_fd': mean_mvs_fd, 'mean_sd_fd': mean_sd_fd, 'mean_mvs_sd': mean_mvs_sd,
        })

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'View':>5s} {'FD%':>6s} {'MVS%':>6s} {'SD%':>6s} | {'MVS-FD':>8s} {'SD-FD':>8s} {'MVS-SD':>8s}")
    print("-" * 60)
    for s in stats_all:
        print(f"{s['view']:5d} {s['cov_fd']:5.1f}% {s['cov_mvs']:5.1f}% {s['cov_sd']:5.1f}% | "
              f"{s['mean_mvs_fd']:7.1f}° {s['mean_sd_fd']:7.1f}° {s['mean_mvs_sd']:7.1f}°")

    # Averages
    avg_mvs_fd = np.mean([s['mean_mvs_fd'] for s in stats_all])
    avg_sd_fd = np.mean([s['mean_sd_fd'] for s in stats_all])
    avg_mvs_sd = np.mean([s['mean_mvs_sd'] for s in stats_all])
    avg_cov_fd = np.mean([s['cov_fd'] for s in stats_all])
    avg_cov_mvs = np.mean([s['cov_mvs'] for s in stats_all])
    avg_cov_sd = np.mean([s['cov_sd'] for s in stats_all])
    print("-" * 60)
    print(f"{'AVG':>5s} {avg_cov_fd:5.1f}% {avg_cov_mvs:5.1f}% {avg_cov_sd:5.1f}% | "
          f"{avg_mvs_fd:7.1f}° {avg_sd_fd:7.1f}° {avg_mvs_sd:7.1f}°")

    print(f"\nHeatmap: blue=0° → yellow=22.5° → red=45°+")
    print(f"Output: {args.output_dir}/")


if __name__ == '__main__':
    main()
