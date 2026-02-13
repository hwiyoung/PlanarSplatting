#!/usr/bin/env python3
import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


LOG_PATTERN = re.compile(
    r"iter=(?P<iter>\d+)\s+"
    r"planes=(?P<planes>\d+)\s+"
    r"depth_loss=(?P<depth_loss>[-+0-9.eE]+)\s+"
    r"normal_l1=(?P<normal_l1>[-+0-9.eE]+)\s+"
    r"normal_cos=(?P<normal_cos>[-+0-9.eE]+)\s+"
    r"plane_loss=(?P<plane_loss>[-+0-9.eE]+)\s+"
    r"total_loss=(?P<total_loss>[-+0-9.eE]+)"
)


def calc_trend(values: np.ndarray, iters: np.ndarray, window: int) -> Tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    n = min(window, len(values))
    v = values[-n:]
    x = iters[-n:].astype(np.float64)
    valid = np.isfinite(v) & np.isfinite(x)
    if valid.sum() < 2:
        return 0.0, 0.0
    v = v[valid]
    x = x[valid]
    slope = float(np.polyfit(x, v, deg=1)[0])
    first = float(v[0])
    last = float(v[-1])
    if abs(first) < 1e-12:
        delta_percent = 0.0
    else:
        delta_percent = (last - first) / abs(first) * 100.0
    return slope, delta_percent


def classify_total_trend(total_slope: float, total_delta_percent: float) -> str:
    if total_slope < -1e-5 and total_delta_percent <= -3.0:
        return "improving"
    if total_slope > 1e-5 and total_delta_percent >= 3.0:
        return "degrading"
    return "stable"


def find_latest_log(base_dir: Path) -> Optional[Path]:
    candidates = list(base_dir.glob("*/*/train.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_log(log_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if not match:
                continue
            row = {
                "iter": int(match.group("iter")),
                "planes": int(match.group("planes")),
                "depth_loss": float(match.group("depth_loss")),
                "normal_l1": float(match.group("normal_l1")),
                "normal_cos": float(match.group("normal_cos")),
                "plane_loss": float(match.group("plane_loss")),
                "total_loss": float(match.group("total_loss")),
            }
            rows.append(row)
    return rows


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, np.nan, dtype=np.float64)
    return np.concatenate([pad, smoothed], axis=0)


def write_csv(rows: List[Dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter",
                "planes",
                "depth_loss",
                "normal_l1",
                "normal_cos",
                "plane_loss",
                "total_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: List[Dict[str, float]], out_path: Path, smooth_window: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iters = np.array([r["iter"] for r in rows], dtype=np.int64)
    planes = np.array([r["planes"] for r in rows], dtype=np.float64)
    depth_loss = np.array([r["depth_loss"] for r in rows], dtype=np.float64)
    normal_l1 = np.array([r["normal_l1"] for r in rows], dtype=np.float64)
    normal_cos = np.array([r["normal_cos"] for r in rows], dtype=np.float64)
    plane_loss = np.array([r["plane_loss"] for r in rows], dtype=np.float64)
    total_loss = np.array([r["total_loss"] for r in rows], dtype=np.float64)

    depth_loss_s = rolling_mean(depth_loss, smooth_window)
    normal_l1_s = rolling_mean(normal_l1, smooth_window)
    normal_cos_s = rolling_mean(normal_cos, smooth_window)
    plane_loss_s = rolling_mean(plane_loss, smooth_window)
    total_loss_s = rolling_mean(total_loss, smooth_window)
    trend_window = max(5, smooth_window * 4)
    total_slope, total_delta = calc_trend(total_loss, iters, trend_window)
    depth_slope, depth_delta = calc_trend(depth_loss, iters, trend_window)
    normal_l1_slope, normal_l1_delta = calc_trend(normal_l1, iters, trend_window)
    normal_cos_slope, normal_cos_delta = calc_trend(normal_cos, iters, trend_window)
    plane_slope, plane_delta = calc_trend(planes, iters, trend_window)
    trend_state = classify_total_trend(total_slope, total_delta)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    axes[0, 0].plot(iters, total_loss, label="total_loss", alpha=0.35)
    axes[0, 0].plot(iters, total_loss_s, label=f"total_loss(ma{smooth_window})", linewidth=2)
    axes[0, 0].set_title(
        f"Total Loss (last {trend_window}: slope={total_slope:.3e}/iter, Δ={total_delta:+.2f}%)"
    )
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(iters, depth_loss, label="depth_loss", alpha=0.35)
    axes[0, 1].plot(iters, depth_loss_s, label=f"depth_loss(ma{smooth_window})", linewidth=2)
    axes[0, 1].plot(iters, plane_loss, label="plane_loss", alpha=0.25)
    axes[0, 1].plot(iters, plane_loss_s, label=f"plane_loss(ma{smooth_window})", linewidth=2)
    axes[0, 1].set_title(
        f"Depth / Plane Loss (depth slope={depth_slope:.3e}, Δ={depth_delta:+.2f}%)"
    )
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(iters, normal_l1, label="normal_l1", alpha=0.35)
    axes[1, 0].plot(iters, normal_l1_s, label=f"normal_l1(ma{smooth_window})", linewidth=2)
    axes[1, 0].plot(iters, normal_cos, label="normal_cos", alpha=0.35)
    axes[1, 0].plot(iters, normal_cos_s, label=f"normal_cos(ma{smooth_window})", linewidth=2)
    axes[1, 0].set_title(
        "Normal Loss "
        f"(l1 slope={normal_l1_slope:.3e}, Δ={normal_l1_delta:+.2f}% | "
        f"cos slope={normal_cos_slope:.3e}, Δ={normal_cos_delta:+.2f}%)"
    )
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(iters, planes, color="tab:green", linewidth=2)
    axes[1, 1].set_title(
        f"Number of Planes (slope={plane_slope:.3e}/iter, Δ={plane_delta:+.2f}%)"
    )
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Plane Count")
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle(
        f"Training Trend: {trend_state.upper()}  "
        f"(total slope={total_slope:.3e}/iter, Δ={total_delta:+.2f}%)",
        fontsize=13,
        fontweight="bold",
    )

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_once(log_path: Path, out_path: Path, csv_path: Path, smooth_window: int) -> int:
    rows = parse_log(log_path)
    if len(rows) == 0:
        print(f"[WARN] No metric rows parsed from: {log_path}")
        print("[WARN] Make sure training is using the updated trainer logger format.")
        return 1

    plot_rows(rows, out_path=out_path, smooth_window=smooth_window)
    write_csv(rows, csv_path=csv_path)

    last = rows[-1]
    iters = np.array([r["iter"] for r in rows], dtype=np.int64)
    total = np.array([r["total_loss"] for r in rows], dtype=np.float64)
    trend_window = max(5, smooth_window * 4)
    total_slope, total_delta = calc_trend(total, iters, trend_window)
    trend_state = classify_total_trend(total_slope, total_delta)
    print(f"[OK] Parsed {len(rows)} rows from {log_path}")
    print(f"[OK] Curve image: {out_path}")
    print(f"[OK] CSV file:    {csv_path}")
    print(
        f"[TREND] window={trend_window} total_slope={total_slope:.3e}/iter "
        f"total_delta={total_delta:+.2f}% state={trend_state}"
    )
    print(
        "[LAST] iter={iter} planes={planes} total={total_loss:.6f} depth={depth_loss:.6f} "
        "normal_l1={normal_l1:.6f} normal_cos={normal_cos:.6f}".format(**last)
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot PlanarSplatting training metrics from train.log")
    parser.add_argument("--log", type=str, default="", help="Path to train.log")
    parser.add_argument("--run_dir", type=str, default="", help="Path to run directory containing train.log")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="planarSplat_ExpRes/expRes",
        help="Base dir used when --log and --run_dir are not provided",
    )
    parser.add_argument("--out", type=str, default="", help="Output curve image path (.png)")
    parser.add_argument("--csv_out", type=str, default="", help="Output CSV path")
    parser.add_argument("--smooth", type=int, default=15, help="Moving average window")
    parser.add_argument("--watch", type=int, default=0, help="Regenerate every N seconds (0 to disable)")
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log).resolve()
    elif args.run_dir:
        log_path = (Path(args.run_dir).resolve() / "train.log")
    else:
        latest = find_latest_log(Path(args.base_dir).resolve())
        if latest is None:
            print(f"[ERR] No train.log found under {Path(args.base_dir).resolve()}")
            return 1
        log_path = latest

    if not log_path.exists():
        print(f"[ERR] Log file not found: {log_path}")
        return 1

    run_dir = log_path.parent
    out_path = Path(args.out).resolve() if args.out else (run_dir / "training_curves.png")
    csv_path = Path(args.csv_out).resolve() if args.csv_out else (run_dir / "training_metrics.csv")

    if args.watch <= 0:
        return run_once(log_path, out_path, csv_path, args.smooth)

    print(f"[INFO] Watching {log_path} every {args.watch}s")
    try:
        while True:
            run_once(log_path, out_path, csv_path, args.smooth)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
