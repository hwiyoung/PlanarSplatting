#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from typing import Optional


def find_latest_run(base_dir: Path) -> Optional[Path]:
    tb_dirs = [p for p in base_dir.rglob("tensorboard") if p.is_dir()]
    if not tb_dirs:
        return None
    latest_tb = max(tb_dirs, key=lambda p: p.stat().st_mtime)
    return latest_tb.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TensorBoard for the latest PlanarSplatting run")
    parser.add_argument("--base_dir", type=str, default="planarSplat_ExpRes")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--run_dir", type=str, default="", help="Optional explicit run dir")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        latest = find_latest_run(Path(args.base_dir).resolve())
        if latest is None:
            print(f"[ERR] No run dir with tensorboard logs found under: {Path(args.base_dir).resolve()}")
            return 1
        run_dir = latest

    logdir = run_dir / "tensorboard"
    if not logdir.exists():
        print(f"[ERR] TensorBoard logdir not found: {logdir}")
        return 1

    cmd = [
        "tensorboard",
        "--logdir",
        str(logdir),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] URL: http://localhost:{args.port}")
    print(f"[INFO] Exec: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
