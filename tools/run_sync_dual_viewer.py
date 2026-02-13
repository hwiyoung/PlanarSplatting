#!/usr/bin/env python3
import argparse
import http.server
import json
import os
import random
import shutil
import socketserver
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _find_latest_run(out_root: Path) -> Optional[Path]:
    run_dirs = [p.parent for p in out_root.rglob("train.log")]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _infer_output_root_from_run(run_dir: Path) -> Optional[Path]:
    parts = run_dir.parts
    if len(parts) < 3:
        return None
    # expected: .../<out_root>/<expname>/<timestamp>
    return run_dir.parent.parent


def _load_data_pth(path: Path) -> Optional[Dict]:
    if torch is None:
        return None
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        return None
    return data


def _resolve_data_pth(args_data_pth: str, run_dir: Path, output_root: Path) -> Path:
    if args_data_pth:
        return Path(args_data_pth).resolve()
    run_local = run_dir / "input_data.pth"
    if run_local.exists():
        return run_local
    return output_root / "data.pth"


def _resolve_pcd_path(args_pcd_path: str, run_dir: Path, output_root: Path) -> Path:
    if args_pcd_path:
        return Path(args_pcd_path).resolve()
    run_local = run_dir / "input_pointcloud.ply"
    if run_local.exists():
        return run_local
    return output_root / "pcd_vggt_downsampled.ply"


def _sample_indices(total: int, max_keep: int, rng: random.Random) -> np.ndarray:
    if total <= max_keep:
        return np.arange(total, dtype=np.int64)
    idx = list(range(total))
    rng.shuffle(idx)
    return np.asarray(idx[:max_keep], dtype=np.int64)


def _load_pointcloud(
    pcd_path: Path,
    max_points: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    if o3d is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.size == 0:
        colors = np.full((points.shape[0], 3), 0.75, dtype=np.float32)

    keep = _sample_indices(points.shape[0], max_points, rng)
    points = points[keep]
    colors = colors[keep]
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return points, colors_u8


def _build_pointcloud_from_data(
    data: Dict,
    stride: int,
    max_points: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    if not all(k in data for k in ("depth", "intrinsics", "extrinsics")):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []
    color_list = data.get("color", None)

    for i, (depth_i, intr_i, c2w_i) in enumerate(zip(data["depth"], data["intrinsics"], data["extrinsics"])):
        depth = np.asarray(depth_i, dtype=np.float32)
        if depth.ndim != 2:
            continue
        h, w = depth.shape
        ys, xs = np.indices((h, w))
        valid = depth > 0.0
        if stride > 1:
            valid = valid & ((ys % stride) == 0) & ((xs % stride) == 0)
        if not np.any(valid):
            continue

        K = np.asarray(intr_i, dtype=np.float32)
        c2w = np.asarray(c2w_i, dtype=np.float32)
        fx = max(float(K[0, 0]), 1e-6)
        fy = max(float(K[1, 1]), 1e-6)
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        z = depth[valid]
        x = (xs[valid].astype(np.float32) - cx) / fx * z
        y = (ys[valid].astype(np.float32) - cy) / fy * z
        pts_cam = np.stack([x, y, z], axis=-1)

        R = c2w[:3, :3]
        t = c2w[:3, 3]
        pts_world = pts_cam @ R.T + t[None, :]
        points_all.append(pts_world.astype(np.float32))

        if color_list is not None and i < len(color_list):
            rgb = np.asarray(color_list[i])
            if rgb.ndim == 3 and rgb.shape[0] == h and rgb.shape[1] == w:
                c = rgb[valid]
                if c.dtype == np.uint8:
                    colors_all.append(c.astype(np.uint8))
                else:
                    colors_all.append(np.clip(c * 255.0, 0, 255).astype(np.uint8))

    if not points_all:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(points_all, axis=0)
    if colors_all:
        colors = np.concatenate(colors_all, axis=0)
        if colors.shape[0] != points.shape[0]:
            colors = np.full((points.shape[0], 3), 192, dtype=np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 192, dtype=np.uint8)

    keep = _sample_indices(points.shape[0], max_points, rng)
    return points[keep], colors[keep]


def _find_mesh_path(run_dir: Path, data: Optional[Dict]) -> Optional[Path]:
    final_mesh_candidates = list(run_dir.glob("*/*_planar_mesh.ply"))
    if final_mesh_candidates:
        return sorted(final_mesh_candidates)[-1]

    plot_mesh_candidates = sorted((run_dir / "plane_plots").glob("planarSplat_*_colorPrim.ply"))
    if plot_mesh_candidates:
        return plot_mesh_candidates[-1]

    if data and "image_paths" in data and len(data["image_paths"]) > 0:
        first_img = Path(str(data["image_paths"][0])).resolve()
        maybe_mono = first_img.parent.parent / "mono_mesh.ply"
        if maybe_mono.exists():
            return maybe_mono
    return None


def _load_mesh(mesh_path: Path, max_faces: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if o3d is None:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
        )
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
        )

    if len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(max_faces)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if colors.size == 0:
        colors = np.full((vertices.shape[0], 3), 0.75, dtype=np.float32)
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return vertices, faces, colors_u8


def _camera_centers(data: Optional[Dict]) -> np.ndarray:
    if data is None or "extrinsics" not in data:
        return np.zeros((0, 3), dtype=np.float32)
    centers = []
    for c2w in data["extrinsics"]:
        c2w_np = np.asarray(c2w, dtype=np.float32)
        if c2w_np.shape == (4, 4):
            centers.append(c2w_np[:3, 3])
    if not centers:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32)


def _compute_scene_bounds(points: np.ndarray, vertices: np.ndarray, cam_centers: np.ndarray) -> Tuple[np.ndarray, float]:
    all_xyz = []
    if points.size > 0:
        all_xyz.append(points)
    if vertices.size > 0:
        all_xyz.append(vertices)
    if cam_centers.size > 0:
        all_xyz.append(cam_centers)
    if not all_xyz:
        return np.zeros(3, dtype=np.float32), 1.0

    xyz = np.concatenate(all_xyz, axis=0)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    radius = float(np.linalg.norm(xyz_max - xyz_min) * 0.5)
    radius = max(radius, 1.0)
    return center.astype(np.float32), radius


def _build_frustum_lines(
    data: Optional[Dict],
    img_h: int,
    img_w: int,
    frustum_depth: float,
    max_cameras: int,
) -> np.ndarray:
    if data is None or "intrinsics" not in data or "extrinsics" not in data:
        return np.zeros((0, 3), dtype=np.float32)

    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]
    n = min(len(intrinsics), len(extrinsics))
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    cam_indices = np.linspace(0, n - 1, num=min(max_cameras, n), dtype=np.int64)
    segments = []
    for idx in cam_indices:
        K = np.asarray(intrinsics[int(idx)], dtype=np.float32)
        c2w = np.asarray(extrinsics[int(idx)], dtype=np.float32)
        if K.shape[0] < 3 or K.shape[1] < 3 or c2w.shape != (4, 4):
            continue

        fx = max(float(K[0, 0]), 1e-6)
        fy = max(float(K[1, 1]), 1e-6)
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        corners_cam = np.array(
            [
                [(0.0 - cx) / fx * frustum_depth, (0.0 - cy) / fy * frustum_depth, frustum_depth],
                [((img_w - 1.0) - cx) / fx * frustum_depth, (0.0 - cy) / fy * frustum_depth, frustum_depth],
                [((img_w - 1.0) - cx) / fx * frustum_depth, ((img_h - 1.0) - cy) / fy * frustum_depth, frustum_depth],
                [(0.0 - cx) / fx * frustum_depth, ((img_h - 1.0) - cy) / fy * frustum_depth, frustum_depth],
            ],
            dtype=np.float32,
        )

        R = c2w[:3, :3]
        t = c2w[:3, 3]
        corners_world = corners_cam @ R.T + t[None, :]
        origin = t.astype(np.float32)

        # pyramid edges
        edges = [
            (origin, corners_world[0]),
            (origin, corners_world[1]),
            (origin, corners_world[2]),
            (origin, corners_world[3]),
            (corners_world[0], corners_world[1]),
            (corners_world[1], corners_world[2]),
            (corners_world[2], corners_world[3]),
            (corners_world[3], corners_world[0]),
        ]
        for a, b in edges:
            segments.append(a)
            segments.append(b)

    if not segments:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32)


def _default_image_size(data: Optional[Dict]) -> Tuple[int, int]:
    if data is None or "color" not in data or not data["color"]:
        return 480, 640
    rgb0 = np.asarray(data["color"][0])
    if rgb0.ndim == 3:
        return int(rgb0.shape[0]), int(rgb0.shape[1])
    return 480, 640


def _prepare_static_assets(target_dir: Path) -> None:
    src_dir = Path(__file__).resolve().parent / "sync_viewer_assets"
    if not src_dir.exists():
        raise FileNotFoundError(f"sync viewer assets dir not found: {src_dir}")

    dst_dir = target_dir / "_viewer_static"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("three.module.js", "OrbitControls.js", "TrackballControls.js"):
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing asset file: {src}")
        dst = dst_dir / name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)


def _write_viewer_files(
    html_path: Path,
    payload: Dict,
) -> Path:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_static_assets(html_path.parent)
    json_path = html_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PlanarSplatting Sync Dual Viewer</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #0f1115;
      color: #e5e7eb;
      font-family: "IBM Plex Sans", "Noto Sans KR", sans-serif;
    }}
    #app {{
      display: grid;
      grid-template-rows: auto 1fr;
      width: 100%;
      height: 100%;
    }}
    #header {{
      display: flex;
      gap: 18px;
      align-items: center;
      padding: 10px 14px;
      border-bottom: 1px solid #1f2430;
      background: #121722;
      font-size: 14px;
    }}
    #views {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      width: 100%;
      height: 100%;
      min-height: 0;
    }}
    .panel {{
      position: relative;
      min-height: 0;
      border-right: 1px solid #1f2430;
    }}
    .panel:last-child {{
      border-right: none;
    }}
    .label {{
      position: absolute;
      left: 10px;
      top: 8px;
      z-index: 10;
      padding: 4px 7px;
      border-radius: 6px;
      background: rgba(8, 10, 14, 0.7);
      border: 1px solid rgba(85, 101, 125, 0.45);
      font-size: 12px;
      letter-spacing: 0.01em;
      pointer-events: none;
    }}
    .hint {{
      opacity: 0.85;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div id="app">
    <div id="header">
      <strong>Sync Dual Viewer</strong>
      <span class="hint">Left: camera + point cloud</span>
      <span class="hint">Right: camera + mesh</span>
      <span class="hint">Rotate/Pan/Zoom one side -> the other side syncs</span>
    </div>
    <div id="views">
      <div id="left" class="panel"><div class="label">Camera + Point Cloud</div></div>
      <div id="right" class="panel"><div class="label">Camera + Mesh</div></div>
    </div>
  </div>

  <script type="module">
    import * as THREE from "./_viewer_static/three.module.js";
    import {{ TrackballControls }} from "./_viewer_static/TrackballControls.js";

    try {{
      const response = await fetch("./{json_path.name}");
      if (!response.ok) {{
        throw new Error(`failed to load JSON: ${{response.status}} ${{response.statusText}}`);
      }}
      const payload = await response.json();

      function flattenVec3(list) {{
        const out = new Float32Array(list.length * 3);
        for (let i = 0; i < list.length; i++) {{
          const v = list[i];
          out[i * 3 + 0] = v[0];
          out[i * 3 + 1] = v[1];
          out[i * 3 + 2] = v[2];
        }}
        return out;
      }}

      function flattenIndices(list) {{
        const out = new Uint32Array(list.length * 3);
        for (let i = 0; i < list.length; i++) {{
          const f = list[i];
          out[i * 3 + 0] = f[0];
          out[i * 3 + 1] = f[1];
          out[i * 3 + 2] = f[2];
        }}
        return out;
      }}

    function makeRenderer(container) {{
      const renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      container.appendChild(renderer.domElement);
      return renderer;
    }}

    function makeBaseScene(container, initCam, controlElement) {{
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0f1115);
      const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, 10000.0);
      camera.position.fromArray(initCam.position);
      camera.up.fromArray(initCam.up);

      const controls = new TrackballControls(camera, controlElement);
      controls.target.fromArray(initCam.target);
      controls.rotateSpeed = 4.0;
      controls.zoomSpeed = 1.5;
      controls.panSpeed = 1.0;
      controls.dynamicDampingFactor = 0.12;
      controls.noRoll = false;
      controls.update();

      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const dir = new THREE.DirectionalLight(0xffffff, 0.75);
      dir.position.set(1.0, 1.2, 1.6);
      scene.add(dir);
      scene.add(new THREE.GridHelper(payload.scene.grid_size, 18, 0x3c465c, 0x273042));
      scene.add(new THREE.AxesHelper(payload.scene.axes_size));
      return {{ scene, camera, controls }};
    }}

    function addPointCloud(scene) {{
      const pos = payload.pointcloud.positions;
      const col = payload.pointcloud.colors;
      if (!pos || pos.length === 0) return;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(flattenVec3(pos), 3));
      const colors = new Float32Array(col.length * 3);
      for (let i = 0; i < col.length; i++) {{
        colors[i * 3 + 0] = col[i][0] / 255.0;
        colors[i * 3 + 1] = col[i][1] / 255.0;
        colors[i * 3 + 2] = col[i][2] / 255.0;
      }}
      g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      const m = new THREE.PointsMaterial({{ size: payload.pointcloud.point_size, vertexColors: true }});
      scene.add(new THREE.Points(g, m));
    }}

    function addMesh(scene) {{
      const v = payload.mesh.vertices;
      const f = payload.mesh.faces;
      if (!v || v.length === 0 || !f || f.length === 0) return;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(flattenVec3(v), 3));
      g.setIndex(new THREE.BufferAttribute(flattenIndices(f), 1));
      const col = payload.mesh.colors;
      if (col && col.length === v.length) {{
        const colors = new Float32Array(col.length * 3);
        for (let i = 0; i < col.length; i++) {{
          colors[i * 3 + 0] = col[i][0] / 255.0;
          colors[i * 3 + 1] = col[i][1] / 255.0;
          colors[i * 3 + 2] = col[i][2] / 255.0;
        }}
        g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      }}
      g.computeVertexNormals();
      const m = new THREE.MeshStandardMaterial({{
        color: 0xc9d2e0,
        side: THREE.DoubleSide,
        roughness: 0.92,
        metalness: 0.02,
        vertexColors: !!g.getAttribute("color"),
        transparent: false
      }});
      scene.add(new THREE.Mesh(g, m));
    }}

    function addFrustums(scene) {{
      const seg = payload.frustums.positions;
      if (!seg || seg.length === 0) return;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(flattenVec3(seg), 3));
      const m = new THREE.LineBasicMaterial({{ color: 0xffc94b, linewidth: 1 }});
      scene.add(new THREE.LineSegments(g, m));
    }}

    function setupPanel(containerId, mode) {{
      const container = document.getElementById(containerId);
      const renderer = makeRenderer(container);
      const {{ scene, camera, controls }} = makeBaseScene(container, payload.default_camera, renderer.domElement);

      if (mode === "pointcloud") addPointCloud(scene);
      if (mode === "mesh") addMesh(scene);
      addFrustums(scene);

      return {{ container, renderer, scene, camera, controls }};
    }}

    const left = setupPanel("left", "pointcloud");
    const right = setupPanel("right", "mesh");

    let isSyncing = false;
    let activePanel = null; // "left" | "right" | null
    let masterPanel = "left";

    function setActivePanel(name) {{
      activePanel = name;
      masterPanel = name;
      left.controls.enabled = (name === "left");
      right.controls.enabled = (name === "right");
    }}

    function clearActivePanel() {{
      activePanel = null;
      left.controls.enabled = true;
      right.controls.enabled = true;
    }}

    function copyPose(src, dst) {{
      isSyncing = true;
      dst.camera.position.copy(src.camera.position);
      dst.camera.quaternion.copy(src.camera.quaternion);
      dst.camera.up.copy(src.camera.up);
      dst.controls.target.copy(src.controls.target);
      dst.camera.updateMatrixWorld(true);
      isSyncing = false;
    }}

    left.controls.addEventListener("start", () => {{
      if (isSyncing) return;
      setActivePanel("left");
    }});
    right.controls.addEventListener("start", () => {{
      if (isSyncing) return;
      setActivePanel("right");
    }});
    left.controls.addEventListener("end", () => {{
      if (isSyncing) return;
      clearActivePanel();
      copyPose(left, right);
    }});
    right.controls.addEventListener("end", () => {{
      if (isSyncing) return;
      clearActivePanel();
      copyPose(right, left);
    }});
    left.controls.addEventListener("change", () => {{
      if (!isSyncing) masterPanel = "left";
    }});
    right.controls.addEventListener("change", () => {{
      if (!isSyncing) masterPanel = "right";
    }});

    function bindInputAsMaster(panel, name) {{
      const el = panel.renderer.domElement;
      const setActive = () => {{
        if (!isSyncing) setActivePanel(name);
      }};
      el.addEventListener("pointerdown", setActive, {{ passive: true }});
      el.addEventListener("mousedown", setActive, {{ passive: true }});
      el.addEventListener("touchstart", setActive, {{ passive: true }});
      el.addEventListener("wheel", setActive, {{ passive: true }});
    }}

    bindInputAsMaster(left, "left");
    bindInputAsMaster(right, "right");
    window.addEventListener("pointerup", () => clearActivePanel(), {{ passive: true }});
    window.addEventListener("mouseup", () => clearActivePanel(), {{ passive: true }});
    window.addEventListener("touchend", () => clearActivePanel(), {{ passive: true }});

    function resizePanel(panel) {{
      const w = panel.container.clientWidth;
      const h = panel.container.clientHeight;
      panel.camera.aspect = w / Math.max(h, 1);
      panel.camera.updateProjectionMatrix();
      panel.renderer.setSize(w, h);
      if (panel.controls.handleResize) panel.controls.handleResize();
    }}
    window.addEventListener("resize", () => {{
      resizePanel(left);
      resizePanel(right);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      if (activePanel === "left") {{
        left.controls.update();
        copyPose(left, right);
      }} else if (activePanel === "right") {{
        right.controls.update();
        copyPose(right, left);
      }} else {{
        if (masterPanel === "left") {{
          left.controls.update();
          copyPose(left, right);
        }} else {{
          right.controls.update();
          copyPose(right, left);
        }}
      }}
      left.renderer.render(left.scene, left.camera);
      right.renderer.render(right.scene, right.camera);
    }}
    animate();
    }} catch (err) {{
      console.error("[sync-viewer] initialization failed:", err);
      const header = document.getElementById("header");
      if (header) {{
        const span = document.createElement("span");
        span.style.color = "#ffb4b4";
        span.textContent = "[sync-viewer] init failed: " + String(err);
        header.appendChild(span);
      }}
    }}
  </script>
  <script nomodule>
    const header = document.getElementById("header");
    if (header) {{
      const span = document.createElement("span");
      span.style.color = "#ffb4b4";
      span.textContent = "[sync-viewer] ES module unsupported browser";
      header.appendChild(span);
    }}
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return json_path


def _serve_directory(root_dir: Path, host: str, port: int) -> None:
    class _TCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    os.chdir(root_dir)
    handler = http.server.SimpleHTTPRequestHandler
    with _TCPServer((host, port), handler) as httpd:
        print(f"[sync-viewer] Serving {root_dir}")
        print(f"[sync-viewer] URL: http://localhost:{port}")
        httpd.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build synced dual 3D viewer (pointcloud vs mesh) and optionally serve it.")
    parser.add_argument("--run_dir", type=str, default="", help="Run directory (.../<exp>/<timestamp>). If omitted, latest run is used.")
    parser.add_argument("--output_root", type=str, default="planarSplat_ExpRes/demo", help="Output root used by run_demo.py")
    parser.add_argument("--data_pth", type=str, default="", help="Explicit data.pth path")
    parser.add_argument("--pcd_path", type=str, default="", help="Explicit point cloud path (.ply)")
    parser.add_argument("--mesh_path", type=str, default="", help="Explicit mesh path (.ply)")
    parser.add_argument(
        "--prefer_saved_pcd",
        action="store_true",
        help="Use saved .ply point cloud first. Default is rebuilding point cloud from data.pth for coordinate consistency.",
    )
    parser.add_argument("--html_path", type=str, default="", help="Target html path (default: <run_dir>/sync_dual_viewer.html)")
    parser.add_argument("--point_stride", type=int, default=20, help="Depth-to-point stride if building from data.pth")
    parser.add_argument("--max_points", type=int, default=200000, help="Maximum point count to visualize")
    parser.add_argument("--max_faces", type=int, default=200000, help="Maximum mesh face count (decimated if larger)")
    parser.add_argument("--max_cameras", type=int, default=96, help="Maximum number of camera frustums")
    parser.add_argument("--frustum_ratio", type=float, default=0.08, help="Frustum depth ratio to scene radius")
    parser.add_argument("--serve", action="store_true", help="Serve html/json with a local HTTP server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if o3d is None:
        print("[sync-viewer][ERR] open3d is not installed. Run this inside the Docker container or install open3d.")
        return 3

    rng = random.Random(args.seed)

    output_root = Path(args.output_root).resolve()
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = _find_latest_run(output_root)
        if run_dir is None:
            print(f"[sync-viewer][ERR] No run found under: {output_root}")
            return 1
    if not run_dir.exists():
        print(f"[sync-viewer][ERR] run_dir does not exist: {run_dir}")
        return 1

    inferred_output_root = _infer_output_root_from_run(run_dir)
    if inferred_output_root is None:
        inferred_output_root = output_root

    data_pth = _resolve_data_pth(args.data_pth, run_dir, inferred_output_root)
    data = _load_data_pth(data_pth)

    pcd_path = _resolve_pcd_path(args.pcd_path, run_dir, inferred_output_root)
    point_source = "data_pth_reprojection"
    if args.prefer_saved_pcd and pcd_path.exists():
        points, pcd_colors = _load_pointcloud(pcd_path, args.max_points, rng)
        point_source = "saved_pcd"
    elif data is not None:
        points, pcd_colors = _build_pointcloud_from_data(data, args.point_stride, args.max_points, rng)
    elif pcd_path.exists():
        points, pcd_colors = _load_pointcloud(pcd_path, args.max_points, rng)
        point_source = "saved_pcd_fallback"
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        pcd_colors = np.zeros((0, 3), dtype=np.uint8)
        point_source = "none"

    if args.mesh_path:
        mesh_path = Path(args.mesh_path).resolve()
    else:
        mesh_path = _find_mesh_path(run_dir, data)
    if mesh_path is None or not mesh_path.exists():
        print("[sync-viewer][ERR] No mesh file found. Use --mesh_path explicitly.")
        return 2
    vertices, faces, mesh_colors = _load_mesh(mesh_path, args.max_faces)

    cam_centers = _camera_centers(data)
    center, radius = _compute_scene_bounds(points, vertices, cam_centers)
    img_h, img_w = _default_image_size(data)
    frustum_depth = max(radius * args.frustum_ratio, 0.05)
    frustum_segments = _build_frustum_lines(
        data=data,
        img_h=img_h,
        img_w=img_w,
        frustum_depth=frustum_depth,
        max_cameras=args.max_cameras,
    )

    default_cam_pos = (center + np.array([radius * 1.8, radius * 1.2, radius * 1.8], dtype=np.float32)).tolist()
    payload = {
        "scene": {
            "grid_size": float(radius * 4.0),
            "axes_size": float(radius * 0.4),
        },
        "default_camera": {
            "position": default_cam_pos,
            "target": center.tolist(),
            "up": [0.0, 1.0, 0.0],
        },
        "pointcloud": {
            "positions": points.tolist(),
            "colors": pcd_colors.tolist(),
            "point_size": max(0.004 * radius, 0.002),
        },
        "mesh": {
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "colors": mesh_colors.tolist(),
        },
        "frustums": {
            "positions": frustum_segments.tolist(),
        },
        "meta": {
            "run_dir": str(run_dir),
            "output_root": str(inferred_output_root),
            "mesh_path": str(mesh_path),
            "data_pth": str(data_pth) if data_pth.exists() else "",
            "pcd_path": str(pcd_path) if pcd_path.exists() else "",
            "point_source": point_source,
            "point_count": int(points.shape[0]),
            "mesh_vertex_count": int(vertices.shape[0]),
            "mesh_face_count": int(faces.shape[0]),
            "frustum_segment_count": int(frustum_segments.shape[0] // 2),
        },
    }

    html_path = Path(args.html_path).resolve() if args.html_path else (run_dir / "sync_dual_viewer.html")
    json_path = _write_viewer_files(html_path, payload)

    print(f"[sync-viewer] run_dir: {run_dir}")
    print(f"[sync-viewer] mesh: {mesh_path}")
    print(f"[sync-viewer] data_pth: {data_pth if data_pth.exists() else '(missing)'}")
    print(f"[sync-viewer] pcd_path: {pcd_path if pcd_path.exists() else '(missing)'}")
    print(f"[sync-viewer] point source: {point_source}")
    print(f"[sync-viewer] html: {html_path}")
    print(f"[sync-viewer] json: {json_path}")
    print(
        "[sync-viewer] counts:",
        f"points={payload['meta']['point_count']},",
        f"mesh_v={payload['meta']['mesh_vertex_count']},",
        f"mesh_f={payload['meta']['mesh_face_count']},",
        f"frustum_lines={payload['meta']['frustum_segment_count']}",
    )

    if args.serve:
        _serve_directory(html_path.parent, args.host, args.port)
    else:
        print("[sync-viewer] To view in browser, run with --serve or host the html/json folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
