"""Generate semantic segmentation maps using Grounded SAM 2.

Hybrid approach (recommended):
  1. Grounded SAM detects "building" and "ground/road" regions
  2. Depth-derived normals + DJI gimbal pitch split surfaces by orientation:
     - Smoothed MVS depth → 3D points → finite-diff normals (camera frame)
     - DJI IMU gimbal pitch → gravity direction in camera frame
     - |dot(normal_cam, gravity_up_cam)| measures surface horizontality
     - > threshold → horizontal (roof/ground), <= threshold → vertical (wall)
  3. Two-threshold system (confident labels only):
     - |dot| > 0.85 → strongly horizontal → roof or ground (decided by height)
     - |dot| <= 0.3 → strongly vertical (wall)
     - Ambiguous (0.3-0.85) → background (no label, deferred to L_mutual)
  4. Height-based roof/ground separation:
     - Global ground level from dataset-wide flat pixel world Y
     - Flat + near ground level → ground (corrects SAM zone misclassification)
     - Flat + elevated → roof
  5. Building zone without valid depth → neighbor propagation:
     - Copies majority class from nearby labeled pixels
     - Correctly handles both nadir (rooftop) and oblique (facade) views
     - Fallback: pitch-dependent default if no neighbors available
  6. Pixels not in any SAM zone → background (ignored in training)

Why depth-derived normals instead of MVS PatchMatch normals:
  MVS PatchMatch normals degenerate to camera-facing direction on textureless
  facades, making all building pixels look similar regardless of actual surface
  orientation. MVS depth is geometrically verified and reliable — deriving normals
  from smoothed depth correctly captures large-scale surface orientation.

Gravity source: DJI EXIF gimbal pitch (--raw_image_dir for raw DJI JPGs).
Fallback: text-only mode for images without DJI EXIF or depth.

Class mapping:
    0 = background (unlabeled)
    1 = building roof
    2 = building wall / facade
    3 = ground / road

Usage:
    # Hybrid mode (recommended): depth-derived normals + DJI pitch
    python scripts/generate_segmentation.py \
        --image_dir user_inputs/testset/0_25x/dense/images \
        --output_dir user_inputs/testset/0_25x/seg_maps \
        --vis_dir user_inputs/testset/0_25x/seg_vis \
        --input_data planarSplat_ExpRes/seongsu_phase1_mvsnormal/input_data.pth \
        --raw_image_dir user_inputs/testset/raw/images

    # Text-only mode (no normal refinement, noisier roof/wall)
    python scripts/generate_segmentation.py \
        --image_dir user_inputs/testset/0_25x/dense/images \
        --output_dir user_inputs/testset/0_25x/seg_maps \
        --vis_dir user_inputs/testset/0_25x/seg_vis
"""

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

# ── Class colors for visualization ─────────────────────────────
CLASS_COLORS = {
    0: (0, 0, 0),        # background
    1: (255, 80, 80),    # roof - red
    2: (80, 80, 255),    # wall - blue
    3: (0, 220, 0),      # ground - green
}

# ── Grounded SAM class definitions ─────────────────────────────
# Hybrid mode: detect "building" + "ground", then split building by normals
HYBRID_CLASSES = [
    {
        "id": "building",  # temporary label, will be split
        "prompt": "building. building facade. building roof.",
        "box_thresh": 0.20,
    },
    {
        "id": "ground",
        "prompt": "road. ground. street. pavement. sidewalk.",
        "box_thresh": 0.20,
    },
]

# Text-only mode (fallback): detect each class independently
TEXTONLY_CLASSES = [
    {"id": 3, "name": "ground",
     "prompt": "road. ground. street. pavement. sidewalk.",
     "box_thresh": 0.20},
    {"id": 1, "name": "building roof",
     "prompt": "building roof. rooftop. building top.",
     "box_thresh": 0.20},
    {"id": 2, "name": "building wall",
     "prompt": "building wall. building facade. building side.",
     "box_thresh": 0.20},
]


def load_models(device: str, gdino_id: str, sam2_id: str):
    """Load Grounding DINO and SAM 2 models."""
    print(f"Loading Grounding DINO: {gdino_id}")
    gdino_processor = AutoProcessor.from_pretrained(gdino_id)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gdino_id
    ).to(device)
    gdino_model.eval()

    print(f"Loading SAM 2: {sam2_id}")
    sam2_processor = Sam2Processor.from_pretrained(sam2_id)
    sam2_model = Sam2Model.from_pretrained(sam2_id).to(device)
    sam2_model.eval()

    return gdino_processor, gdino_model, sam2_processor, sam2_model


def detect_masks(
    image: Image.Image,
    text_prompt: str,
    gdino_processor, gdino_model,
    sam2_processor, sam2_model,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple:
    """Run Grounding DINO + SAM 2 for one text prompt.

    Returns:
        masks: (N, H, W) bool array
        scores: (N,) float array
    """
    w, h = image.size

    inputs = gdino_processor(
        images=image, text=text_prompt, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )

    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()

    if len(boxes) == 0:
        return np.zeros((0, h, w), dtype=bool), np.array([])

    input_boxes = [boxes.tolist()]
    sam2_inputs = sam2_processor(
        images=image, input_boxes=input_boxes, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        sam2_outputs = sam2_model(**sam2_inputs, multimask_output=False)

    low_masks = sam2_outputs.pred_masks[0, :, 0]
    high_masks = torch.nn.functional.interpolate(
        low_masks.unsqueeze(0).float().cpu(), size=(h, w),
        mode="bilinear", align_corners=False,
    )[0]
    masks = (high_masks > 0.0).numpy()

    return masks, scores



def read_dji_gimbal_pitch(image_path: str) -> float | None:
    """Read DJI gimbal pitch from EXIF XMP metadata.

    Args:
        image_path: Path to raw DJI JPG image.

    Returns:
        Gimbal pitch in degrees (negative = looking down), or None if unavailable.
    """
    try:
        with open(image_path, "rb") as f:
            # Read first 64KB where XMP is typically stored
            header = f.read(65536)
        # Search for DJI XMP tag
        text = header.decode("latin-1")
        match = re.search(r'drone-dji:GimbalPitchDegree="([^"]+)"', text)
        if match:
            return float(match.group(1))
        # Try alternate format (element style)
        match = re.search(r'<drone-dji:GimbalPitchDegree>([^<]+)</drone-dji:GimbalPitchDegree>', text)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def compute_gravity_up_cam(pitch_deg: float) -> np.ndarray:
    """Compute gravity-up direction in camera frame from DJI gimbal pitch.

    In OpenCV camera convention (X=right, Y=down, Z=forward):
    - pitch=0° (level): gravity-up = [0, -1, 0] (camera -Y)
    - pitch=-90° (nadir): gravity-up = [0, 0, -1] (camera -Z)
    - pitch=-45° (oblique): gravity-up = [0, -0.707, -0.707]

    Derivation: Camera pitches down by theta from level (DJI pitch = -theta).
    R_w2c @ gravity_up_level = [0, -cos(theta), -sin(theta)]
    Substituting theta = -pitch_deg: [0, -cos(pitch), sin(pitch)]
    """
    pitch_rad = math.radians(pitch_deg)
    return np.array([0.0, -math.cos(pitch_rad), math.sin(pitch_rad)])


def compute_depth_normals(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    sigma: float = 3.0,
) -> tuple:
    """Compute surface normals from smoothed depth map.

    MVS PatchMatch normals degenerate on textureless facades (camera-facing bias).
    Depth is geometrically verified and reliable — deriving normals from smoothed
    depth correctly captures large-scale surface orientation.

    Args:
        depth: (H, W) float32 depth map (0 = invalid).
        intrinsics: (3, 3) camera intrinsic matrix.
        sigma: Gaussian smoothing sigma for depth (larger = smoother normals).

    Returns:
        normal_cam: (3, H, W) float32 unit normals in camera frame, [-1, 1].
        valid: (H, W) bool mask of valid normals.
        d_smooth: (H, W) float64 smoothed depth map.
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # NaN-aware Gaussian smoothing of depth
    d = depth.copy().astype(np.float64)
    mask = d > 0
    d[~mask] = 0.0
    d_smooth = gaussian_filter(d, sigma=sigma)
    w_smooth = gaussian_filter(mask.astype(np.float64), sigma=sigma)
    d_smooth = np.where(w_smooth > 0.1, d_smooth / w_smooth, 0.0)

    # Unproject to 3D points in camera frame
    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                        np.arange(H, dtype=np.float64))
    X = (u - cx) * d_smooth / fx
    Y = (v - cy) * d_smooth / fy
    Z = d_smooth

    # Surface normal = cross(dP/du, dP/dv)
    dXdu, dXdv = np.gradient(X, axis=1), np.gradient(X, axis=0)
    dYdu, dYdv = np.gradient(Y, axis=1), np.gradient(Y, axis=0)
    dZdu, dZdv = np.gradient(Z, axis=1), np.gradient(Z, axis=0)

    nx = dYdu * dZdv - dZdu * dYdv
    ny = dZdu * dXdv - dXdu * dZdv
    nz = dXdu * dYdv - dYdu * dXdv

    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    valid = (d_smooth > 0) & (mag > 1e-8)

    normal_cam = np.zeros((3, H, W), dtype=np.float32)
    normal_cam[0] = np.where(valid, nx / (mag + 1e-8), 0).astype(np.float32)
    normal_cam[1] = np.where(valid, ny / (mag + 1e-8), 0).astype(np.float32)
    normal_cam[2] = np.where(valid, nz / (mag + 1e-8), 0).astype(np.float32)

    return normal_cam, valid, d_smooth


def unproject_to_world_Y(
    depth_smooth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Unproject masked pixels to world Y coordinate (elevation).

    In COLMAP world frame, gravity ≈ -Y, so lower Y = higher elevation.

    Returns:
        world_Y_map: (H, W) float32, NaN where mask is False.
    """
    H, W = depth_smooth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                        np.arange(H, dtype=np.float64))
    Xc = np.where(mask, (u - cx) * depth_smooth / fx, 0)
    Yc = np.where(mask, (v - cy) * depth_smooth / fy, 0)
    Zc = np.where(mask, depth_smooth, 0)

    # P_world = c2w @ P_cam; extract Y row
    world_Y = c2w[1, 0] * Xc + c2w[1, 1] * Yc + c2w[1, 2] * Zc + c2w[1, 3]
    world_Y_map = np.where(mask, world_Y, np.nan).astype(np.float32)
    return world_Y_map


def _propagate_from_neighbors(
    seg_map: np.ndarray,
    unlabeled: np.ndarray,
    pitch_deg: float,
    radius: int = 15,
) -> np.ndarray:
    """Fill unlabeled building pixels from nearby labeled pixels.

    For each unlabeled pixel, find the majority class among labeled pixels
    within a square window of given radius. Falls back to pitch-based default
    if no labeled neighbors exist.

    Uses uniform_filter on per-class indicator maps for efficient computation.
    """
    from scipy.ndimage import uniform_filter

    seg = seg_map.copy()
    size = 2 * radius + 1

    # Count labeled neighbors for each class
    labeled = seg > 0
    counts = {}
    for c in [1, 2, 3]:
        indicator = ((seg == c) & labeled).astype(np.float32)
        counts[c] = uniform_filter(indicator, size=size, mode='constant')

    # For each unlabeled pixel, pick the majority class
    best_class = np.zeros_like(seg)
    best_count = np.zeros(seg.shape, dtype=np.float32)
    for c in [1, 2, 3]:
        better = counts[c] > best_count
        best_class[better] = c
        best_count[better] = counts[c][better]

    # Apply: propagate where we have neighbors, otherwise pitch-based fallback
    has_neighbors = best_count > 0
    seg[unlabeled & has_neighbors] = best_class[unlabeled & has_neighbors]

    # Fallback for isolated pixels with no labeled neighbors
    still_unlabeled = unlabeled & ~has_neighbors
    if still_unlabeled.any():
        seg[still_unlabeled] = 1 if pitch_deg <= -60.0 else 2

    return seg


def process_image_hybrid(
    image_path: str,
    normal_cam: np.ndarray,
    valid_normal: np.ndarray,
    gravity_up_cam: np.ndarray,
    pitch_deg: float,
    depth_smooth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    ground_Y_ref: float,
    ground_Y_tol: float,
    gdino_processor, gdino_model,
    sam2_processor, sam2_model,
    device: str,
    text_threshold: float,
    horiz_thresh: float = 0.85,
    wall_thresh: float = 0.3,
) -> np.ndarray:
    """Hybrid: Grounded SAM + normal classification + height-based roof/ground.

    Classification pipeline:
    1. Grounded SAM detects building/ground zones
    2. Depth-derived normals classify horizontal/vertical/ambiguous
    3. Height (world Y) separates roof from ground among horizontal surfaces:
       - Flat + near ground level → ground (corrects SAM zone misclassification)
       - Flat + elevated above ground → roof
    4. Building zone without valid normals → neighbor propagation:
       - Copies majority class from nearby labeled pixels (radius=15)
       - Crosswalk no-depth pixels → neighbors are ground → ground
       - Rooftop no-depth pixels → neighbors are roof → roof
       - Facade no-depth pixels → neighbors are wall → wall
       - Fallback if no neighbors: oblique → wall, near-nadir → roof

    Two-threshold system with uncertain-as-unlabeled:
    - |dot| > horiz_thresh (0.85): strongly horizontal → roof or ground (by height)
    - |dot| <= wall_thresh (0.3): strongly vertical → wall
    - Ambiguous (0.3 < |dot| ≤ 0.85): left as background (no L_sem supervision)
      → avoids multi-view consistent wrong labels; defers to L_mutual geometric prior
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # Step 1: Detect building and ground masks with scores
    building_mask = np.zeros((h, w), dtype=bool)
    ground_mask = np.zeros((h, w), dtype=bool)
    building_scores = np.zeros((h, w), dtype=np.float32)
    ground_scores = np.zeros((h, w), dtype=np.float32)

    for cls in HYBRID_CLASSES:
        masks, scores = detect_masks(
            image, cls["prompt"],
            gdino_processor, gdino_model,
            sam2_processor, sam2_model,
            device, cls["box_thresh"], text_threshold,
        )
        if len(scores) == 0:
            continue
        if cls["id"] == "building":
            for i in range(len(scores)):
                update = masks[i] & (scores[i] > building_scores)
                building_mask[update] = True
                building_scores[update] = scores[i]
        else:
            for i in range(len(scores)):
                update = masks[i] & (scores[i] > ground_scores)
                ground_mask[update] = True
                ground_scores[update] = scores[i]

    # Step 2: Compute dot(normal_cam, gravity_up_cam) directly in camera frame
    nh, nw = normal_cam.shape[1], normal_cam.shape[2]

    dot_up_map = np.abs(
        normal_cam[0] * gravity_up_cam[0]
        + normal_cam[1] * gravity_up_cam[1]
        + normal_cam[2] * gravity_up_cam[2]
    )
    valid_map = valid_normal

    # Resize masks and scores to normal resolution if needed
    if (h, w) != (nh, nw):
        building_r = np.array(Image.fromarray(building_mask.astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST)).astype(bool)
        ground_r = np.array(Image.fromarray(ground_mask.astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST)).astype(bool)
        building_sc = np.array(Image.fromarray(building_scores).resize(
            (nw, nh), Image.NEAREST))
        ground_sc = np.array(Image.fromarray(ground_scores).resize(
            (nw, nh), Image.NEAREST))
    else:
        building_r = building_mask
        ground_r = ground_mask
        building_sc = building_scores
        ground_sc = ground_scores

    # Step 3: Compute world Y (elevation) for height-based classification
    world_Y_map = unproject_to_world_Y(depth_smooth, intrinsics, c2w,
                                        mask=(depth_smooth > 0))
    # Resize to normal resolution if needed
    if (h, w) != (nh, nw):
        world_Y_map = np.array(Image.fromarray(world_Y_map).resize(
            (nw, nh), Image.NEAREST))

    is_at_ground_level = ~np.isnan(world_Y_map) & (world_Y_map > ground_Y_ref - ground_Y_tol)

    # Step 4: Compose seg map at normal resolution
    seg_map = np.zeros((nh, nw), dtype=np.uint8)

    # Zone classification with score-based overlap resolution
    overlap_raw = building_r & ground_r
    bld_wins = overlap_raw & (building_sc > ground_sc)
    gnd_wins = overlap_raw & ~bld_wins
    only_building = (building_r & ~ground_r) | bld_wins
    only_ground = (ground_r & ~building_r) | gnd_wins

    # Normal orientation classification
    is_strong_horiz = valid_map & (dot_up_map > horiz_thresh)
    is_strong_wall = valid_map & (dot_up_map <= wall_thresh)
    is_ambiguous = valid_map & ~is_strong_horiz & ~is_strong_wall

    # Ground-only zones: trust SAM, override only for strongly vertical normals
    seg_map[only_ground & is_strong_wall] = 2         # strong wall overrides SAM
    seg_map[only_ground & ~is_strong_wall] = 3        # everything else → ground

    # Building zones: normal + height-based classification
    # Strong horizontal: use height to decide roof vs ground
    bld_horiz = only_building & is_strong_horiz
    seg_map[bld_horiz & is_at_ground_level] = 3       # flat + ground level → ground
    seg_map[bld_horiz & ~is_at_ground_level] = 1      # flat + elevated → roof
    # Strong vertical: wall regardless of height
    seg_map[only_building & is_strong_wall] = 2       # strongly vertical → wall
    # Ambiguous normals (0.3 < |dot| ≤ 0.85): leave as background (no label)
    # Rationale: forcing wall/roof here creates multi-view consistent wrong labels
    # that L_sem reinforces and L_mutual can't correct. Instead, defer to L_mutual's
    # geometric prior to determine the class during training optimization.
    # (is_ambiguous pixels in building zone stay seg_map=0 → masked from L_sem)

    # Building zone without valid normal → propagate from nearby labeled pixels
    # This fills depth-gap regions (textureless facades, MVS failures) from neighbors,
    # but does NOT fill ambiguous-normal pixels (which are intentionally unlabeled).
    no_label_no_normal = (seg_map == 0) & only_building & ~valid_map
    if no_label_no_normal.any():
        seg_map = _propagate_from_neighbors(
            seg_map, no_label_no_normal, pitch_deg, radius=15)

    # Resize back to original image resolution
    if (h, w) != (nh, nw):
        seg_map = np.array(Image.fromarray(seg_map).resize((w, h), Image.NEAREST))

    return seg_map


def process_image_textonly(
    image_path: str,
    gdino_processor, gdino_model,
    sam2_processor, sam2_model,
    device: str,
    text_threshold: float,
) -> np.ndarray:
    """Text-only mode: per-class detection with score-based overlap resolution."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    seg_map = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.float32)

    for cls in TEXTONLY_CLASSES:
        masks, scores = detect_masks(
            image, cls["prompt"],
            gdino_processor, gdino_model,
            sam2_processor, sam2_model,
            device, cls["box_thresh"], text_threshold,
        )
        if len(scores) == 0:
            continue
        for i in range(len(scores)):
            update = masks[i] & (scores[i] > score_map)
            seg_map[update] = cls["id"]
            score_map[update] = scores[i]

    return seg_map


def create_vis(image_path: str, seg_map: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Create visualization overlay."""
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image, dtype=np.float32)

    overlay = np.zeros_like(img_array)
    for cid, color in CLASS_COLORS.items():
        mask = seg_map == cid
        if mask.any():
            overlay[mask] = color

    fg_mask = seg_map > 0
    blended = img_array.copy()
    blended[fg_mask] = (1 - alpha) * img_array[fg_mask] + alpha * overlay[fg_mask]

    vis = Image.fromarray(blended.astype(np.uint8))

    draw = ImageDraw.Draw(vis)
    legend = [("Roof (1)", CLASS_COLORS[1]),
              ("Wall (2)", CLASS_COLORS[2]),
              ("Ground (3)", CLASS_COLORS[3])]
    y = 10
    for text, color in legend:
        draw.rectangle([10, y, 30, y + 20], fill=color, outline=(255, 255, 255))
        draw.text((35, y + 2), text, fill=(255, 255, 255))
        y += 25

    return vis


def compute_stats(seg_map: np.ndarray) -> dict:
    total = seg_map.size
    stats = {}
    for cid, name in [(1, "building roof"), (2, "building wall"), (3, "ground")]:
        count = int((seg_map == cid).sum())
        stats[name] = {"pixels": count, "ratio": count / total}
    stats["background"] = {"pixels": int((seg_map == 0).sum()),
                           "ratio": (seg_map == 0).sum() / total}
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate segmentation maps using Grounded SAM 2")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vis_dir", default=None)
    parser.add_argument("--input_data", default=None,
                        help="Path to input_data.pth for hybrid mode (image_paths mapping)")
    parser.add_argument("--raw_image_dir", default=None,
                        help="Path to raw DJI images for EXIF gimbal pitch extraction")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gdino_model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2_model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--text_threshold", type=float, default=0.25)
    args = parser.parse_args()

    # Collect images
    image_dir = Path(args.image_dir)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in img_exts
    )
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    print(f"Found {len(image_paths)} images in {image_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Load DJI gimbal pitch from raw images ──
    pitch_lookup = {}
    if args.raw_image_dir:
        raw_dir = Path(args.raw_image_dir)
        raw_exts = {".jpg", ".jpeg", ".dng"}
        raw_files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in raw_exts)
        for rp in raw_files:
            pitch = read_dji_gimbal_pitch(str(rp))
            if pitch is not None:
                pitch_lookup[rp.stem] = pitch
        pitches = list(pitch_lookup.values())
        if pitches:
            print(f"  Loaded DJI pitch for {len(pitch_lookup)} images: "
                  f"mean={np.mean(pitches):.1f}° range=[{np.min(pitches):.1f}°, {np.max(pitches):.1f}°]")
        else:
            print(f"  WARNING: No DJI pitch found in {raw_dir}")

    # ── Load depth + intrinsics + extrinsics for hybrid mode ──
    hybrid_mode = False
    depth_lookup = {}
    intrinsics_lookup = {}
    c2w_lookup = {}

    if args.input_data and Path(args.input_data).exists():
        print(f"Loading depth + intrinsics + extrinsics from {args.input_data} (hybrid mode)")
        data = torch.load(args.input_data, map_location="cpu")

        for idx in range(len(data["image_paths"])):
            stem = os.path.splitext(os.path.basename(data["image_paths"][idx]))[0]
            d = data["depth"][idx]
            k = data["intrinsics"][idx]
            e = data["extrinsics"][idx]
            if isinstance(d, torch.Tensor):
                d = d.numpy()
            if isinstance(k, torch.Tensor):
                k = k.numpy()
            if isinstance(e, torch.Tensor):
                e = e.numpy()
            depth_lookup[stem] = d
            intrinsics_lookup[stem] = k
            c2w_lookup[stem] = e
        print(f"  Loaded depth for {len(depth_lookup)} images")

        hybrid_mode = True

    # ── Pre-compute global ground level from all flat pixels ──
    ground_Y_ref = 0.0
    ground_Y_tol = 0.15  # pixels within this tolerance of ground level → ground

    if hybrid_mode and pitch_lookup:
        print("  Computing global ground level from dataset...")
        all_flat_Y = []
        for stem in depth_lookup:
            if stem not in pitch_lookup:
                continue
            d = depth_lookup[stem]
            k = intrinsics_lookup[stem]
            c2w = c2w_lookup[stem]
            pitch = pitch_lookup[stem]

            gravity_up = compute_gravity_up_cam(pitch)
            normal_cam, valid_n, d_smooth = compute_depth_normals(d, k, sigma=3.0)

            dot_abs = np.abs(
                normal_cam[0] * gravity_up[0]
                + normal_cam[1] * gravity_up[1]
                + normal_cam[2] * gravity_up[2]
            )
            is_flat = valid_n & (dot_abs > 0.85)

            if is_flat.sum() < 100:
                continue

            world_Y_map = unproject_to_world_Y(d_smooth, k, c2w,
                                                mask=(d_smooth > 0))
            flat_Y = world_Y_map[is_flat]
            flat_Y = flat_Y[~np.isnan(flat_Y)]

            # Subsample for memory
            if len(flat_Y) > 5000:
                flat_Y = np.random.choice(flat_Y, 5000, replace=False)
            all_flat_Y.extend(flat_Y.tolist())

        if all_flat_Y:
            all_flat_Y = np.array(all_flat_Y)
            # Ground is the most common flat surface (roads) → high Y values
            # Use 75th percentile as ground level reference
            ground_Y_ref = float(np.percentile(all_flat_Y, 75))
            print(f"  Global ground level: Y_ref={ground_Y_ref:.3f} "
                  f"(tol={ground_Y_tol}, median={np.median(all_flat_Y):.3f}, "
                  f"75pct={np.percentile(all_flat_Y, 75):.3f}, "
                  f"95pct={np.percentile(all_flat_Y, 95):.3f})")
        else:
            print("  WARNING: No flat pixels found, using ground_Y_ref=0.0")

    # Load models
    gdino_proc, gdino_model, sam2_proc, sam2_model = load_models(
        args.device, args.gdino_model, args.sam2_model
    )

    # Process
    all_stats = []
    n_hybrid = 0
    n_textonly = 0

    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] {img_path.name}", end=" ... ")

        stem = img_path.stem
        has_depth = hybrid_mode and stem in depth_lookup
        has_pitch = stem in pitch_lookup

        if has_depth and has_pitch:
            # Compute depth-derived normals (smoothed depth → 3D → finite diff)
            normal_cam, valid_normal, depth_smooth = compute_depth_normals(
                depth_lookup[stem], intrinsics_lookup[stem], sigma=3.0)
            gravity_up_cam = compute_gravity_up_cam(pitch_lookup[stem])
            seg_map = process_image_hybrid(
                str(img_path),
                normal_cam, valid_normal, gravity_up_cam,
                pitch_lookup[stem],
                depth_smooth, intrinsics_lookup[stem], c2w_lookup[stem],
                ground_Y_ref, ground_Y_tol,
                gdino_proc, gdino_model, sam2_proc, sam2_model,
                args.device, args.text_threshold,
            )
            n_hybrid += 1
            mode_str = f"hybrid(pitch={pitch_lookup[stem]:.1f}°)"
        else:
            seg_map = process_image_textonly(
                str(img_path),
                gdino_proc, gdino_model, sam2_proc, sam2_model,
                args.device, args.text_threshold,
            )
            n_textonly += 1
            reason = []
            if not has_depth:
                reason.append("no depth")
            if not has_pitch:
                reason.append("no pitch")
            mode_str = f"text({','.join(reason)})"

        seg_img = Image.fromarray(seg_map, mode="L")
        seg_img.save(output_dir / f"{stem}.png")

        if vis_dir:
            vis = create_vis(str(img_path), seg_map)
            vis.save(vis_dir / f"{stem}.png")

        stats = compute_stats(seg_map)
        all_stats.append(stats)

        cov = 1.0 - stats["background"]["ratio"]
        print(f"[{mode_str}] cov={cov:.1%}  "
              f"roof={stats['building roof']['ratio']:.1%}  "
              f"wall={stats['building wall']['ratio']:.1%}  "
              f"ground={stats['ground']['ratio']:.1%}")

    # Summary
    if all_stats:
        print(f"\n=== Summary ({n_hybrid} hybrid, {n_textonly} text-only) ===")
        for name in ["building roof", "building wall", "ground", "background"]:
            ratios = [s[name]["ratio"] for s in all_stats]
            print(f"  {name:15s}: mean={np.mean(ratios):.1%}  "
                  f"min={np.min(ratios):.1%}  max={np.max(ratios):.1%}")
        covs = [1.0 - s["background"]["ratio"] for s in all_stats]
        print(f"  {'coverage':15s}: mean={np.mean(covs):.1%}  "
              f"min={np.min(covs):.1%}  max={np.max(covs):.1%}")


if __name__ == "__main__":
    main()
