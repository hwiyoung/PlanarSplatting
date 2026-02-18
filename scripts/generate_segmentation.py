"""Generate semantic segmentation maps using Grounded SAM 2.

MVS Hybrid approach (recommended):
  1. Grounded SAM detects "building" and "ground/road" regions
  2. Normal + DJI gimbal pitch splits surfaces by orientation:
     - All computation in camera frame (no COLMAP world frame needed)
     - DJI IMU gimbal pitch → gravity direction in camera frame
     - |dot(normal_cam, gravity_up_cam)| = sin(angle_from_vertical)
     - > threshold → horizontal (roof/ground), <= threshold → vertical (wall)
  3. Normal overrides SAM in ALL zones:
     - Building-only + horizontal → roof; + vertical → wall
     - Ground-only + horizontal → ground; + vertical → wall (SAM override)
     - Overlap + horizontal → ground; + vertical → wall
  4. Pixels without valid normals → background (ignored in training)

Normal source: Normals from input_data.pth (processed by colmap_to_ps.py, verified correct).
Both MVS and Metric3D normals are stored in (3,H,W) [0,1] format in input_data.pth.

Gravity source: DJI EXIF gimbal pitch (--raw_image_dir for raw DJI JPGs).
Fallback: text-only mode for images without DJI EXIF or normals.

Class mapping:
    0 = background (unlabeled)
    1 = building roof
    2 = building wall / facade
    3 = ground / road

Usage:
    # MVS Hybrid mode (recommended): input_data.pth normals + DJI pitch
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


def process_image_hybrid(
    image_path: str,
    normal_cam: np.ndarray,
    gravity_up_cam: np.ndarray,
    gdino_processor, gdino_model,
    sam2_processor, sam2_model,
    device: str,
    text_threshold: float,
    horiz_thresh: float = 0.85,
    wall_thresh: float = 0.3,
) -> np.ndarray:
    """Hybrid: Grounded SAM (building/ground) + two-threshold normal classification.

    All computation in camera frame. Normals from input_data.pth ([0,1] encoded).
    DJI pitch gives gravity direction in camera frame.

    Two-threshold system handles degenerate MVS normals on facades
    (which have |dot| ≈ |sin(pitch)|, falling in the ambiguous zone):
    - |dot| > horiz_thresh (0.85): strongly horizontal → roof/ground
    - |dot| <= wall_thresh (0.3): strongly vertical → wall
    - Ambiguous: zone-dependent default
      - Building zone → wall (oblique view: visible building surface = facade)
      - Ground zone → ground (trust SAM)
      - Overlap zone → ground (trust SAM)
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
    # normal_cam: (3, H_n, W_n) in [0,1] format (from input_data.pth)
    normal_dec = 2.0 * normal_cam - 1.0  # decode [0,1] → [-1,1]
    nh, nw = normal_dec.shape[1], normal_dec.shape[2]

    # Normalize normals in camera frame (no coordinate transform needed)
    n_flat = normal_dec.reshape(3, -1)  # (3, N) in camera frame
    n_mag = np.sqrt((n_flat ** 2).sum(axis=0))
    valid = n_mag > 0.1
    n_norm = n_flat / (n_mag[None, :] + 1e-8)

    # dot(normal_cam, gravity_up_cam): measures alignment with gravity
    # abs() handles normal sign ambiguity (z>0 vs z<0)
    dot_up = (n_norm * gravity_up_cam[:, None]).sum(axis=0)  # (N,)
    dot_up_map = np.abs(dot_up.reshape(nh, nw))
    valid_map = valid.reshape(nh, nw)

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

    # Step 3: Compose seg map at normal resolution
    seg_map = np.zeros((nh, nw), dtype=np.uint8)

    # Zone classification with score-based overlap resolution
    overlap_raw = building_r & ground_r
    bld_wins = overlap_raw & (building_sc > ground_sc)
    gnd_wins = overlap_raw & ~bld_wins
    only_building = (building_r & ~ground_r) | bld_wins
    only_ground = (ground_r & ~building_r) | gnd_wins

    # Two-threshold system: zone-dependent classification
    # |dot| > horiz_thresh (0.85): strongly horizontal (flat roof/ground)
    # |dot| <= wall_thresh (0.3): strongly vertical (definite wall)
    # Between: ambiguous (degenerate MVS normals on facades have |dot|≈|sin(pitch)|)
    is_strong_horiz = valid_map & (dot_up_map > horiz_thresh)
    is_strong_wall = valid_map & (dot_up_map <= wall_thresh)
    is_ambiguous = valid_map & ~is_strong_horiz & ~is_strong_wall

    # Ground-only zones: trust SAM, override only for strongly vertical normals
    seg_map[only_ground & is_strong_wall] = 2         # strong wall overrides SAM
    seg_map[only_ground & ~is_strong_wall] = 3        # everything else → ground

    # Building zones (including overlap where building score wins):
    # Default ambiguous to wall (oblique view = visible building surface is facade)
    seg_map[only_building & is_strong_horiz] = 1      # strongly horizontal → roof
    seg_map[only_building & is_strong_wall] = 2       # strongly vertical → wall
    seg_map[only_building & is_ambiguous] = 2         # ambiguous → wall (facade likely)
    # Building without valid normal → stay 0 (background, ignored in training)

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

    # ── Load normals for hybrid mode ──
    hybrid_mode = False
    normal_lookup = {}

    if args.input_data and Path(args.input_data).exists():
        print(f"Loading normals from {args.input_data} (hybrid mode)")
        data = torch.load(args.input_data, map_location="cpu")

        # Load normals from input_data.pth: (3, H, W) in [0,1] format
        # Both MVS and Metric3D normals are stored in this format by colmap_to_ps.py
        for idx in range(len(data["image_paths"])):
            stem = os.path.splitext(os.path.basename(data["image_paths"][idx]))[0]
            n = data["normal"][idx]
            if isinstance(n, torch.Tensor):
                n = n.numpy()
            normal_lookup[stem] = n
        print(f"  Loaded normals for {len(normal_lookup)} images")

        hybrid_mode = True

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
        has_normal = hybrid_mode and stem in normal_lookup
        has_pitch = stem in pitch_lookup

        if has_normal and has_pitch:
            gravity_up_cam = compute_gravity_up_cam(pitch_lookup[stem])
            seg_map = process_image_hybrid(
                str(img_path),
                normal_lookup[stem], gravity_up_cam,
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
            if not has_normal:
                reason.append("no normal")
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
