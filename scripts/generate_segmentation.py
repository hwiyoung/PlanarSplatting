"""Generate semantic segmentation maps using Grounded SAM 2.

MVS Hybrid approach (recommended):
  1. Grounded SAM detects "building" and "ground/road" regions
  2. Within building regions, COLMAP MVS normal splits into roof vs wall:
     - dot(normal, world_up) > 0.5 → roof (upward-facing horizontal surface)
     - otherwise → wall (binary split, no ambiguous zone)
  3. Building-ground overlap zones: upward normal → ground, non-upward → wall
  4. Building pixels without valid normals → wall (safer default for facades)

Normal source: COLMAP MVS normal (--normal_source mvs) recommended over Metric3D.
Metric3D has foreshortening bias on oblique drone views (facades misclassified as roof).

Class mapping:
    0 = background (unlabeled)
    1 = building roof
    2 = building wall / facade
    3 = ground / road

Usage:
    # MVS Hybrid mode (recommended): COLMAP MVS normals
    python scripts/generate_segmentation.py \
        --image_dir user_inputs/testset/0_25x/dense/images \
        --output_dir user_inputs/testset/0_25x/seg_maps \
        --vis_dir user_inputs/testset/0_25x/seg_vis \
        --input_data planarSplat_ExpRes/seongsu_phase1/input_data.pth \
        --normal_source mvs \
        --mvs_normal_dir user_inputs/testset/0_25x/dense/stereo/normal_maps

    # Metric3D Hybrid mode (fallback): normals from input_data.pth
    python scripts/generate_segmentation.py \
        --image_dir user_inputs/testset/0_25x/dense/images \
        --output_dir user_inputs/testset/0_25x/seg_maps \
        --vis_dir user_inputs/testset/0_25x/seg_vis \
        --input_data planarSplat_ExpRes/seongsu_phase1/input_data.pth

    # Text-only mode (no normal refinement, noisier roof/wall)
    python scripts/generate_segmentation.py \
        --image_dir user_inputs/testset/0_25x/dense/images \
        --output_dir user_inputs/testset/0_25x/seg_maps \
        --vis_dir user_inputs/testset/0_25x/seg_vis
"""

import argparse
import os
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


def compute_world_up(extrinsics: list) -> np.ndarray:
    """Estimate world UP direction from camera extrinsics.

    For downward-looking drones, UP ≈ negative of mean camera viewing direction.
    """
    view_dirs = []
    for e in extrinsics:
        if isinstance(e, torch.Tensor):
            e = e.numpy()
        R = e[:3, :3]
        view_dir = -R.T[:, 2]  # camera -z in world
        view_dirs.append(view_dir)
    mean_view = np.array(view_dirs).mean(axis=0)
    world_up = -mean_view
    world_up /= np.linalg.norm(world_up) + 1e-8
    return world_up


def process_image_hybrid(
    image_path: str,
    normal_cam: np.ndarray,
    R_w2c: np.ndarray,
    world_up: np.ndarray,
    gdino_processor, gdino_model,
    sam2_processor, sam2_model,
    device: str,
    text_threshold: float,
    roof_thresh: float = 0.5,
    normal_source: str = "metric3d",
) -> np.ndarray:
    """Hybrid: Grounded SAM (building/ground) + normal-based roof/wall split.

    Strategy:
    - Grounded SAM detects "building" and "ground" regions
    - Within building-only zones: upward normal → roof, everything else → wall
    - Within ground-only zones: → ground
    - In overlap zones (both building AND ground detected):
      use normals to decide — upward → ground, not upward → wall
    - Building pixels without valid normals → wall (safer default for facades)
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

    # Step 2: Transform normal to world and compute dot with UP
    # normal_cam: (3, H_n, W_n) — Metric3D: [0,1] range, MVS: [-1,1] range
    # normal_source: "metric3d" or "mvs"
    if normal_source == "metric3d":
        normal_dec = 2.0 * normal_cam - 1.0  # decode [0,1] → [-1,1]
    else:
        normal_dec = normal_cam  # MVS already [-1,1]
    nh, nw = normal_dec.shape[1], normal_dec.shape[2]

    n_flat = normal_dec.reshape(3, -1)
    n_world = R_w2c.T @ n_flat  # (3, N) in world frame
    n_mag = np.sqrt((n_world ** 2).sum(axis=0))
    valid = n_mag > 0.1
    n_world_norm = n_world / (n_mag[None, :] + 1e-8)

    # dot(normal, world_up): positive = upward, negative = downward
    dot_up = (n_world_norm * world_up[:, None]).sum(axis=0)  # (N,)
    dot_up_map = dot_up.reshape(nh, nw)
    valid_map = valid.reshape(nh, nw)

    # Both Metric3D and COLMAP MVS normals need negation
    # (empirically verified: road surfaces give negative dot without negation)
    dot_up_map = -dot_up_map

    # Resize masks to normal resolution if needed
    if (h, w) != (nh, nw):
        building_r = np.array(Image.fromarray(building_mask.astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST)).astype(bool)
        ground_r = np.array(Image.fromarray(ground_mask.astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST)).astype(bool)
    else:
        building_r = building_mask
        ground_r = ground_mask

    # Step 3: Compose seg map at normal resolution
    seg_map = np.zeros((nh, nw), dtype=np.uint8)

    # Zone classification
    only_building = building_r & ~ground_r
    only_ground = ground_r & ~building_r
    overlap = building_r & ground_r

    # Ground-only zones → ground
    seg_map[only_ground] = 3

    # Building-only zones: split by normal
    # upward-facing (dot_up > thresh) → roof, everything else → wall
    is_upward = valid_map & (dot_up_map > roof_thresh)
    seg_map[only_building & is_upward] = 1       # roof
    seg_map[only_building & ~is_upward] = 2      # wall (default for building)

    # Overlap zones: use normal to decide
    # If surface faces up → ground (horizontal = road/ground surface)
    # If surface doesn't face up → wall (vertical = building facade)
    seg_map[overlap & is_upward] = 3             # ground
    seg_map[overlap & ~is_upward] = 2            # wall

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
                        help="Path to input_data.pth for hybrid mode (normal-based roof/wall split)")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gdino_model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2_model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--normal_source", default="metric3d",
                        choices=["metric3d", "mvs"],
                        help="Normal source: metric3d (from input_data.pth) or mvs (COLMAP normal_maps)")
    parser.add_argument("--mvs_normal_dir", default=None,
                        help="Path to COLMAP normal_maps dir (required if --normal_source=mvs)")
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

    normal_source = args.normal_source

    # Load training data for hybrid mode
    hybrid_mode = False
    normal_lookup = {}
    R_lookup = {}
    world_up = None

    if args.input_data and Path(args.input_data).exists():
        print(f"Loading data from {args.input_data} (hybrid mode, normal_source={normal_source})")
        data = torch.load(args.input_data, map_location="cpu")
        world_up = compute_world_up(data["extrinsics"])
        print(f"  World UP direction: [{world_up[0]:.3f}, {world_up[1]:.3f}, {world_up[2]:.3f}]")

        # Load extrinsics (needed for both sources)
        for idx in range(len(data["image_paths"])):
            stem = os.path.basename(data["image_paths"][idx]).replace(".JPG", "").replace(".jpg", "").replace(".png", "")
            e = data["extrinsics"][idx]
            if isinstance(e, torch.Tensor):
                e = e.numpy()
            R_lookup[stem] = e[:3, :3]

        if normal_source == "metric3d":
            # Metric3D normals from input_data.pth: (3, H, W) in [0,1]
            for idx in range(len(data["image_paths"])):
                stem = os.path.basename(data["image_paths"][idx]).replace(".JPG", "").replace(".jpg", "").replace(".png", "")
                n = data["normal"][idx]
                if isinstance(n, torch.Tensor):
                    n = n.numpy()
                normal_lookup[stem] = n
            print(f"  Loaded Metric3D normals for {len(normal_lookup)} images")

        elif normal_source == "mvs":
            # COLMAP MVS normals from .geometric.bin files: (H, W, 3) in [-1,1]
            mvs_dir = args.mvs_normal_dir
            if mvs_dir is None:
                # Try default path relative to image_dir
                mvs_dir = str(image_dir.parent / "stereo" / "normal_maps")
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
            from colmap_to_ps import read_colmap_array
            n_loaded = 0
            for idx in range(len(data["image_paths"])):
                img_basename = os.path.basename(data["image_paths"][idx])
                stem = img_basename.replace(".JPG", "").replace(".jpg", "").replace(".png", "")
                mvs_path = os.path.join(mvs_dir, f"{img_basename}.geometric.bin")
                if os.path.exists(mvs_path):
                    mvs_n = read_colmap_array(mvs_path)  # (H, W, 3) in [-1,1]
                    # Transpose to (3, H, W) to match Metric3D format
                    normal_lookup[stem] = mvs_n.transpose(2, 0, 1)
                    n_loaded += 1
            print(f"  Loaded MVS normals for {n_loaded}/{len(data['image_paths'])} images")

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
        if hybrid_mode and stem in normal_lookup:
            seg_map = process_image_hybrid(
                str(img_path),
                normal_lookup[stem], R_lookup[stem], world_up,
                gdino_proc, gdino_model, sam2_proc, sam2_model,
                args.device, args.text_threshold,
                normal_source=normal_source,
            )
            n_hybrid += 1
            mode_str = "hybrid"
        else:
            seg_map = process_image_textonly(
                str(img_path),
                gdino_proc, gdino_model, sam2_proc, sam2_model,
                args.device, args.text_threshold,
            )
            n_textonly += 1
            mode_str = "text"

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
