import torch
import torch.nn as nn
import torch.nn.functional as F

def metric_depth_loss(depth_pred, depth_gt, mask, max_depth=4.0, weight=None):
    depth_mask = torch.logical_and(depth_gt<=max_depth, depth_gt>0)
    depth_mask = torch.logical_and(depth_mask, mask)
    if depth_mask.sum() == 0:
        depth_loss = torch.tensor([0.]).mean().cuda()
    else:
        if weight is None:
            depth_loss = torch.mean(torch.abs((depth_pred - depth_gt)[depth_mask]))
        else:
            depth_loss = torch.mean((weight * torch.abs(depth_pred - depth_gt))[depth_mask])
    return depth_loss

def normal_loss(normal_pred, normal_gt, mask):
    normal_pred = F.normalize(normal_pred, dim=-1)
    normal_gt = F.normalize(normal_gt, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)[mask].mean()
    cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1))[mask].mean()
    return l1, cos


# ==================== Phase 3-B': Photometric loss ====================

def photo_loss(rendered_rgb, gt_rgb, mask=None, lambda_ssim=0.2):
    """L_photo: L1 + SSIM photometric loss.

    Args:
        rendered_rgb: (3, H, W) rendered RGB image
        gt_rgb: (H*W, 3) or (3, H, W) ground truth RGB
        mask: (H*W,) valid pixel mask (optional)
        lambda_ssim: SSIM weight (default 0.2)

    Returns:
        loss scalar
    """
    H, W = rendered_rgb.shape[1], rendered_rgb.shape[2]

    # Normalize GT to [0, 1]
    if gt_rgb.dim() == 2:
        gt_img = gt_rgb.reshape(H, W, 3).permute(2, 0, 1)  # (3, H, W)
    else:
        gt_img = gt_rgb
    if gt_img.max() > 1.0:
        gt_img = gt_img / 255.0

    pred_img = rendered_rgb  # (3, H, W), already [0,1] from sigmoid

    if mask is not None:
        mask_2d = mask.reshape(H, W)
        # Apply mask: zero out invalid regions
        pred_masked = pred_img * mask_2d.unsqueeze(0)
        gt_masked = gt_img * mask_2d.unsqueeze(0)
    else:
        pred_masked = pred_img
        gt_masked = gt_img
        mask_2d = torch.ones(H, W, device=rendered_rgb.device, dtype=torch.bool)

    # L1 loss (only on valid pixels)
    n_valid = mask_2d.sum().clamp(min=1)
    l1 = (pred_masked - gt_masked).abs().sum() / (n_valid * 3)

    # SSIM loss (window-based, tolerates some masked regions)
    ssim_val = _ssim(pred_masked.unsqueeze(0), gt_masked.unsqueeze(0))

    loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)
    return loss


def _ssim(img1, img2, window_size=11):
    """Compute SSIM between two (1, 3, H, W) images."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channel = img1.shape[1]

    # Gaussian window
    kernel_1d = torch.exp(-torch.arange(window_size, dtype=torch.float32, device=img1.device)
                          .sub(window_size // 2).pow(2) / (2 * 1.5 ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    window = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ==================== Phase 2-B: Semantic losses ====================

def semantic_loss(rendered_features, seg_gt, mask=None):
    """L_sem: CrossEntropyLoss with ignore_index=0 (background).

    Args:
        rendered_features: (C, H, W) raw alpha-blended features from rasterizer
        seg_gt: (H*W,) long tensor of class indices (0=bg, 1=roof, 2=wall, 3=ground)
        mask: optional (H*W,) bool mask for valid pixels

    Returns:
        loss scalar
    """
    C, H, W = rendered_features.shape
    # CrossEntropyLoss expects (N, C) logits, (N,) targets
    logits = rendered_features.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
    targets = seg_gt.reshape(-1)  # (H*W,)

    if mask is not None:
        logits = logits[mask]
        targets = targets[mask]

    if targets.numel() == 0:
        return torch.tensor(0., device=rendered_features.device, requires_grad=True)

    ce = F.cross_entropy(logits, targets, ignore_index=0, reduction='mean')
    return ce


# ==================== Phase 3-A: L_mutual ====================

def mutual_loss(semantic_features, plane_normals, e_gravity, tau=0.15, mode='full'):
    """L_mutual: bidirectional geometric-semantic consistency loss.

    Per-primitive loss encouraging consistency between semantic class
    probabilities (from f_i) and geometric normal orientations (from R_i).
    Operates directly at the primitive level (no rendering involved).

    Three geometric terms:
      - L_vert(n)  = (n . e_gravity)^2         -- 0 when horizontal (walls)
      - L_horiz(n) = (1 - |n . e_gravity|)^2   -- 0 when vertical (ground)
      - L_slope(n) = relu(tau - (n.e_gravity)^2)^2  -- one-sided wall exclusion (roofs)

    L_mutual = mean_i [ p_wall * L_vert + p_roof * L_slope + p_ground * L_horiz ]

    Args:
        semantic_features: (N, C) raw semantic logits, C=4 (bg/roof/wall/ground)
        plane_normals: (N, 3) per-primitive normal vectors in world frame
        e_gravity: (3,) gravity direction unit vector, e.g. [0, -1, 0]
        tau: threshold for L_slope (default 0.15)
        mode: 'full' -- bidirectional gradient (no detach)
              'sem2geo' -- detach softmax(f_i), only R_i gets gradient
              'geo2sem' -- detach n_i, only f_i gets gradient
              'none' -- returns zero (disabled)

    Returns:
        loss scalar
    """
    if mode == 'none':
        return torch.tensor(0., device=semantic_features.device, requires_grad=True)

    # Class probabilities: (N, C) where C=4 (bg=0, roof=1, wall=2, ground=3)
    p = F.softmax(semantic_features, dim=-1)
    p_roof = p[:, 1]
    p_wall = p[:, 2]
    p_ground = p[:, 3]

    if mode == 'sem2geo':
        # Only R_i gets gradient (semantics -> geometry direction)
        p_roof = p_roof.detach()
        p_wall = p_wall.detach()
        p_ground = p_ground.detach()

    # Per-primitive normal dot gravity
    n = F.normalize(plane_normals, dim=-1)
    if mode == 'geo2sem':
        # Only f_i gets gradient (geometry -> semantics direction)
        n = n.detach()

    dot = (n * e_gravity.to(n.device)).sum(dim=-1)  # (N,)

    # Geometric terms
    L_vert = dot ** 2                          # 0 when horizontal
    L_horiz = (1.0 - dot.abs()) ** 2           # 0 when vertical
    L_slope = F.relu(tau - dot ** 2) ** 2      # one-sided wall exclusion

    # Weighted sum per primitive, mean over all primitives
    loss = (p_wall * L_vert + p_roof * L_slope + p_ground * L_horiz).mean()

    return loss


def normal_consistency_loss(depth, normal_rendered, intrinsic, mask=None):
    """L_geo (L_normal_consistency): rendered normal vs depth-derived normal.

    Computes normals from depth via finite difference, then compares with
    rendered normals. Excludes depth discontinuity edges (large depth gradient).

    Based on 2DGS/PGSR standard implementation.

    Args:
        depth: (H, W) rendered depth map
        normal_rendered: (H, W, 3) rendered normal in camera frame
        intrinsic: (3, 3) or (4, 4) camera intrinsic matrix
        mask: optional (H, W) bool mask

    Returns:
        loss scalar
    """
    H, W = depth.shape
    device = depth.device

    # Unproject depth to 3D points in camera frame
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                           torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    points = torch.stack([x, y, depth], dim=-1)  # (H, W, 3)

    # Finite difference normals
    # dx = points[i, j+1] - points[i, j-1], dy = points[i+1, j] - points[i-1, j]
    dx = torch.zeros_like(points)
    dy = torch.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1, :] = points[2:, :] - points[:-2, :]

    normal_derived = torch.cross(dx, dy, dim=-1)  # (H, W, 3)
    normal_derived = F.normalize(normal_derived, dim=-1)

    # Depth discontinuity mask: exclude edges where depth changes rapidly
    depth_grad_x = torch.zeros_like(depth)
    depth_grad_y = torch.zeros_like(depth)
    depth_grad_x[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]).abs()
    depth_grad_y[1:-1, :] = (depth[2:, :] - depth[:-2, :]).abs()
    depth_grad = torch.max(depth_grad_x, depth_grad_y)
    # Threshold: relative to local depth (2DGS convention)
    edge_mask = depth_grad < (depth * 0.05)

    # Valid mask: combine with input mask, exclude borders, require positive depth
    valid = (depth > 0.01) & edge_mask
    valid[:1, :] = False
    valid[-1:, :] = False
    valid[:, :1] = False
    valid[:, -1:] = False
    if mask is not None:
        valid = valid & mask

    if valid.sum() == 0:
        return torch.tensor(0., device=device, requires_grad=True)

    # Normal consistency: 1 - dot(n_render, n_derived)
    normal_r = F.normalize(normal_rendered, dim=-1)
    cos_sim = (normal_r * normal_derived).sum(dim=-1)
    loss = (1.0 - cos_sim)[valid].mean()
    return loss