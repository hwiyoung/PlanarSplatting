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