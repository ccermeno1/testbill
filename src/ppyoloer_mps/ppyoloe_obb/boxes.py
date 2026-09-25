"""
Oriented box geometry in plain PyTorch. Runs on CPU, CUDA and MPS.

Conventions, same as PP-YOLOE-R in PaddleDetection:
  rbox = (cx, cy, w, h, angle), angle in RADIANS, clockwise with y pointing down
  poly = (x1, y1, x2, y2, x3, y3, x4, y4), four consecutive corners
"""
from __future__ import annotations

import math
import os

import torch

__all__ = [
    "box2corners",
    "corners2poly",
    "rbox2poly",
    "poly2rbox",
    "check_points_in_rotated_boxes",
    "probiou",
    "probiou_loss",
    "rotated_iou",
    "pairwise_rotated_iou",
]

# How many (a, b) pairs rotated_iou handles at once. The temporaries are roughly
# pairs x 24 x 24 floats, so lowering this cuts peak VRAM without changing the result.
# Override with PPYOLOER_MAX_PAIRS.
MAX_PAIRS = int(os.environ.get("PPYOLOER_MAX_PAIRS", 40_000))


def box2corners(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 5) -> (..., 4, 2). Matches box2corners in ppdet/modeling/rbox_utils.py."""
    x, y, w, h, alpha = boxes.unbind(-1)
    x4 = boxes.new_tensor([0.5, 0.5, -0.5, -0.5]) * w.unsqueeze(-1)
    y4 = boxes.new_tensor([-0.5, 0.5, 0.5, -0.5]) * h.unsqueeze(-1)
    corners = torch.stack([x4, y4], dim=-1)  # (..., 4, 2)
    sin, cos = torch.sin(alpha), torch.cos(alpha)
    # transposed rotation matrix, applied on the right: [[cos, sin], [-sin, cos]]
    rot_t = torch.stack(
        [torch.stack([cos, sin], -1), torch.stack([-sin, cos], -1)], dim=-2
    )  # (..., 2, 2)
    rotated = corners @ rot_t
    return rotated + torch.stack([x, y], dim=-1).unsqueeze(-2)


def corners2poly(corners: torch.Tensor) -> torch.Tensor:
    """(..., 4, 2) -> (..., 8)."""
    return corners.flatten(-2)


def rbox2poly(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 5) -> (..., 8)."""
    return corners2poly(box2corners(boxes))


def poly2rbox(polys: torch.Tensor) -> torch.Tensor:
    """(..., 8) -> (..., 5) with the angle in [0, pi/2), like Poly2RBox with rbox_type='oc'.

    Edge 0->1 is the reference side. The angle is brought into the first quadrant by
    swapping w and h when needed, which is the OpenCV minAreaRect convention.
    """
    pts = polys.reshape(*polys.shape[:-1], 4, 2)
    edge1 = pts[..., 1, :] - pts[..., 0, :]
    edge2 = pts[..., 2, :] - pts[..., 1, :]
    w = edge1.norm(dim=-1)
    h = edge2.norm(dim=-1)
    angle = torch.atan2(edge1[..., 1], edge1[..., 0])
    center = pts.mean(dim=-2)
    half_pi = math.pi / 2
    # fold the angle into [0, pi/2), swapping w and h accordingly
    angle = torch.remainder(angle, math.pi)
    swap = angle >= half_pi
    angle = torch.where(swap, angle - half_pi, angle)
    w_new = torch.where(swap, h, w)
    h_new = torch.where(swap, w, h)
    return torch.cat(
        [center, w_new.unsqueeze(-1), h_new.unsqueeze(-1), angle.unsqueeze(-1)], dim=-1
    )


def check_points_in_rotated_boxes(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """points (1, L, 2), boxes (B, N, 5) -> (B, N, L) boolean mask."""
    corners = box2corners(boxes)  # (B, N, 4, 2)
    pts = points.unsqueeze(0)  # (1, 1, L, 2)
    a, b, _c, d = corners.unbind(-2)
    a = a.unsqueeze(-2)
    ab = (b - a.squeeze(-2)).unsqueeze(-2)
    ad = (d - a.squeeze(-2)).unsqueeze(-2)
    ap = pts - a
    norm_ab = (ab * ab).sum(-1)
    norm_ad = (ad * ad).sum(-1)
    ap_ab = (ap * ab).sum(-1)
    ap_ad = (ap * ad).sum(-1)
    return (ap_ab >= 0) & (ap_ab <= norm_ab) & (ap_ad >= 0) & (ap_ad <= norm_ad)


# --------------------------------------------------------------------------------------
# ProbIoU (https://arxiv.org/abs/2106.06072), same as ppdet/modeling/losses/probiou_loss.py
# --------------------------------------------------------------------------------------
def _gbb_form(boxes: torch.Tensor):
    return boxes[..., 0], boxes[..., 1], boxes[..., 2].pow(2) / 12.0, boxes[..., 3].pow(2) / 12.0, boxes[..., 4]


def _rotated_form(a_, b_, angles):
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    a = a_ * cos_a.pow(2) + b_ * sin_a.pow(2)
    b = a_ * sin_a.pow(2) + b_ * cos_a.pow(2)
    c = (a_ - b_) * cos_a * sin_a
    return a, b, c


def probiou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """ProbIoU distance ('l1' mode) between paired rboxes: (N, 5) x (N, 5) -> (N,)."""
    x1, y1, a1_, b1_, c1_ = _gbb_form(pred)
    x2, y2, a2_, b2_, c2_ = _gbb_form(target)
    a1, b1, c1 = _rotated_form(a1_, b1_, c1_)
    a2, b2, c2 = _rotated_form(a2_, b2_, c2_)

    t1 = 0.25 * ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) + 0.5 * (
        (c1 + c2) * (x2 - x1) * (y1 - y2)
    )
    # t2 is positive for valid boxes. Clamping guards against degenerate ones, which would
    # otherwise give log(0) -> -inf and NaN once it propagates
    t2 = ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)).clamp(min=eps)
    t3_ = (a1 * b1 - c1 * c1) * (a2 * b2 - c2 * c2)
    t3 = 0.5 * torch.log(t2 / (4 * torch.sqrt(torch.relu(t3_)) + eps))

    bd = (t1 / t2) + t3
    bd = bd.clamp(min=eps, max=100.0)
    return torch.sqrt(1.0 - torch.exp(-bd) + eps)


def probiou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return probiou(pred, target, eps)


# --------------------------------------------------------------------------------------
# Exact rotated IoU (intersection of two convex rectangles), vectorised.
# Same algorithm as PaddleDetection's rbox_iou custom op, itself adapted from detectron2:
# candidate points = edge-edge intersections + corners inside; sort by angle; shoelace.
# --------------------------------------------------------------------------------------
def _edge_intersections(c1: torch.Tensor, c2: torch.Tensor):
    """c1, c2: (P, 4, 2) -> points (P, 16, 2) and mask (P, 16)."""
    p1 = c1.unsqueeze(2)
    d1 = (torch.roll(c1, -1, dims=1) - c1).unsqueeze(2)
    p3 = c2.unsqueeze(1)
    d2 = (torch.roll(c2, -1, dims=1) - c2).unsqueeze(1)
    dp = p3 - p1

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    den = cross(d1, d2)
    nonpar = den.abs() > 1e-12
    den_safe = torch.where(nonpar, den, torch.ones_like(den))
    t = cross(dp, d2) / den_safe
    u = cross(dp, d1) / den_safe
    valid = nonpar & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    pts = p1 + t.unsqueeze(-1) * d1
    p = c1.shape[0]
    return pts.reshape(p, 16, 2), valid.reshape(p, 16)


def _corners_inside(pts: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Which of pts (P, 4, 2) fall inside the rectangle box (P, 4, 2) -> (P, 4)."""
    a = box[:, 0:1, :]
    ab = box[:, 1:2, :] - a
    ad = box[:, 3:4, :] - a
    ap = pts - a
    proj_ab = (ap * ab).sum(-1)
    proj_ad = (ap * ad).sum(-1)
    len_ab = (ab * ab).sum(-1)
    len_ad = (ad * ad).sum(-1)
    eps = 1e-6
    return (
        (proj_ab >= -eps) & (proj_ab <= len_ab + eps) & (proj_ad >= -eps) & (proj_ad <= len_ad + eps)
    )


def _convex_area(pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Area of the convex polygon formed by the valid points. pts (P, K, 2), mask (P, K)."""
    p, k, _ = pts.shape
    maskf = mask.to(pts.dtype).unsqueeze(-1)
    num_valid = maskf.sum(dim=1)
    center = (pts * maskf).sum(dim=1) / num_valid.clamp(min=1.0)
    diff = pts - center.unsqueeze(1)
    ang = torch.atan2(diff[..., 1], diff[..., 0])
    ang = torch.where(mask, ang, torch.full_like(ang, 1e6))

    # angular ordering without a sort: rank_i = how many valid points come before i
    idx = torch.arange(k, device=pts.device)
    tie = (idx.unsqueeze(1) > idx.unsqueeze(0)).unsqueeze(0)
    before = ((ang.unsqueeze(1) < ang.unsqueeze(2)) | ((ang.unsqueeze(1) == ang.unsqueeze(2)) & tie))
    before = before & mask.unsqueeze(1)
    rank = before.to(pts.dtype).sum(dim=2)

    succ_rank = rank + 1
    succ_rank = torch.where(succ_rank >= num_valid, torch.zeros_like(succ_rank), succ_rank)
    onehot = (rank.unsqueeze(1) == succ_rank.unsqueeze(2)) & mask.unsqueeze(1)
    succ = torch.matmul(onehot.to(pts.dtype), pts)
    cross = pts[..., 0] * succ[..., 1] - pts[..., 1] * succ[..., 0]
    area = 0.5 * (cross * maskf.squeeze(-1)).sum(dim=1).abs()
    return torch.where(num_valid.squeeze(-1) >= 3, area, torch.zeros_like(area))


def pairwise_rotated_iou(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Exact rotated IoU between matched pairs: (P, 5) x (P, 5) -> (P,)."""
    if r1.shape[0] == 0:
        return r1.new_zeros(0)
    c1 = box2corners(r1)
    c2 = box2corners(r2)
    inter_pts, inter_mask = _edge_intersections(c1, c2)
    pts = torch.cat([inter_pts, c1, c2], dim=1)
    mask = torch.cat([inter_mask, _corners_inside(c1, c2), _corners_inside(c2, c1)], dim=1)
    inter = _convex_area(pts, mask)
    area1 = (r1[:, 2] * r1[:, 3]).abs()
    area2 = (r2[:, 2] * r2[:, 3]).abs()
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-9)
    iou = torch.where((area1 < 1e-14) | (area2 < 1e-14), torch.zeros_like(iou), iou)
    return iou.clamp(0.0, 1.0)


def rotated_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Exact rotated IoU, all against all: (M, 5) x (N, 5) -> (M, N)."""
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    m, n = boxes1.shape[0], boxes2.shape[0]
    if m == 0 or n == 0:
        return boxes1.new_zeros((m, n))
    rows = max(1, MAX_PAIRS // max(n, 1))
    outs = []
    for i in range(0, m, rows):
        chunk = boxes1[i : i + rows]
        c = chunk.shape[0]
        a = chunk.unsqueeze(1).expand(c, n, 5).reshape(c * n, 5)
        b = boxes2.unsqueeze(0).expand(c, n, 5).reshape(c * n, 5)
        outs.append(pairwise_rotated_iou(a, b).reshape(c, n))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def batch_rotated_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """(B, M, 5) x (B, N, 5) -> (B, M, N). Equivalent to ppdet's rotated_iou_similarity."""
    return torch.stack([rotated_iou(a, b) for a, b in zip(boxes1, boxes2)], dim=0)
