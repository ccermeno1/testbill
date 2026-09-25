"""Rotated-box geometry in pure PyTorch (works on CPU, CUDA and MPS).

Replaces the compiled kernels of YOLOX_OBB (``yolox/ops``: ``convex_sort``,
``box_iou_rotated``, ``nms_rotated``):

* :func:`diff_iou_rotated_2d` - differentiable IoU of aligned box pairs (for the
  ``PolyIoULoss``). Port of ``mmcv/ops/diff_iou_rotated.py`` where the CUDA
  ``sort_vertices`` kernel is replaced by an angular ``argsort``.
* :func:`box_iou_rotated` - pairwise (M x N) IoU / IoF, no gradient (SimOTA, mAP).
* :func:`nms_rotated` / :func:`batched_nms_rotated` - greedy rotated NMS.

Boxes are ``(cx, cy, w, h, angle_rad)``; the angle convention only has to be
consistent between the two inputs (IoU is invariant to mirroring both).
"""
import math

import numpy as np
import torch

EPS = 1e-8

CHUNK_PAIRS = {'cuda': 1 << 17, 'cpu': 1 << 15, 'mps': 1 << 13}


def default_chunk_pairs(device: torch.device) -> int:
    return CHUNK_PAIRS.get(device.type, 1 << 13)


def box2corners(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 5) -> (..., 4, 2) corner coordinates (same order as mmcv)."""
    x, y, w, h, a = boxes.unbind(-1)
    cos, sin = torch.cos(a), torch.sin(a)
    sx = boxes.new_tensor([0.5, -0.5, -0.5, 0.5])
    sy = boxes.new_tensor([0.5, 0.5, -0.5, -0.5])
    dx = sx * w[..., None]
    dy = sy * h[..., None]
    px = dx * cos[..., None] - dy * sin[..., None] + x[..., None]
    py = dx * sin[..., None] + dy * cos[..., None] + y[..., None]
    return torch.stack([px, py], -1)


def _corners_in_box(c1: torch.Tensor, c2: torch.Tensor, tol_px: float) -> torch.Tensor:
    """(..., 4, 2), (..., 4, 2) -> (..., 4) bool: corner i of box1 inside box2."""
    a = c2[..., 0:1, :]
    b = c2[..., 1:2, :]
    d = c2[..., 3:4, :]
    ab = b - a
    ad = d - a
    am = c1 - a
    prod_ab = (ab * am).sum(-1)
    norm_ab = (ab * ab).sum(-1)
    prod_ad = (ad * am).sum(-1)
    norm_ad = (ad * ad).sum(-1)
    # Tolerance of ``tol_px`` along each edge direction so that corners lying
    # on the boundary (identical / touching boxes) still count as inside
    # despite float32 rounding. Expressed on the projections to avoid a division.
    tol_ab = 1e-6 * norm_ab + tol_px * norm_ab.sqrt()
    tol_ad = 1e-6 * norm_ad + tol_px * norm_ad.sqrt()
    return ((prod_ab > -tol_ab) & (prod_ab < norm_ab + tol_ab) &
            (prod_ad > -tol_ad) & (prod_ad < norm_ad + tol_ad))


def polygon_intersection_area(c1: torch.Tensor, c2: torch.Tensor, tol_px: float = 1e-2) -> torch.Tensor:
    """Intersection area of two convex quadrilaterals, differentiable.

    Args:
        c1, c2: corners ``(..., 4, 2)`` with identical leading dims.
        tol_px: a corner within this distance outside the other box still
            counts as a vertex. 1e-2 px makes identical/touching boxes robust
            (IoU exactly 1); the loss uses a much smaller value so that the
            gradient is only affected in a vanishingly thin band.
    Returns:
        ``(...)`` intersection areas.
    """
    # edges as (x1, y1, x2, y2)
    l1 = torch.cat([c1, c1.roll(-1, dims=-2)], -1)  # (..., 4, 4)
    l2 = torch.cat([c2, c2.roll(-1, dims=-2)], -1)
    x1, y1, x2, y2 = l1[..., :, None, :].unbind(-1)  # (..., 4, 1)
    x3, y3, x4, y4 = l2[..., None, :, :].unbind(-1)  # (..., 1, 4)

    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)  # (..., 4, 4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    den_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    parallel = num == 0
    safe_num = torch.where(parallel, torch.ones_like(num), num)
    t = torch.where(parallel, torch.full_like(num, -1.0), den_t / safe_num)
    u = torch.where(parallel, torch.full_like(num, -1.0), -den_u / safe_num)
    inter_mask = (t > 0) & (t < 1) & (u > 0) & (u < 1)  # (..., 4, 4)

    t = den_t / (num + EPS)
    inter = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], -1)  # (..., 4, 4, 2)
    inter = inter * inter_mask[..., None].to(inter.dtype)

    c1_in_2 = _corners_in_box(c1, c2, tol_px)
    c2_in_1 = _corners_in_box(c2, c1, tol_px)

    lead = c1.shape[:-2]
    vertices = torch.cat([c1, c2, inter.reshape(*lead, 16, 2)], -2)  # (..., 24, 2)
    mask = torch.cat([c1_in_2, c2_in_1, inter_mask.reshape(*lead, 16)], -1)  # (..., 24)
    num_valid = mask.sum(-1)  # (...)

    # Sort valid vertices counter-clockwise around their centroid (no gradient
    # needed for the ordering). All 24 slots are kept: coincident vertices (a
    # corner of one box lying on an edge of the other, identical boxes...) end
    # up adjacent and contribute nothing to the shoelace sum, so no
    # de-duplication is needed.
    with torch.no_grad():
        maskf = mask.to(vertices.dtype)
        mean = (vertices * maskf[..., None]).sum(-2, keepdim=True) / num_valid.clamp(min=1)[..., None, None].to(vertices.dtype)
        vn = vertices - mean
        ang = torch.atan2(vn[..., 1], vn[..., 0])
        ang = torch.where(mask, ang, torch.full_like(ang, 1e4))  # invalid last
        order = ang.argsort(dim=-1)  # (..., 24)
        slot = torch.arange(24, device=vertices.device)[(None,) * len(lead)]
        slot_valid = (slot < num_valid[..., None]).to(vertices.dtype)
        close_onehot = torch.nn.functional.one_hot(num_valid.long(), 25).to(vertices.dtype)  # (..., 25)

    sorted_v = torch.gather(vertices, -2, order[..., None].expand(*lead, 24, 2))  # (..., 24, 2)
    sel = sorted_v * slot_valid[..., None]
    sel = torch.cat([sel, torch.zeros_like(sel[..., :1, :])], -2)  # (..., 25, 2)
    sel = sel + sorted_v[..., 0:1, :] * close_onehot[..., None]  # close the polygon
    cross = sel[..., :-1, 0] * sel[..., 1:, 1] - sel[..., :-1, 1] * sel[..., 1:, 0]
    return cross.sum(-1).abs() / 2


def local_corners(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Corners of both boxes in a frame centred on ``boxes1``.

    With pixel coordinates in the thousands, float32 rounding of the corner
    coordinates would otherwise be of the order of the size of small boxes.
    The intersection area is translation invariant, so detaching the shift
    leaves the gradients exact.
    """
    origin = boxes1[..., :2].detach()
    shift = torch.cat([origin, torch.zeros_like(boxes1[..., 2:])], -1)
    return box2corners(boxes1 - shift), box2corners(boxes2 - shift)


def diff_iou_rotated_2d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Differentiable IoU of aligned pairs. ``(..., 5)`` x ``(..., 5)`` -> ``(...)``."""
    c1, c2 = local_corners(boxes1, boxes2)
    inter = polygon_intersection_area(c1, c2, tol_px=1e-4)
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    return inter / (area1 + area2 - inter)


def _hbb(corners: torch.Tensor) -> torch.Tensor:
    """(N, 4, 2) corners -> (N, 4) enclosing horizontal box x1 y1 x2 y2."""
    return torch.cat([corners.min(-2).values, corners.max(-2).values], -1)


@torch.no_grad()
def box_iou_rotated(boxes1: torch.Tensor, boxes2: torch.Tensor, mode: str = 'iou',
                    aligned: bool = False, chunk_pairs: int = None) -> torch.Tensor:
    """Pairwise rotated IoU. ``(M, 5)`` x ``(N, 5)`` -> ``(M, N)`` (or ``(M,)`` if aligned).

    ``mode='iof'`` divides by the area of ``boxes1``. Only pairs whose enclosing
    horizontal boxes overlap are evaluated (the rest are exactly 0), in chunks
    of ``chunk_pairs`` so memory stays bounded.
    """
    assert mode in ('iou', 'iof')
    if chunk_pairs is None:
        chunk_pairs = default_chunk_pairs(boxes1.device)
    boxes1 = boxes1.float().clone()
    boxes2 = boxes2.float().clone()
    # degenerate boxes make the corner-in-box test divide by zero (as in mmrotate)
    boxes1[:, 2:4].clamp_(min=1e-3)
    boxes2[:, 2:4].clamp_(min=1e-3)
    if aligned:
        assert boxes1.shape[0] == boxes2.shape[0]
        if boxes1.shape[0] == 0:
            return boxes1.new_zeros((0,))
        c1, c2 = local_corners(boxes1, boxes2)
        inter = polygon_intersection_area(c1, c2)
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        denom = area1 if mode == 'iof' else area1 + area2 - inter
        return inter / denom.clamp(min=EPS)

    m, n = boxes1.shape[0], boxes2.shape[0]
    out = boxes1.new_zeros((m, n))
    if m == 0 or n == 0:
        return out
    c1 = box2corners(boxes1)  # (M, 4, 2)
    c2 = box2corners(boxes2)  # (N, 4, 2)
    h1, h2 = _hbb(c1), _hbb(c2)
    overlap = ((h1[:, None, 0] < h2[None, :, 2]) & (h1[:, None, 2] > h2[None, :, 0]) &
               (h1[:, None, 1] < h2[None, :, 3]) & (h1[:, None, 3] > h2[None, :, 1]))
    pairs = overlap.nonzero()  # (P, 2)
    if pairs.shape[0] == 0:
        return out
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    for chunk in pairs.split(chunk_pairs):
        i, j = chunk.unbind(1)
        inter = polygon_intersection_area(*local_corners(boxes1[i], boxes2[j]))
        denom = area1[i] if mode == 'iof' else area1[i] + area2[j] - inter
        out[i, j] = inter / denom.clamp(min=EPS)
    return out


@torch.no_grad()
def nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    """Greedy rotated NMS. Returns ``(dets (k, 6) = boxes+score sorted by score, keep idx)``.

    The IoU matrix is computed on ``boxes.device``; the sequential suppression
    loop runs on CPU with numpy (it is O(n) python steps on a boolean matrix).
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 6)), boxes.new_zeros((0,), dtype=torch.long)
    order = scores.argsort(descending=True)
    boxes_s = boxes[order]
    iou = box_iou_rotated(boxes_s, boxes_s).cpu().numpy()
    n = iou.shape[0]
    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > iou_threshold
    keep = order[torch.as_tensor(keep, dtype=torch.long, device=order.device)]
    dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
    return dets, keep


@torch.no_grad()
def batched_nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
                        iou_threshold: float):
    """Class-aware rotated NMS (boxes of different classes never suppress each other)."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 6)), boxes.new_zeros((0,), dtype=torch.long)
    # shift each class to its own region of the plane so one NMS call suffices
    max_coord = boxes[:, :2].abs().max() + boxes[:, 2:4].max() + 1
    offsets = labels.to(boxes.dtype) * max_coord
    shifted = boxes.clone()
    shifted[:, 0] += offsets
    _, keep = nms_rotated(shifted, scores, iou_threshold)
    dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
    return dets, keep
