"""How to measure how much two rotated boxes resemble each other, three ways.

The loss recipes do not share a way of measuring overlap, and that difference
IS the difference between them: the own one uses the axis-aligned envelope (in
`assign.py`), the YOLOX-OBB fork and Ultralytics use Gaussians, and DDGRCF uses
the exact polygon IoU. This module holds the two that are not the envelope.

All in pure, differentiable torch. Nothing compiled: that was the constraint,
and the three ways fit in the training loop (measured: the exact pairwise IoU
of a whole assigner, 3549 x 2, takes ~19 ms).

=== Gaussians: KLD and ProbIoU ===
Rotated boxes as Gaussians: KLD and ProbIoU. Pure torch, nothing compiled.

The two foreign recipes this project reproduces measure the distance between
boxes by turning each one into a bivariate normal distribution: the center is
the mean and the rotated rectangle gives the covariance. Comparing two boxes
becomes comparing two bells, and that has a closed form. It is what avoids
intersecting polygons in the training loop.

Licenses, and why this was written from the papers
--------------------------------------------------
- **KLD**: Yang et al., "Learning High-Precision Bounding Box for Rotated Object
  Detection via Kullback-Leibler Divergence", NeurIPS 2021. Used by the fork
  `buzhidaoshenme/YOLOX-OBB` (Apache-2.0), whose implementation was used only
  to numerically VERIFY the one here (`tests/test_overlap.py`).
- **ProbIoU**: Llerena et al., "Gaussian Bounding Boxes and Probabilistic
  Intersection-over-Union for Object Detection", 2021. Used by Ultralytics,
  which is **AGPL**: nothing of its code has been read or copied. Everything
  below comes from the equations of the paper.

Two covariance conventions, and it is not a detail
--------------------------------------------------
The two papers turn `(w, h)` into variances differently:

    KLD      sigma_x^2 = w^2 / 4      (the box as 2-sigma of the Gaussian)
    ProbIoU  sigma_x^2 = w^2 / 12     (the box as the support of a uniform)

Mixing them changes the numbers without changing the name. Each function uses
the one of its paper, and `box_to_gaussian` takes the divisor explicitly so it
cannot be called "plainly".

=== Exact polygon IoU ===
EXACT IoU between rotated rectangles, in pure, differentiable torch.

It is what replaces the compiled operator (`box_iou_rotated` / `convex`) of
DDGRCF/YOLOX_OBB, which uses it in two places: the SimOTA cost and the box loss
(PolyIoU). Without this the port does not reproduce their recipe; with the
Gaussians it would change it.

How it is computed
------------------
For two convex quadrilaterals, the intersection is a convex polygon whose
vertices are of three kinds: corners of the first inside the second, corners
of the second inside the first, and edge crossings. The 24 candidates
(4 + 4 + 16) are collected with a validity mask, sorted by angle around their
center, and the masked shoelace formula is applied. At most 8 are valid.

It is differentiable with respect to the coordinates: corners and crossings
are smooth functions of the boxes; only the ORDER is computed without
gradient, and the order is piecewise constant, so it does not need one. It is
the same approach as Lanxiao Li's "Rotated IoU" (MIT), rewritten here from the
geometry.

Cost: for the assigner it is `cells x banknotes` pairs per image (3549 x ~2),
all vectorized. Measured, it is on the order of milliseconds: not the "does
not fit in the loop" of shapely, which goes pair by pair in Python.
"""

from __future__ import annotations

import torch

# --------------------------------------------------------------------------
# Gaussians
# --------------------------------------------------------------------------

#: `sigma^2 = side^2 / divisor`. See the module docstring.
KLD_VARIANCE_DIVISOR = 4.0
PROBIOU_VARIANCE_DIVISOR = 12.0

_EPS = 1e-8


def box_to_gaussian(
    boxes: torch.Tensor, *, variance_divisor: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(N, 5)` with `cx, cy, w, h, theta` -> mean and covariance `(a, b, c)`.

    The covariance is `R diag(w^2/d, h^2/d) R^T`, expanded:

        a = (w^2 cos^2 + h^2 sin^2) / d      variance in x
        b = (w^2 sin^2 + h^2 cos^2) / d      variance in y
        c = (w^2 - h^2) cos sin / d          covariance

    They are returned loose and not as a matrix because the formulas below use
    them loose, and building `(N, 2, 2)` to take it apart again is noise.
    """
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    w2, h2 = (w * w) / variance_divisor, (h * h) / variance_divisor
    a = w2 * cos * cos + h2 * sin * sin
    b = w2 * sin * sin + h2 * cos * cos
    c = (w2 - h2) * cos * sin
    return torch.stack((cx, cy), dim=-1), a, b, c


# --- KLD (Yang et al., 2021) ----------------------------------------------


def kld_divergence(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL divergence `D(N_p || N_t)` between the Gaussians of two boxes. `(N,)`.

    Closed form for two bivariate normals:

        D = 1/2 (mu_p - mu_t)^T S_t^-1 (mu_p - mu_t)
          + 1/2 tr(S_t^-1 S_p)
          + 1/2 ln(|S_t| / |S_p|)
          - 1

    Computed in the axis system of the TARGET box, which is where its
    covariance is diagonal and everything is written with sines and cosines of
    the relative angle. It is what the paper does and what the fork does; with
    matrices the same would come out, slower.

    It is NOT symmetric: `D(p||t) != D(t||p)`. For a loss that is what is
    wanted -- the target is fixed and the prediction moves towards it -- but
    it is worth knowing before using it as a "distance".
    """
    cx_p, cy_p, w_p, h_p, t_p = predicted.unbind(dim=-1)
    cx_t, cy_t, w_t, h_t, t_t = target.unbind(dim=-1)
    d = KLD_VARIANCE_DIVISOR

    dx, dy = cx_p - cx_t, cy_p - cy_t
    cos_t, sin_t = torch.cos(t_t), torch.sin(t_t)
    # Center offset projected onto the target's axes.
    along = dx * cos_t + dy * sin_t
    across = dy * cos_t - dx * sin_t

    var_w_t, var_h_t = (w_t * w_t) / d, (h_t * h_t) / d
    var_w_p, var_h_p = (w_p * w_p) / d, (h_p * h_p) / d
    delta = t_p - t_t
    sin2, cos2 = torch.sin(delta) ** 2, torch.cos(delta) ** 2

    mahalanobis = 0.5 * (along * along / var_w_t + across * across / var_h_t)
    trace = 0.5 * (
        var_h_p / var_w_t * sin2
        + var_w_p / var_h_t * sin2
        + var_h_p / var_h_t * cos2
        + var_w_p / var_w_t * cos2
    )
    log_det = 0.5 * (
        torch.log(var_h_t / var_h_p.clamp(min=_EPS))
        + torch.log(var_w_t / var_w_p.clamp(min=_EPS))
    )
    return mahalanobis + trace + log_det - 1.0


def kld_loss(
    predicted: torch.Tensor, target: torch.Tensor, *, tau: float = 1.0
) -> torch.Tensor:
    """The loss of the paper: `1 - 1 / (tau + ln(D + 1))`. `(N,)`, in `[0, 1)`.

    The raw divergence is unbounded and grows without brake with the center
    error; the logarithmic wrapper flattens it so a box very far away does not
    dominate the batch. `tau = 1` is the value of the paper and of the fork.
    """
    divergence = kld_divergence(predicted, target).clamp(min=0.0)
    return 1.0 - 1.0 / (tau + torch.log1p(divergence))


# --- ProbIoU (Llerena et al., 2021) ----------------------------------------


def bhattacharyya_distance(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """`B_D` between the Gaussians of two boxes, with the equations of the paper.

    With `(a, b, c)` the covariances and `(x, y)` the centers of each one:

        B_D = 1/4 * [ (a1+a2)(y1-y2)^2 + (b1+b2)(x1-x2)^2 ] / [ (a1+a2)(b1+b2) - (c1+c2)^2 ]
            + 1/2 * [ (c1+c2)(x2-x1)(y1-y2) ]             / [ (a1+a2)(b1+b2) - (c1+c2)^2 ]
            + 1/2 * ln( [ (a1+a2)(b1+b2) - (c1+c2)^2 ] / (4 sqrt((a1 b1 - c1^2)(a2 b2 - c2^2))) )

    It IS symmetric, unlike the KL.
    """
    mu1, a1, b1, c1 = box_to_gaussian(predicted, variance_divisor=PROBIOU_VARIANCE_DIVISOR)
    mu2, a2, b2, c2 = box_to_gaussian(target, variance_divisor=PROBIOU_VARIANCE_DIVISOR)
    x1, y1 = mu1.unbind(dim=-1)
    x2, y2 = mu2.unbind(dim=-1)

    a, b, c = a1 + a2, b1 + b2, c1 + c2
    denominator = (a * b - c * c).clamp(min=_EPS)
    t1 = 0.25 * (a * (y1 - y2) ** 2 + b * (x1 - x2) ** 2) / denominator
    t2 = 0.5 * (c * (x2 - x1) * (y1 - y2)) / denominator
    det1 = (a1 * b1 - c1 * c1).clamp(min=_EPS)
    det2 = (a2 * b2 - c2 * c2).clamp(min=_EPS)
    t3 = 0.5 * torch.log(denominator / (4.0 * torch.sqrt(det1 * det2)).clamp(min=_EPS))
    return (t1 + t2 + t3).clamp(min=_EPS, max=100.0)


def probiou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """`ProbIoU = 1 - H_D`, with `H_D = sqrt(1 - exp(-B_D))` the Hellinger one. `(N,)`.

    It is 1 for two equal boxes and drops towards 0 as they move apart. It is
    what the Ultralytics recipe uses as "IoU" both in the loss (`1 - probiou`)
    and in the alignment metric of the assigner.
    """
    hellinger = torch.sqrt(1.0 - torch.exp(-bhattacharyya_distance(predicted, target)) + _EPS)
    return 1.0 - hellinger


def pairwise_probiou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`(N, M)` of ProbIoU between every box of `a` and every one of `b`."""
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return a.new_zeros((n, m))
    left = a[:, None, :].expand(n, m, 5).reshape(-1, 5)
    right = b[None, :, :].expand(n, m, 5).reshape(-1, 5)
    return probiou(left, right).view(n, m)


def pairwise_kld_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`(N, M)` of `kld_loss(a_i, b_j)`: the prediction in rows, the target in
    columns. The order matters because the KL is not symmetric."""
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return a.new_zeros((n, m))
    left = a[:, None, :].expand(n, m, 5).reshape(-1, 5)
    right = b[None, :, :].expand(n, m, 5).reshape(-1, 5)
    return kld_loss(left, right).view(n, m)


# --------------------------------------------------------------------------
# Polygons
# --------------------------------------------------------------------------


def box_corners(boxes: torch.Tensor) -> torch.Tensor:
    """`(N, 5)` cx, cy, w, h, theta -> `(N, 4, 2)` corners in clockwise order
    (in image coordinates, with y pointing down)."""
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    dx = torch.stack((-w, w, w, -w), dim=-1) / 2
    dy = torch.stack((-h, -h, h, h), dim=-1) / 2
    x = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    y = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    return torch.stack((x, y), dim=-1)


def _cross(o, a, b):
    """2D cross product of (a - o) x (b - o). Sign = which side b is on."""
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (
        b[..., 0] - o[..., 0]
    )


def _edge_intersections(c1: torch.Tensor, c2: torch.Tensor):
    """Crossings between the 4 edges of each box: `(N, 16, 2)` and mask `(N, 16)`."""
    a = c1  # (N, 4, 2)
    b = torch.roll(c1, -1, dims=1)
    c = c2
    d = torch.roll(c2, -1, dims=1)
    # All combinations edge_i of 1 x edge_j of 2.
    a = a[:, :, None, :].expand(-1, 4, 4, -1)
    b = b[:, :, None, :].expand(-1, 4, 4, -1)
    c = c[:, None, :, :].expand(-1, 4, 4, -1)
    d = d[:, None, :, :].expand(-1, 4, 4, -1)
    # Parameters t (along ab) and u (along cd) of the line crossing.
    ab = b - a
    cd = d - c
    ac = c - a
    denominator = ab[..., 0] * cd[..., 1] - ab[..., 1] * cd[..., 0]
    parallel = denominator.abs() < _EPS
    safe = torch.where(parallel, torch.ones_like(denominator), denominator)
    t = (ac[..., 0] * cd[..., 1] - ac[..., 1] * cd[..., 0]) / safe
    u = (ac[..., 0] * ab[..., 1] - ac[..., 1] * ab[..., 0]) / safe
    valid = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    points = a + t[..., None] * ab
    return points.reshape(-1, 16, 2), valid.reshape(-1, 16)


def _points_inside(points: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
    """`(N, P)`: whether each point falls inside the convex quadrilateral `corners`.

    Inside = on the same side of the four edges. Works for both orientations
    because the sign is compared between edges, not against a fixed direction.
    """
    a = corners[:, None, :, :]  # (N, 1, 4, 2)
    b = torch.roll(corners, -1, dims=1)[:, None, :, :]
    p = points[:, :, None, :]  # (N, P, 1, 2)
    side = _cross(a, b, p)  # (N, P, 4)
    return (side >= -_EPS).all(dim=-1) | (side <= _EPS).all(dim=-1)


def intersection_area(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Intersection area of each pair `(N, 5)` x `(N, 5)`. Differentiable."""
    c1, c2 = box_corners(boxes_a), box_corners(boxes_b)
    crossings, crossing_valid = _edge_intersections(c1, c2)
    inside_1 = _points_inside(c1, c2)  # corners of A inside B
    inside_2 = _points_inside(c2, c1)

    vertices = torch.cat((c1, c2, crossings), dim=1)  # (N, 24, 2)
    valid = torch.cat((inside_1, inside_2, crossing_valid), dim=1)  # (N, 24)
    count = valid.sum(dim=1, keepdim=True).clamp(min=1)

    # Angular order around the center of the valid ones. No gradient: the
    # order is piecewise constant, and what is differentiated is the
    # coordinates.
    with torch.no_grad():
        mask = valid[..., None].to(vertices.dtype)
        center = (vertices * mask).sum(dim=1, keepdim=True) / count[..., None]
        angles = torch.atan2(vertices[..., 1] - center[..., 1], vertices[..., 0] - center[..., 0])
        angles = torch.where(valid, angles, torch.full_like(angles, 10.0))  # invalid ones last
        order = angles.argsort(dim=1)
    sorted_vertices = torch.gather(vertices, 1, order[..., None].expand(-1, -1, 2))
    sorted_valid = torch.gather(valid, 1, order)

    # Masked shoelace: each valid one with the next valid one, and the last
    # valid one closes against the first.
    n = sorted_valid.sum(dim=1)  # (N,)
    index = torch.arange(24, device=vertices.device)[None, :]
    next_index = torch.where(index + 1 < n[:, None], index + 1, torch.zeros_like(index))
    next_vertices = torch.gather(sorted_vertices, 1, next_index[..., None].expand(-1, -1, 2))
    cross = (
        sorted_vertices[..., 0] * next_vertices[..., 1]
        - next_vertices[..., 0] * sorted_vertices[..., 1]
    )
    area = 0.5 * (cross * sorted_valid.to(cross.dtype)).sum(dim=1).abs()
    return torch.where(n >= 3, area, torch.zeros_like(area))


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return boxes[:, 2] * boxes[:, 3]


def rotated_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Exact pairwise IoU `(N,)`, differentiable. It is DDGRCF's `PolyIoU`."""
    inter = intersection_area(boxes_a, boxes_b)
    union = box_area(boxes_a) + box_area(boxes_b) - inter
    return (inter / (union + eps)).clamp(min=0.0, max=1.0)


def pairwise_rotated_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`(N, M)` of exact IoU. For the assigner; called without gradient."""
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return a.new_zeros((n, m))
    left = a[:, None, :].expand(n, m, 5).reshape(-1, 5)
    right = b[None, :, :].expand(n, m, 5).reshape(-1, 5)
    return rotated_iou(left, right).view(n, m)


__all__ = [
    "KLD_VARIANCE_DIVISOR",
    "PROBIOU_VARIANCE_DIVISOR",
    "bhattacharyya_distance",
    "box_corners",
    "box_to_gaussian",
    "intersection_area",
    "kld_divergence",
    "kld_loss",
    "pairwise_kld_loss",
    "pairwise_probiou",
    "pairwise_rotated_iou",
    "probiou",
    "rotated_iou",
]
