"""SimOTA: which cell becomes responsible for which banknote.

An anchor-free detector has ~3500 cells at 416x416 and, in our images, one or
two banknotes. Almost everything is background. Deciding BADLY which cells are
positive is what costs the most in a one-stage detector: pick too few and it
does not learn, pick too many and the background dominates.

SimOTA solves it in two steps: filter candidates geometrically, and among those
choose by cost, with a number of positives that ADAPTS to each banknote instead
of being fixed. A well-defined banknote takes more cells than a doubtful one.

Why shapely's rotated IoU is not used here
------------------------------------------
Because it does not fit in the loop. Our rotated IoU builds polygons and
intersects them; to evaluate 100 images once it is perfect, but here it would
take `candidates x banknotes` intersections per image and per iteration.
Measured in the project: shapely takes milliseconds per pair.

So the assignment cost uses an APPROXIMATION in torch: IoU of the axis-aligned
enclosing boxes plus the distance between centers. It is a proxy, and it is
declared as such. Two things make it acceptable:

1. The candidate filter IS exact: it checks whether the cell center falls
   inside the ROTATED rectangle, and that is cheap in torch.
2. The real metric is still shapely's rotated IoU at evaluation. If the proxy
   assigns badly, the final number gives it away. We train with the proxy and
   measure with the real thing, which is the right order for the two.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

#: Radius, in cells, of the central region that is also accepted as candidate.
#: A very thin banknote may contain the center of NO cell at the coarse levels;
#: without this window it would have no positives and learn nothing.
CENTER_RADIUS = 2.5

#: How many of the best IoUs are summed to decide how many positives each
#: banknote takes. It is SimOTA's `dynamic k`.
TOP_CANDIDATES = 10

#: Cost added to whatever is not a geometric candidate. Large, but finite: with
#: infinity the `topk` selection propagates NaN if a banknote is left without
#: candidates, and we prefer a bad positive to a poisoned gradient.
BLOCKED_COST = 1e5


@dataclass(frozen=True, slots=True)
class AnchorGrid:
    """Cell centers of every level, already in image pixels."""

    #: (N, 2) -- center of each cell.
    centers: torch.Tensor
    #: (N,) -- spatial stride of the level each cell belongs to.
    strides: torch.Tensor

    def __len__(self) -> int:
        return self.centers.shape[0]


def build_anchor_grid(
    sizes: list[tuple[int, int]],
    strides: tuple[int, ...],
    device: torch.device | None = None,
) -> AnchorGrid:
    """One point per cell, at the CENTER of the cell, not its corner.

    The half pixel matters: without it every box comes out biased half a cell
    up and to the left, which at stride 32 is 16 pixels.
    """
    centers, all_strides = [], []
    for (height, width), stride in zip(sizes, strides):
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        grid = torch.stack(((xs + 0.5) * stride, (ys + 0.5) * stride), dim=-1)
        centers.append(grid.reshape(-1, 2))
        all_strides.append(
            torch.full((height * width,), float(stride), device=device)
        )
    return AnchorGrid(torch.cat(centers), torch.cat(all_strides))


def points_in_rotated_boxes(
    points: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    """(N, M) -- whether point n falls inside rotated rectangle m.

    Exact and cheap: each point is taken to the box's frame by rotating -theta,
    and there the question is a comparison against w/2 and h/2. No polygon
    needs to be built.

    `boxes` is (M, 5) with `cx, cy, w, h, theta`.
    """
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    offset = points[:, None, :] - torch.stack((cx, cy), dim=-1)[None, :, :]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    # Inverse rotation: rotate the point by -theta around the box center.
    local_x = offset[..., 0] * cos_t + offset[..., 1] * sin_t
    local_y = -offset[..., 0] * sin_t + offset[..., 1] * cos_t
    return (local_x.abs() <= w / 2) & (local_y.abs() <= h / 2)


def points_near_centers(
    points: torch.Tensor, boxes: torch.Tensor, strides: torch.Tensor
) -> torch.Tensor:
    """Square window around the center, measured in cells of the level.

    In cells and not pixels on purpose: a coarse level needs a physically
    larger window to get the same number of candidates.
    """
    centers = boxes[:, :2]
    radius = (CENTER_RADIUS * strides)[:, None]
    delta = (points[:, None, :] - centers[None, :, :]).abs()
    return (delta[..., 0] <= radius) & (delta[..., 1] <= radius)


def enclosing_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """(M, 4) -- axis-aligned envelope `x1, y1, x2, y2` of each rotated box.

    It is the approximation the cost rests on. For an aligned box it is exact;
    for one rotated 45 degrees it is generous. The bias is bounded and always
    goes the same way, which is what makes it usable as a RELATIVE cost.
    """
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    cos_t, sin_t = torch.cos(theta).abs(), torch.sin(theta).abs()
    half_w = (w * cos_t + h * sin_t) / 2
    half_h = (w * sin_t + h * cos_t) / 2
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N, M) -- IoU of axis-aligned boxes `x1, y1, x2, y2`."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    left_top = torch.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


@dataclass(frozen=True, slots=True)
class Assignment:
    """Result: which cell goes with which banknote."""

    #: (N,) bool -- positive cells.
    positive: torch.Tensor
    #: (N,) long -- index of the assigned banknote. Only valid where `positive`.
    matched: torch.Tensor
    #: (N,) -- IoU of the pair, to weight the objectness loss.
    matched_iou: torch.Tensor
    #: (N, C) -- TAL only: SOFT classification target per cell, with the
    #: normalized alignment metric. With SimOTA it is None and the target is
    #: hard (one-hot, or IoU-weighted one-hot in the fork's recipe).
    target_scores: torch.Tensor | None = None

    @property
    def num_positives(self) -> int:
        return int(self.positive.sum())


#: (N, 5) x (M, 5) -> (N, M) of "likeness" in [0, 1]: 1 is the same box.
OverlapFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def enclosing_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The default overlap: IoU of the axis-aligned envelopes."""
    return pairwise_iou(enclosing_boxes(predicted), enclosing_boxes(target))


def simota_assign(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    grid: AnchorGrid,
    *,
    iou_weight: float = 3.0,
    overlap: OverlapFn = enclosing_iou,
    overlap_cost: str = "neg_log",
) -> Assignment:
    """Assign cells to banknotes. Everything in pixels.

    `predicted_boxes` and `target_boxes` are (·, 5) with `cx, cy, w, h, theta`.
    `predicted_scores` is (N,) with the confidence already as a probability.

    `overlap` and `overlap_cost` exist to reproduce the YOLOX-OBB fork's recipe
    WITHOUT touching our own:

        own    overlap = envelope IoU,    cost = -log(overlap)     (YOLOX)
        fork   overlap = 1 - kld_loss,    cost = 1 - overlap       (= kld_loss)

    The `dynamic k` uses `overlap` in both cases, which is what the fork does
    (`pair_wise_iou_approximate = 1 - kld_loss`).

    The class cost is `-log(score)`. In the fork it is a BCE against the
    class one-hot; with ONE class, `BCE(p, 1) = -log(p)` and it is the same.
    With more classes it would stop being so, and this project has one.
    """
    n_points = len(grid)
    device = grid.centers.device
    empty = Assignment(
        positive=torch.zeros(n_points, dtype=torch.bool, device=device),
        matched=torch.zeros(n_points, dtype=torch.long, device=device),
        matched_iou=torch.zeros(n_points, device=device),
    )
    if target_boxes.numel() == 0:
        # Image with no banknotes: everything is background and there is
        # nothing to assign. A legitimate case, not an error -- the area filter
        # can empty an image that only had strips.
        return empty

    inside = points_in_rotated_boxes(grid.centers, target_boxes)
    near = points_near_centers(grid.centers, target_boxes, grid.strides)
    candidate = inside | near
    if not candidate.any():
        return empty

    iou = overlap(predicted_boxes, target_boxes)
    if overlap_cost == "neg_log":
        # The log of the IoU punishes bad overlap with increasing harshness,
        # which is what we want -- between 0.8 and 0.9 the difference matters
        # little; between 0.1 and 0.2, a lot.
        pair_cost = -torch.log(iou.clamp(min=1e-8))
    elif overlap_cost == "one_minus":
        pair_cost = 1.0 - iou
    else:
        raise ValueError(f"unknown overlap_cost: {overlap_cost!r}")
    cost = (
        -torch.log(predicted_scores[:, None].clamp(min=1e-8))
        + iou_weight * pair_cost
        + (~candidate) * BLOCKED_COST
    )

    matching = _dynamic_k_matching(cost, iou, candidate)
    positive = matching.any(dim=1)
    matched = matching.float().argmax(dim=1)
    return Assignment(
        positive=positive,
        matched=matched,
        matched_iou=iou.gather(1, matched[:, None]).squeeze(1) * positive,
    )


def _dynamic_k_matching(
    cost: torch.Tensor, iou: torch.Tensor, candidate: torch.Tensor
) -> torch.Tensor:
    """(N, M) bool -- the "OTA" part of SimOTA.

    Each banknote takes `k` cells, and `k` comes from the sum of its best IoUs:
    a banknote the model already localizes well gets more positives than one
    it barely finds. A fixed `k` would treat the easy and the hard case alike.
    """
    n_points, n_targets = cost.shape
    matching = torch.zeros_like(cost, dtype=torch.bool)

    top = min(TOP_CANDIDATES, n_points)
    top_iou, _ = torch.topk(iou * candidate, top, dim=0)
    # At least one: a banknote with no positive at all produces no gradient
    # and is as if it were not annotated.
    dynamic_k = top_iou.sum(dim=0).int().clamp(min=1)

    for target in range(n_targets):
        k = int(dynamic_k[target])
        _, indices = torch.topk(cost[:, target], k, largest=False)
        matching[indices, target] = True

    # A cell cannot serve two banknotes: it keeps the cheapest. Without this
    # the cell would receive two different targets and learn the average,
    # which is neither.
    conflicts = matching.sum(dim=1) > 1
    if conflicts.any():
        best = cost[conflicts].argmin(dim=1)
        matching[conflicts] = False
        matching[conflicts, best] = True
    return matching


def tal_assign(
    predicted_boxes: torch.Tensor,
    predicted_class_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    grid: AnchorGrid,
    *,
    overlap: OverlapFn,
    topk: int = 10,
    alpha: float = 0.5,
    beta: float = 6.0,
) -> Assignment:
    """Task-Aligned Assigner (Feng et al., "TOOD", ICCV 2021). Everything in pixels.

    It is the assigner of the Ultralytics recipe, implemented from the paper:
    its code is AGPL and has not been read. The defaults (`topk=10`,
    `alpha=0.5`, `beta=6.0`) are the ones Ultralytics DOCUMENTS for its
    `TaskAlignedAssigner`; they do not come from the paper, which uses alpha=1.

    How it decides:

    1. **Alignment metric** per cell-banknote pair:
       `t = score^alpha * overlap^beta`, with `score` the probability of the
       right class and `overlap` the ProbIoU. It rewards classifying well AND
       localizing well at once, which is TOOD's idea: that the positive cells
       be the good ones at BOTH tasks and not at one.
    2. **Candidates**: only cells whose center falls inside the rotated rectangle.
    3. **Top-k** per banknote, by metric.
    4. **Conflicts**: a cell claimed by two banknotes keeps the one with more overlap.
    5. **Soft target**: `t` normalized per banknote to `[0, max overlap]` and
       spread by class. It is what makes the class BCE and the box loss weight
       carry the localization quality inside.

    `predicted_class_scores` is (N, C) as probabilities.
    """
    n_points = len(grid)
    n_classes = predicted_class_scores.shape[-1]
    device = grid.centers.device
    empty = Assignment(
        positive=torch.zeros(n_points, dtype=torch.bool, device=device),
        matched=torch.zeros(n_points, dtype=torch.long, device=device),
        matched_iou=torch.zeros(n_points, device=device),
        target_scores=torch.zeros(n_points, n_classes, device=device),
    )
    if target_boxes.numel() == 0:
        return empty

    overlaps = overlap(predicted_boxes, target_boxes).clamp(min=0.0)  # (N, M)
    score_of_class = predicted_class_scores[:, target_classes]  # (N, M)
    metric = score_of_class.clamp(min=0.0) ** alpha * overlaps**beta

    inside = points_in_rotated_boxes(grid.centers, target_boxes)  # (N, M)
    if not inside.any():
        return empty

    k = min(topk, n_points)
    masked = torch.where(inside, metric, torch.zeros_like(metric))
    top_values, top_indices = torch.topk(masked, k, dim=0)  # (k, M)
    matching = torch.zeros_like(inside)
    # Only the top-k with metric > 0 count: a banknote with fewer than k
    # candidates must not take cells from outside its box as filler.
    valid = top_values > 0
    for column in range(matching.shape[1]):
        matching[top_indices[valid[:, column], column], column] = True

    conflicts = matching.sum(dim=1) > 1
    if conflicts.any():
        best = overlaps[conflicts].argmax(dim=1)
        matching[conflicts] = False
        matching[conflicts, best] = True

    positive = matching.any(dim=1)
    matched = matching.float().argmax(dim=1)
    if not positive.any():
        return empty

    # Per-banknote normalization: the best-aligned cell of each receives
    # exactly its best overlap as target, and the rest in proportion.
    aligned = metric * matching
    max_metric = aligned.max(dim=0, keepdim=True).values
    max_overlap = (overlaps * matching).max(dim=0, keepdim=True).values
    normalized = aligned * max_overlap / max_metric.clamp(min=1e-9)  # (N, M)
    per_cell = normalized.gather(1, matched[:, None]).squeeze(1) * positive
    target_scores = torch.zeros(n_points, n_classes, device=device)
    target_scores[positive, target_classes[matched[positive]]] = per_cell[positive]

    return Assignment(
        positive=positive,
        matched=matched,
        matched_iou=overlaps.gather(1, matched[:, None]).squeeze(1) * positive,
        target_scores=target_scores,
    )


__all__ = [
    "BLOCKED_COST",
    "CENTER_RADIUS",
    "TOP_CANDIDATES",
    "AnchorGrid",
    "Assignment",
    "OverlapFn",
    "build_anchor_grid",
    "enclosing_boxes",
    "enclosing_iou",
    "pairwise_iou",
    "points_in_rotated_boxes",
    "points_near_centers",
    "simota_assign",
    "tal_assign",
]
