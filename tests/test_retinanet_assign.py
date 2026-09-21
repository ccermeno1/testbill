"""Rotated RetinaNet assignment (not the shared two-stage matcher)."""

import math
from unittest.mock import patch

import pytest
import torch

from oriented_det.geometry.rbox import RBox
from oriented_det.models.retinanet_assign import (
    _paired_exact_rotated_iou,
    match_retinanet_anchors_to_gt,
)
from oriented_det.ops.diff_iou_rotated import diff_iou_rotated_2d
from oriented_det.ops.iou import rbox_iou
from oriented_det.ops.gpu_ops import oriented_box_hbb_iou_gpu


def _box(cx, cy, w, h, angle=0.0):
    return torch.tensor([[cx, cy, w, h, angle]], dtype=torch.float32)


def test_identical_anchor_is_positive_far_anchor_is_background():
    anchors = torch.tensor(
        [
            [64.0, 64.0, 32.0, 16.0, 0.0],
            [400.0, 400.0, 32.0, 16.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = _box(64.0, 64.0, 32.0, 16.0, 0.0)
    labels, matched = match_retinanet_anchors_to_gt(anchors, gt)
    assert labels[0].item() == 1
    assert matched[0].item() == 0
    assert labels[1].item() == 0
    assert matched[1].item() == -1


def test_zero_iou_low_quality_match_stays_background():
    """min_pos_iou=0 must not promote index 0 when every IoU is 0."""
    anchors = torch.tensor(
        [
            [10.0, 10.0, 8.0, 8.0, 0.0],
            [30.0, 30.0, 8.0, 8.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = _box(800.0, 800.0, 16.0, 8.0, 0.4)
    labels, matched = match_retinanet_anchors_to_gt(
        anchors, gt, min_pos_iou=0.0, match_low_quality=True
    )
    assert torch.equal(labels, torch.zeros(2, dtype=torch.long))
    assert torch.equal(matched, torch.full((2,), -1, dtype=torch.long))


def test_concat_assign_does_not_force_coarse_level_when_fine_level_fits():
    """Per-level min_pos_iou=0 would mark a huge P7 prior; concat keeps only the P3 best."""
    p3 = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.0]], dtype=torch.float32)
    p7 = torch.tensor([[64.0, 64.0, 256.0, 256.0, 0.0]], dtype=torch.float32)
    gt = _box(64.0, 64.0, 32.0, 16.0, 0.0)

    labels_p7, _ = match_retinanet_anchors_to_gt(
        p7, gt, min_pos_iou=0.0, match_low_quality=True
    )
    assert labels_p7[0].item() == 1

    labels_cat, matched_cat = match_retinanet_anchors_to_gt(
        torch.cat([p3, p7], dim=0), gt, min_pos_iou=0.0, match_low_quality=True
    )
    assert labels_cat[0].item() == 1
    assert matched_cat[0].item() == 0
    assert labels_cat[1].item() == 0
    assert matched_cat[1].item() == -1


def test_low_quality_match_assigns_best_anchor_when_all_ious_below_pos():
    """min_pos_iou=0 still promotes the best-overlapping anchor per GT."""
    anchors = torch.tensor(
        [
            [80.0, 64.0, 32.0, 16.0, 0.0],
            [400.0, 400.0, 32.0, 16.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = _box(64.0, 64.0, 32.0, 16.0, 0.0)
    labels, matched = match_retinanet_anchors_to_gt(
        anchors, gt, positive_iou_threshold=0.99, negative_iou_threshold=0.4
    )
    assert labels[0].item() == 1
    assert matched[0].item() == 0
    assert labels[1].item() == 0


def test_empty_gt_is_all_background():
    anchors = torch.zeros((8, 5), dtype=torch.float32)
    gt = torch.zeros((0, 5), dtype=torch.float32)
    labels, matched = match_retinanet_anchors_to_gt(anchors, gt)
    assert torch.equal(labels, torch.zeros(8, dtype=torch.long))
    assert torch.equal(matched, torch.full((8,), -1, dtype=torch.long))


def test_ignore_region_marks_non_positive_as_ignore():
    anchors = torch.tensor(
        [
            [64.0, 64.0, 32.0, 16.0, 0.0],
            [200.0, 200.0, 32.0, 16.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = torch.zeros((0, 5), dtype=torch.float32)
    ign = _box(200.0, 200.0, 32.0, 16.0, 0.0)
    labels, matched = match_retinanet_anchors_to_gt(
        anchors, gt, gt_boxes_ignore=ign, ignore_iou_threshold=0.5
    )
    assert labels[0].item() == 0
    assert labels[1].item() == -1
    assert matched[1].item() == -1


def test_lookalike_overrides_ignore_to_background():
    anchors = torch.tensor(
        [
            [200.0, 200.0, 32.0, 16.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt = torch.zeros((0, 5), dtype=torch.float32)
    region = _box(200.0, 200.0, 32.0, 16.0, 0.0)
    labels, matched = match_retinanet_anchors_to_gt(
        anchors,
        gt,
        gt_boxes_ignore=region,
        ignore_iou_threshold=0.5,
        gt_boxes_lookalike=region,
        lookalike_iou_threshold=0.5,
    )
    assert labels[0].item() == 0
    assert matched[0].item() == -1


def test_hbb_path_delegates_to_shared_matcher():
    anchors = _box(64.0, 64.0, 32.0, 16.0, 0.3)
    gt = _box(64.0, 64.0, 32.0, 16.0, 1.0)
    sentinel_labels = torch.tensor([1], dtype=torch.long)
    sentinel_matched = torch.tensor([0], dtype=torch.long)
    with patch(
        "oriented_det.models.retinanet_assign.match_oriented_anchors_to_gt",
        return_value=(sentinel_labels, sentinel_matched),
    ) as mocked:
        labels, matched = match_retinanet_anchors_to_gt(
            anchors, gt, use_hbb_for_matching=True
        )
    mocked.assert_called_once()
    assert mocked.call_args.kwargs["use_hbb_for_matching"] is True
    assert labels is sentinel_labels
    assert matched is sentinel_matched


def test_obb_rejects_thin_rotated_gt_that_hbb_accepts():
    """Horizontal prior vs thin ~45° GT: circum-HBB ≥0.5 but exact OBB IoU <0.4."""
    anchors = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.0]], dtype=torch.float32)
    gt = torch.tensor(
        [[64.0, 64.0, 32.0, 8.0, math.radians(45.0)]], dtype=torch.float32
    )
    hbb = float(oriented_box_hbb_iou_gpu(anchors, gt)[0, 0])
    obb = float(diff_iou_rotated_2d(anchors, gt))
    assert hbb >= 0.5
    assert obb < 0.4

    labels_obb, _ = match_retinanet_anchors_to_gt(
        anchors,
        gt,
        use_hbb_for_matching=False,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        match_low_quality=False,
    )
    labels_hbb, _ = match_retinanet_anchors_to_gt(
        anchors,
        gt,
        use_hbb_for_matching=True,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        match_low_quality=False,
    )
    assert labels_obb[0].item() == 0
    assert labels_hbb[0].item() == 1


def _paired_sample_iou(boxes_a, boxes_b, num_samples=100):
    """Old RetinaNet assigner estimator (fixed 100-sample grid)."""
    from oriented_det.ops.gpu_ops import (
        _box_vertices,
        _generate_box_samples,
        _points_in_paired_boxes,
    )

    area_a = boxes_a[:, 2] * boxes_a[:, 3]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]
    samples_a = _generate_box_samples(boxes_a, num_samples)
    samples_b = _generate_box_samples(boxes_b, num_samples)
    verts_a = _box_vertices(boxes_a)
    verts_b = _box_vertices(boxes_b)
    count_in_b = _points_in_paired_boxes(samples_a, verts_b).sum(dim=1).float()
    count_in_a = _points_in_paired_boxes(samples_b, verts_a).sum(dim=1).float()
    inter = torch.maximum(
        (count_in_b / num_samples) * area_a,
        (count_in_a / num_samples) * area_b,
    )
    inter = torch.minimum(inter, torch.minimum(area_a, area_b))
    union = (area_a + area_b - inter).clamp(min=1e-8)
    return torch.clamp(inter / union, 0.0, 1.0)


def test_sample_iou_underestimates_exact_and_would_miss_positive():
    """100-sample ranking (Sep 11 OBB run) drops a true ≥0.5 pair below 0.5."""
    anchors = torch.tensor([[64.0, 64.0, 32.0, 16.0, 0.0]], dtype=torch.float32)
    gt = torch.tensor(
        [[68.0, 64.0, 32.0, 18.0, math.radians(35.0)]], dtype=torch.float32
    )
    exact = float(_paired_exact_rotated_iou(anchors, gt))
    shapely = float(
        rbox_iou(
            RBox(*[float(x) for x in anchors[0]]),
            RBox(*[float(x) for x in gt[0]]),
            intersection_backend="auto",
        )
    )
    sampled = float(_paired_sample_iou(anchors, gt))
    assert exact == pytest.approx(shapely, abs=3e-3)
    assert exact >= 0.5
    assert sampled < 0.5

    labels, _ = match_retinanet_anchors_to_gt(
        anchors,
        gt,
        use_hbb_for_matching=False,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        match_low_quality=False,
    )
    assert labels[0].item() == 1


def test_assigner_exact_iou_matches_diff_iou_rotated_and_shapely():
    a = torch.tensor([[10.0, 20.0, 40.0, 12.0, 0.4]], dtype=torch.float32)
    b = torch.tensor([[14.0, 18.0, 36.0, 14.0, -0.25]], dtype=torch.float32)
    got = float(_paired_exact_rotated_iou(a, b))
    ref = float(diff_iou_rotated_2d(a, b))
    shapely = float(
        rbox_iou(
            RBox(*[float(x) for x in a[0]]),
            RBox(*[float(x) for x in b[0]]),
            intersection_backend="auto",
        )
    )
    assert got == pytest.approx(ref, abs=1e-6)
    assert got == pytest.approx(shapely, abs=3e-3)


def test_dense_p3_grid_finishes_without_materializing_full_sampling():
    """27-anchor P3-scale grid vs a handful of GTs must stay cheap on CPU."""
    h = w = 64
    num_anchors = 27
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    cx = (xx + 0.5) * 8.0
    cy = (yy + 0.5) * 8.0
    n_loc = h * w
    anchors = torch.stack(
        [
            cx.reshape(-1).repeat(num_anchors),
            cy.reshape(-1).repeat(num_anchors),
            torch.full((n_loc * num_anchors,), 32.0),
            torch.full((n_loc * num_anchors,), 16.0),
            torch.zeros(n_loc * num_anchors),
        ],
        dim=1,
    )
    gt = torch.tensor(
        [
            [64.0, 64.0, 40.0, 18.0, 0.4],
            [200.0, 180.0, 28.0, 12.0, -0.3],
            [400.0, 400.0, 48.0, 20.0, 0.7],
        ],
        dtype=torch.float32,
    )
    labels, matched = match_retinanet_anchors_to_gt(anchors, gt)
    assert labels.shape == (n_loc * num_anchors,)
    assert (labels == 1).any()
    assert (labels == 0).any()
    assert matched[labels == 1].min().item() >= 0
