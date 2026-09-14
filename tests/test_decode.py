"""Decoding, rotated NMS and the step to canonical quads."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="the own candidate needs torch")

from shapely.geometry import Polygon

from testbank.dataio.image_sizes import ImageSize
from testbank.models.decode import (
    box_to_polygon,
    boxes_to_quads,
    decode_outputs,
    detections,
    rotated_nms,
)
from testbank.models.yolox_obb import YoloxObb

SIZE = ImageSize(416, 416)


def box(cx, cy, w, h, theta=0.0):
    return torch.tensor([cx, cy, w, h, theta], dtype=torch.float32)


# --- the polygon of a box -------------------------------------------------


def test_an_aligned_box_gives_the_expected_rectangle():
    polygon = box_to_polygon(box(100, 100, 40, 20, 0.0))
    minx, miny, maxx, maxy = polygon.bounds
    assert (minx, miny, maxx, maxy) == pytest.approx((80.0, 90.0, 120.0, 110.0))


def test_the_area_does_not_change_when_rotating():
    """Rotating neither creates nor destroys surface. If it changed, the
    rotation would be misapplied."""
    straight = box_to_polygon(box(100, 100, 60, 20, 0.0)).area
    rotated = box_to_polygon(box(100, 100, 60, 20, 0.9)).area
    assert rotated == pytest.approx(straight, rel=1e-6)


def test_rotating_90_degrees_swaps_the_sides():
    polygon = box_to_polygon(box(100, 100, 60, 20, math.pi / 2))
    minx, miny, maxx, maxy = polygon.bounds
    assert (maxx - minx) == pytest.approx(20.0, abs=1e-4)
    assert (maxy - miny) == pytest.approx(60.0, abs=1e-4)


# --- rotated NMS ----------------------------------------------------------


def test_two_equal_detections_are_reduced_to_one():
    boxes = torch.stack([box(100, 100, 60, 30), box(100, 100, 60, 30)])
    kept = rotated_nms(boxes, torch.tensor([0.9, 0.8]))
    assert kept == [0], "the most confident one survives"


def test_two_far_detections_both_survive():
    boxes = torch.stack([box(50, 50, 40, 20), box(350, 350, 40, 20)])
    assert len(rotated_nms(boxes, torch.tensor([0.9, 0.8]))) == 2


def test_rotated_nms_keeps_fanned_banknotes():
    """The reason the whole project is OBB.

    Two elongated banknotes crossed at different angles have ALIGNED
    envelopes that overlap almost entirely: a standard NMS would suppress a
    true detection. The rotated one sees that the rectangles barely touch.
    """
    # At +45 and at -45: crossed, but with the SAME square envelope. It is the
    # case that breaks the aligned NMS. Perpendicular at 0 and 90 would not
    # do: their envelopes are a horizontal bar and a vertical one, which
    # barely overlap.
    a = box(200, 200, 160, 30, math.pi / 4)
    b = box(200, 200, 160, 30, -math.pi / 4)
    boxes = torch.stack([a, b])

    envelopes = [box_to_polygon(x).envelope for x in (a, b)]
    aligned_iou = envelopes[0].intersection(envelopes[1]).area / envelopes[0].union(envelopes[1]).area
    assert aligned_iou > 0.99, "the envelopes are the same square"

    # But the rotated IoU is low, and both survive.
    assert len(rotated_nms(boxes, torch.tensor([0.9, 0.85]))) == 2


def test_no_boxes_does_not_fail():
    assert rotated_nms(torch.zeros((0, 5)), torch.zeros(0)) == []


def test_the_detection_cap_is_respected():
    boxes = torch.stack([box(20 + 40 * i, 20, 20, 10) for i in range(10)])
    scores = torch.linspace(0.9, 0.1, 10)
    assert len(rotated_nms(boxes, scores, max_detections=3)) == 3


def test_the_output_order_is_by_confidence():
    boxes = torch.stack([box(50, 50, 30, 20), box(300, 300, 30, 20)])
    kept = rotated_nms(boxes, torch.tensor([0.3, 0.95]))
    assert kept[0] == 1


# --- step to quads --------------------------------------------------------


def test_quads_come_out_normalized():
    quads = boxes_to_quads(torch.stack([box(208, 208, 100, 50)]), SIZE)
    assert len(quads) == 1
    for x, y in quads[0].points:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_the_quad_preserves_the_box_area():
    b = box(208, 208, 100, 50, 0.6)
    quad = boxes_to_quads(torch.stack([b]), SIZE)[0]
    quad_area = Polygon([(x * SIZE.width, y * SIZE.height) for x, y in quad.points]).area
    assert quad_area == pytest.approx(100 * 50, rel=1e-3)


def test_the_quad_comes_out_in_canonical_order():
    """If it were not, the metrics would compare unpaired vertices."""
    from testbank.geometry.quad import is_canonical

    quad = boxes_to_quads(torch.stack([box(208, 208, 120, 60, 0.4)]), SIZE)[0]
    assert is_canonical(quad, aspect=SIZE.width / SIZE.height)


# --- the full path --------------------------------------------------------


def _outputs():
    model = YoloxObb("nano", num_classes=1).eval()
    with torch.no_grad():
        return model(torch.randn(1, 3, SIZE.height, SIZE.width))


def test_decoding_gives_one_box_per_cell():
    outputs = _outputs()
    boxes, scores = decode_outputs(outputs, SIZE)
    expected = sum(o.distances.shape[-1] * o.distances.shape[-2] for o in outputs)
    assert boxes.shape == (expected, 5)
    assert scores.shape == (expected,)


def test_decoded_sides_are_positive():
    boxes, _ = decode_outputs(_outputs(), SIZE)
    assert torch.all(boxes[:, 2] >= 0)
    assert torch.all(boxes[:, 3] >= 0)


def test_the_decoded_angle_falls_within_half_a_turn():
    boxes, _ = decode_outputs(_outputs(), SIZE)
    assert torch.all(boxes[:, 4] >= 0.0)
    assert torch.all(boxes[:, 4] < math.pi + 1e-6)


def test_scores_are_probabilities():
    _, scores = decode_outputs(_outputs(), SIZE)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_a_freshly_created_network_barely_detects_anything():
    """The initial bias leaves p(object) = 0.01, so with threshold 0.25 almost
    nothing should come out. If it did, the bias would not be applied."""
    found = detections(_outputs(), SIZE, confidence=0.25)
    assert len(found) < 5


def test_detections_returns_predictions_usable_by_the_metrics():
    # The initial bias leaves objectness and class at 0.01, and the score is
    # their product: 1e-4. The threshold has to stay below that.
    found = detections(_outputs(), SIZE, confidence=1e-8, max_detections=10)
    assert found, "with a near-zero threshold something has to come out"
    for prediction in found:
        assert 0.0 <= prediction.score <= 1.0
        assert prediction.class_id == 0
        assert len(prediction.quad.points) == 4


def test_detections_come_out_ordered_by_confidence():
    found = detections(_outputs(), SIZE, confidence=1e-8, max_detections=20)
    scores = [p.score for p in found]
    assert scores == sorted(scores, reverse=True)


def test_a_prediction_far_outside_the_frame_has_no_geometry():
    """REGRESSION. With the COCO backbone freshly loaded and the head
    untrained, the network predicted a vertex at -0.54 normalized and `Quad`
    rejected it with a QuadError. It is emitted without geometry instead:
    `main` counts it as a false positive, the app skips it."""
    outside = box(-300, 208, 100, 50)  # center half a frame left of the image
    inside = box(208, 208, 100, 50)
    quads = boxes_to_quads(torch.stack([outside, inside]), SIZE)
    assert quads[0] is None and quads[1] is not None
