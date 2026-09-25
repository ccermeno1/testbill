"""
Rotated box geometry and NMS.

Rotated IoU is checked against Shapely, an independent implementation. The assigner, the
metrics and the NMS all sit on top of it, so a bug here spreads everywhere without
announcing itself.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ppyoloer_mps.ppyoloe_obb.boxes import box2corners, poly2rbox, probiou, rbox2poly, rotated_iou
from ppyoloer_mps.ppyoloe_obb.ops import batched_postprocess, rotated_nms

from conftest import shapely_iou


def test_rbox_poly_roundtrip():
    rb = torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.3]])
    back = poly2rbox(rbox2poly(rb))
    assert torch.allclose(back[:, :2], rb[:, :2], atol=1e-3)
    assert torch.allclose(back[:, 2:4], rb[:, 2:4], atol=1e-3)
    assert abs(float(back[0, 4]) - 0.3) < 1e-3


def test_poly2rbox_survives_mirroring_and_reordering():
    """The derived rbox must describe the SAME rectangle as the input polygon, even when it
    comes mirrored or with the vertices rolled: a flip reverses polygon orientation."""
    from shapely.geometry import Polygon

    rng = np.random.default_rng(0)
    transforms = {
        "identity": lambda p: p,
        "horizontal mirror": lambda p: (p.reshape(4, 2) * [-1, 1] + [640, 0]).reshape(8),
        "reversed order": lambda p: p.reshape(4, 2)[::-1].reshape(8),
        "rolled vertices": lambda p: np.roll(p.reshape(4, 2), 1, 0).reshape(8),
    }
    for name, transform in transforms.items():
        for _ in range(50):
            rb = rng.uniform([100, 100, 20, 20, -math.pi], [500, 500, 300, 300, math.pi]).astype(np.float32)
            poly = rbox2poly(torch.tensor(rb[None])).numpy()[0]
            q = np.ascontiguousarray(transform(poly), dtype=np.float32)
            back = rbox2poly(poly2rbox(torch.from_numpy(q[None])))[0].numpy()
            pa, pb = Polygon(q.reshape(4, 2)), Polygon(back.reshape(4, 2))
            inter = pa.intersection(pb).area
            iou = inter / (pa.area + pb.area - inter)
            assert iou > 0.99, f"{name}: IoU {iou:.3f} between the polygon and its rbox"


def test_box2corners_is_rectangle():
    corners = box2corners(torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]]))[0]
    sides = (corners - torch.roll(corners, -1, dims=0)).norm(dim=-1)
    assert pytest.approx(float(sides[0]), abs=1e-4) == 2.0
    assert pytest.approx(float(sides[1]), abs=1e-4) == 4.0


def test_rotated_iou_known_cases():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    assert pytest.approx(float(rotated_iou(a, a)[0, 0]), abs=1e-5) == 1.0
    assert float(rotated_iou(a, torch.tensor([[100.0, 100.0, 10.0, 10.0, 0.0]]))[0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert pytest.approx(float(rotated_iou(a, torch.tensor([[0.0, 0.0, 5.0, 5.0, 0.0]]))[0, 0]), abs=1e-4) == 0.25
    rot90 = torch.tensor([[0.0, 0.0, 10.0, 10.0, math.pi / 2]])
    assert pytest.approx(float(rotated_iou(a, rot90)[0, 0]), abs=1e-4) == 1.0


def test_rotated_iou_matches_reference():
    pytest.importorskip("shapely")
    rng = np.random.default_rng(0)
    boxes1 = rng.uniform([0, 0, 5, 5, -math.pi], [200, 200, 80, 80, math.pi], (6, 5)).astype(np.float32)
    boxes2 = rng.uniform([0, 0, 5, 5, -math.pi], [200, 200, 80, 80, math.pi], (7, 5)).astype(np.float32)
    got = rotated_iou(torch.tensor(boxes1), torch.tensor(boxes2)).numpy()
    ref = np.array([[shapely_iou(a, b) for b in boxes2] for a in boxes1])
    assert np.abs(got - ref).max() < 1e-4


def test_rotated_iou_chunking_is_transparent():
    """The chunking that caps VRAM must not change the result."""
    from ppyoloer_mps.ppyoloe_obb import boxes as B

    rng = np.random.default_rng(1)
    a = torch.tensor(rng.uniform(0, 300, (7, 5)).astype(np.float32))
    b = torch.tensor(rng.uniform(0, 300, (40, 5)).astype(np.float32))
    old = B.MAX_PAIRS
    try:
        B.MAX_PAIRS = 10_000
        full = rotated_iou(a, b)
        B.MAX_PAIRS = 41          # force several chunks
        chunked = rotated_iou(a, b)
    finally:
        B.MAX_PAIRS = old
    assert torch.allclose(full, chunked, atol=1e-6)


def test_probiou_zero_for_identical_boxes():
    box = torch.tensor([[10.0, 10.0, 20.0, 8.0, 0.4]])
    assert float(probiou(box, box)) < 0.05


def test_probiou_finite_for_degenerate_boxes():
    """A degenerate box (side near 0) must not produce NaN: it used to sink training."""
    boxes = torch.tensor([[10.0, 10.0, 0.0, 8.0, 0.0],
                          [10.0, 10.0, 20.0, 1e-6, 0.3],
                          [0.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([[10.0, 10.0, 20.0, 8.0, 0.0]] * 3)
    assert torch.isfinite(probiou(boxes, target)).all()


def test_rotated_nms_suppresses_overlaps():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0],
                          [0.5, 0.5, 10.0, 10.0, 0.0],   # near duplicate, gets suppressed
                          [100.0, 100.0, 10.0, 10.0, 0.0]])
    keep = rotated_nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def test_postprocess_rescales_to_original():
    scores = torch.zeros(1, 1, 2)
    scores[0, 0, 0] = 0.9
    rboxes = torch.tensor([[[100.0, 100.0, 40.0, 20.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]]])
    det = batched_postprocess(scores, rboxes, score_threshold=0.5,
                              scale_factor=torch.tensor([[0.5, 0.5]]))[0]
    assert len(det) == 1
    assert pytest.approx(float(det["rboxes"][0, 0]), abs=1e-4) == 200.0
    assert pytest.approx(float(det["rboxes"][0, 2]), abs=1e-4) == 80.0
