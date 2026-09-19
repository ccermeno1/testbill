"""Score prefilter + NMS (numpy; no torch)."""

from __future__ import annotations

import numpy as np

from export.postprocess import (
    any_pair_iou_at_least,
    finalize_detections_numpy,
    max_pairwise_rbox_iou,
)


def _kwargs(**overrides):
    base = {
        "nms_class_agnostic": True,
        "final_nms_iou_threshold": 0.5,
        "max_detections_per_image": 100,
        "final_nms_use_cpu": True,
        "score_threshold": 0.2,
        "per_class_score_threshold": None,
        "class_id_to_name": {1: "a", 2: "b"},
        "max_output_slots": 100,
    }
    base.update(overrides)
    return base


def test_finalize_prefilter_matches_postfilter_only_path():
    """Pre-NMS score floor must not change greedy-NMS results vs post-only filter."""
    rng = np.random.RandomState(0)
    n = 200
    boxes = rng.rand(n, 5).astype(np.float32)
    boxes[:, 2:4] = boxes[:, 2:4] * 40 + 5
    scores = rng.rand(n).astype(np.float32)
    scores[:80] = scores[:80] * 0.15  # below 0.2
    scores[80:] = scores[80:] * 0.8 + 0.2
    labels = rng.randint(1, 3, size=(n,)).astype(np.int64)

    out_new, num_new = finalize_detections_numpy(boxes, scores, labels, n, **_kwargs())

    out_all, num_all = finalize_detections_numpy(
        boxes, scores, labels, n, **_kwargs(score_threshold=0.0)
    )
    keep = out_all[:num_all, 5] >= 0.2
    ref = out_all[:num_all][keep]
    num_ref = int(keep.sum())

    assert num_new == num_ref
    if num_new:
        assert np.allclose(out_new[:num_new], ref, atol=1e-5)


def test_max_pairwise_rbox_iou_identical_and_separated() -> None:
    same = np.array([[10.0, 10.0, 8.0, 4.0, 0.0], [10.0, 10.0, 8.0, 4.0, 0.0]], dtype=np.float32)
    assert any_pair_iou_at_least(same, 0.5)
    assert max_pairwise_rbox_iou(same) > 0.99
    sep = np.array([[10.0, 10.0, 8.0, 4.0, 0.0], [80.0, 80.0, 8.0, 4.0, 0.0]], dtype=np.float32)
    assert not any_pair_iou_at_least(sep, 0.5)
    assert max_pairwise_rbox_iou(sep) < 0.05
    assert max_pairwise_rbox_iou(same[:1]) == 0.0
