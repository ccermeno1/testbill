"""
ONNX export and the client-side pre/post-processing.

These cover what actually breaks a deployment: the exported graph and the numpy
post-processing must reproduce the PyTorch detections, corner convention included. Getting
the angle sign wrong gives plausible-looking boxes that are rotated the wrong way.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "ppyoloer_mps"))

from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r
from ppyoloer_mps.ppyoloe_obb.boxes import rbox2poly
from ppyoloer_mps.ppyoloe_obb.ops import batched_postprocess


def _onnx_example():
    return pytest.importorskip("onnx_example", reason="needs the repo package on sys.path")


def test_numpy_rbox2poly_matches_torch():
    """The client's polygon conversion must match the model's."""
    mod = _onnx_example()
    rng = np.random.default_rng(0)
    boxes = np.concatenate(
        [
            rng.uniform([0, 0, 10, 10, -np.pi], [500, 500, 300, 300, np.pi], (200, 5)),
            np.array([[100.0, 100.0, 50.0, 20.0, 0.0], [0.0, 0.0, 10.0, 10.0, np.pi / 4]]),
        ]
    ).astype(np.float32)
    got = mod.rbox2poly(boxes)
    ref = rbox2poly(torch.from_numpy(boxes)).numpy()
    assert np.abs(got - ref).max() < 1e-3


def test_numpy_rotated_nms_matches_torch():
    mod = _onnx_example()
    from ppyoloer_mps.ppyoloe_obb.ops import rotated_nms as nms_t

    boxes = np.array(
        [
            [100.0, 100.0, 60.0, 30.0, 0.0],
            [102.0, 101.0, 60.0, 30.0, 0.05],   # near duplicate
            [300.0, 300.0, 60.0, 30.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep_np = mod.rotated_nms(boxes, scores, 0.1)
    keep_t = nms_t(torch.from_numpy(boxes), torch.from_numpy(scores), 0.1).tolist()
    assert keep_np == keep_t == [0, 2]


def test_preprocess_shapes_and_scale(tmp_path):
    mod = _onnx_example()
    import cv2

    path = tmp_path / "im.jpg"
    cv2.imwrite(str(path), np.full((200, 400, 3), 120, np.uint8))
    x, sx, sy = mod.preprocess(str(path), img_size=640)
    assert x.shape[0] == 1 and x.shape[1] == 3
    assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0
    assert x.dtype == np.float32 and 0.0 <= x.min() and x.max() <= 255.0
    assert pytest.approx(sx, abs=1e-3) == 1.6 and pytest.approx(sy, abs=1e-3) == 1.6


@pytest.mark.slow
def test_onnx_export_matches_torch(tmp_path):
    """The exported graph must give the same detections as the PyTorch model."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    mod = _onnx_example()
    from ppyoloer_mps.export_onnx import ExportWrapper

    torch.manual_seed(0)
    model = build_ppyoloe_r(num_classes=1, size="s").eval()
    wrapper = ExportWrapper(model).eval()
    out = str(tmp_path / "m.onnx")
    example = torch.zeros(1, 3, 320, 320)
    with torch.no_grad():
        torch.onnx.export(wrapper, example, out, input_names=["images_rgb_0_255"],
                          output_names=["scores", "boxes"], opset_version=16, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(out))

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (1, 3, 320, 320)).astype(np.float32)
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    s_onnx, b_onnx = sess.run(None, {sess.get_inputs()[0].name: x})
    with torch.no_grad():
        s_t, b_t = wrapper(torch.from_numpy(x))
    assert np.abs(s_onnx - s_t.numpy()).max() < 1e-4
    assert np.abs(b_onnx - b_t.numpy()).max() < 1e-2

    # and the numpy post-processing must match PyTorch's on those same outputs
    rb_np, _, sc_np, _ = mod.postprocess(s_onnx, b_onnx, 1.0, 1.0, score_thr=0.05, nms_iou=0.1)
    det = batched_postprocess(s_t, b_t, score_threshold=0.05, nms_threshold=0.1)[0]
    assert len(sc_np) == len(det["scores"])
    if len(sc_np):
        assert np.abs(np.sort(sc_np) - np.sort(det["scores"].numpy())).max() < 1e-5
