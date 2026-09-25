"""Model input/output shapes, and running on an accelerator."""
from __future__ import annotations

import pytest
import torch

from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r
from ppyoloer_mps.ppyoloe_obb.engine import resolve_device


def _num_anchors(h: int, w: int) -> int:
    return sum((h // s) * (w // s) for s in (32, 16, 8))


def test_model_eval_shapes(model):
    with torch.no_grad():
        scores, rboxes = model(torch.zeros(1, 3, 320, 320))
    n = _num_anchors(320, 320)
    assert scores.shape == (1, 1, n)
    assert rboxes.shape == (1, n, 5)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0


def test_model_train_shapes(model):
    model.train()
    try:
        cls, dist, angle, anchors, num_list, strides = model(torch.zeros(2, 3, 320, 320))
        n = _num_anchors(320, 320)
        assert cls.shape == (2, n, 1)
        assert dist.shape == (2, n, 4)
        assert angle.shape == (2, n, 91)
        assert anchors.shape == (1, n, 2)
        assert sum(num_list) == n
        assert strides.shape == (1, n, 1)
    finally:
        model.eval()


def test_model_accepts_non_square_input(model):
    with torch.no_grad():
        scores, rboxes = model(torch.zeros(1, 3, 320, 640))
    n = _num_anchors(320, 640)
    assert scores.shape[-1] == n and rboxes.shape[1] == n


def test_angle_projection_is_fixed(model):
    """The DFL angle projection is a constant, not a trainable parameter."""
    w = model.yolo_head.angle_proj_conv.weight
    assert not w.requires_grad
    assert pytest.approx(float(w.max()), abs=1e-5) == pytest.approx(3.14159 / 2, abs=1e-4)


def test_build_sizes_have_expected_scale():
    small = sum(p.numel() for p in build_ppyoloe_r(1, "s").parameters())
    medium = sum(p.numel() for p in build_ppyoloe_r(1, "m").parameters())
    assert 7e6 < small < 9e6          # PP-YOLOE-R-s is about 8.1 M params
    assert medium > small


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


@pytest.mark.parametrize("device_name", ["cuda", "mps"])
def test_forward_on_accelerator(device_name):
    """Skipped when the accelerator is not there; on a Mac this covers MPS."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    device = torch.device(device_name)
    m = build_ppyoloe_r(num_classes=1, size="s").to(device).eval()
    with torch.no_grad():
        scores, rboxes = m(torch.zeros(1, 3, 320, 320, device=device))
    assert scores.device.type == device_name
    assert torch.isfinite(rboxes).all()
