"""CAMs: the map peaks where the gradient says, the model is left untouched."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn

from testbank.models.explain import (
    METHODS,
    capture_head,
    eigencam_maps,
    explain,
    gradcam_maps,
    nearest_cell_score,
    normalize,
    overlay,
)


def test_gradcam_peaks_where_gradient_and_activation_agree():
    f = torch.zeros(1, 4, 8, 8)
    f[0, 1, 2, 5] = 3.0  # one active channel at (y=2, x=5)
    f[0, 2, 6, 1] = 3.0  # another active elsewhere, but no gradient there
    f.grad = torch.zeros_like(f)
    f.grad[0, 1, 2, 5] = 1.0
    f.grad[0, 3, 6, 1] = 1.0  # gradient on a channel that is not active
    cam = gradcam_maps([f], (64, 64))
    assert cam.shape == (64, 64) and cam.max() == pytest.approx(1.0)
    y, x = np.unravel_index(int(cam.argmax()), cam.shape)
    assert abs(y - 2 * 8 - 4) <= 8 and abs(x - 5 * 8 - 4) <= 8
    assert cam[6 * 8 + 4, 1 * 8 + 4] == 0.0, "activation without gradient, or gradient without activation, is nothing"


def test_gradcam_without_any_gradient_is_all_zero_not_an_error():
    f = torch.ones(1, 2, 4, 4)
    assert not gradcam_maps([f], (16, 16)).any()


def test_eigencam_finds_the_active_region():
    f = torch.zeros(1, 6, 10, 10)
    f[0, :, 6:9, 2:5] = torch.rand(6, 3, 3) + 1.0  # a bright patch, all channels
    cam = eigencam_maps([f], (100, 100))
    y, x = np.unravel_index(int(cam.argmax()), cam.shape)
    assert 55 <= y <= 95 and 15 <= x <= 55
    assert cam.max() == pytest.approx(1.0) and cam.min() >= 0.0


def test_normalize_rectifies_and_scales():
    cam = normalize(np.array([[-1.0, 2.0], [4.0, 0.0]]))
    assert cam.tolist() == [[0.0, 0.5], [1.0, 0.0]]
    assert not normalize(np.zeros((2, 2))).any()


def test_nearest_cell_score_prefers_the_best_cell_within_the_radius():
    scores = torch.tensor([0.9, 0.2, 0.7, 0.1])
    centres = torch.tensor([[100.0, 100.0], [10.0, 10.0], [14.0, 10.0], [0.0, 0.0]])
    # Near (12, 10): cells 1 and 2 are within 5 px; the better one wins.
    assert float(nearest_cell_score(scores, centres, (12.0, 10.0), 5.0)) == pytest.approx(0.7)
    # Nothing within 1 px: the nearest cell (cell 1, at 2 px).
    assert float(nearest_cell_score(scores, centres, (11.0, 10.0), 0.5)) == pytest.approx(0.2)
    assert float(nearest_cell_score(scores, centres, (99.0, 99.0), 0.5)) == pytest.approx(0.9)


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        # Positive weights on ReLU'd features: the score grows with every
        # activation, so gradient x activation is non-negative and the map
        # cannot vanish under the ReLU for the wrong reason.
        with torch.no_grad():
            self.conv.weight.abs_()

    def forward(self, features):
        return [self.conv(f) for f in features]


class _Net(nn.Module):
    """Two 'levels' of features and a head that reads a list of them."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 3, 3, padding=1)
        self.head = _Head()

    def forward(self, x):
        a = torch.relu(self.stem(x))
        b = nn.functional.avg_pool2d(a, 2)
        return self.head([a, b])


def test_explain_runs_both_methods_and_leaves_the_model_as_it_was():
    torch.manual_seed(0)
    net = _Net().eval()
    x = torch.rand(1, 3, 16, 16)

    def target(seen):
        out = seen.outputs[0]  # (1, 1, 16, 16)
        return out[0, 0, 4, 9]

    before = [p.detach().clone() for p in net.parameters()]
    cam = explain(net, net.head, x, method="gradcam", target=target)
    assert cam.shape == (16, 16) and 0.0 <= cam.min() and cam.max() <= 1.0
    assert cam.max() == pytest.approx(1.0)
    eig = explain(net, net.head, x, method="eigencam")
    assert eig.shape == (16, 16) and eig.max() == pytest.approx(1.0)
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters()))
    assert all(p.grad is None for p in net.parameters()), "no gradient left on the weights"
    assert not net.training


def test_explain_restores_training_mode_per_module():
    net = _Net()
    net.train()
    net.head.eval()  # mixed on purpose
    explain(net, net.head, torch.rand(1, 3, 8, 8), method="eigencam")
    assert net.training and net.stem.training and not net.head.training


def test_explain_gradcam_with_a_single_cell_target_lights_up_that_cell():
    """A 1x1 head: the score at (y, x) depends on the features at (y, x)
    only, so the map must peak there."""
    torch.manual_seed(1)
    net = _Net().eval()
    x = torch.rand(1, 3, 16, 16)
    cam = explain(net, net.head, x, method="gradcam", target=lambda seen: seen.outputs[0][0, 0, 3, 12])
    y, xx = np.unravel_index(int(cam.argmax()), cam.shape)
    assert abs(y - 3) <= 1 and abs(xx - 12) <= 1


def test_explain_refuses_bad_arguments():
    net = _Net().eval()
    with pytest.raises(ValueError, match="method"):
        explain(net, net.head, torch.rand(1, 3, 8, 8), method="lime")
    with pytest.raises(ValueError, match="target"):
        explain(net, net.head, torch.rand(1, 3, 8, 8), method="gradcam")
    assert METHODS == ("gradcam", "eigencam")


def test_capture_head_removes_its_hook():
    net = _Net().eval()
    with capture_head(net.head, retain_grad=False) as seen:
        net(torch.rand(1, 3, 8, 8))
    assert len(seen.features) == 2 and len(net.head._forward_hooks) == 0


def test_overlay_keeps_the_image_size_and_colours_the_peak_red():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[2, 7] = 1.0
    out = overlay(image, cam, alpha=1.0)
    assert out.shape == image.shape
    b, _, r = out[2 * 4 + 2, 7 * 6 + 3]
    assert r > 200 and b < 50, "JET: the peak is red"
    b, _, r = out[38, 2]
    assert b > 100 and r < 50, "JET: zero is blue"


# --- through the adapter and serve ----------------------------------------


def test_the_own_adapter_explains_a_detection_and_serve_overlays_it(tmp_path):
    import cv2
    from conftest import rotated_rect_points

    from testbank.config import Config
    from testbank.geometry.quad import Quad, canonicalize
    from testbank.models.yolox_obb import YoloxObb
    from testbank.prediction import Prediction
    from testbank.serve import TrainedModel, explain_image, load_weights

    network = YoloxObb("nano", num_classes=1)
    weights = tmp_path / "w" / "best.pt"
    weights.parent.mkdir()
    torch.save({
        "model": network.state_dict(), "variant": "nano", "head": network.head_spec.to_dict(),
        "arch": "yolox_obb", "num_classes": 1, "image_size": 64, "epoch": 0,
    }, weights)
    base = Config()
    config = base.model_copy(update={"detector": base.detector.model_copy(update={"image_size": 64})})
    model = TrainedModel(directory=tmp_path, name="t", detector="yolox-obb-nano", weights=weights, config=config)
    image = np.random.default_rng(0).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    quad = canonicalize(Quad.from_xy(rotated_rect_points(0.5, 0.5, 0.25, 2.0, 0.3)))
    loaded = load_weights(model)
    heat = explain_image(model, image, Prediction(quad=quad, score=0.5), method="gradcam", loaded=loaded)
    assert heat.shape == image.shape and heat.dtype == np.uint8
    eig = explain_image(model, image, None, method="eigencam", loaded=loaded)
    assert eig.shape == image.shape
    assert not np.array_equal(heat, eig)
    assert cv2.absdiff(heat, image).mean() > 0, "something was blended"


def test_an_adapter_without_feature_maps_gives_no_map(tmp_path):
    from testbank.config import Config
    from testbank.detectors.base import BaseDetector, register
    from testbank.serve import TrainedModel, explain_image

    @register
    class Opaque(BaseDetector):
        name = "test-opaque"

    model = TrainedModel(directory=tmp_path, name="o", detector="test-opaque", weights=tmp_path / "w.pt", config=Config())
    assert explain_image(model, np.zeros((8, 8, 3), np.uint8), None, method="eigencam") is None
