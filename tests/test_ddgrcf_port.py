"""The DDGRCF/YOLOX_OBB port, the polygon IoU and the weight loaders.

The tensor-by-tensor comparison against THEIR actually built model was done
outside the suite (it needs their clone and a stub of their operators): 426
identical keys and shapes, `strict=True` loads, and with their weights the
output matches theirs to 3e-5 px. Here what can be anchored without the clone
is anchored: the parameter count that came out of that comparison, the names
their checkpoint expects, and that the loaders fail loudly when something
does not fit.
"""

from __future__ import annotations

import math

import pytest
import torch
from shapely.geometry import Polygon

from testbank.models.ddgrcf import DdgrcfYoloxObb
from testbank.models.ddgrcf import load_pretrained as load_ddgrcf
from testbank.models.losses import regularize_angle_ddgrcf
from testbank.models.overlap import box_corners, pairwise_rotated_iou, rotated_iou
from testbank.models.pretrained import load_megvii_yolox, remap_megvii_key
from testbank.models.yolox_obb import YoloxObb

torch.manual_seed(0)

# --- the port ----------------------------------------------------------------


def test_the_port_has_exactly_the_parameters_of_their_model():
    """8,051,797: the number THEIR `Model` built with their yamls gave."""
    assert DdgrcfYoloxObb(num_classes=1).parameter_count() == 8_051_797


def test_the_keys_follow_their_naming_scheme():
    keys = set(DdgrcfYoloxObb(num_classes=1).state_dict())
    assert "model.0.conv.weight" in keys
    assert "model.2.m.0.cv1.conv.weight" in keys
    # The head stems go WITHOUT Sequential: `round(2 * 0.33) = 1`.
    assert "model.27.conv.weight" in keys and "model.27.0.conv.weight" not in keys
    assert "model.33.cls_preds.0.bias" in keys and "model.33.reg_preds.2.weight" in keys


def test_the_port_outputs_three_levels_with_their_regression_and_angle():
    outputs = DdgrcfYoloxObb(num_classes=1)(torch.zeros(1, 3, 128, 128))
    assert [o.stride for o in outputs] == [8, 16, 32]
    for o in outputs:
        assert o.regression == "yolox" and o.angle_mode == "radians"
        assert o.distances.shape[1] == 4 and o.angle.shape[1] == 1
        assert o.objectness is not None and o.classes.shape[1] == 1


def test_loading_their_dota_weights_skips_only_the_class_layer():
    """Their weights have 15 classes; the port, one. Everything else, strict."""
    fifteen = DdgrcfYoloxObb(num_classes=15).state_dict()
    model = DdgrcfYoloxObb(num_classes=1)
    skipped = load_ddgrcf(model, _saved(fifteen))
    assert skipped and all("cls_preds" in k for k in skipped)
    assert torch.equal(model.state_dict()["model.0.conv.weight"], fifteen["model.0.conv.weight"])


def test_a_checkpoint_that_does_not_cover_the_port_is_an_error():
    partial = {k: v for k, v in DdgrcfYoloxObb(num_classes=1).state_dict().items() if "model.1" not in k}
    with pytest.raises(RuntimeError, match="does not cover the port"):
        load_ddgrcf(DdgrcfYoloxObb(num_classes=1), _saved(partial))


def _saved(state):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "w.pth"
    torch.save({"model": state}, path)
    return path


# --- the exact IoU in torch ----------------------------------------------------


def _random_boxes(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack(
        [
            torch.rand(n, generator=g) * 300 + 50,
            torch.rand(n, generator=g) * 300 + 50,
            20 + torch.rand(n, generator=g) * 150,
            10 + torch.rand(n, generator=g) * 80,
            (torch.rand(n, generator=g) - 0.5) * math.pi,
        ],
        -1,
    )


def _shapely(a, b):
    out = []
    for x, y in zip(box_corners(a).tolist(), box_corners(b).tolist()):
        pa, pb = Polygon(x), Polygon(y)
        inter = pa.intersection(pb).area
        union = pa.area + pb.area - inter
        out.append(inter / union if union > 0 else 0.0)
    return torch.tensor(out)


def test_the_polygon_iou_matches_shapely():
    a, b = _random_boxes(500, 1), _random_boxes(500, 2)
    b[:, :2] = a[:, :2] + (torch.rand(500, 2) - 0.5) * 80  # so they overlap
    assert torch.allclose(rotated_iou(a, b), _shapely(a, b), atol=1e-5)


def test_iou_edge_cases():
    b = _random_boxes(20, 3)
    assert torch.allclose(rotated_iou(b, b), torch.ones(20), atol=1e-5)
    far = b.clone()
    far[:, 0] += 1000
    assert rotated_iou(b, far).max().item() == 0.0
    inside = b.clone()
    inside[:, 2:4] *= 0.5
    assert torch.allclose(rotated_iou(b, inside), torch.full((20,), 0.25), atol=1e-5)


def test_the_iou_is_differentiable_with_finite_gradient():
    p = _random_boxes(50, 4).requires_grad_(True)
    (1 - rotated_iou(p, _random_boxes(50, 5))).sum().backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_the_pairwise_version_matches_the_direct_one():
    a, b = _random_boxes(6, 6), _random_boxes(3, 7)
    pw = pairwise_rotated_iou(a, b)
    for i in range(6):
        for j in range(3):
            assert pw[i, j].item() == pytest.approx(rotated_iou(a[i : i + 1], b[j : j + 1]).item(), abs=1e-6)


# --- their angle convention ----------------------------------------------------


def test_regularizing_the_angle_leaves_the_same_rectangle_in_their_range():
    boxes = _random_boxes(200, 8)
    reg = regularize_angle_ddgrcf(boxes)
    assert (reg[:, 4] > -math.pi / 4 - 1e-6).all() and (reg[:, 4] <= math.pi / 4 + 1e-6).all()
    assert torch.allclose(rotated_iou(boxes, reg), torch.ones(200), atol=1e-4), "it is the same rectangle"


# --- Megvii's COCO into the own head -------------------------------------------


def test_the_megvii_mapping_covers_backbone_and_neck_and_discards_their_head():
    """Their checkpoint is simulated by renaming ours the other way round: the
    round trip has to cover the 354 backbone+neck tensors and none of the head."""
    model = YoloxObb("small")
    from testbank.models.pretrained import _MEGVII_NECK

    inverse = {v: k for k, v in _MEGVII_NECK.items()}
    fake = {}
    for key, value in model.state_dict().items():
        if key.startswith("backbone."):
            fake["backbone.backbone." + key[len("backbone.") :]] = value * 0 + 7.0
        elif key.startswith("neck."):
            module, _, tail = key[len("neck.") :].partition(".")
            fake[f"backbone.{inverse[module]}.{tail}"] = value * 0 + 7.0
    fake["head.cls_preds.0.weight"] = torch.zeros(80, 128, 1, 1)  # their COCO head
    report = load_megvii_yolox(model, _saved(fake))
    assert report["loaded"] == 354 and report["skipped_head"] == 1
    assert (model.state_dict()["neck.p4.conv1.conv.weight"] == 7.0).all()


def test_loading_the_wrong_variant_is_an_error_and_not_half_measures():
    small = YoloxObb("small").state_dict()
    fake = {"backbone.backbone." + k[len("backbone.") :]: v for k, v in small.items() if k.startswith("backbone.")}
    with pytest.raises(RuntimeError, match="does not fit"):
        load_megvii_yolox(YoloxObb("nano"), _saved(fake))


def test_an_unknown_neck_key_is_not_swallowed():
    with pytest.raises(KeyError, match="unknown neck module"):
        remap_megvii_key("backbone.made_up_module.conv.weight")
    assert remap_megvii_key("head.obj_preds.0.bias") is None
