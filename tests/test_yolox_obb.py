"""Own OBB head on YOLOX: architecture and angle representation."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="the own candidate needs torch")

from testbank.models.yolox_obb import (
    STRIDES,
    VARIANTS,
    YoloxObb,
    decode_angle,
    encode_angle,
)

IMAGE = 416


# --- the angle ------------------------------------------------------------


def test_theta_and_theta_plus_180_give_the_same_encoding():
    """The property that motivates the whole representation.

    A rectangle rotated t and another rotated t+180 are the SAME rectangle.
    With `(sin 2t, cos 2t)` they fall on the same point of the circle, so the
    ambiguity disappears by construction instead of being corrected later.
    """
    theta = torch.tensor([0.1, 1.0, 2.5, 3.0])
    a = encode_angle(theta)
    b = encode_angle(theta + math.pi)
    assert torch.allclose(a, b, atol=1e-6)


def test_encoding_and_decoding_returns_the_angle():
    theta = torch.tensor([0.0, 0.3, 1.2, 2.0, 3.0])
    assert torch.allclose(decode_angle(encode_angle(theta)), theta, atol=1e-6)


def test_the_decoded_angle_always_falls_within_half_a_turn():
    """Beyond pi it repeats: there is nothing to distinguish there."""
    theta = torch.linspace(-10.0, 10.0, 200)
    out = decode_angle(encode_angle(theta))
    assert torch.all(out >= 0.0)
    assert torch.all(out < math.pi + 1e-6)


def test_the_encoding_is_continuous_when_crossing_zero():
    """When crossing 0/180 the encoded distance is still small.

    It is the reason for not regressing theta directly. A banknote at 179.4
    degrees and another at 0.6 are 1.2 degrees apart for real, but in raw
    theta they are 178.8 apart: the model would receive a huge gradient for
    being almost exactly right. Here the encoded distance is proportional to
    the real one.
    """
    a, b = math.pi - 0.01, 0.01
    encoded_distance = float(
        torch.norm(encode_angle(torch.tensor([a])) - encode_angle(torch.tensor([b])))
    )
    raw_distance = abs(a - b)

    # 0.02 rad of real difference -> 0.04 on the doubled circle (~2 * 0.02).
    assert encoded_distance == pytest.approx(0.04, abs=0.005)
    # And against the 3.12 rad a direct regression on theta would see.
    assert raw_distance > 3.0
    assert encoded_distance < raw_distance / 50


def test_truly_different_angles_do_not_collide():
    """The doubling joins t with t+180, and NOTHING else: if it joined more,
    the model could not tell a lying banknote from a standing one."""
    a = encode_angle(torch.tensor([0.0]))
    b = encode_angle(torch.tensor([math.pi / 2]))
    assert not torch.allclose(a, b, atol=0.1)


# --- the architecture -----------------------------------------------------


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_each_variant_builds_and_runs(variant):
    model = YoloxObb(variant, num_classes=1).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, IMAGE, IMAGE))
    assert len(outputs) == 3


def test_unknown_variant_says_which_exist():
    with pytest.raises(ValueError, match="unknown variant"):
        YoloxObb("giant")


def test_the_shapes_follow_the_strides():
    model = YoloxObb("nano", num_classes=1).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(2, 3, IMAGE, IMAGE))
    for output, stride in zip(outputs, STRIDES):
        side = IMAGE // stride
        assert output.stride == stride
        assert output.distances.shape == (2, 4, side, side)
        assert output.angle.shape == (2, 2, side, side)
        assert output.objectness.shape == (2, 1, side, side)
        assert output.classes.shape == (2, 1, side, side)


def test_nano_fits_on_a_phone():
    """Under a million parameters. It is the reason for choosing this variant."""
    assert YoloxObb("nano", 1).parameter_count() < 1_000_000


def test_the_variants_grow_in_order():
    counts = [YoloxObb(v, 1).parameter_count() for v in ("nano", "tiny", "small")]
    assert counts == sorted(counts)
    # nano has to be MUCH smaller, not a bit: it is separable convolution, not
    # only fewer channels.
    assert counts[1] > 4 * counts[0]


def test_distances_are_positive():
    """They are distances to the cell center: negative ones mean nothing."""
    model = YoloxObb("nano", 1).eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, IMAGE, IMAGE))
    for output in outputs:
        assert torch.all(output.distances >= 0)


def test_at_start_it_predicts_almost_nothing():
    """Without the initial bias, the network predicts object in every cell and
    the gradient of thousands of false positives dominates the first iterations."""
    model = YoloxObb("nano", 1).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, IMAGE, IMAGE))
    probability = torch.sigmoid(outputs[0].objectness).mean()
    assert probability < 0.05


def test_the_head_shares_weights_across_levels():
    """With ~450 images, three independent heads triple the parameters of the
    part that overfits most easily."""
    model = YoloxObb("nano", 1)
    # A single set of branches, not one per level: the stems ARE per level
    # because each receives a different number of channels.
    assert len(model.head.stems) == len(STRIDES)
    assert isinstance(model.head.cls_pred, torch.nn.Conv2d)
    assert model.head.cls_pred.out_channels == 1
    assert model.head.angle_pred.out_channels == 2


def test_the_gradient_reaches_every_branch():
    """If a branch were disconnected, it would train silently without learning."""
    model = YoloxObb("nano", 1)
    outputs = model(torch.randn(1, 3, 128, 128))
    loss = sum(
        o.distances.sum() + o.angle.sum() + o.objectness.sum() + o.classes.sum()
        for o in outputs
    )
    loss.backward()
    for name in ("reg_pred", "angle_pred", "obj_pred", "cls_pred"):
        layer = getattr(model.head, name)
        assert layer.weight.grad is not None, name
        assert torch.any(layer.weight.grad != 0), name
