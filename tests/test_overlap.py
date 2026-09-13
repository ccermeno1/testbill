"""KLD and ProbIoU: properties, and numerical fidelity to the original where possible.

The KLD is checked against the formula of the fork `buzhidaoshenme/YOLOX-OBB`
(Apache-2.0), transcribed here with attribution: it is the implementation
the recipe we are reproducing uses, so "it gives the same" is exactly the
claim to anchor. It takes the angle in DEGREES; ours in radians.

ProbIoU has NO code reference: Ultralytics is AGPL and has not been read. It
is checked against the properties the paper guarantees and against an
independent matrix implementation of the Bhattacharyya distance, which only
uses `torch.linalg` and shares no line with the production one.
"""

from __future__ import annotations

import math

import pytest
import torch

from testbank.models.overlap import (
    KLD_VARIANCE_DIVISOR,
    PROBIOU_VARIANCE_DIVISOR,
    bhattacharyya_distance,
    box_to_gaussian,
    kld_divergence,
    kld_loss,
    pairwise_kld_loss,
    pairwise_probiou,
    probiou,
)

torch.manual_seed(0)


def _boxes(n, *, seed):
    g = torch.Generator().manual_seed(seed)
    cx = torch.rand(n, generator=g) * 400
    cy = torch.rand(n, generator=g) * 400
    w = 20 + torch.rand(n, generator=g) * 200
    h = 10 + torch.rand(n, generator=g) * 100
    theta = (torch.rand(n, generator=g) - 0.5) * math.pi
    return torch.stack((cx, cy, w, h, theta), dim=-1)


# --- KLD: fidelity to the fork -------------------------------------------


def _fork_kld(pred, target, taf=1.0):
    """`KLD_loss.kld_loss` of buzhidaoshenme/YOLOX-OBB (Apache-2.0), transcribed
    as is except for the variable names. Angles in DEGREES."""
    delta_x = pred[:, 0] - target[:, 0]
    delta_y = pred[:, 1] - target[:, 1]
    pre = math.pi * pred[:, 4] / 180.0
    tgt = math.pi * target[:, 4] / 180.0
    delta = pre - tgt
    kld = (
        0.5
        * (
            4 * (delta_x * torch.cos(tgt) + delta_y * torch.sin(tgt)) ** 2 / target[:, 2] ** 2
            + 4 * (delta_y * torch.cos(tgt) - delta_x * torch.sin(tgt)) ** 2 / target[:, 3] ** 2
        )
        + 0.5
        * (
            pred[:, 3] ** 2 / target[:, 2] ** 2 * torch.sin(delta) ** 2
            + pred[:, 2] ** 2 / target[:, 3] ** 2 * torch.sin(delta) ** 2
            + pred[:, 3] ** 2 / target[:, 3] ** 2 * torch.cos(delta) ** 2
            + pred[:, 2] ** 2 / target[:, 2] ** 2 * torch.cos(delta) ** 2
        )
        + 0.5
        * (
            torch.log(target[:, 3] ** 2 / pred[:, 3] ** 2)
            + torch.log(target[:, 2] ** 2 / pred[:, 2] ** 2)
        )
        - 1.0
    )
    return 1 - 1 / (taf + torch.log(kld + 1))


def test_kld_matches_the_fork_formula():
    p, t = _boxes(500, seed=1), _boxes(500, seed=2)
    ours = kld_loss(p, t)
    in_degrees = lambda b: torch.cat((b[:, :4], torch.rad2deg(b[:, 4:5])), dim=1)
    theirs = _fork_kld(in_degrees(p), in_degrees(t))
    assert torch.allclose(ours, theirs, atol=1e-5), (ours - theirs).abs().max()


def test_kld_is_zero_for_the_same_box():
    b = _boxes(50, seed=3)
    assert torch.allclose(kld_divergence(b, b), torch.zeros(50), atol=1e-5)
    assert torch.allclose(kld_loss(b, b), torch.zeros(50), atol=1e-5)


def test_kld_is_not_symmetric_and_it_is_known():
    """It is not a bug: it is the KL. The test exists so nobody uses it as a
    distance without knowing."""
    p, t = _boxes(50, seed=4), _boxes(50, seed=5)
    assert not torch.allclose(kld_divergence(p, t), kld_divergence(t, p))


def test_kld_grows_with_the_center_error():
    t = torch.tensor([[100.0, 100.0, 80.0, 40.0, 0.3]])
    values = [kld_loss(t + torch.tensor([[dx, 0, 0, 0, 0]]), t).item() for dx in (0, 5, 20, 80)]
    assert values == sorted(values) and values[0] < 1e-5


def test_kld_is_invariant_to_a_180_degree_rotation():
    """`(w, h, t)` and `(w, h, t + pi)` are the same rectangle. So is the
    Gaussian: the covariance is quadratic in sine and cosine."""
    t = _boxes(50, seed=6)
    rotated = t.clone()
    rotated[:, 4] += math.pi
    assert torch.allclose(kld_loss(rotated, t), torch.zeros(50), atol=1e-5)


def test_kld_is_bounded_in_zero_one():
    p, t = _boxes(300, seed=7), _boxes(300, seed=8)
    v = kld_loss(p, t)
    assert (v >= 0).all() and (v < 1).all()


# --- ProbIoU: properties and matrix cross-check ----------------------------


def _matrix_bhattacharyya(p, t):
    """Independent reference: the B_D with matrices, without the `a,b,c` expansion."""
    def cov(b):
        _, a, bb, c = box_to_gaussian(b, variance_divisor=PROBIOU_VARIANCE_DIVISOR)
        return torch.stack((torch.stack((a, c), -1), torch.stack((c, bb), -1)), -2)

    s1, s2 = cov(p), cov(t)
    s = 0.5 * (s1 + s2)
    d = (p[:, :2] - t[:, :2]).unsqueeze(-1)
    term1 = 0.125 * (d.transpose(1, 2) @ torch.linalg.inv(s) @ d).squeeze(-1).squeeze(-1)
    term2 = 0.5 * torch.log(
        torch.linalg.det(s) / torch.sqrt(torch.linalg.det(s1) * torch.linalg.det(s2))
    )
    return term1 + term2


def test_bhattacharyya_matches_the_matrix_version():
    p, t = _boxes(400, seed=9), _boxes(400, seed=10)
    ours = bhattacharyya_distance(p, t)
    ref = _matrix_bhattacharyya(p, t).clamp(min=1e-7, max=100.0)
    assert torch.allclose(ours, ref, atol=1e-4, rtol=1e-4), (ours - ref).abs().max()


def test_probiou_is_one_for_the_same_box():
    b = _boxes(50, seed=11)
    assert torch.allclose(probiou(b, b), torch.ones(50), atol=1e-3)


def test_probiou_is_symmetric():
    p, t = _boxes(100, seed=12), _boxes(100, seed=13)
    assert torch.allclose(probiou(p, t), probiou(t, p), atol=1e-6)


def test_probiou_drops_when_moving_away_and_is_in_zero_one():
    t = torch.tensor([[100.0, 100.0, 80.0, 40.0, 0.3]])
    values = [probiou(t + torch.tensor([[dx, 0, 0, 0, 0]]), t).item() for dx in (0, 5, 20, 80, 300)]
    assert values == sorted(values, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in values)


def test_probiou_is_invariant_to_a_180_degree_rotation():
    t = _boxes(50, seed=14)
    rotated = t.clone()
    rotated[:, 4] += math.pi
    assert torch.allclose(probiou(rotated, t), torch.ones(50), atol=1e-3)


def test_a_rotated_square_box_changes_nothing():
    """A square is the same at any angle, and its Gaussian is isotropic: both
    measures have to see it. It is the case that breaks the `(w, h, theta)`
    representation and that the Gaussians solve on their own."""
    square = torch.tensor([[100.0, 100.0, 60.0, 60.0, 0.0]])
    rotated = torch.tensor([[100.0, 100.0, 60.0, 60.0, 0.7]])
    assert probiou(rotated, square).item() == pytest.approx(1.0, abs=1e-3)
    assert kld_loss(rotated, square).item() == pytest.approx(0.0, abs=1e-5)


# --- covariance conventions, which cannot be mixed ------------------------


def test_the_two_variance_conventions_are_those_of_their_papers():
    assert KLD_VARIANCE_DIVISOR == 4.0
    assert PROBIOU_VARIANCE_DIVISOR == 12.0


def test_box_to_gaussian_requires_the_divisor():
    with pytest.raises(TypeError):
        box_to_gaussian(_boxes(1, seed=0))  # type: ignore[call-arg]


# --- gradients and pairs ---------------------------------------------------


def test_gradients_are_finite_even_in_the_square_case():
    p = _boxes(30, seed=15).requires_grad_(True)
    t = _boxes(30, seed=16)
    t[:5, 2] = t[:5, 3]  # exact squares
    for fn in (kld_loss, lambda a, b: 1 - probiou(a, b)):
        p.grad = None
        fn(p, t).sum().backward()
        assert torch.isfinite(p.grad).all()


def test_the_pairwise_versions_match_the_one_to_one_ones():
    a, b = _boxes(7, seed=17), _boxes(4, seed=18)
    pk, pp = pairwise_kld_loss(a, b), pairwise_probiou(a, b)
    assert pk.shape == pp.shape == (7, 4)
    for i in range(7):
        for j in range(4):
            assert pk[i, j].item() == pytest.approx(kld_loss(a[i : i + 1], b[j : j + 1]).item(), abs=1e-6)
            assert pp[i, j].item() == pytest.approx(probiou(a[i : i + 1], b[j : j + 1]).item(), abs=1e-6)
