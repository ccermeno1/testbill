"""The five invariants of the horizontal flip.

Derivation (verified before writing the test, because this is where it is
easy to write a test that makes a correct implementation fail):

With q = p0,p1,p2,p3 canonical clockwise and (p0,p1) the long side, M inverts
the turning direction, so the clockwise path of the flipped one is the
inverted cycle

    M(p0) -> M(p3) -> M(p2) -> M(p1) -> M(p0)

whose edges are the images of (p3,p0), (p2,p3), (p1,p2), (p0,p1). The long
ones are the second and the fourth, opened by M(p3) and M(p1). Hence
invariant 4: the anchor of the flipped one is the mirror image of a vertex
that CLOSES a long side, never of one that opens it.
"""

from __future__ import annotations

import pytest
from conftest import rotated_rects
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from testbank.geometry.quad import (
    Quad,
    canonical_order,
    canonicalize,
    flip_horizontal,
    snap,
)


def mirror(point: tuple[float, float]) -> tuple[float, float]:
    return (1.0 - point[0], point[1])


# -- invariant 1 ------------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=500)
def test_1_flip_is_an_exact_involution(quad: Quad):
    """Exact equality, not approximate. The dyadic grid is what allows it."""
    assert flip_horizontal(flip_horizontal(quad)).points == quad.points


def test_1_exact_involution_in_the_case_that_breaks_float64():
    """1-(1-0.1) gives 0.09999999999999998 without the grid."""
    quad = Quad.from_xy([0.1, 0.1, 0.7, 0.1, 0.7, 0.4, 0.1, 0.4])
    assert flip_horizontal(flip_horizontal(quad)).points == quad.points


# -- invariant 2 ------------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=300)
def test_2_preserves_the_set_of_points(quad: Quad):
    canon = canonicalize(quad)
    flipped = canonicalize(flip_horizontal(canon))
    assert set(flipped.points) == {mirror(p) for p in canon.points}


# -- invariant 3 ------------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=300)
def test_3_the_flip_inverts_the_turn_and_recanonicalization_restores_it(quad: Quad):
    canon = canonicalize(quad)
    assert canon.is_clockwise()
    raw_flip = flip_horizontal(canon)
    assert not raw_flip.is_clockwise(), "the reflection has to invert the direction"
    assert canonicalize(raw_flip).is_clockwise()


# -- invariant 4 ------------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=500)
def test_4_the_anchor_of_the_flipped_one_is_the_image_of_a_vertex_closing_a_long_side(
    quad: Quad,
):
    canon = canonicalize(quad)
    p0, p1, p2, p3 = canon.points
    anchor = canonicalize(flip_horizontal(canon)).points[0]
    assert anchor in {mirror(p1), mirror(p3)}
    assert anchor not in {mirror(p0), mirror(p2)}


def test_4_explicit_case_by_hand():
    """Aligned rectangle 0.4 x 0.2. Canonical [A,B,C,D]; flipped anchors at M(p1)."""
    quad = Quad.from_xy([0.1, 0.1, 0.5, 0.1, 0.5, 0.3, 0.1, 0.3])
    canon = canonicalize(quad)
    assert canon.points[0] == (snap(0.1), snap(0.1))
    anchor = canonicalize(flip_horizontal(canon)).points[0]
    assert anchor == mirror(canon.points[1]) == (1.0 - snap(0.5), snap(0.1))


# -- invariant 5 ------------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=300)
def test_5_repeated_determinism(quad: Quad):
    a = canonicalize(flip_horizontal(canonicalize(quad))).points
    b = canonicalize(flip_horizontal(canonicalize(quad))).points
    assert a == b


@given(rotated_rects(min_ratio=1.6), st.integers(min_value=0, max_value=255))
@settings(max_examples=400)
def test_5_determinism_under_sub_tolerance_perturbation(quad: Quad, seed: int):
    """Stability of the DECISION, not of the coordinates.

    The perturbation moves the points, so comparing values says nothing. What
    has to stay put is which vertex acts as anchor. It is conditioned on the
    tie-break margin exceeding the perturbation: below it the choice can
    legitimately change, and that boundary is unavoidable.
    """
    canon = canonicalize(quad)
    flipped = flip_horizontal(canon)
    perm, margin = canonical_order(flipped)
    jitter = 1e-5
    assume(margin > 8.0 * jitter)

    moved = []
    for i, (x, y) in enumerate(flipped.points):
        sign = 1.0 if (seed >> i) & 1 else -1.0
        moved.append((x + sign * jitter, y - sign * jitter))
    perturbed_perm, _ = canonical_order(Quad.from_xy(moved))
    assert perturbed_perm == perm


# -- guards -----------------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=200)
def test_double_canonical_flip_returns_to_the_original_canonical(quad: Quad):
    canon = canonicalize(quad)
    there = canonicalize(flip_horizontal(canon))
    back = canonicalize(flip_horizontal(there))
    assert back.points == canon.points


@given(rotated_rects())
@settings(max_examples=200)
def test_the_flip_preserves_area_and_lengths(quad: Quad):
    canon = canonicalize(quad)
    flipped = canonicalize(flip_horizontal(canon))
    assert abs(flipped.signed_area()) == pytest.approx(abs(canon.signed_area()))
    assert sorted(flipped.edge_lengths()) == pytest.approx(sorted(canon.edge_lengths()))
