from __future__ import annotations

import itertools
import math

import pytest
from conftest import rotated_rect_points, rotated_rects
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from testbank.geometry.quad import (
    LONG_SIDE_TIE_TOL,
    CoordinateRangeWarning,
    Quad,
    QuadError,
    QuadShapeWarning,
    canonical_order,
    canonicalize,
    is_canonical,
    snap,
)

# -- construction and validation --------------------------------------------


def test_rejects_wrong_number_of_coordinates():
    with pytest.raises(QuadError, match="8 coordinates"):
        Quad.from_xy([0.1, 0.1, 0.5, 0.1, 0.5, 0.3])


def test_rejects_non_finite():
    with pytest.raises(QuadError, match="non-finite"):
        Quad.from_xy([0.1, 0.1, float("nan"), 0.1, 0.5, 0.3, 0.1, 0.3])


def test_rejects_outside_the_tolerant_range():
    with pytest.raises(QuadError, match="outside the tolerant range"):
        Quad.from_xy([0.1, 0.1, 1.9, 0.1, 1.9, 0.3, 0.1, 0.3])


def test_warns_but_accepts_outside_0_1():
    """A banknote crossing the border legitimately has vertices outside [0,1]."""
    with pytest.warns(CoordinateRangeWarning):
        quad = Quad.from_xy([-0.2, 0.1, 0.5, 0.1, 0.5, 0.3, -0.2, 0.3])
    assert quad.points[0][0] == pytest.approx(-0.2)


def test_rejects_degenerate():
    with pytest.raises(QuadError):
        Quad.from_xy([0.1, 0.1, 0.5, 0.1, 0.9, 0.1, 0.3, 0.1])


def test_warns_unstable_ratio():
    """Ratio < 1.1: the longest-side anchor is decided by noise."""
    quad = Quad.from_xy(rotated_rect_points(0.5, 0.5, 0.2, 1.02, 0.3))
    with pytest.warns(QuadShapeWarning, match="unstable"):
        canonicalize(quad)


# -- dyadic grid ------------------------------------------------------------


@given(st.floats(min_value=-0.5, max_value=1.5))
def test_snap_is_idempotent(value: float):
    assert snap(snap(value)) == snap(value)


@given(st.floats(min_value=-0.5, max_value=1.5))
def test_exact_reflection_on_the_grid(value: float):
    """1-x does not round on the grid: it is what makes the involution exact."""
    x = snap(value)
    assert 1.0 - (1.0 - x) == x


def test_the_grid_error_is_negligible():
    assert abs(snap(0.123456789) - 0.123456789) < 2.0**-31


# -- canonical order --------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=300)
def test_canonical_is_clockwise(quad: Quad):
    assert canonicalize(quad).is_clockwise()


@given(rotated_rects())
@settings(max_examples=300)
def test_canonical_is_idempotent(quad: Quad):
    once = canonicalize(quad)
    assert canonicalize(once).points == once.points
    assert is_canonical(once)


@given(rotated_rects())
@settings(max_examples=300)
def test_anchor_opens_the_longest_side(quad: Quad):
    canon = canonicalize(quad)
    lengths = canon.edge_lengths()
    assert lengths[0] >= max(lengths) * (1.0 - LONG_SIDE_TIE_TOL)


@given(rotated_rects())
@settings(max_examples=200)
def test_canonical_does_not_depend_on_input_order(quad: Quad):
    """The 24 permutations of the same 4 points give the same canonical."""
    reference = canonicalize(quad).points
    for perm in itertools.permutations(quad.points):
        assert canonicalize(Quad(points=perm)).points == reference


@given(rotated_rects())
@settings(max_examples=200)
def test_canonical_preserves_the_set_of_points(quad: Quad):
    assert set(canonicalize(quad).points) == set(quad.points)


def test_tie_break_at_45_degrees_is_deterministic():
    """x+y ties when the diagonal is perpendicular to (1,1). Full chain."""
    quad = Quad.from_xy(rotated_rect_points(0.5, 0.5, 0.2, 2.0, math.pi / 4))
    first = canonicalize(quad).points
    for perm in itertools.permutations(quad.points):
        assert canonicalize(Quad(points=perm)).points == first


# -- determinism ------------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=200)
def test_repeated_determinism(quad: Quad):
    assert canonicalize(quad).points == canonicalize(quad).points


@given(rotated_rects(min_ratio=1.6), st.integers(min_value=0, max_value=255))
@settings(max_examples=400)
def test_determinism_under_sub_tolerance_perturbation(quad: Quad, seed: int):
    """The literal form of the specification is impossible.

    "An input perturbed below the tolerance gives the same output" cannot hold
    in general: every discrete selection over a continuous input has a
    boundary, and the tolerance moves it, it does not remove it. What is true
    and is what is tested: if the separation between candidates exceeds the
    size of the perturbation, the perturbation does not change the anchor.

    The permutation is compared, not the coordinates: the perturbation moves
    the points by definition, and what must stay put is the decision.

    With ratio >= 1.6 the short sides are at 62% of the long one, well above
    the 5% tie tolerance, so the candidate set is stable and the only decision
    at stake is the x+y tie-break.
    """
    perm, margin = canonical_order(quad)
    jitter = 1e-5
    assume(margin > 8.0 * jitter)

    jittered = []
    for i, (x, y) in enumerate(quad.points):
        sign = 1.0 if (seed >> i) & 1 else -1.0
        jittered.append((x + sign * jitter, y - sign * jitter))
    perturbed_perm, _ = canonical_order(Quad.from_xy(jittered))
    assert perturbed_perm == perm
