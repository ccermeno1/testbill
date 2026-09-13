from __future__ import annotations

import math

import pytest
from conftest import rotated_rect_points
from shapely.geometry import Polygon

from testbank.checks.visibility import (
    check_quads,
    max_real_violations,
    quad_to_polygon,
)
from testbank.geometry.quad import Quad, canonicalize


def rect(cx, cy, half_long=0.2, ratio=2.0, theta=0.0) -> Quad:
    return canonicalize(Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta)))


def axis_rect(x0, y0, x1, y1) -> Quad:
    """Axis-aligned rectangle, clockwise in image coordinates."""
    return canonicalize(Quad.from_xy([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))


def test_a_single_quad_is_never_flagged():
    assert check_quads([rect(0.5, 0.5)], "s") == []


def test_separate_quads_are_not_flagged():
    quads = [rect(0.25, 0.25), rect(0.75, 0.75)]
    assert check_quads(quads, "s") == []


def test_an_almost_fully_covered_quad_is_flagged():
    covered = rect(0.5, 0.5, half_long=0.10, ratio=2.0)
    coverer = rect(0.5, 0.5, half_long=0.30, ratio=2.0)
    findings = check_quads([covered, coverer], "s")
    assert [f.annotation_index for f in findings] == [0]
    assert findings[0].visible_fraction < 0.25


def test_occlusion_is_computed_against_the_union_not_pairwise():
    """Covered 51% by one and 51% by another: no pair detects it."""
    victim = axis_rect(0.28, 0.39, 0.72, 0.61)
    left = axis_rect(0.20, 0.30, 0.505, 0.70)
    right = axis_rect(0.495, 0.30, 0.80, 0.70)

    victim_poly = quad_to_polygon(victim)
    for other in (left, right):
        pairwise = (
            victim_poly.intersection(quad_to_polygon(other)).area / victim_poly.area
        )
        assert pairwise < 0.75, "the setup must be undetectable pairwise"

    findings = check_quads([victim, left, right], "s")
    assert {f.annotation_index for f in findings} == {0}


def test_the_threshold_is_parametrizable():
    """Victim covered 90%: visible 0.10, right between the two thresholds."""
    victim = axis_rect(0.28, 0.39, 0.72, 0.61)
    coverer = axis_rect(0.28, 0.25, 0.676, 0.75)
    quads = [victim, coverer]

    strict = check_quads(quads, "s", visibility_threshold=0.25)
    assert {f.annotation_index for f in strict} == {0}
    assert strict[0].visible_fraction == pytest.approx(0.10, abs=0.01)

    assert check_quads(quads, "s", visibility_threshold=0.05) == []


# -- bound of real violations -----------------------------------------------


def test_the_top_one_is_never_covered():
    """Two quads covering each other: at most one can be real."""
    a = rect(0.5, 0.5, half_long=0.2, ratio=2.0, theta=0.0)
    b = rect(0.52, 0.5, half_long=0.2, ratio=2.0, theta=0.05)
    polygons = [quad_to_polygon(a), quad_to_polygon(b)]
    findings = check_quads([a, b], "s")
    flagged = {f.annotation_index for f in findings}
    if len(flagged) == 2:
        assert max_real_violations(polygons, flagged) == 1


def test_without_flags_the_bound_is_zero():
    polygons = [
        quad_to_polygon(rect(0.25, 0.25)),
        quad_to_polygon(rect(0.75, 0.75)),
    ]
    assert max_real_violations(polygons, set()) == 0


def test_the_bound_never_exceeds_the_number_of_flags():
    quads = [
        rect(0.5, 0.5, half_long=0.10),
        rect(0.5, 0.5, half_long=0.12, theta=0.3),
        rect(0.5, 0.5, half_long=0.30),
    ]
    polygons = [quad_to_polygon(q) for q in quads]
    findings = check_quads(quads, "s")
    flagged = {f.annotation_index for f in findings}
    assert max_real_violations(polygons, flagged) <= len(flagged)


def test_the_bound_is_invariant_to_input_order():
    quads = [
        rect(0.45, 0.5, half_long=0.18, theta=0.2),
        rect(0.55, 0.5, half_long=0.18, theta=0.9),
        rect(0.50, 0.5, half_long=0.30, theta=1.4),
    ]
    polygons = [quad_to_polygon(q) for q in quads]
    flagged = {f.annotation_index for f in check_quads(quads, "s")}
    first = max_real_violations(polygons, flagged)

    order = [2, 0, 1]
    reordered = [quads[i] for i in order]
    polys2 = [Polygon(q.points) for q in reordered]
    flagged2 = {f.annotation_index for f in check_quads(reordered, "s")}
    assert max_real_violations(polys2, flagged2) == first


def test_area_ratios_do_not_depend_on_aspect():
    """The check works with area ratios, invariant under affine scaling."""
    quads = [rect(0.5, 0.5, half_long=0.10), rect(0.5, 0.5, half_long=0.30)]
    base = check_quads(quads, "s")

    squashed = []
    for quad in quads:
        squashed.append(Quad.from_xy([(x, y * 0.5 + 0.25) for x, y in quad.points]))
    other = check_quads(squashed, "s")

    assert [f.annotation_index for f in base] == [f.annotation_index for f in other]
    assert math.isclose(
        base[0].visible_fraction, other[0].visible_fraction, abs_tol=1e-6
    )
