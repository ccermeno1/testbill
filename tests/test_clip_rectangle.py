"""`clip` has to return RECTANGLES inside the frame.

Why this file exists
--------------------
`clip` used to pin each vertex to [0,1] on its own. It is the obvious thing
and it is wrong: a ROTATED rectangle cut against a straight frame gives a
trapezoid. Measured on the real data, 82 of 679 annotations (12%) stopped
being rectangles, and the only run recorded until then trained that way.

It matters because the model predicts `(cx, cy, w, h, angle)`. A trapezoid as
ground truth is an unreachable target: it can only be learned as the
rectangle that fits it least badly.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon, box

from testbank.dataio.prepare import clip_quad
from testbank.geometry.quad import Quad

# Clipping a banknote that sticks far out leaves it nearly square, and then
# the longest-side anchor is unstable -- which is exactly what
# `QuadShapeWarning` warns about. Here it is the EXPECTED result, not a
# symptom: the cases in this file are deliberately extreme. And
# `CoordinateRangeWarning` fires when BUILDING the input quads, which are
# outside the frame on purpose.
pytestmark = pytest.mark.filterwarnings(
    "ignore::testbank.geometry.quad.QuadShapeWarning",
    "ignore::testbank.geometry.quad.CoordinateRangeWarning",
)


def _quad(cx, cy, half_long, half_short, theta, aspect=1.0):
    """Rotated rectangle built in the space with real proportions."""
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    points = [
        (cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)
        for lx, ly in (
            (-half_long, -half_short),
            (half_long, -half_short),
            (half_long, half_short),
            (-half_long, half_short),
        )
    ]
    return Quad.from_xy([(x / aspect, y) for x, y in points])


def _sides(quad, aspect=1.0):
    points = [(x * aspect, y) for x, y in quad.points]
    return [math.dist(points[i], points[(i + 1) % 4]) for i in range(4)]


def _angle(quad, aspect=1.0):
    points = [(x * aspect, y) for x, y in quad.points]
    return math.degrees(
        math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    ) % 180.0


@pytest.mark.parametrize("theta", [0.0, 0.2, 0.5, 0.9, 1.3, -0.4, -1.1])
@pytest.mark.parametrize("cx,cy", [(0.05, 0.5), (0.95, 0.5), (0.5, 0.02), (0.02, 0.04)])
def test_what_comes_out_is_always_a_rectangle(theta, cx, cy):
    clipped = clip_quad(_quad(cx, cy, 0.30, 0.12, theta))
    a, b, c, d = _sides(clipped)
    assert a == pytest.approx(c, abs=1e-6)
    assert b == pytest.approx(d, abs=1e-6)


@pytest.mark.parametrize("theta", [0.0, 0.2, 0.5, 0.9, 1.3, -0.4, -1.1])
@pytest.mark.parametrize("cx,cy", [(0.05, 0.5), (0.95, 0.5), (0.5, 0.02), (0.02, 0.04)])
def test_what_comes_out_always_fits_in_the_frame(theta, cx, cy):
    """The reason `clip` exists: so the trainer does not throw the image away."""
    for x, y in clip_quad(_quad(cx, cy, 0.30, 0.12, theta)).points:
        assert -1e-9 <= x <= 1.0 + 1e-9
        assert -1e-9 <= y <= 1.0 + 1e-9


def test_a_quad_that_already_fits_is_not_touched():
    inside = _quad(0.5, 0.5, 0.2, 0.1, 0.3)
    output = clip_quad(inside)
    for (ax, ay), (bx, by) in zip(inside.points, output.points):
        assert ax == pytest.approx(bx, abs=1e-9)
        assert ay == pytest.approx(by, abs=1e-9)


@pytest.mark.parametrize("theta", [0.0, 0.15, 0.6, -0.35])
def test_the_angle_survives_the_clipping(theta):
    """The annotation's angle is data; a trapezoid's is an artifact.

    If the clipping recomputed it, it would inject noise into the only
    quantity measured apart from IoU.
    """
    original = _quad(0.05, 0.5, 0.30, 0.12, theta)
    assert _angle(clip_quad(original)) == pytest.approx(_angle(original), abs=1e-6)


def test_a_small_violation_does_not_eat_the_long_side():
    """REGRESSION of the failure not visible by reading the code, only by measuring.

    Real case `Multiple_Euro_154`: a nearly horizontal banknote sticking out
    0.026 at the top. Since the long axis had `uy = 0.0315`, the constraint
    `y >= 0` could also be satisfied by shrinking the LONG side -- and that is
    what it did: from 0.890 to 0.064, keeping 8% of the banknote. It met every
    constraint.

    The rule that fixes it is assigning each constraint to the axis most
    aligned with it. Here it is anchored: the long side is barely touched, the
    short one absorbs the clipping.
    """
    original = Quad.from_xy(
        [(0.096, -0.026), (0.985, 0.002), (0.974, 0.332), (0.085, 0.304)]
    )
    before = _sides(original)
    after = _sides(clip_quad(original))
    long_before, long_after = max(before), max(after)
    assert long_after > 0.95 * long_before, "the long side must barely be touched"

    visible = Polygon(original.points).intersection(box(0.0, 0.0, 1.0, 1.0))
    rectangle = Polygon(clip_quad(original).points)
    covered = rectangle.intersection(visible).area / visible.area
    assert covered > 0.90, f"it only covers {covered:.1%} of the visible banknote"


def test_with_non_square_aspect_the_rectangle_is_one_in_PIXELS():
    """In normalized coordinates a rotated rectangle is a PARALLELOGRAM.

    Without passing the aspect, this would "fix" the shape in the wrong space:
    it would come out rectangular in [0,1]x[0,1] and skewed in the real image.
    """
    aspect = 2.5
    original = _quad(0.04 * aspect, 0.5, 0.30, 0.12, 0.4, aspect=aspect)
    clipped = clip_quad(original, aspect=aspect)
    a, b, c, d = _sides(clipped, aspect)
    assert a == pytest.approx(c, abs=1e-6)
    assert b == pytest.approx(d, abs=1e-6)
    # And in normalized coordinates it is NOT a rectangle, which is the reason
    # for passing the aspect. The ANGLE between sides is checked, not their
    # length: a parallelogram also has equal opposite sides, so comparing them
    # distinguishes nothing. (It was checked that way and the test passed by
    # chance.)
    points = list(clipped.points)
    ax = points[1][0] - points[0][0], points[1][1] - points[0][1]
    bx = points[2][0] - points[1][0], points[2][1] - points[1][1]
    cosine = (ax[0] * bx[0] + ax[1] * bx[1]) / (
        math.hypot(*ax) * math.hypot(*bx)
    )
    assert abs(cosine) > 1e-3, "in normalized coordinates the corners are not 90 degrees"


def test_a_quad_almost_entirely_outside_does_not_blow_up():
    """The limit of what `Quad` admits: beyond it cannot even be built.

    `Quad.from_xy` rejects coordinates outside [-0.5, 1.5], so "entirely
    outside" is not a reachable state. This is the extreme case that is.
    """
    outside = _quad(-0.18, 0.5, 0.30, 0.12, 0.2)
    output = clip_quad(outside)
    assert all(math.isfinite(v) for v in output.flat())
    for x, y in output.points:
        assert -1e-9 <= x <= 1.0 + 1e-9
        assert -1e-9 <= y <= 1.0 + 1e-9
