from __future__ import annotations

import math

from hypothesis import strategies as st

from testbank.geometry.quad import Quad

# ratio >= 1.2 keeps the short sides outside the tie tolerance
# (1/(1-0.05) = 1.053) and above the unstable-anchor threshold (1.1).
SAFE_MIN_RATIO = 1.2


def rotated_rect_points(
    cx: float, cy: float, half_long: float, ratio: float, theta: float
) -> list[tuple[float, float]]:
    """Rotated rectangle in image coordinates (y pointing down).

    Returns the vertices in visual clockwise order before canonicalizing.
    """
    half_short = half_long / ratio
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    local = [
        (-half_long, -half_short),
        (+half_long, -half_short),
        (+half_long, +half_short),
        (-half_long, +half_short),
    ]
    return [
        (cx + lx * cos_t - ly * sin_t, cy + lx * sin_t + ly * cos_t)
        for lx, ly in local
    ]


@st.composite
def rotated_rects(draw, min_ratio: float = SAFE_MIN_RATIO, max_ratio: float = 5.0):
    """Rotated rectangles that fit comfortably within the tolerant range."""
    half_long = draw(st.floats(min_value=0.04, max_value=0.22))
    ratio = draw(st.floats(min_value=min_ratio, max_value=max_ratio))
    theta = draw(st.floats(min_value=0.0, max_value=math.pi, exclude_max=True))
    margin = half_long * 1.45 + 0.02  # half diagonal <= half_long*sqrt(2)
    cx = draw(st.floats(min_value=margin, max_value=1.0 - margin))
    cy = draw(st.floats(min_value=margin, max_value=1.0 - margin))
    return Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta))
