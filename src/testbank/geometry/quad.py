"""Canonical quad: 4 normalized vertices, the pivot of every conversion.

N formats mean 2N converters, not N^2: everything goes through here.

Dyadic grid
-----------
Coordinates are snapped to multiples of 2^-SNAP_BITS. It is the only way for
`flip(flip(q)) == q` to hold EXACTLY, as the specification demands: in float64
`1 - (1 - 0.1)` gives 0.09999999999999998, so x -> 1-x is not an involution. On
the grid, 1-x is representable without rounding and the involution is bit for
bit. The error introduced is <= 2^-31 normalized (2e-6 px on a 4000 px image),
six orders of magnitude below the precision of an "approximate rectangle"
annotation. The annotation files on disk are never modified.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

Point = tuple[float, float]

SNAP_BITS = 30
_GRID = float(2**SNAP_BITS)

# Tolerant range: some banknotes cross the image border and their vertices
# legitimately fall outside [0,1].
COORD_MIN = -0.5
COORD_MAX = 1.5

# Two sides count as tied in length if they differ by less than this, relative.
# Without a tolerance, the anchor of a nearly exact rectangle (long sides that
# differ by 0.3%) is chosen by noise and a 1 px jitter rotates it 180 degrees.
LONG_SIDE_TIE_TOL = 0.05

# Below this side ratio the longest-side anchor is unstable.
MIN_STABLE_SIDE_RATIO = 1.1

_MIN_AREA = 1e-12

#: Width/height ratio of the image in pixels. Normalized coordinates divide x by
#: the width and y by the height, which is an ANISOTROPIC scaling: in that space
#: the longest side, the angle and the ratio are not the geometric ones. A 2:1
#: banknote lying flat in a 16:9 image has normalized ratio 1.13, and in 20:9 it
#: drops below 1 and the anchor jumps to the short side. That is why every
#: length comparison accepts the image aspect.
DEFAULT_ASPECT = 1.0


class QuadShapeWarning(UserWarning):
    """The quad is geometrically degenerate for the canonical anchoring."""


class CoordinateRangeWarning(UserWarning):
    """Vertex outside [0,1]. Legitimate if the banknote crosses the border."""


class QuadError(ValueError):
    """The quad is unusable."""


def snap(value: float) -> float:
    """Snap to the dyadic grid. Exact: round() gives an integer, /2^k is exact."""
    return round(value * _GRID) / _GRID


@dataclass(frozen=True, slots=True)
class Quad:
    """Four normalized vertices, already snapped to the grid.

    Does not guarantee canonical order: use `canonicalize`. Construction
    validates range and non-degeneracy but does not reorder, so that `flip` can
    be an exact involution on the raw sequence.
    """

    points: tuple[Point, Point, Point, Point]

    @classmethod
    def from_xy(cls, coords) -> Quad:
        """Build from 8 floats or 4 pairs. Snaps to the grid and validates."""
        flat: list[float] = []
        for item in coords:
            if isinstance(item, (tuple, list)):
                flat.extend(float(v) for v in item)
            else:
                flat.append(float(item))
        if len(flat) != 8:
            raise QuadError(f"a quad is 8 coordinates, got {len(flat)}")
        for v in flat:
            if not math.isfinite(v):
                raise QuadError(f"non-finite coordinate: {v!r}")
            if not (COORD_MIN <= v <= COORD_MAX):
                raise QuadError(
                    f"coordinate {v!r} outside the tolerant range [{COORD_MIN}, {COORD_MAX}]"
                )
        if any(not (0.0 <= v <= 1.0) for v in flat):
            warnings.warn(
                "quad with vertices outside [0,1]; accepted (banknote crossing "
                "the image border)",
                CoordinateRangeWarning,
                stacklevel=2,
            )
        snapped = [snap(v) for v in flat]
        pts = tuple((snapped[i], snapped[i + 1]) for i in range(0, 8, 2))
        quad = cls(points=pts)  # type: ignore[arg-type]
        quad._reject_degenerate()
        return quad

    # -- basic geometry -----------------------------------------------------

    def flat(self) -> tuple[float, ...]:
        return tuple(c for p in self.points for c in p)

    def centroid(self) -> Point:
        return (
            sum(p[0] for p in self.points) / 4.0,
            sum(p[1] for p in self.points) / 4.0,
        )

    def signed_area(self) -> float:
        """Shoelace. In image coordinates (y pointing down), clockwise > 0."""
        pts = self.points
        total = 0.0
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % 4]
            total += x0 * y1 - x1 * y0
        return total / 2.0

    def is_clockwise(self) -> bool:
        return self.signed_area() > 0.0

    def edge_lengths(
        self, aspect: float = DEFAULT_ASPECT
    ) -> tuple[float, float, float, float]:
        """Length of the side OPENED by each vertex: L[i] = |p_i -> p_{i+1}|.

        `aspect` is width/height in pixels. With the default 1.0 the measure is
        taken in normalized space, which only matches the geometric one when
        the image is square.
        """
        pts = self.points
        out = []
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % 4]
            out.append(math.hypot((x1 - x0) * aspect, y1 - y0))
        return tuple(out)  # type: ignore[return-value]

    def side_ratio(self, aspect: float = DEFAULT_ASPECT) -> float:
        """Longer side / shorter side, averaging opposite pairs."""
        lengths = sorted(self.edge_lengths(aspect))
        short = (lengths[0] + lengths[1]) / 2.0
        long_ = (lengths[2] + lengths[3]) / 2.0
        if short <= 0.0:
            return math.inf
        return long_ / short

    def _reject_degenerate(self) -> None:
        if abs(self.signed_area()) < _MIN_AREA:
            raise QuadError(f"degenerate quad, zero area: {self.points}")
        if len(set(self.points)) != 4:
            raise QuadError(f"quad with repeated vertices: {self.points}")

    def angle_deg(self, aspect: float = DEFAULT_ASPECT) -> float:
        """Orientation of the long side p0->p1, in [0,180). Requires canonical."""
        (x0, y0), (x1, y1) = self.points[0], self.points[1]
        return math.degrees(math.atan2(y1 - y0, (x1 - x0) * aspect)) % 180.0


# -- canonical order --------------------------------------------------------


def canonical_order(
    quad: Quad,
    *,
    tie_tol: float = LONG_SIDE_TIE_TOL,
    aspect: float = DEFAULT_ASPECT,
) -> tuple[tuple[int, int, int, int], float]:
    """Canonical permutation of the input indices, and the margin of the decision.

    Returning the permutation separately allows testing the STABILITY of the
    discrete decision without confusing it with the coordinate values: a
    perturbation moves the points, and what has to stay put is which vertex
    acts as anchor, not its value.

    The margin is the separation in the tiebreak key between the chosen
    candidate and the next one. It is the only honest quantity to measure
    stability against: below it the choice may change, and that is not a
    failure, it is the boundary every discrete selection necessarily has.
    """
    cx, cy = quad.centroid()

    def polar(i: int) -> float:
        x, y = quad.points[i]
        return math.atan2(y - cy, (x - cx) * aspect)

    keyed = sorted(range(4), key=polar)
    for a, b in zip(keyed, keyed[1:] + keyed[:1]):
        if polar(a) == polar(b):
            raise QuadError(
                f"two vertices collinear with the centroid, ambiguous order: {quad.points}"
            )

    ordered_pts = [quad.points[i] for i in keyed]
    ordered = Quad(points=tuple(ordered_pts))  # type: ignore[arg-type]

    ratio = ordered.side_ratio(aspect)
    if ratio < MIN_STABLE_SIDE_RATIO:
        warnings.warn(
            f"quad with side ratio {ratio:.3f} < {MIN_STABLE_SIDE_RATIO}: "
            "the longest-side anchor is unstable",
            QuadShapeWarning,
            stacklevel=3,
        )

    lengths = ordered.edge_lengths(aspect)
    threshold = max(lengths) * (1.0 - tie_tol)
    candidates = [i for i, L in enumerate(lengths) if L >= threshold]

    def tiebreak(i: int) -> tuple[float, float, float]:
        x, y = ordered_pts[i]
        return (x + y, x, y)

    ranked = sorted(candidates, key=tiebreak)
    start = ranked[0]
    margin = (
        tiebreak(ranked[1])[0] - tiebreak(start)[0] if len(ranked) > 1 else math.inf
    )

    perm = tuple(keyed[(start + i) % 4] for i in range(4))
    return perm, margin  # type: ignore[return-value]


def canonicalize(
    quad: Quad,
    *,
    tie_tol: float = LONG_SIDE_TIE_TOL,
    aspect: float = DEFAULT_ASPECT,
) -> Quad:
    """Canonical order: clockwise, starting at the vertex that opens the longest side.

    Tiebreak: smallest x+y, then smallest x, then smallest y. The full chain is
    needed because x+y ties on rectangles whose diagonal is perpendicular to
    (1,1) -- the 45 degree case, exactly the one that rules out "the corner
    closest to the origin".

    "Corner closest to the origin" is not used: it is discontinuous near 45
    degrees and a 2 px jitter rotates the labels by 90 degrees.
    """
    perm, _ = canonical_order(quad, tie_tol=tie_tol, aspect=aspect)
    return Quad(points=tuple(quad.points[i] for i in perm))  # type: ignore[arg-type]


def is_canonical(
    quad: Quad,
    *,
    tie_tol: float = LONG_SIDE_TIE_TOL,
    aspect: float = DEFAULT_ASPECT,
) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuadShapeWarning)
        return canonicalize(quad, tie_tol=tie_tol, aspect=aspect).points == quad.points


# -- horizontal flip --------------------------------------------------------


def flip_horizontal(quad: Quad) -> Quad:
    """Reflection x -> 1-x, preserving the raw vertex sequence.

    Does NOT recanonicalize: the reflection inverts the winding, so the result
    is counter-clockwise and must go through `canonicalize` to return to the
    canonical form. Keeping it raw is what makes the involution exact.

    On the dyadic grid 1-x does not round, and the range [-0.5, 1.5] is
    symmetric about 0.5, so the reflection cannot push a vertex out of range.
    """
    flipped = tuple((1.0 - x, y) for x, y in quad.points)
    return Quad(points=flipped)  # type: ignore[arg-type]
