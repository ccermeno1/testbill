"""Geometric core of the metrics. Everything in PIXELS, no exceptions.

Why pixels and not normalized
-----------------------------
Normalizing divides x by the width and y by the height: an ANISOTROPIC scaling.
Area ratios (IoU, coverage, contamination) survive any affinity and would come
out the same in both spaces, but the ANGLE and the PER-VERTEX DISTANCE do not.
A 2:1 banknote lying flat in a 20:9 image changes angle and ratio when
normalized, and the canonical anchor can even jump sides.

Mixing spaces depending on the metric would be asking for the error. The rule
is a single one: in here everything is in pixels, and the conversion happens
at the boundary.

No compiled operators: rotated IoU and rotated NMS via shapely. Slow, and
enough for 500 images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import Polygon
from shapely.ops import unary_union

from testbank.dataio.formats import ImageSize
from testbank.geometry.quad import Quad

#: Angles that differ by 180 degrees describe the same rectangle.
HALF_TURN = 180.0


@dataclass(frozen=True, slots=True)
class Prediction:
    """A detection. `score` orders the greedy matching.

    `quad` may be None: a box the network predicted so far outside the image
    that `Quad` will not accept it (more than half a frame). It is still a
    detection the model made, so it COUNTS as a false positive at its score --
    it matches nothing because it has nothing to match with -- instead of being
    discarded, which would gift the metric an error the model did commit.
    """

    quad: Quad | None
    score: float
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class ImageEval:
    """Everything needed to evaluate ONE image.

    `ignored` are the quads the relative area filter left out. They are neither
    truth to be detected nor background to penalize: they are real annotations
    we decided not to use. See `matching.py`.
    """

    sample_id: str
    size: ImageSize
    truths: tuple[Quad, ...] = ()
    predictions: tuple[Prediction, ...] = ()
    ignored: tuple[Quad, ...] = ()

    def __post_init__(self) -> None:
        if self.size.width <= 0 or self.size.height <= 0:
            raise ValueError(f"{self.sample_id}: invalid image size")


def to_polygon(quad: Quad | None, size: ImageSize) -> Polygon:
    """Normalized quad -> polygon in pixels, sanitized.

    None -> empty polygon: intersects nothing, zero area, zero IoU with
    everything. That is how a prediction without geometry stays unmatched and
    becomes a false positive without touching the matching.
    """
    if quad is None:
        return Polygon()
    polygon = Polygon(
        [(x * size.width, y * size.height) for x, y in quad.points]
    )
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def iou(a: Polygon, b: Polygon) -> float:
    """Rotated IoU. Area ratios do not depend on the space, but the rest of the
    module does, so it takes pixels like everything else."""
    if not a.intersects(b):
        return 0.0
    intersection = a.intersection(b).area
    if intersection <= 0.0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def angle_error_deg(a: Quad, b: Quad, size: ImageSize) -> float:
    """Angle error modulo 180: `min(|d|, 180 - |d|)`.

    A rectangle rotated 179 degrees and one rotated 1 degree are almost the
    same rectangle, not two that differ by 178. Without the modulo those cases
    dominate the mean and the metric stops measuring anything.
    """
    delta = abs(a.angle_deg(size.aspect) - b.angle_deg(size.aspect)) % HALF_TURN
    return min(delta, HALF_TURN - delta)


def vertex_distances_px(a: Quad, b: Quad, size: ImageSize) -> list[float]:
    """Per-vertex distance, already in the canonical order of each.

    Diagnostic, not a success criterion: the specification says exact
    geometric precision is not the goal.
    """
    pa = [(x * size.width, y * size.height) for x, y in a.points]
    pb = [(x * size.width, y * size.height) for x, y in b.points]
    return [math.dist(p, q) for p, q in zip(pa, pb)]


def longest_side_px(quad: Quad, size: ImageSize) -> float:
    points = [(x * size.width, y * size.height) for x, y in quad.points]
    return max(
        math.dist(points[i], points[(i + 1) % 4]) for i in range(4)
    )


def expand(polygon: Polygon, margin: float) -> Polygon:
    """Crop with margin: scale the rectangle about its center.

    `margin` is the fraction of EACH SIDE added on EACH BORDER, so the total
    side is multiplied by `1 + 2*margin`. With margin=0.05 the crop carries 5%
    of slack per side, not 5% shared out.

    It scales instead of dilating with `buffer` because `buffer` rounds the
    corners and the crop has to stay a quadrilateral so it can be rectified by
    homography.
    """
    if margin < 0:
        raise ValueError(f"the margin cannot be negative: {margin}")
    if margin == 0:
        return polygon
    from shapely import affinity

    factor = 1.0 + 2.0 * margin
    return affinity.scale(polygon, xfact=factor, yfact=factor, origin="center")


def union_of(polygons) -> Polygon | None:
    """Union of a list, or None if empty. Keeps the special case out of callers."""
    valid = [p for p in polygons if p is not None and p.area > 0]
    if not valid:
        return None
    merged = unary_union(valid)
    return merged if merged.area > 0 else None


@dataclass
class PolygonCache:
    """Pixel polygons of one image, computed once.

    The greedy matching and the crop metrics walk the same quads several
    times, and `Polygon` is not free. The bootstrap resamples 2000 times over
    the SAME images, so without a cache everything would be recomputed two
    thousand times.
    """

    size: ImageSize
    truths: list[Polygon] = field(default_factory=list)
    predictions: list[Polygon] = field(default_factory=list)
    ignored: list[Polygon] = field(default_factory=list)

    @classmethod
    def build(cls, item: ImageEval) -> PolygonCache:
        return cls(
            size=item.size,
            truths=[to_polygon(q, item.size) for q in item.truths],
            predictions=[to_polygon(p.quad, item.size) for p in item.predictions],
            ignored=[to_polygon(q, item.size) for q in item.ignored],
        )
