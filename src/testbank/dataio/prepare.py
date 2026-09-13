"""The single gate between the annotations on disk and what any trainer sees.

Two steps, in this order, the same for EVERY format and candidate:

1. **Load and filter** (`load_samples`): read the labels and apply the relative
   area filter. What the filter drops is not deleted: it stays as `dropped`,
   with its original index, so it can be pointed at in Roboflow.
2. **Prepare the view** (`prepare`): apply the border policy (`clip`, `pad`,
   `keep`) and fix the image size that pixel-based formats need.

They used to be two modules (`loader.py` and `view.py`) and were merged in the
refactor: they are the same idea -- that two candidates cannot train on
different truths -- and keeping them apart meant reading two places to
understand one thing.

=== Loading and filtering ===
Single entry point to a sample's annotations.

`read_label_file` reads a file and validates its format. Here it is composed
with the image aspect and with the relative area filter, and it is what the
visualisation, the checks and everything downstream consume. One place decides
which annotations are in force, just as `splits.py` is the only one deciding
which split a sample belongs to.

Relative area filter
--------------------
In every image the banknote in front is kept: every annotation whose area is
smaller than `min_relative_area` times the area of the largest annotation in
that same image is dropped.

It covers both situations without having to tell them apart. In a fan, the
visible strip of a covered banknote is much smaller than the one in front and
falls out. In a photo of two or three banknotes side by side, all have a
similar size and all are kept.

It is a FILTER ON LOAD, not a deletion: the annotation files are never touched,
and what is dropped is reported with image, index and relative area percentage
so it can be reviewed and fixed at the source.

Why the image size is not needed
--------------------------------
The criterion is a ratio of areas within one image. Going from normalized
coordinates to pixels multiplies both areas by the same W*H, so the ratio does
not change. The aspect is still requested, but only so the canonical order of
the quads is right, not for the filter.

=== Border policy and view ===
Common preparation of the samples before writing them in any format.

Between the source file and what a trainer sees there are three steps that
HAVE to be the same for all formats:

1. The relative area filter (above).
2. The policy for banknotes crossing the border (`clip`, `pad` or `keep`).
3. The image size, which pixel-based formats need.

They used to live inside `detectors/dataset.py`, which writes the Ultralytics
view. Adding the DOTA and COCO exporters would have meant repeating them, and a
divergence there would be silent: two candidates training on different truths
and a table comparing them as if they were the same. So they live here and
both places call the same thing.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from testbank.config import Config, OutOfBoundsPolicy
from testbank.data.discover import Sample
from testbank.dataio.formats import ImageSize
from testbank.dataio.image_sizes import SizeIndex
from testbank.dataio.obb_yolo import Annotation, LabelFormatError, read_label_file
from testbank.geometry.quad import DEFAULT_ASPECT, Quad, canonicalize

# --------------------------------------------------------------------------
# Loading and area filter
# --------------------------------------------------------------------------

#: Fraction of the largest annotation's area below which one is dropped. An
#: approximation to the 25% visibility policy, not a measure of occlusion.
DEFAULT_MIN_RELATIVE_AREA = 0.25


@dataclass(frozen=True, slots=True)
class DroppedAnnotation:
    """An annotation the filter leaves out. It is still in the source file."""

    sample_id: str
    #: Index of the annotation within its image, counting from 0, in the same
    #: order it appears in the file. It is the index the visibility report
    #: uses, so the two reports can be cross-referenced.
    index: int
    #: area / area of the largest annotation in the image, in [0, 1).
    relative_area: float
    label_path: Path
    quad: Quad

    def describe(self) -> str:
        return (
            f"{self.sample_id}: annotation #{self.index} dropped, "
            f"relative area {self.relative_area:.1%}"
        )


@dataclass(frozen=True, slots=True)
class LoadedSample:
    """A sample's annotations in force, plus what was left out and why."""

    sample_id: str
    label_path: Path
    aspect: float
    annotations: tuple[Annotation, ...]
    #: Original position of each kept annotation within its file, counting
    #: from 0. Filtering renumbers, and a report saying "annotation #2" about
    #: the already-filtered position would point at the wrong annotation when
    #: opening the file in Roboflow. Every downstream report uses these
    #: indices, so the filter's and the visibility check's cross-reference.
    kept_indices: tuple[int, ...] = ()
    dropped: tuple[DroppedAnnotation, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def quads(self) -> tuple[Quad, ...]:
        return tuple(a.quad for a in self.annotations)


@dataclass
class FilterReport:
    """Tally of what the filter leaves out, to review it in Roboflow."""

    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA
    images_loaded: int = 0
    annotations_read: int = 0
    dropped: list[DroppedAnnotation] = field(default_factory=list)

    @property
    def annotations_kept(self) -> int:
        return self.annotations_read - len(self.dropped)

    @property
    def images_affected(self) -> int:
        return len({d.sample_id for d in self.dropped})

    def to_dict(self) -> dict:
        return {
            "min_relative_area": self.min_relative_area,
            "images_loaded": self.images_loaded,
            "annotations_read": self.annotations_read,
            "annotations_kept": self.annotations_kept,
            "annotations_dropped": len(self.dropped),
            "images_affected": self.images_affected,
            "note": (
                "Filter on load, an approximation to the 25% visibility "
                "policy: it does not measure occlusion. The listed annotations "
                "are still in the source dataset; nothing was deleted."
            ),
            "dropped": [
                {
                    "sample_id": d.sample_id,
                    "index": d.index,
                    "relative_area": round(d.relative_area, 6),
                    "label_path": str(d.label_path),
                }
                for d in sorted(self.dropped, key=lambda d: (d.sample_id, d.index))
            ],
        }

    def summary_lines(self) -> list[str]:
        threshold = (
            f"Relative area filter: threshold {self.min_relative_area:.0%} "
            f"of the largest banknote in each image"
        )
        dropped = (
            f"Annotations dropped:  {len(self.dropped)} "
            f"in {self.images_affected} images"
        )
        return [
            threshold,
            f"Images loaded:        {self.images_loaded}",
            f"Annotations read:     {self.annotations_read}",
            f"Annotations kept:     {self.annotations_kept}",
            dropped,
        ]


def filter_by_relative_area(
    annotations: tuple[Annotation, ...] | list[Annotation],
    *,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
) -> tuple[tuple[tuple[int, Annotation], ...], tuple[tuple[int, float], ...]]:
    """Keep the banknote in front and whatever is close to it in size.

    Returns `(kept, dropped)`, where `kept` are `(original index, annotation)`
    pairs and `dropped` are `(original index, relative area)` pairs. The
    indices are always the position within the file, not the position after
    filtering, so that any report points at the annotation one actually has
    to open in Roboflow.

    The largest annotation has relative area 1.0 and is therefore never
    dropped, not even with `min_relative_area = 1.0`: the comparison is strict,
    so threshold 1.0 keeps the largest and its exact ties.
    """
    if not 0.0 <= min_relative_area <= 1.0:
        raise ValueError(
            f"min_relative_area must be in [0, 1], got {min_relative_area}"
        )
    annotations = tuple(annotations)
    if not annotations:
        return (), ()

    areas = [abs(a.quad.signed_area()) for a in annotations]
    largest = max(areas)
    if largest <= 0.0:  # pragma: no cover - the reader already rejects zero area
        return tuple(enumerate(annotations)), ()

    kept: list[tuple[int, Annotation]] = []
    dropped: list[tuple[int, float]] = []
    for index, (annotation, area) in enumerate(zip(annotations, areas)):
        relative = area / largest
        if relative < min_relative_area:
            dropped.append((index, relative))
        else:
            kept.append((index, annotation))
    return tuple(kept), tuple(dropped)


def load_sample(
    sample: Sample,
    *,
    aspect: float = DEFAULT_ASPECT,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
) -> LoadedSample:
    """Read a sample's annotations and apply the area filter."""
    label = read_label_file(sample.label_path, aspect=aspect)
    kept, dropped = filter_by_relative_area(
        label.annotations, min_relative_area=min_relative_area
    )
    return LoadedSample(
        sample_id=sample.sample_id,
        label_path=sample.label_path,
        aspect=aspect,
        annotations=tuple(a for _, a in kept),
        kept_indices=tuple(i for i, _ in kept),
        dropped=tuple(
            DroppedAnnotation(
                sample_id=sample.sample_id,
                index=index,
                relative_area=relative,
                label_path=sample.label_path,
                quad=label.annotations[index].quad,
            )
            for index, relative in dropped
        ),
        warnings=label.warnings,
    )


def load_samples(
    samples,
    *,
    sizes=None,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
) -> tuple[tuple[LoadedSample, ...], FilterReport]:
    """Load a set of samples and accumulate the filter report.

    `sizes` is a `SizeIndex`; without it the canonical order is computed in
    normalized space, which is not the geometric one. Pass it whenever you
    have it.
    """
    report = FilterReport(min_relative_area=min_relative_area)
    loaded: list[LoadedSample] = []
    problems: list[str] = []
    for sample in samples:
        aspect = sizes.aspect(sample.sample_id) if sizes else DEFAULT_ASPECT
        try:
            item = load_sample(
                sample, aspect=aspect, min_relative_area=min_relative_area
            )
        except LabelFormatError as exc:
            # Same as within a file: walk all of them and fail at the end with
            # the full list. With 500 labels, learning about one error per run
            # turns the cleanup into a loop.
            problems.extend(exc.problems)
            continue
        loaded.append(item)
        report.images_loaded += 1
        report.annotations_read += len(item.annotations) + len(item.dropped)
        report.dropped.extend(item.dropped)

    if problems:
        raise LabelFormatError.combine(problems, "invalid annotations in the dataset")

    return tuple(loaded), report


# --------------------------------------------------------------------------
# Border policy and prepared view
# --------------------------------------------------------------------------

#: Orders in which the extents are visited. See `_fit_extents`.
_TRIM_ORDERS = (
    ("u1", "u0", "v1", "v0"),
    ("v1", "v0", "u1", "u0"),
    ("u0", "u1", "v0", "v1"),
    ("v0", "v1", "u0", "u1"),
)


def _trim(u0, u1, v0, v1, axes, aspect, order, passes):
    """Gauss-Seidel over the four extents, in a given order.

    The golden rule: **an extent never crosses its opposite**. If the bound it
    gets would demand it, it is left as is and the other axis absorbs it on the
    next pass. Without that guard the extent was pinched against its opposite,
    the dimension collapsed to zero and, since the fit only shrinks, there was
    no way back: a degenerate quad came out instead of a small rectangle.
    """
    (ux, uy), (vx, vy) = axes
    # (coef_u, coef_v, cap) of each frame constraint.
    limits = ((ux, vx, aspect), (uy, vy, 1.0))

    for index in range(passes):
        # Only the LAST pass lets any axis absorb any constraint. Before that,
        # alignment rules (see `_fit_extents`).
        cheapest_only = index < passes - 1
        for which in order:
            on_u = which[0] == "u"
            low, high = -math.inf, math.inf
            others = (v0, v1) if on_u else (u0, u1)
            for coefficient_u, coefficient_v, cap in limits:
                own = coefficient_u if on_u else coefficient_v
                fixed = coefficient_v if on_u else coefficient_u
                if abs(own) < 1e-12:
                    continue
                if cheapest_only and abs(own) < abs(fixed):
                    continue
                for other in others:
                    offset = fixed * other
                    a, b = (0.0 - offset) / own, (cap - offset) / own
                    lo, hi = (a, b) if own > 0 else (b, a)
                    low, high = max(low, lo), min(high, hi)

            if which == "u1" and high > u0:
                u1 = min(u1, high)
            elif which == "u0" and low < u1:
                u0 = max(u0, low)
            elif which == "v1" and high > v0:
                v1 = min(v1, high)
            elif which == "v0" and low < v1:
                v0 = max(v0, low)
    return u0, u1, v0, v1


def _inside(u0, u1, v0, v1, axes, aspect, tolerance=1e-9):
    (ux, uy), (vx, vy) = axes
    for u in (u0, u1):
        for v in (v0, v1):
            x, y = u * ux + v * vx, u * uy + v * vy
            if not (-tolerance <= x <= aspect + tolerance):
                return False
            if not (-tolerance <= y <= 1.0 + tolerance):
                return False
    return True


def _fit_extents(u0, u1, v0, v1, axes, aspect, *, passes=6):
    """Shrink the box extents until its four corners fit.

    With the angle fixed, each corner is a LINEAR function of the four extents,
    so each frame constraint (`0 <= x <= aspect`, `0 <= y <= 1`) solves as a
    bound on the extent being adjusted.

    Two things are needed, and both came from measuring, not reasoning
    ------------------------------------------------------------------
    **Each constraint goes to the axis MOST ALIGNED with it.** Without that the
    fit is valid and still disastrous. Real case (`Multiple_Euro_154`): a nearly
    horizontal banknote sticking out 0.029 at the top. Since the long axis has
    `uy = 0.0315`, the constraint `y >= 0` can also be satisfied by shrinking
    the LONG side... from 0.890 to 0.064. It met everything and kept 8% of the
    banknote.

    **Several orders are tried and the largest area wins.** With a single order,
    coordinate ascent gets stuck at angles near 45 degrees, where both axes
    weigh almost the same in both constraints, and can collapse a dimension to
    zero. Four orders cost nothing and remove the problem.

    It is still not the maximum-area inscribed rectangle: that would need an
    LP, one more dependency to refine something already within the noise of
    the clipping itself.
    """
    best = None
    for order in _TRIM_ORDERS:
        candidate = _trim(u0, u1, v0, v1, axes, aspect, order, passes)
        # A candidate that does not fit is worthless however large: the
        # anti-collapse guard in `_trim` may leave a constraint unsatisfied.
        if not _inside(*candidate, axes, aspect):
            continue
        a0, a1, b0, b1 = candidate
        area = (a1 - a0) * (b1 - b0)
        if best is None or area > best[0]:
            best = (area, candidate)
    return best[1] if best is not None else None


def _visible_bounds(points, aspect):
    """AXIS-ALIGNED envelope of the visible part. `clip_quad`'s last resort.

    It loses the angle, which is why it is the last resort and not the rule.
    In exchange it is a rectangle and fits the frame by construction, always.
    """
    from shapely.geometry import Polygon, box

    visible = Polygon(points).intersection(box(0.0, 0.0, aspect, 1.0))
    if visible.is_empty or visible.area <= 0.0:
        # Entirely outside. Pinch and let the rest of the project decide: the
        # area filter or the quad's own validation.
        return canonicalize(
            Quad.from_xy(
                [
                    (min(max(x / aspect, 0.0), 1.0), min(max(y, 0.0), 1.0))
                    for x, y in points
                ]
            ),
            aspect=aspect,
        )
    x0, y0, x1, y1 = visible.bounds
    return canonicalize(
        Quad.from_xy(
            [
                (x0 / aspect, y0),
                (x1 / aspect, y0),
                (x1 / aspect, y1),
                (x0 / aspect, y1),
            ]
        ),
        aspect=aspect,
    )


def clip_quad(quad: Quad, *, aspect: float = 1.0) -> Quad:
    """Clip to the frame keeping the ANGLE, and return a RECTANGLE.

    This used to pinch each vertex to [0,1] separately. It is fast and keeps
    four vertices, but a ROTATED rectangle cut against a straight frame does
    not give a smaller rectangle: it gives a trapezoid. Measured on the real
    data, **82 of 679 annotations (12%) stopped being rectangles**, and the
    only recorded run trained like that.

    That is a target the model cannot reach: it predicts `(cx, cy, w, h,
    angle)`, i.e. rectangles. A trapezoid as ground truth can only be learned
    as the rectangle that fits it least badly.

    What it does now, in the box's own axes (the long side and its
    perpendicular):

    1. Start from the original box.
    2. Shrink its four extents until the four corners fit the frame, with the
       angle intact.

    Why "the visible part" is NOT enveloped
    ---------------------------------------
    The first attempt was to intersect the polygon with the frame and take its
    envelope in those same axes. **It does not work, and silently so**: cutting
    a CORNER leaves the other three intact, and those fix the envelope.
    Measured, the 99 out-of-frame annotations were still out, by up to 23% of
    the side. It is written down because the failure cannot be seen by reading
    the code, only by measuring.

    Why the angle is kept
    ---------------------
    The annotation's angle is data; a clipped trapezoid's is an artifact of the
    cut. Recomputing it with a minimum-area box would add noise to the only
    quantity we measure besides IoU.

    `aspect` is not truly optional: in normalized coordinates a rotated
    rectangle is a parallelogram, so "being a rectangle" only means something
    in pixels.
    """
    points = [(x * aspect, y) for x, y in quad.points]
    (x0, y0), (x1, y1) = points[0], points[1]
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0.0:
        return canonicalize(Quad.from_xy(quad.points), aspect=aspect)

    # The canonical order anchors p0->p1 on the longest side: those are the axes.
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    vx, vy = -uy, ux
    us = [px * ux + py * uy for px, py in points]
    vs = [px * vx + py * vy for px, py in points]

    extents = _fit_extents(
        min(us), max(us), min(vs), max(vs), ((ux, uy), (vx, vy)), aspect
    )
    if extents is None:
        # No rectangle with THAT angle fits the frame. It happens when the
        # banknote is almost entirely outside, and then keeping the angle is no
        # longer what matters: return the axis-aligned envelope of the visible
        # part, which is a rectangle and fits by construction.
        #
        # These are the cases `pad` handles well and `clip` cannot: none appear
        # in the real data, and the relative area filter takes out banknotes
        # that covered.
        return _visible_bounds(points, aspect)

    u0, u1, v0, v1 = extents
    rectangle = [
        (u * ux + v * vx, u * uy + v * vy)
        for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    ]
    return canonicalize(
        Quad.from_xy(
            # The final pinch is against rounding error, not geometry: without
            # it a 1.0000000002 trips the quad validation.
            [(min(max(x / aspect, 0.0), 1.0), min(max(y, 0.0), 1.0))
             for x, y in rectangle]
        ),
        aspect=aspect,
    )


def pad_geometry(fraction: float) -> tuple[float, float]:
    """Scale and offset when normalizing over the padded image.

    With a border of `fraction` on each side, the total side is multiplied by
    `1 + 2f`, and the original image's origin lands at `f` of the new one.
    """
    scale = 1.0 + 2.0 * fraction
    return 1.0 / scale, fraction / scale


def pad_quad(quad: Quad, fraction: float) -> Quad:
    """Re-place a quad in the coordinates of the already padded image."""
    scale, offset = pad_geometry(fraction)
    return canonicalize(
        Quad.from_xy([(x * scale + offset, y * scale + offset) for x, y in quad.points])
    )


def pad_image(source: Path, target: Path, fraction: float) -> None:
    """Black border. Constant and not reflected on purpose: the border is
    invented content and has to LOOK like it, not mimic texture that was never
    photographed."""
    import cv2

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read {source}")
    height, width = image.shape[:2]
    top = bottom = round(height * fraction)
    left = right = round(width * fraction)
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), padded)


def out_of_bounds(quad: Quad) -> bool:
    return any(not (0.0 <= v <= 1.0) for v in quad.flat())


@dataclass(frozen=True, slots=True)
class PreparedSample:
    """A sample ready to be written, in whatever format."""

    sample: Sample
    #: Size of the image THAT GETS WRITTEN. With `pad` it is the padded one, not
    #: the original: pixel-based formats have to match the image.
    size: ImageSize
    quads: tuple[Quad, ...]
    class_ids: tuple[int, ...]
    #: Whether the image must be rewritten (padding) or linking is enough.
    needs_rewrite: bool = False
    #: How much padding it carries. It lives HERE and not as an argument of
    #: whoever writes: every exporter would have to remember to pass it, and
    #: forgetting it with policy `pad` would write the image without a border
    #: while the labels assume one. The sample knows what it is; the writer
    #: does not have to.
    pad_fraction: float = 0.0

    @property
    def sample_id(self) -> str:
        return self.sample.sample_id


@dataclass
class PreparationReport:
    policy: str
    pad_fraction: float = 0.0
    #: Annotations the area filter left out.
    dropped: int = 0
    #: Annotations that crossed the border.
    adjusted: int = 0
    #: Only with `pad`: the ones still outside after padding.
    clipped_after_pad: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        extra = ""
        if self.policy == OutOfBoundsPolicy.PAD.value:
            extra = (
                f", pad={self.pad_fraction:.0%}, "
                f"{self.clipped_after_pad} clipped anyway"
            )
        return (
            f"{counts}; {self.dropped} annotations filtered; "
            f"border={self.policy}, {self.adjusted} out of frame{extra}"
        )


def prepare(
    samples_by_split: dict[str, list],
    config: Config | None = None,
) -> tuple[dict[str, list[PreparedSample]], PreparationReport]:
    """Area filter + border policy, the same for every format."""
    config = config or Config()
    # Deliberate coercion: `Config.model_copy(update=...)` does NOT validate, so
    # an `out_of_bounds="clip"` set somewhere arrives as a str and blows up
    # later with an unrelated-looking AttributeError.
    policy = OutOfBoundsPolicy(config.detector.out_of_bounds)
    fraction = config.detector.pad_fraction
    report = PreparationReport(
        policy=policy.value,
        pad_fraction=fraction if policy is OutOfBoundsPolicy.PAD else 0.0,
    )

    out: dict[str, list[PreparedSample]] = {}
    for split, samples in samples_by_split.items():
        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        loaded, filter_report = load_samples(
            samples,
            sizes=sizes,
            min_relative_area=config.annotation_policy.min_relative_area,
        )
        report.dropped += len(filter_report.dropped)

        by_id = {s.sample_id: s for s in samples}
        prepared: list[PreparedSample] = []
        for item in loaded:
            width, height = sizes.size(item.sample_id)
            # The aspect goes to `clip_quad`: in normalized coordinates a
            # rotated rectangle is a parallelogram, and "clipping to a
            # rectangle" without it would happen in the wrong space.
            aspect = width / height
            quads: list[Quad] = []
            for annotation in item.annotations:
                quad = annotation.quad
                if out_of_bounds(quad):
                    report.adjusted += 1
                    if policy is OutOfBoundsPolicy.CLIP:
                        quad = clip_quad(quad, aspect=aspect)
                if policy is OutOfBoundsPolicy.PAD:
                    # ALL of them are re-placed, whether they stuck out or not:
                    # the image changed size and the coordinate system with it.
                    quad = pad_quad(quad, fraction)
                    if out_of_bounds(quad):
                        # The border was not enough. Clip instead of leaving
                        # it out, because leaving it out makes the trainer
                        # drop the WHOLE image.
                        report.clipped_after_pad += 1
                        # With padding the image is a different one: the aspect
                        # does not change (the border is proportional on both
                        # sides), but it is passed explicitly so it does not
                        # depend on remembering that.
                        quad = clip_quad(quad, aspect=aspect)
                quads.append(quad)

            if policy is OutOfBoundsPolicy.PAD:
                width = width + 2 * round(width * fraction)
                height = height + 2 * round(height * fraction)

            prepared.append(
                PreparedSample(
                    sample=by_id[item.sample_id],
                    size=ImageSize(int(width), int(height)),
                    quads=tuple(quads),
                    class_ids=tuple(a.class_id for a in item.annotations),
                    needs_rewrite=policy is OutOfBoundsPolicy.PAD,
                    pad_fraction=fraction if policy is OutOfBoundsPolicy.PAD else 0.0,
                )
            )
        out[split] = prepared
        report.counts[split] = len(prepared)
    return out, report


def place_image(item: PreparedSample, target: Path) -> None:
    """Pad if the sample asks for it; otherwise hard link, and if that fails, copy.

    A symlink would be better but on Windows it needs permissions that are not
    always there, and failing for that when starting to train would be absurd.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if item.needs_rewrite:
        pad_image(item.sample.image_path, target, item.pad_fraction)
        return
    if target.exists():
        return
    try:
        os.link(item.sample.image_path, target)
    except OSError:
        # Different volume, filesystem without hard links, or permissions.
        shutil.copy2(item.sample.image_path, target)


__all__ = [
    "DEFAULT_MIN_RELATIVE_AREA",
    "DroppedAnnotation",
    "FilterReport",
    "LoadedSample",
    "PreparationReport",
    "PreparedSample",
    "clip_quad",
    "filter_by_relative_area",
    "load_sample",
    "load_samples",
    "out_of_bounds",
    "pad_geometry",
    "pad_image",
    "pad_quad",
    "place_image",
    "prepare",
]
