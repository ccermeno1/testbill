"""Training-time augmentation in the Ultralytics recipe. Off by default.

The defaults mirror Ultralytics' `default.yaml` (the ones its OBB models
train with, and what the `ultralytics-yolo-obb` reference row uses):

    mosaic 1.0 (off for the last `close_mosaic = 10` epochs)
    scale +-0.5, translate +-0.1
    HSV gains: hue 0.015, saturation 0.7, value 0.4
    horizontal flip 0.5, vertical flip 0.0

Reproduced from its public documentation and default hyperparameters, not
from its code (AGPL: none of it has been read). Same order of operations as
theirs: mosaic -> random perspective -> HSV -> flips.

Where this recipe departs from theirs, because the model was missing the
banknotes that are not horizontal or vertical (the training photos mostly
are): the random perspective ROTATES by any angle (`degrees`, +-180 by
default: a banknote is valid in every orientation), adds a little shear and
a very small perspective. Ultralytics has the three knobs but ships them at
0. And when a rotation would push a banknote out of the frame, instead of
cutting it (their behaviour) the image is zoomed out until it fits and the
empty canvas is filled with a BACKGROUND: the median colour of the photo's
border, which is the table the banknote lies on (`keep_whole`, `fill`). A
cut banknote teaches a truncated box; a whole one on more table teaches the
rotation, which is what was missing.

Rotations by multiples of 90 degrees (`rotations`) are kept as an exact,
separate option, off by default: with `degrees` they are redundant.

Geometry of the quads
---------------------
Every pixel operation is applied to the quad with the same map, in
homogeneous coordinates for the perspective. Rotation keeps a rectangle a
rectangle; shear and perspective make it a slight quadrilateral, and the
quad IS that quadrilateral (the label follows the pixels, not the other way
round). A quad pushed partly out of the frame goes through `clip_quad` (the
same rectangle-preserving clip as the `clip` border policy), and it is
dropped when less than `min_visible` of it survives (Ultralytics'
`area_thr`) or it becomes thinner than 2 px. Tested by painting the quad as
a mask, warping the mask and comparing with the mask of the warped quad.

Who and when
------------
Only the `train` split, only in this project's loop (own variants, DDGRCF
port, Rotated FCOS). `valid` and `test` are measured on the real photos.
Deterministic: the dataset owns a `random.Random(seed)` drawn in sampler
order; with `num_workers = 0` two runs with the same seed see the same
images. The dataset knows the epoch (`set_epoch`) for `close_mosaic`.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Callable, Sequence

import cv2
import numpy as np
from shapely.geometry import Polygon, box

from testbank.dataio.prepare import clip_quad
from testbank.geometry.quad import (
    COORD_MAX,
    COORD_MIN,
    CoordinateRangeWarning,
    Quad,
    QuadShapeWarning,
    canonicalize,
    flip_horizontal,
)

#: Gray of the padding, as in Ultralytics.
FILL = 114
#: Minimum side of a surviving box, in pixels (their `wh_thr`).
MIN_SIDE_PX = 2.0
#: With `keep_whole`, the boxes are kept this far from the frame, as a
#: fraction of the side: a box touching the border is indistinguishable
#: from a cut one.
KEEP_WHOLE_MARGIN = 0.02
#: Width of the ring of border pixels the background colour is read from.
BORDER_RING = 8


# --- exact geometry ---------------------------------------------------------


def flip_vertical(quad: Quad) -> Quad:
    """Reflection y -> 1-y, raw like `flip_horizontal`: exact on the grid."""
    return Quad(points=tuple((x, 1.0 - y) for x, y in quad.points))  # type: ignore[arg-type]


def rotate90(quad: Quad, k: int) -> Quad:
    """`k` quarter turns COUNTER-clockwise on a square image, matching
    `np.rot90(image, k)`: one turn maps `(x, y) -> (y, 1 - x)`."""
    points = list(quad.points)
    for _ in range(k % 4):
        points = [(y, 1.0 - x) for x, y in points]
    return Quad(points=tuple(points))  # type: ignore[arg-type]


# --- HSV ----------------------------------------------------------------------


def hsv_jitter(image: np.ndarray, rng: random.Random, *, h: float, s: float, v: float) -> np.ndarray:
    """Ultralytics-style HSV: random gains `1 + U(-x, x)` on each channel.
    Hue wraps, saturation and value clip. uint8 BGR in and out."""
    if not (h or s or v):
        return image
    gains = np.array([rng.uniform(-h, h), rng.uniform(-s, s), rng.uniform(-v, v)]) + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_hue = ((x * gains[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * gains[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * gains[2], 0, 255).astype(np.uint8)
    merged = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


# --- affine and the quads that survive it ------------------------------------


def _survivors(
    corners_px: Sequence[np.ndarray],
    side: int,
    *,
    min_visible: float,
) -> list[Quad]:
    with warnings.catch_warnings():
        return _survivors_unguarded(corners_px, side, min_visible=min_visible)


def _survivors_unguarded(
    corners_px: Sequence[np.ndarray],
    side: int,
    *,
    min_visible: float,
) -> list[Quad]:
    """Quads (normalized to `side`) of the boxes whose corners `corners_px`
    (4x2 each, in output pixels) still show enough of themselves.

    Three steps, in order: drop what shows less than `min_visible` of its
    area inside the frame or is thinner than 2 px; shrink along the box's own
    axes into the tolerant range `Quad` accepts (a coarse clip, same
    function, wider frame); then the rectangle-preserving clip to [0, 1].
    """
    frame = box(0.0, 0.0, float(side), float(side))
    out: list[Quad] = []
    # `Quad`'s warnings judge ANNOTATIONS (a vertex outside the frame, an
    # unstable anchor). A transformed box is not an annotation; here those
    # states are the normal outcome of a crop, and with pytest they would be
    # errors.
    warnings.simplefilter("ignore", QuadShapeWarning)
    warnings.simplefilter("ignore", CoordinateRangeWarning)
    for pts in corners_px:
        polygon = Polygon(pts)
        if not polygon.is_valid or polygon.area <= 0:
            continue
        visible = polygon.intersection(frame)
        if visible.area / polygon.area < min_visible:
            continue
        sides = [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        if min(sides) < MIN_SIDE_PX:
            continue
        norm = [(float(x) / side, float(y) / side) for x, y in pts]
        if any(not (COORD_MIN <= v <= COORD_MAX) for xy in norm for v in xy):
            # Coarse clip to the tolerant box [-0.5, 1.5]: express it as
            # [0, 1] of a frame twice as large, clip there, come back.
            wide = [((x + 0.5) / 2.0, (y + 0.5) / 2.0) for x, y in norm]
            if any(not (COORD_MIN <= v <= COORD_MAX) for xy in wide for v in xy):
                continue  # more than a frame and a half away: gone
            clipped = clip_quad(Quad.from_xy(wide))
            norm = [(2.0 * x - 0.5, 2.0 * y - 0.5) for x, y in clipped.points]
        quad = Quad.from_xy(norm)
        if any(not (0.0 <= v <= 1.0) for xy in norm for v in xy):
            quad = clip_quad(quad)
        out.append(canonicalize(quad))
    return out


def background_colour(image: np.ndarray, ring: int = BORDER_RING) -> tuple[int, int, int]:
    """Median BGR of the border ring: on these photos, the table."""
    h, w = image.shape[:2]
    r = max(1, min(ring, h // 2, w // 2))
    strips = [image[:r], image[-r:], image[:, :r], image[:, -r:]]
    pixels = np.concatenate([s.reshape(-1, 3) for s in strips])
    return tuple(int(v) for v in np.median(pixels, axis=0))


_HALF = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
_MINUS_HALF = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])


def _apply(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """3x3 map on 4x2 corners, homogeneous: divides by w for the perspective."""
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def perspective_matrix(
    rng: random.Random,
    *,
    width: int,
    height: int,
    out_side: int,
    degrees: float,
    scale: float,
    translate: float,
    shear: float,
    perspective: float,
) -> np.ndarray:
    """One random draw of Ultralytics' `random_perspective` as a 3x3 matrix:
    centre, perspective, rotation+scale, shear, translation, composed in
    that order. `degrees` and `shear` in degrees, `perspective` on the
    projective row (0.0005 is already a lot), `translate` as a fraction of
    the output side."""
    centre = np.eye(3)
    centre[0, 2], centre[1, 2] = -width / 2.0, -height / 2.0
    projective = np.eye(3)
    projective[2, 0] = rng.uniform(-perspective, perspective)
    projective[2, 1] = rng.uniform(-perspective, perspective)
    rotation = np.eye(3)
    angle = rng.uniform(-degrees, degrees)
    s = rng.uniform(1.0 - scale, 1.0 + scale)
    rotation[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0.0, 0.0), scale=s)
    shearing = np.eye(3)
    shearing[0, 1] = math.tan(math.radians(rng.uniform(-shear, shear)))
    shearing[1, 0] = math.tan(math.radians(rng.uniform(-shear, shear)))
    translation = np.eye(3)
    translation[0, 2] = (0.5 + rng.uniform(-translate, translate)) * out_side
    translation[1, 2] = (0.5 + rng.uniform(-translate, translate)) * out_side
    return translation @ shearing @ rotation @ projective @ centre


def fit_inside(
    matrix: np.ndarray,
    corners_px: Sequence[np.ndarray],
    *,
    out_side: int,
    centre: tuple[float, float],
    margin: float = KEEP_WHOLE_MARGIN,
) -> np.ndarray:
    """Zooms `matrix` out, about `centre` (where the source centre lands, in
    output pixels), just enough for every box to land inside
    `[margin, 1 - margin]` of the output. Never zooms in. The empty canvas
    that appears is the background's job.

    `centre` has to be INSIDE the frame for the ratios below to be positive;
    it is, by construction (translate <= 0.5). It was once read off the
    matrix as where (0, 0) lands, which is the source's top-left corner,
    not its centre: a large rotation put it outside, the factor went
    negative and the image collapsed to a point.
    """
    if not corners_px:
        return matrix
    cx, cy = centre
    if not (0.0 <= cx <= out_side and 0.0 <= cy <= out_side):
        raise ValueError(f"the centre must land inside the frame, got {centre}")
    low, high = margin * out_side, (1.0 - margin) * out_side
    factor = 1.0
    for pts in corners_px:
        for px, py in _apply(matrix, pts):
            for value, origin in ((px, cx), (py, cy)):
                offset = value - origin
                if offset > 0 and origin + offset > high:
                    factor = min(factor, (high - origin) / offset)
                elif offset < 0 and origin + offset < low:
                    factor = min(factor, (low - origin) / offset)
    if factor >= 1.0:
        return matrix
    factor = max(factor, 0.0)
    zoom = np.array(
        [[factor, 0.0, cx * (1.0 - factor)], [0.0, factor, cy * (1.0 - factor)], [0.0, 0.0, 1.0]]
    )
    return zoom @ matrix


def random_perspective(
    image: np.ndarray,
    corners_px: Sequence[np.ndarray],
    rng: random.Random,
    *,
    out_side: int,
    degrees: float = 0.0,
    scale: float = 0.5,
    translate: float = 0.1,
    shear: float = 0.0,
    perspective: float = 0.0,
    min_visible: float = 0.1,
    keep_whole: bool = False,
    fill: tuple[int, int, int] = (FILL, FILL, FILL),
) -> tuple[np.ndarray, list[Quad]]:
    """Rotation, scale, shear, perspective and translation in one warp into
    an `out_side` square, the corners mapped with the same matrix.

    With `keep_whole`, boxes that are entirely inside the source stay
    entirely inside the output: the draw is zoomed out as needed and the
    canvas that appears takes `fill`. Boxes already cut by the source frame
    (a mosaic tile's edge) are not protected: they were cut before.
    """
    h, w = image.shape[:2]
    # OpenCV warps in pixel-CENTRE coordinates (pixel i sits at i); the quads
    # are in pixel-EDGE coordinates (0..side, pixel i spans i..i+1). The same
    # map expressed in edge coordinates is the matrix conjugated by a half
    # pixel; without it every box lands half a pixel off the pixels.
    centre_matrix = perspective_matrix(
        rng, width=w, height=h, out_side=out_side, degrees=degrees, scale=scale,
        translate=translate, shear=shear, perspective=perspective,
    )
    matrix = _HALF @ centre_matrix @ _MINUS_HALF
    if keep_whole:
        source = box(0.0, 0.0, float(w), float(h))
        whole = [pts for pts in corners_px if source.contains(Polygon(pts))]
        landed = _apply(matrix, np.array([[w / 2.0, h / 2.0]]))[0]
        matrix = fit_inside(
            matrix, whole, out_side=out_side, centre=(float(landed[0]), float(landed[1]))
        )
        centre_matrix = _MINUS_HALF @ matrix @ _HALF
    if perspective:
        warped = cv2.warpPerspective(
            image, centre_matrix, (out_side, out_side), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=fill,
        )
    else:
        warped = cv2.warpAffine(
            image, centre_matrix[:2], (out_side, out_side), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=fill,
        )
    moved = [_apply(matrix, pts) for pts in corners_px]
    return warped, _survivors(moved, out_side, min_visible=min_visible)


def random_affine(
    image: np.ndarray,
    corners_px: Sequence[np.ndarray],
    rng: random.Random,
    *,
    out_side: int,
    scale: float,
    translate: float,
    min_visible: float,
) -> tuple[np.ndarray, list[Quad]]:
    """Scale and translate only: Ultralytics' defaults. The general case is
    `random_perspective`."""
    return random_perspective(
        image, corners_px, rng, out_side=out_side, scale=scale, translate=translate,
        min_visible=min_visible,
    )


def _corners_px(quad: Quad, side: int) -> np.ndarray:
    return np.array([[x * side, y * side] for x, y in quad.points], dtype=np.float64)


# --- mosaic ----------------------------------------------------------------


def mosaic4(
    tiles: Sequence[tuple[np.ndarray, Sequence[Quad]]],
    rng: random.Random,
    *,
    side: int,
    min_visible: float = 0.1,
    fill: tuple[int, int, int] = (FILL, FILL, FILL),
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Four `side x side` images on a `2 side x 2 side` canvas around a random
    center, as Ultralytics does: the first image ends at the center
    (bottom-right corner), the second starts to its right, the third below,
    the fourth diagonally. Returns the canvas and the box corners in canvas
    pixels (the perspective that follows crops it back to `side`)."""
    canvas = np.full((2 * side, 2 * side, 3), fill, dtype=np.uint8)
    xc = rng.uniform(0.5 * side, 1.5 * side)
    yc = rng.uniform(0.5 * side, 1.5 * side)
    corners: list[np.ndarray] = []
    for index, (image, quads) in enumerate(tiles[:4]):
        h, w = image.shape[:2]
        if index == 0:
            x1, y1 = xc - w, yc - h
        elif index == 1:
            x1, y1 = xc, yc - h
        elif index == 2:
            x1, y1 = xc - w, yc
        else:
            x1, y1 = xc, yc
        # Paste the part of the tile that falls inside the canvas.
        cx1, cy1 = int(max(x1, 0)), int(max(y1, 0))
        cx2, cy2 = int(min(x1 + w, 2 * side)), int(min(y1 + h, 2 * side))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        sx1, sy1 = cx1 - int(x1), cy1 - int(y1)
        canvas[cy1:cy2, cx1:cx2] = image[sy1 : sy1 + (cy2 - cy1), sx1 : sx1 + (cx2 - cx1)]
        offset = np.array([int(x1), int(y1)], dtype=np.float64)
        corners.extend(_corners_px(q, side) + offset for q in quads)
    # Clip to the canvas, as Ultralytics clips its labels to the mosaic: a
    # tile that overflowed the canvas lost those pixels, and the affine that
    # follows (down to half scale) would otherwise see its box over gray.
    kept = _survivors(corners, 2 * side, min_visible=min_visible)
    return canvas, [_corners_px(q, 2 * side) for q in kept]


# --- the composite ----------------------------------------------------------


def augment(
    image: np.ndarray,
    quads: Sequence[Quad],
    rng: random.Random,
    *,
    others: Callable[[], tuple[np.ndarray, Sequence[Quad]]] | None = None,
    mosaic: float = 1.0,
    scale: float = 0.5,
    translate: float = 0.1,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    flip_h: float = 0.5,
    flip_v: float = 0.0,
    rotations: bool = False,
    min_visible: float = 0.1,
    degrees: float = 0.0,
    shear: float = 0.0,
    perspective: float = 0.0,
    keep_whole: bool = False,
    fill: str = "gray",
) -> tuple[np.ndarray, list[Quad]]:
    """One random draw of the whole recipe on a SQUARE BGR uint8 image with
    normalized quads. `others()` supplies another random training image (with
    its quads) for the mosaic; without it, or with `mosaic = 0`, the single
    image goes through the random perspective alone. `fill` is `gray`
    (Ultralytics' 114) or `border` (the photo's own background). Returns the
    image and the quads, canonical again."""
    side = image.shape[0]
    if image.shape[1] != side:
        raise ValueError(f"augment expects a square image, got {image.shape[:2]}")
    if fill not in ("gray", "border"):
        raise ValueError(f"fill must be 'gray' or 'border', got {fill!r}")
    with warnings.catch_warnings():
        # See `_survivors`: transformed boxes are not annotations.
        warnings.simplefilter("ignore", QuadShapeWarning)
        warnings.simplefilter("ignore", CoordinateRangeWarning)
        return _augment(
            image, quads, rng, side=side, others=others, mosaic=mosaic, scale=scale,
            translate=translate, hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v, flip_h=flip_h,
            flip_v=flip_v, rotations=rotations, min_visible=min_visible, degrees=degrees,
            shear=shear, perspective=perspective, keep_whole=keep_whole, fill=fill,
        )


def _augment(image, quads, rng, *, side, others, mosaic, scale, translate, hsv_h, hsv_s,
             hsv_v, flip_h, flip_v, rotations, min_visible, degrees, shear, perspective,
             keep_whole, fill):
    colour = background_colour(image) if fill == "border" else (FILL, FILL, FILL)
    if others is not None and mosaic > 0 and rng.random() < mosaic:
        tiles = [(image, list(quads))] + [others() for _ in range(3)]
        canvas, corners = mosaic4(tiles, rng, side=side, min_visible=min_visible, fill=colour)
        # A mosaic is cut by construction (2 sides into 1): keeping every
        # box whole would mean zooming the whole canvas out to half. The
        # tiles keep their cuts, as in Ultralytics.
        whole = False
    else:
        canvas, corners = image, [_corners_px(q, side) for q in quads]
        whole = keep_whole
    out, moved = random_perspective(
        canvas, corners, rng, out_side=side, degrees=degrees, scale=scale,
        translate=translate, shear=shear, perspective=perspective,
        min_visible=min_visible, keep_whole=whole, fill=colour,
    )
    out = hsv_jitter(out, rng, h=hsv_h, s=hsv_s, v=hsv_v)
    if flip_v and rng.random() < flip_v:
        out = out[::-1]
        moved = [canonicalize(flip_vertical(q)) for q in moved]
    if flip_h and rng.random() < flip_h:
        out = out[:, ::-1]
        moved = [canonicalize(flip_horizontal(q)) for q in moved]
    if rotations:
        k = rng.randrange(4)
        if k:
            out = np.rot90(out, k)
            moved = [canonicalize(rotate90(q, k)) for q in moved]
    return np.ascontiguousarray(out), moved


__all__ = [
    "FILL",
    "KEEP_WHOLE_MARGIN",
    "augment",
    "background_colour",
    "fit_inside",
    "flip_vertical",
    "hsv_jitter",
    "mosaic4",
    "perspective_matrix",
    "random_affine",
    "random_perspective",
    "rotate90",
]
