"""Training-time augmentation in the Ultralytics recipe. Off by default.

The defaults mirror Ultralytics' `default.yaml` (the ones its OBB models
train with, and what the `ultralytics-yolo-obb` reference row uses):

    mosaic 1.0 (off for the last `close_mosaic = 10` epochs)
    scale +-0.5, translate +-0.1, no rotation, no shear
    HSV gains: hue 0.015, saturation 0.7, value 0.4
    horizontal flip 0.5, vertical flip 0.0

Reproduced from its public documentation and default hyperparameters, not
from its code (AGPL: none of it has been read). Same order of operations as
theirs: mosaic -> affine -> HSV -> flips.

Two things of our own, OFF by default so the default is theirs: vertical
flip and rotations by multiples of 90 degrees, which are exact for banknotes
(valid in any orientation) and cost nothing.

Geometry of the quads
---------------------
Every pixel operation is applied to the quad with the same map. The affine
can push a quad partly out of the frame; then it goes through `clip_quad`
(the same rectangle-preserving clip as the `clip` border policy), and it is
dropped when less than `min_visible` of it survives (Ultralytics'
`area_thr`) or it becomes thinner than 2 px. Tested by painting the quad as a
mask, warping the mask and comparing with the mask of the warped quad.

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
    """Scale by `U(1 - scale, 1 + scale)` about the canvas center and
    translate by `U(-translate, translate)` of the output side, into an
    `out_side` square. No rotation, no shear: Ultralytics' defaults."""
    h, w = image.shape[:2]
    s = rng.uniform(1.0 - scale, 1.0 + scale)
    tx = (0.5 + rng.uniform(-translate, translate)) * out_side
    ty = (0.5 + rng.uniform(-translate, translate)) * out_side
    matrix = np.array([[s, 0.0, tx - s * w / 2.0], [0.0, s, ty - s * h / 2.0]], dtype=np.float64)
    warped = cv2.warpAffine(
        image, matrix, (out_side, out_side), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(FILL, FILL, FILL),
    )
    moved = [pts @ matrix[:, :2].T + matrix[:, 2] for pts in corners_px]
    return warped, _survivors(moved, out_side, min_visible=min_visible)


def _corners_px(quad: Quad, side: int) -> np.ndarray:
    return np.array([[x * side, y * side] for x, y in quad.points], dtype=np.float64)


# --- mosaic ----------------------------------------------------------------


def mosaic4(
    tiles: Sequence[tuple[np.ndarray, Sequence[Quad]]],
    rng: random.Random,
    *,
    side: int,
    min_visible: float = 0.1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Four `side x side` images on a `2 side x 2 side` canvas around a random
    center, as Ultralytics does: the first image ends at the center
    (bottom-right corner), the second starts to its right, the third below,
    the fourth diagonally. Returns the canvas and the box corners in canvas
    pixels (the affine that follows crops it back to `side`)."""
    canvas = np.full((2 * side, 2 * side, 3), FILL, dtype=np.uint8)
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
) -> tuple[np.ndarray, list[Quad]]:
    """One random draw of the whole recipe on a SQUARE BGR uint8 image with
    normalized quads. `others()` supplies another random training image (with
    its quads) for the mosaic; without it, or with `mosaic = 0`, the single
    image goes through the affine alone. Returns the image and the quads,
    canonical again."""
    side = image.shape[0]
    if image.shape[1] != side:
        raise ValueError(f"augment expects a square image, got {image.shape[:2]}")
    with warnings.catch_warnings():
        # See `_survivors`: transformed boxes are not annotations.
        warnings.simplefilter("ignore", QuadShapeWarning)
        warnings.simplefilter("ignore", CoordinateRangeWarning)
        return _augment(
            image, quads, rng, side=side, others=others, mosaic=mosaic, scale=scale,
            translate=translate, hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v, flip_h=flip_h,
            flip_v=flip_v, rotations=rotations, min_visible=min_visible,
        )


def _augment(image, quads, rng, *, side, others, mosaic, scale, translate, hsv_h, hsv_s,
             hsv_v, flip_h, flip_v, rotations, min_visible):
    if others is not None and mosaic > 0 and rng.random() < mosaic:
        tiles = [(image, list(quads))] + [others() for _ in range(3)]
        canvas, corners = mosaic4(tiles, rng, side=side, min_visible=min_visible)
    else:
        canvas, corners = image, [_corners_px(q, side) for q in quads]
    out, moved = random_affine(
        canvas, corners, rng, out_side=side, scale=scale, translate=translate,
        min_visible=min_visible,
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
    "augment",
    "flip_vertical",
    "hsv_jitter",
    "mosaic4",
    "random_affine",
    "rotate90",
]
