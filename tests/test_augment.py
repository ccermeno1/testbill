"""Training-time augmentation: the quad follows the pixels, exactly.

The check that matters is geometric: paint the quad as a mask, transform
the MASK with the same image operation, and compare it with the mask of the
transformed QUAD. If they differ, the boxes and the pixels have parted ways
and the model learns a target that is not in the image.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
import pytest
from conftest import rotated_rect_points

from testbank.config import AugmentConfig
from testbank.geometry.quad import Quad, canonicalize, flip_horizontal, is_canonical
from testbank.models.augment import (
    augment,
    flip_vertical,
    hsv_jitter,
    mosaic4,
    random_affine,
    rotate90,
)

SIDE = 128


def _quad(cx, cy, half_long=0.25, ratio=2.0, theta=0.4) -> Quad:
    return canonicalize(Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta)))


def _mask(quad: Quad, side: int = SIDE, *, subpixel: bool = False) -> np.ndarray:
    """`subpixel`: fixed point with 8 fractional bits, for the warps, where
    truncating the corners to integers biases the mask up to a pixel (a
    tenth of the IoU on a 20 px box). The exact-geometry tests mirror or
    turn a raster and compare it with the raster of the mirrored quad, and
    there the truncated version is what stays symmetric (a rasterizer's
    fill rule is not)."""
    canvas = np.zeros((side, side), dtype=np.uint8)
    if subpixel:
        pts = np.array([[x * side * 256, y * side * 256] for x, y in quad.points]).astype(np.int32)
        cv2.fillPoly(canvas, [pts], 1, shift=8)
    else:
        pts = np.array([[x * side, y * side] for x, y in quad.points], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], 1)
    return canvas


def _corners(quad: Quad, side: int = SIDE) -> np.ndarray:
    return np.array([[x * side, y * side] for x, y in quad.points], dtype=np.float64)


def _iou(a, b) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 1.0


def _image_of(quad: Quad, side: int = SIDE) -> np.ndarray:
    return np.repeat(_mask(quad, side)[..., None] * 255, 3, axis=2).astype(np.uint8)


# --- exact geometry -------------------------------------------------------


def test_flip_horizontal_follows_the_pixels():
    q = _quad(0.4, 0.55)
    assert _iou(_mask(q)[:, ::-1], _mask(canonicalize(flip_horizontal(q)))) > 0.97


def test_flip_vertical_follows_the_pixels():
    q = _quad(0.4, 0.55)
    assert _iou(_mask(q)[::-1], _mask(canonicalize(flip_vertical(q)))) > 0.97


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotate90_follows_np_rot90(k):
    q = _quad(0.3, 0.6, theta=0.9)
    assert _iou(np.rot90(_mask(q), k), _mask(canonicalize(rotate90(q, k)))) > 0.97


# --- the affine -----------------------------------------------------------


def test_the_affine_moves_the_quad_with_the_pixels():
    q = _quad(0.45, 0.5, half_long=0.2, theta=0.7)
    rng = random.Random(0)
    for _ in range(30):
        out, quads = random_affine(
            _image_of(q), [_corners(q)], rng, out_side=SIDE, scale=0.5, translate=0.1, min_visible=0.1,
        )
        assert out.shape == (SIDE, SIDE, 3)
        if not quads:
            continue  # scaled or pushed mostly out: dropped, legitimately
        assert is_canonical(quads[0])
        painted = out[..., 0] > 127
        # The warped mask and the quad's mask must agree where the quad is.
        assert _iou(painted, _mask(quads[0], subpixel=True)) > 0.9


def test_a_box_pushed_out_of_the_frame_is_clipped_to_a_rectangle_inside():
    q = _quad(0.5, 0.5, half_long=0.3, theta=0.3)
    # A translation of nearly a full side: most of the box leaves.
    pts = _corners(q) + np.array([0.7 * SIDE, 0.0])
    from testbank.models.augment import _survivors

    kept = _survivors([pts], SIDE, min_visible=0.1)
    assert len(kept) == 1
    assert all(0.0 <= v <= 1.0 for xy in kept[0].points for v in xy)


def test_a_box_that_keeps_too_little_is_dropped():
    q = _quad(0.5, 0.5, half_long=0.3)
    pts = _corners(q) + np.array([0.98 * SIDE, 0.0])
    from testbank.models.augment import _survivors

    assert _survivors([pts], SIDE, min_visible=0.1) == []


# --- the mosaic -----------------------------------------------------------


def test_mosaic_places_four_tiles_and_their_quads_where_the_pixels_are():
    quads = [_quad(0.5, 0.5, half_long=0.2, theta=t) for t in (0.2, 0.9, 1.5, 2.3)]
    tiles = [(_image_of(q), [q]) for q in quads]
    rng = random.Random(4)
    canvas, corners = mosaic4(tiles, rng, side=SIDE)
    assert canvas.shape == (2 * SIDE, 2 * SIDE, 3)
    assert len(corners) == 4
    painted = canvas[..., 0] > 127
    for pts in corners:
        mask = np.zeros((2 * SIDE, 2 * SIDE), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        inside = mask.astype(bool)
        # Pixels of this quad that are on the canvas must be painted.
        assert painted[inside].mean() > 0.95 if inside.any() else True


def test_the_full_recipe_keeps_quads_on_the_pixels_and_inside_the_frame():
    q = _quad(0.5, 0.5, half_long=0.2, theta=1.1)
    rng = random.Random(1)
    n_boxes = []
    for _ in range(30):
        out, quads = augment(
            _image_of(q), [q], rng, others=lambda: (_image_of(q), [q]),
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        )
        assert out.shape == (SIDE, SIDE, 3) and out.dtype == np.uint8
        n_boxes.append(len(quads))
        painted = out[..., 0] > 127
        for quad in quads:
            assert is_canonical(quad)
            assert all(0.0 <= v <= 1.0 for xy in quad.points for v in xy)
            m = _mask(quad).astype(bool)
            touches_border = any(v <= 0.01 or v >= 0.99 for xy in quad.points for v in xy)
            # A box cut by the frame went through `clip_quad`, the border
            # policy of the annotations: the rectangle that keeps the angle
            # (or the envelope when nothing else fits) covers some gray.
            # Measured on the real export it keeps a median 0.985 of the
            # visible part with p05 0.79; here (median 0.987, one draw at
            # 0.48) the cut boxes are the extreme ones, so the bar is lower.
            # Whole boxes must sit on the pixels (min 0.93 measured).
            assert painted[m].mean() > (0.45 if touches_border else 0.85)
    assert max(n_boxes) > 1, "a mosaic has to bring boxes from the other tiles"


def test_without_others_there_is_no_mosaic():
    q = _quad(0.5, 0.5, half_long=0.2)
    rng = random.Random(2)
    for _ in range(10):
        _, quads = augment(_image_of(q), [q], rng, others=None, hsv_s=0.0, hsv_v=0.0, hsv_h=0.0)
        assert len(quads) <= 1


def test_augment_refuses_a_non_square_image():
    with pytest.raises(ValueError, match="square"):
        augment(np.zeros((64, 80, 3), np.uint8), [], random.Random(0))


# --- HSV ------------------------------------------------------------------


def test_hsv_jitter_changes_pixels_and_stays_uint8():
    image = np.full((SIDE, SIDE, 3), 120, dtype=np.uint8)
    image[:, : SIDE // 2] = (30, 90, 200)
    out = hsv_jitter(image, random.Random(1), h=0.015, s=0.7, v=0.4)
    assert out.dtype == np.uint8 and out.shape == image.shape
    assert not np.array_equal(out, image)


def test_hsv_jitter_with_zero_gains_is_the_identity():
    image = np.random.default_rng(0).integers(0, 255, (SIDE, SIDE, 3), dtype=np.uint8)
    assert np.array_equal(hsv_jitter(image, random.Random(0), h=0, s=0, v=0), image)


# --- determinism, config and scope ----------------------------------------


def test_the_same_seed_gives_the_same_draws():
    q = _quad(0.4, 0.5)
    image = np.random.default_rng(0).integers(0, 255, (SIDE, SIDE, 3), dtype=np.uint8)
    a = [augment(image, [q], random.Random(7), others=lambda: (image, [q]))[0] for _ in range(3)]
    b = [augment(image, [q], random.Random(7), others=lambda: (image, [q]))[0] for _ in range(3)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_defaults_are_ultralytics_and_off():
    cfg = AugmentConfig()
    assert cfg.enabled is False
    assert (cfg.mosaic, cfg.close_mosaic, cfg.scale, cfg.translate) == (1.0, 10, 0.5, 0.1)
    assert (cfg.hsv_h, cfg.hsv_s, cfg.hsv_v) == (0.015, 0.7, 0.4)
    assert (cfg.flip_horizontal, cfg.flip_vertical, cfg.rotations) == (0.5, 0.0, False)


def test_the_flag_switches_it_on():
    from testbank.cli import build_parser

    assert build_parser().parse_args(["train", "yolox-obb-nano", "--augment"]).augment is True
    assert build_parser().parse_args(["train", "yolox-obb-nano"]).augment is False


def _dataset(tmp_path, epochs=12, **aug):
    torch = pytest.importorskip("torch")  # noqa: F841
    from PIL import Image

    from testbank.config import Config
    from testbank.data.discover import Sample
    from testbank.dataio.formats import get as get_format
    from testbank.models.data import build_datasets

    def write(split, sid):
        d = tmp_path / split
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        img = d / "images" / f"{sid}.jpg"
        Image.new("RGB", (64, 64), (60, 60, 60)).save(img)
        lab = d / "labels" / f"{sid}.txt"
        lab.write_text("0 " + get_format("obb_yolo").from_quad(_quad(0.5, 0.5, 0.2)).payload + "\n")
        return Sample(sample_id=sid, image_path=img, label_path=lab)

    base = Config()
    config = base.model_copy(update={
        "data": base.data.model_copy(update={"derived_dir": tmp_path / "d"}),
        "detector": base.detector.model_copy(update={
            "image_size": 64, "epochs": epochs, "augment": AugmentConfig(enabled=True, **aug),
        }),
    })
    return build_datasets(
        {"train": [write("train", f"a{i}") for i in range(4)], "valid": [write("valid", "b")]},
        config,
    )


def test_only_the_train_dataset_augments(tmp_path):
    sets = _dataset(tmp_path)
    assert sets["train"].augment is not None
    assert sets["valid"].augment is None
    _, boxes, _, _ = sets["train"][0]
    assert boxes.shape[1] == 5


def test_close_mosaic_switches_the_mosaic_off_for_the_last_epochs(tmp_path):
    train = _dataset(tmp_path, epochs=12, close_mosaic=10)["train"]
    train.set_epoch(0)
    assert train.mosaic_active
    train.set_epoch(1)
    assert train.mosaic_active
    train.set_epoch(2)
    assert not train.mosaic_active, "12 - 10 = 2: from epoch 2 on, no mosaic"
    train.set_epoch(11)
    assert not train.mosaic_active


# --- rotation, shear, perspective and the background --------------------------


def _sides(quad: Quad, side: int = SIDE):
    pts = _corners(quad, side)
    return [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]


def test_any_angle_rotation_follows_the_pixels_and_keeps_the_rectangle():
    from testbank.models.augment import random_perspective

    q = _quad(0.5, 0.5, half_long=0.2, ratio=2.0, theta=0.0)
    rng = random.Random(3)
    angles_seen = set()
    for _ in range(40):
        out, quads = random_perspective(
            _image_of(q), [_corners(q)], rng, out_side=SIDE, degrees=180.0, scale=0.0,
            translate=0.0, keep_whole=True,
        )
        assert len(quads) == 1, "nothing may be dropped: the box is centred and kept whole"
        painted = out[..., 0] > 127
        assert _iou(painted, _mask(quads[0], subpixel=True)) > 0.9
        a, b, c, d = _sides(quads[0])
        assert a == pytest.approx(c, rel=0.03) and b == pytest.approx(d, rel=0.03)
        assert max(a, c) / max(b, d) == pytest.approx(2.0, rel=0.05), "a rotation keeps the aspect"
        pts = _corners(quads[0])
        angles_seen.add(round(math.degrees(math.atan2(*(pts[1] - pts[0])[::-1])) / 30))
    assert len(angles_seen) >= 6, "the angles have to spread over the circle, not stick to 0/90"


def test_a_banknote_near_the_border_is_zoomed_out_whole_instead_of_cut():
    from testbank.models.augment import KEEP_WHOLE_MARGIN, random_perspective

    # Long banknote hugging the right edge: rotated by 45 degrees it would
    # leave the frame on two sides.
    q = _quad(0.72, 0.5, half_long=0.26, ratio=3.0, theta=0.0)
    side = 256  # the zoomed-out box is thin: 128 px leaves ~20 px of it, all rasterization
    cut = whole = 0
    for seed in range(30):
        image = _image_of(q, side)
        _, without = random_perspective(
            image, [_corners(q, side)], random.Random(seed), out_side=side, degrees=180.0,
            scale=0.0, translate=0.1, keep_whole=False,
        )
        out, with_keep = random_perspective(
            image, [_corners(q, side)], random.Random(seed), out_side=side, degrees=180.0,
            scale=0.0, translate=0.1, keep_whole=True,
        )
        assert len(with_keep) == 1
        low, high = KEEP_WHOLE_MARGIN - 1e-6, 1 - KEEP_WHOLE_MARGIN + 1e-6
        assert all(low <= v <= high for xy in with_keep[0].points for v in xy)
        a, b, _, _ = _sides(with_keep[0], side)
        assert max(a, b) / min(a, b) == pytest.approx(3.0, rel=0.05), "whole, same aspect"
        assert _iou(out[..., 0] > 127, _mask(with_keep[0], side, subpixel=True)) > 0.9
        whole += 1
        if not without or any(v <= 0.001 or v >= 0.999 for xy in without[0].points for v in xy):
            cut += 1
    assert cut > 5, "without keep_whole the same draws cut or drop the banknote"
    assert whole == 30


def test_the_background_fill_is_the_photo_border_not_gray():
    from testbank.models.augment import background_colour, random_perspective

    table = (30, 90, 200)
    image = np.full((SIDE, SIDE, 3), table, dtype=np.uint8)
    q = _quad(0.7, 0.5, half_long=0.28, ratio=2.5, theta=0.0)
    cv2.fillPoly(image, [_corners(q).astype(np.int32)], (255, 255, 255))
    assert background_colour(image) == table
    out, quads = random_perspective(
        image, [_corners(q)], random.Random(5), out_side=SIDE, degrees=45.0, scale=0.0,
        translate=0.0, keep_whole=True, fill=background_colour(image),
    )
    assert len(quads) == 1
    corners = [out[0, 0], out[0, -1], out[-1, 0], out[-1, -1]]
    assert all(tuple(int(v) for v in c) == table for c in corners), "the new canvas is table"
    assert not (out == 114).all(axis=2).any(), "no Ultralytics gray anywhere"


def test_shear_and_perspective_keep_the_label_on_the_pixels():
    from testbank.models.augment import random_perspective

    q = _quad(0.5, 0.5, half_long=0.22, ratio=2.0, theta=0.5)
    rng = random.Random(8)
    for _ in range(30):
        out, quads = random_perspective(
            _image_of(q), [_corners(q)], rng, out_side=SIDE, degrees=0.0, scale=0.0,
            translate=0.0, shear=5.0, perspective=0.0005, keep_whole=True,
        )
        assert len(quads) == 1
        assert _iou(out[..., 0] > 127, _mask(quads[0], subpixel=True)) > 0.9


def test_fit_inside_only_zooms_out_and_lands_the_corners_within_the_margin():
    from testbank.models.augment import KEEP_WHOLE_MARGIN, _apply, fit_inside

    identity = np.eye(3)
    centre = (SIDE / 2.0, SIDE / 2.0)
    inside = [np.array([[40.0, 40.0], [90.0, 40.0], [90.0, 70.0], [40.0, 70.0]])]
    assert np.array_equal(fit_inside(identity, inside, out_side=SIDE, centre=centre), identity)
    outside = [np.array([[100.0, 10.0], [300.0, 10.0], [300.0, 60.0], [100.0, 60.0]])]
    fitted = fit_inside(identity, outside, out_side=SIDE, centre=centre)
    moved = _apply(fitted, outside[0])
    assert moved[:, 0].max() <= (1 - KEEP_WHOLE_MARGIN) * SIDE + 1e-6
    assert moved[:, 0].min() >= KEEP_WHOLE_MARGIN * SIDE - 1e-6
    assert fitted[0, 0] == fitted[1, 1] < 1.0
    with pytest.raises(ValueError, match="centre"):
        fit_inside(identity, outside, out_side=SIDE, centre=(-10.0, 0.0))


def test_a_banknote_filling_the_whole_photo_is_rotated_whole_not_collapsed():
    """REGRESSION: the zoom-out was centred on where the source's top-left
    corner landed; a large rotation put it outside the frame, the factor
    went negative and 20% of the draws came out as a blank tile with no
    labels. Two half-frame banknotes (a real training photo) must come out
    rotated, whole, and visible."""
    from testbank.models.augment import random_perspective

    image = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    image[: SIDE // 2] = (255, 255, 255)
    image[SIDE // 2 :] = (0, 0, 255)
    top = np.array([[0.0, 0.0], [SIDE, 0.0], [SIDE, SIDE / 2], [0.0, SIDE / 2]])
    bottom = top + np.array([0.0, SIDE / 2])
    for seed in range(60):
        out, quads = random_perspective(
            image, [top, bottom], random.Random(seed), out_side=SIDE, degrees=180.0,
            scale=0.5, translate=0.1, shear=2.0, perspective=0.0002, keep_whole=True,
            fill=(114, 114, 114),
        )
        assert len(quads) == 2, f"seed {seed}: both banknotes must survive"
        assert out.std() > 30, f"seed {seed}: the image collapsed"
        assert all(0.0 <= v <= 1.0 for q in quads for xy in q.points for v in xy)


def test_the_full_recipe_with_rotation_keeps_every_quad_inside_and_canonical():
    q = _quad(0.5, 0.5, half_long=0.2, theta=1.1)
    rng = random.Random(11)
    for _ in range(30):
        out, quads = augment(
            _image_of(q), [q], rng, others=lambda: (_image_of(q), [q]),
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=180.0, shear=2.0,
            perspective=0.0002, keep_whole=True, fill="border",
        )
        assert out.shape == (SIDE, SIDE, 3)
        for quad in quads:
            assert is_canonical(quad)
            assert all(0.0 <= v <= 1.0 for xy in quad.points for v in xy)


def test_augment_rejects_an_unknown_fill():
    with pytest.raises(ValueError, match="fill"):
        augment(np.zeros((64, 64, 3), np.uint8), [], random.Random(0), fill="pink")
