"""Format converters: registry, round trip, and loss where it belongs."""

from __future__ import annotations

import hashlib
import math

import pytest
from conftest import rotated_rect_points

from testbank.dataio import formats
from testbank.dataio.formats import (
    DEFAULT_CLASS_NAMES,
    Converter,
    FormatError,
    ImageSize,
    Record,
    convert,
    get,
    register,
)
from testbank.geometry.quad import Quad, QuadShapeWarning, canonicalize

SIZE = ImageSize(width=640, height=480)

LOSSLESS = ("obb_yolo", "dota")
LOSSY = ("bbox_coco",)


def quad(cx=0.5, cy=0.5, half_long=0.2, ratio=2.0, theta=0.0) -> Quad:
    """Rotated rectangle in PIXELS, returned normalized.

    Building it in normalized coordinates would be a subtle mistake:
    normalizing divides x by the width and y by the height, which is an
    ANISOTROPIC scaling, so a rectangle rotated in normalized coordinates is
    a PARALLELOGRAM in pixels and vice versa. Banknotes are rectangles in the
    image, which is pixel space, and that is where we measure them: the 762
    annotations of the export give 90.0000 degrees in pixels.
    """
    half_long_px = half_long * SIZE.width
    points_px = rotated_rect_points(
        cx * SIZE.width, cy * SIZE.height, half_long_px, ratio, theta
    )
    return canonicalize(
        Quad.from_xy(
            [(x / SIZE.width, y / SIZE.height) for x, y in points_px]
        ),
        aspect=SIZE.aspect,
    )


def max_pixel_error(a: Quad, b: Quad) -> float:
    return max(
        math.dist(
            (pa[0] * SIZE.width, pa[1] * SIZE.height),
            (pb[0] * SIZE.width, pb[1] * SIZE.height),
        )
        for pa, pb in zip(a.points, b.points)
    )


# --- the registry ---------------------------------------------------------


def test_all_formats_are_there():
    assert formats.formats() == ["bbox_coco", "dota", "obb_yolo"]


@pytest.mark.parametrize("name", LOSSLESS + LOSSY)
def test_all_satisfy_the_protocol(name):
    assert isinstance(get(name), Converter)


@pytest.mark.parametrize(
    "name,lossy,needs_size",
    [
        ("obb_yolo", False, False),
        ("dota", False, True),
        ("bbox_coco", True, True),
    ],
)
def test_declared_metadata(name, lossy, needs_size):
    converter = get(name)
    assert converter.lossy is lossy
    assert converter.requires_image_size is needs_size


def test_unknown_format_says_which_exist():
    with pytest.raises(FormatError, match="unknown format"):
        get("does_not_exist")


def test_registering_a_repeated_name_is_an_error():
    with pytest.raises(FormatError, match="duplicate"):

        @register
        class Other:
            name = "obb_yolo"
            lossy = False
            requires_image_size = False


def test_adding_a_format_does_not_touch_convert():
    """The registry is the only thing that decides: `convert` knows no name."""

    @register
    class Double:
        name = "_test_double"
        lossy = False
        requires_image_size = False

        def to_quad(self, record, *, size=None):
            tokens = [float(t) / 2 for t in str(record.payload).split()]
            return canonicalize(Quad.from_xy(tokens))

        def from_quad(self, q, *, class_id=0, size=None, class_names=DEFAULT_CLASS_NAMES):
            return Record(class_id, " ".join(f"{v * 2:.17g}" for v in q.flat()))

    try:
        source = get("obb_yolo").from_quad(quad())
        out = convert(source, source="obb_yolo", target="_test_double")
        back = convert(out, source="_test_double", target="obb_yolo")
        assert get("obb_yolo").to_quad(back).points == quad().points
    finally:
        del formats.REGISTRY["_test_double"]


# --- round trip -----------------------------------------------------------


@pytest.mark.parametrize("name", LOSSLESS)
@pytest.mark.parametrize("theta", [0.0, 0.3, 1.2, 2.4, 3.0])
def test_lossless_formats_round_trip(name, theta):
    original = quad(theta=theta)
    record = get("obb_yolo").from_quad(original)
    other = convert(record, source="obb_yolo", target=name, size=SIZE)
    back = convert(other, source=name, target="obb_yolo", size=SIZE)
    assert max_pixel_error(original, get("obb_yolo").to_quad(back)) < 1e-3


def test_obb_yolo_round_trips_exactly():
    """Without a change of units there is no rounding: equality is exact."""
    original = quad(theta=0.7)
    record = get("obb_yolo").from_quad(original)
    assert get("obb_yolo").to_quad(record).points == original.points


def test_the_class_survives_the_conversion():
    record = Record(class_id=0, payload=get("obb_yolo").from_quad(quad()).payload)
    for name in LOSSLESS + LOSSY:
        out = convert(record, source="obb_yolo", target=name, size=SIZE)
        assert out.class_id == 0


# --- loss where it belongs ------------------------------------------------


@pytest.mark.parametrize("name", LOSSY)
def test_lossy_formats_lose_the_orientation(name):
    """A rotated quad does not come back the same: that is the point of them being lossy."""
    original = quad(theta=0.6)
    record = get("obb_yolo").from_quad(original)
    other = convert(record, source="obb_yolo", target=name, size=SIZE)
    back = convert(other, source=name, target="obb_yolo", size=SIZE)
    assert max_pixel_error(original, get("obb_yolo").to_quad(back)) > 1.0


def test_at_45_degrees_the_aligned_envelope_is_a_square():
    """The worst case of bbox_coco, and it warns on its own.

    The aligned envelope of a 2:1 rectangle rotated 45 degrees has both sides
    equal, so its canonical anchor is unstable and `QuadShapeWarning` fires.
    It is not a converter failure: it is exactly the diagnostic aligned
    formats exist for. That the warning reaches here confirms it is not
    swallowed along the way.
    """
    original = quad(theta=math.pi / 4)
    record = convert(
        get("obb_yolo").from_quad(original),
        source="obb_yolo",
        target="bbox_coco",
        size=SIZE,
    )
    with pytest.warns(QuadShapeWarning, match="longest-side anchor"):
        convert(record, source="bbox_coco", target="obb_yolo", size=SIZE)


@pytest.mark.parametrize("name", LOSSY)
def test_an_already_aligned_box_survives_the_lossy_ones(name):
    """With no orientation to lose, nothing is lost."""
    original = quad(theta=0.0)
    record = get("obb_yolo").from_quad(original)
    other = convert(record, source="obb_yolo", target=name, size=SIZE)
    back = convert(other, source=name, target="obb_yolo", size=SIZE)
    assert max_pixel_error(original, get("obb_yolo").to_quad(back)) < 1e-3


# --- the image size -------------------------------------------------------


@pytest.mark.parametrize("name", ["dota", "bbox_coco"])
def test_without_size_the_pixel_formats_fail_clearly(name):
    record = get("obb_yolo").from_quad(quad())
    with pytest.raises(FormatError, match="image size"):
        convert(record, source="obb_yolo", target=name)


@pytest.mark.parametrize("name", ["obb_yolo"])
def test_normalized_formats_do_not_ask_for_size(name):
    record = get("obb_yolo").from_quad(quad())
    assert convert(record, source="obb_yolo", target=name) is not None


def test_the_right_size_changes_the_result():
    """If the size were ignored, two different aspects would give the same thing."""
    q = quad(theta=0.5)
    wide = get("dota").from_quad(q, size=ImageSize(1000, 200))
    tall = get("dota").from_quad(q, size=ImageSize(200, 1000))
    assert wide.payload != tall.payload


# --- the source annotations are read-only ---------------------------------


def test_the_original_annotations_do_not_change(tmp_path):
    """Immutability by hash after running ALL the conversions."""
    paths = []
    for i, theta in enumerate((0.0, 0.6, 1.9)):
        q = quad(theta=theta)
        path = tmp_path / f"label_{i}.txt"
        path.write_text(
            "0 " + get("obb_yolo").from_quad(q).payload + "\n", encoding="utf-8"
        )
        paths.append(path)

    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    for path in paths:
        payload = path.read_text(encoding="utf-8").split(maxsplit=1)[1].strip()
        record = Record(class_id=0, payload=payload)
        for target in formats.formats():
            other = convert(record, source="obb_yolo", target=target, size=SIZE)
            convert(other, source=target, target="obb_yolo", size=SIZE)

    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before == after
