"""Format converters, registered by decorator.

N formats mean 2N converters, not N^2: everything goes through the canonical
quad. Adding a format is writing `to_quad` / `from_quad` in a class and
decorating it with `@register`. There is not a single `if format ==` here.

About the image size
--------------------
The specification's table flagged `requires_image_size` only for `bbox_coco`.
Measured against the real formats, **dota** and **bbox_coco** both work in
absolute pixels, not normalized coordinates. Only `obb_yolo` is normalized and
gets away without it.

The size index already exists (`dataio/image_sizes.py`, headers only, via
Pillow) because the canonical order needed it too, so it costs nothing:
`convert` demands it when the converter declares it and fails clearly otherwise.

Formats that were here and left
-------------------------------
`voc_xml` (roLabelImg's `<robndbox>`), `bbox_yolo` and the VOC variant of the
YOLOX-OBB fork were removed in the refactor: no candidate consumed them. Their
lesson stays: a five-degree-of-freedom format only represents rectangles, and
this dataset's quads ARE exact rectangles (measured over the 762 annotations:
opposite sides match to machine precision, all 3048 corners give 90.0000
degrees), but a converter must check that instead of rounding silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from testbank.geometry.quad import Quad, canonicalize

#: Default class name, the one in the export's `data.yaml`.
DEFAULT_CLASS_NAMES: tuple[str, ...] = ("euro_banknote",)


class FormatError(ValueError):
    """The record cannot be converted to/from the requested format."""


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / self.height


@dataclass(frozen=True, slots=True)
class Record:
    """One annotation in a concrete format, plus the class it belongs to.

    `payload` is the format's natural representation: a text line in line
    formats, a dict in bbox_coco. Forcing a common representation on all of
    them would only add one more translation.
    """

    class_id: int
    payload: Any


@runtime_checkable
class Converter(Protocol):
    name: ClassVar[str]
    lossy: ClassVar[bool]
    #: What gets lost, in one line. Mandatory: without it, a new lossy format
    #: would come out warning about what ANOTHER one loses.
    lossy_reason: ClassVar[str]
    requires_image_size: ClassVar[bool]

    def to_quad(self, record: Record, *, size: ImageSize | None = None) -> Quad: ...

    def from_quad(
        self,
        quad: Quad,
        *,
        class_id: int = 0,
        size: ImageSize | None = None,
        class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    ) -> Record: ...


REGISTRY: dict[str, Converter] = {}


def register(cls: type) -> type:
    """Registration decorator. A duplicate name is a programming error."""
    if cls.name in REGISTRY:
        raise FormatError(f"duplicate format in the registry: {cls.name!r}")
    REGISTRY[cls.name] = cls()
    return cls


def get(name: str) -> Converter:
    try:
        return REGISTRY[name]
    except KeyError:
        raise FormatError(
            f"unknown format: {name!r}; registered: {sorted(REGISTRY)}"
        ) from None


def formats() -> list[str]:
    return sorted(REGISTRY)


# --- shared helpers --------------------------------------------------------


def _require_size(converter: Converter, size: ImageSize | None) -> ImageSize:
    if size is None:
        raise FormatError(
            f"format {converter.name!r} works in absolute pixels and needs the "
            f"image size; pass it with size= or generate "
            f"data/derived/image_sizes.json"
        )
    return size


def _to_pixels(quad: Quad, size: ImageSize) -> list[tuple[float, float]]:
    return [(x * size.width, y * size.height) for x, y in quad.points]


def _to_normalized(points, size: ImageSize) -> list[tuple[float, float]]:
    return [(x / size.width, y / size.height) for x, y in points]


def _class_name(class_id: int, class_names: tuple[str, ...]) -> str:
    try:
        return class_names[class_id]
    except IndexError:
        raise FormatError(
            f"class id {class_id} outside {list(class_names)}"
        ) from None


def _class_id(name: str, class_names: tuple[str, ...]) -> int:
    try:
        return class_names.index(name)
    except ValueError:
        raise FormatError(
            f"unknown class {name!r}; expected {list(class_names)}"
        ) from None


def _axis_aligned_bounds(points) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# --- formats ---------------------------------------------------------------


@register
class ObbYolo:
    """`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized. The canonical one, the export's."""

    name = "obb_yolo"
    lossy = False
    #: What exactly gets lost. Empty if nothing does.
    lossy_reason = ""
    requires_image_size = False

    def to_quad(self, record: Record, *, size: ImageSize | None = None) -> Quad:
        tokens = str(record.payload).split()
        if len(tokens) != 8:
            raise FormatError(
                f"obb_yolo expects 8 coordinates, found {len(tokens)}"
            )
        return canonicalize(Quad.from_xy([float(t) for t in tokens]))

    def from_quad(
        self,
        quad: Quad,
        *,
        class_id: int = 0,
        size: ImageSize | None = None,
        class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    ) -> Record:
        return Record(
            class_id=class_id,
            payload=" ".join(f"{v:.17g}" for v in quad.flat()),
        )


@register
class Dota:
    """`x1 y1 x2 y2 x3 y3 x4 y4 category difficult`, in absolute PIXELS.

    It is what RTMDet-R (via BboxToolkit) and the YOLOX-OBB forks expect.
    `difficult` is always 0: the annotation policy does not single out hard
    cases.
    """

    name = "dota"
    lossy = False
    #: What exactly gets lost. Empty if nothing does.
    lossy_reason = ""
    requires_image_size = True

    def to_quad(self, record: Record, *, size: ImageSize | None = None) -> Quad:
        size = _require_size(self, size)
        tokens = str(record.payload).split()
        if len(tokens) < 9:
            raise FormatError(
                f"dota expects 8 coordinates + category (+ difficult), "
                f"found {len(tokens)} fields"
            )
        values = [float(t) for t in tokens[:8]]
        points = [(values[i * 2], values[i * 2 + 1]) for i in range(4)]
        return canonicalize(
            Quad.from_xy(_to_normalized(points, size)), aspect=size.aspect
        )

    def from_quad(
        self,
        quad: Quad,
        *,
        class_id: int = 0,
        size: ImageSize | None = None,
        class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    ) -> Record:
        size = _require_size(self, size)
        coords = " ".join(
            f"{v:.6f}" for point in _to_pixels(quad, size) for v in point
        )
        name = _class_name(class_id, class_names)
        return Record(class_id=class_id, payload=f"{coords} {name} 0")


@register
class BboxCoco:
    """`{"bbox": [x, y, w, h], ...}` in PIXELS, top-left corner.

    LOSSY: it keeps only the axis-aligned envelope, and it needs the image size.
    """

    name = "bbox_coco"
    lossy = True
    #: What exactly gets lost. Empty if nothing does.
    lossy_reason = (
        "keeps only the axis-aligned envelope: LOSES THE ORIENTATION, so it "
        "cannot train an OBB detector"
    )
    requires_image_size = True

    def to_quad(self, record: Record, *, size: ImageSize | None = None) -> Quad:
        size = _require_size(self, size)
        payload = record.payload
        try:
            x, y, w, h = (float(v) for v in payload["bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"bbox_coco expects 'bbox' with 4 numbers: {exc}") from exc
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return canonicalize(
            Quad.from_xy(_to_normalized(points, size)), aspect=size.aspect
        )

    def from_quad(
        self,
        quad: Quad,
        *,
        class_id: int = 0,
        size: ImageSize | None = None,
        class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    ) -> Record:
        size = _require_size(self, size)
        x0, y0, x1, y1 = _axis_aligned_bounds(_to_pixels(quad, size))
        width, height = x1 - x0, y1 - y0
        return Record(
            class_id=class_id,
            payload={
                "category_id": class_id,
                "bbox": [round(x0, 4), round(y0, 4), round(width, 4), round(height, 4)],
                "area": round(width * height, 4),
                "iscrowd": 0,
            },
        )


def convert(
    record: Record,
    *,
    source: str,
    target: str,
    size: ImageSize | None = None,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> Record:
    """Convert through the canonical quad. There are no direct routes."""
    quad = get(source).to_quad(record, size=size)
    return get(target).from_quad(
        quad, class_id=record.class_id, size=size, class_names=class_names
    )
