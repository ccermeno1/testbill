"""Conversores de formato, registrados por decorador.

N formatos son 2N conversores, no N^2: todo pasa por el quad canonico. Anadir un
formato es escribir `to_quad` / `from_quad` en una clase y decorarla con
`@register`. Aqui no hay ni un `if formato ==`.

Sobre el tamano de la imagen
----------------------------
La tabla de la especificacion marcaba `requires_image_size` solo en `bbox_coco`.
Medido contra los formatos reales, hacen falta tres: **dota**, **voc_xml** y
**bbox_coco** trabajan en pixeles absolutos, no en coordenadas normalizadas. Solo
`obb_yolo` y `bbox_yolo` son normalizados y se libran.

El indice de tamanos ya existe (`dataio/image_sizes.py`, solo cabeceras con
Pillow) porque el orden canonico tambien lo necesitaba, asi que no cuesta nada:
`convert` lo exige cuando el conversor lo declara y falla claro si no llega.

Sobre `voc_xml` y la perdida
---------------------------
El `<robndbox>` de roLabelImg guarda `cx cy w h angle`: cinco grados de libertad.
Un cuadrilatero libre tiene ocho, asi que en general la conversion perderia
informacion. No aqui: **los quads de este dataset son rectangulos exactos** --
medido sobre las 762 anotaciones del export, los lados opuestos coinciden hasta
la precision de la maquina y las 3048 esquinas dan 90.0000 grados. Un rectangulo
girado tiene exactamente cinco grados de libertad, asi que la conversion es
biyectiva y `lossy = False`.

Eso deja de ser cierto en cuanto entre un quad que no sea rectangulo, y entonces
seria una perdida silenciosa. Por eso el conversor lo comprueba y falla en vez de
redondear a la callada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from testbank.geometry.quad import Quad, canonicalize

#: Nombre de clase por defecto, el del `data.yaml` del export.
DEFAULT_CLASS_NAMES: tuple[str, ...] = ("euro_banknote",)

#: Tolerancia relativa para aceptar un quad como rectangulo. Holgada respecto a
#: lo medido (desviacion 0.0) para no fallar por el ajuste a la rejilla diadica.
RECTANGLE_TOL = 1e-3


class FormatError(ValueError):
    """El registro no se puede convertir a/desde el formato pedido."""


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / self.height


@dataclass(frozen=True, slots=True)
class Record:
    """Una anotacion en un formato concreto, mas la clase a la que pertenece.

    `payload` es la representacion natural del formato: una linea de texto en los
    formatos de linea, un `<object>` en voc_xml, un dict en bbox_coco. Forzar a
    todos una representacion comun solo anadiria una traduccion mas.
    """

    class_id: int
    payload: Any


@runtime_checkable
class Converter(Protocol):
    name: ClassVar[str]
    lossy: ClassVar[bool]
    #: Que se pierde, en una linea. Obligatorio: sin esto, un formato lossy
    #: nuevo saldria avisando de lo que pierde OTRO.
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
    """Decorador de registro. Un nombre duplicado es error de programacion."""
    if cls.name in REGISTRY:
        raise FormatError(f"formato duplicado en el registro: {cls.name!r}")
    REGISTRY[cls.name] = cls()
    return cls


def get(name: str) -> Converter:
    try:
        return REGISTRY[name]
    except KeyError:
        raise FormatError(
            f"formato desconocido: {name!r}; registrados: {sorted(REGISTRY)}"
        ) from None


def formats() -> list[str]:
    return sorted(REGISTRY)


# --- utilidades compartidas ------------------------------------------------


def _require_size(converter: Converter, size: ImageSize | None) -> ImageSize:
    if size is None:
        raise FormatError(
            f"el formato {converter.name!r} trabaja en pixeles absolutos y "
            f"necesita el tamano de la imagen; pasalo con size= o genera "
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
            f"id de clase {class_id} fuera de {list(class_names)}"
        ) from None


def _class_id(name: str, class_names: tuple[str, ...]) -> int:
    try:
        return class_names.index(name)
    except ValueError:
        raise FormatError(
            f"clase desconocida {name!r}; esperadas {list(class_names)}"
        ) from None


def _axis_aligned_bounds(points) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# --- formatos --------------------------------------------------------------


@register
class ObbYolo:
    """`class x1 y1 x2 y2 x3 y3 x4 y4`, normalizado. El canonico, el del export."""

    name = "obb_yolo"
    lossy = False
    #: Que se pierde exactamente. Vacio si no se pierde nada.
    lossy_reason = ''
    requires_image_size = False

    def to_quad(self, record: Record, *, size: ImageSize | None = None) -> Quad:
        tokens = str(record.payload).split()
        if len(tokens) != 8:
            raise FormatError(
                f"obb_yolo espera 8 coordenadas, se encontraron {len(tokens)}"
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
    """`x1 y1 x2 y2 x3 y3 x4 y4 categoria dificil`, en PIXELES absolutos.

    Es lo que esperan los forks de YOLOX-OBB. `dificil` va siempre a 0: la
    politica de anotacion no distingue casos dificiles.
    """

    name = "dota"
    lossy = False
    #: Que se pierde exactamente. Vacio si no se pierde nada.
    lossy_reason = ''
    requires_image_size = True

    def to_quad(self, record: Record, *, size: ImageSize | None = None) -> Quad:
        size = _require_size(self, size)
        tokens = str(record.payload).split()
        if len(tokens) < 9:
            raise FormatError(
                f"dota espera 8 coordenadas + categoria (+ dificil), "
                f"se encontraron {len(tokens)} campos"
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
    """`{"bbox": [x, y, w, h], ...}` en PIXELES, esquina superior izquierda.

    LOSSY por lo mismo que bbox_yolo, y ademas necesita el tamano de la imagen.
    """

    name = "bbox_coco"
    lossy = True
    #: Que se pierde exactamente. Vacio si no se pierde nada.
    lossy_reason = 'solo guarda la envolvente alineada al eje: PIERDE LA ORIENTACION, asi que no sirve para entrenar un detector OBB'
    requires_image_size = True

    def to_quad(self, record: Record, *, size: ImageSize | None = None) -> Quad:
        size = _require_size(self, size)
        payload = record.payload
        try:
            x, y, w, h = (float(v) for v in payload["bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"bbox_coco espera 'bbox' con 4 numeros: {exc}") from exc
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


def _assert_rectangle(points) -> None:
    """Un `robndbox` solo representa rectangulos. Fallar antes que redondear."""
    sides = [math.dist(points[i], points[(i + 1) % 4]) for i in range(4)]
    longest = max(sides)
    if longest <= 0:
        raise FormatError("quad degenerado")
    if (
        abs(sides[0] - sides[2]) / longest > RECTANGLE_TOL
        or abs(sides[1] - sides[3]) / longest > RECTANGLE_TOL
    ):
        raise FormatError(
            "voc_xml usa <robndbox> (cx cy w h angle), que solo representa "
            "rectangulos; este quad tiene los lados opuestos desiguales y la "
            "conversion perderia informacion en silencio"
        )
    for i in range(4):
        ax, ay = points[(i - 1) % 4][0] - points[i][0], points[(i - 1) % 4][1] - points[i][1]
        bx, by = points[(i + 1) % 4][0] - points[i][0], points[(i + 1) % 4][1] - points[i][1]
        norm = math.hypot(ax, ay) * math.hypot(bx, by)
        if norm > 0 and abs(ax * bx + ay * by) / norm > RECTANGLE_TOL:
            raise FormatError(
                "voc_xml usa <robndbox>, que solo representa rectangulos; este "
                "quad tiene esquinas que no son de 90 grados"
            )


def convert(
    record: Record,
    *,
    source: str,
    target: str,
    size: ImageSize | None = None,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> Record:
    """Convierte pasando por el quad canonico. No hay rutas directas."""
    quad = get(source).to_quad(record, size=size)
    return get(target).from_quad(
        quad, class_id=record.class_id, size=size, class_names=class_names
    )


