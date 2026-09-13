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
import xml.etree.ElementTree as ET
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
class VocXml:
    """VOC XML con `<robndbox>` (roLabelImg), en PIXELES. Lo espera YOLOX-OBB.

    Convenio, que es donde se cuelan los errores: `w` es el lado LARGO, `h` el
    corto, y `angle` la rotacion del lado largo respecto al eje x, en RADIANES,
    normalizada a [0, pi). Como el quad canonico ancla p0->p1 en el lado mas
    largo, el mapeo sale directo y el orden se recupera intacto al volver.
    """

    name = "voc_xml"
    lossy = False
    #: Que se pierde exactamente. Vacio si no se pierde nada.
    lossy_reason = ''
    requires_image_size = True

    def to_quad(self, record: Record, *, size: ImageSize | None = None) -> Quad:
        size = _require_size(self, size)
        element = record.payload
        box = element.find("robndbox")
        if box is None:
            raise FormatError("el <object> no tiene <robndbox>")

        def field(tag: str) -> float:
            node = box.find(tag)
            if node is None or node.text is None:
                raise FormatError(f"falta <{tag}> en <robndbox>")
            return float(node.text)

        cx, cy = field("cx"), field("cy")
        half_w, half_h = field("w") / 2.0, field("h") / 2.0
        angle = field("angle")
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        local = [
            (-half_w, -half_h),
            (+half_w, -half_h),
            (+half_w, +half_h),
            (-half_w, +half_h),
        ]
        points = [
            (cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)
            for lx, ly in local
        ]
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
        points = _to_pixels(quad, size)
        _assert_rectangle(points)

        cx = sum(p[0] for p in points) / 4.0
        cy = sum(p[1] for p in points) / 4.0
        long_side = math.dist(points[0], points[1])
        short_side = math.dist(points[1], points[2])
        angle = math.atan2(
            points[1][1] - points[0][1], points[1][0] - points[0][0]
        ) % math.pi

        element = ET.Element("object")
        ET.SubElement(element, "name").text = _class_name(class_id, class_names)
        ET.SubElement(element, "type").text = "robndbox"
        ET.SubElement(element, "difficult").text = "0"
        box = ET.SubElement(element, "robndbox")
        for tag, value in (
            ("cx", cx),
            ("cy", cy),
            ("w", long_side),
            ("h", short_side),
            ("angle", angle),
        ):
            ET.SubElement(box, tag).text = f"{value:.6f}"
        return Record(class_id=class_id, payload=element)


@register
class BboxYolo:
    """`class cx cy w h` normalizado, alineado al eje. Diagnostico y comparacion.

    LOSSY: pierde la orientacion. Es justo el punto de la especificacion -- sirve
    para medir cuanto se pierde al usar cajas alineadas, no para produccion.
    """

    name = "bbox_yolo"
    lossy = True
    #: Que se pierde exactamente. Vacio si no se pierde nada.
    lossy_reason = 'solo guarda la envolvente alineada al eje: PIERDE LA ORIENTACION, asi que no sirve para entrenar un detector OBB'
    requires_image_size = False

    def to_quad(self, record: Record, *, size: ImageSize | None = None) -> Quad:
        tokens = str(record.payload).split()
        if len(tokens) != 4:
            raise FormatError(
                f"bbox_yolo espera cx cy w h, se encontraron {len(tokens)}"
            )
        cx, cy, w, h = (float(t) for t in tokens)
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        return canonicalize(
            Quad.from_xy([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        )

    def from_quad(
        self,
        quad: Quad,
        *,
        class_id: int = 0,
        size: ImageSize | None = None,
        class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    ) -> Record:
        x0, y0, x1, y1 = _axis_aligned_bounds(quad.points)
        return Record(
            class_id=class_id,
            payload=(
                f"{(x0 + x1) / 2:.17g} {(y0 + y1) / 2:.17g} "
                f"{x1 - x0:.17g} {y1 - y0:.17g}"
            ),
        )


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


@register
class YoloxObbVoc:
    """El XML de `buzhidaoshenme/YOLOX-OBB`. Verificado contra SU generador.

    No es `<robndbox>` ni una envolvente alineada, aunque lo parezca. Leido su
    `custom tools/DOTA2VOC_obb.py`, los cuatro campos de VOC guardan el ancho y
    el alto de la caja GIRADA colocados alrededor del centro:

        xmin = cx - w/2      xmax = cx + w/2      -> w = xmax - xmin
        ymin = cy - h/2      ymax = cy + h/2      -> h = ymax - ymin

    O sea un `(cx, cy, w, h, angulo)` normal, escrito en los campos equivocados.
    Y su generador garantiza `w >= h` intercambiando y restando 90 grados, asi
    que `w` es siempre el lado LARGO -- el mismo invariante que nuestro orden
    canonico, lo que hace la conversion directa.

    El angulo va en GRADOS, no radianes.

    La base de coordenadas: se sigue al LECTOR, no al generador
    -------------------------------------------------------------
    El fork se contradice consigo mismo por un pixel. Su lector aplica la base 1
    de VOC:

        cur_pt = int(bbox.find(pt).text) - 1     # dota_obb.py:60

    pero su generador escribe en base 0, sin sumar nada (`int(c_x - w/2)`). O
    sea que sus propias etiquetas llegan a la red desplazadas -1 px.

    Aqui se escribe en **base 1** (`+ 1`) y se lee restando 1. Se elige al lector
    porque es el que fabrica los objetivos de entrenamiento: reproducir el fallo
    del generador desplazaria cada caja un pixel sin ganar nada. La consecuencia
    es que leer ficheros generados por SU herramienta da un desfase de 1 px, que
    es el desfase que el fork ya tiene por dentro.

    Ojo tambien: su lector usa `int(texto)`, no `float(texto)`, asi que las
    cuatro coordenadas TIENEN que ser literales enteros. Un `123.5` seria un
    ValueError al cargar. Y lee `<difficult>` sin comprobar que exista, asi que
    omitirlo da un AttributeError: por eso se escribe siempre.

    LOSSY, y por dos motivos distintos de la orientacion:

    1. Su generador trunca a entero (`int(c_x - w/2)`), asi que se pierde la
       precision subpixel. Aqui se escribe redondeado para no acumular sesgo
       hacia abajo, que es lo que hace `int()` con valores positivos.
    2. Recorta al marco de la imagen, asi que un billete que cruza el borde
       pierde la parte de fuera -- igual que la politica `clip`.
    """

    name = "yolox_obb_voc"
    lossy = True
    #: Que se pierde exactamente. Vacio si no se pierde nada.
    lossy_reason = 'conserva la orientacion; lo que pierde es precision subpixel al redondear a entero (IoU 0.9965 de mediana sobre las 762 reales)'
    requires_image_size = True

    def to_quad(self, record: Record, *, size: ImageSize | None = None) -> Quad:
        size = _require_size(self, size)
        element = record.payload
        box = element.find("bndbox")
        if box is None:
            raise FormatError("el <object> no tiene <bndbox>")

        def field(tag: str) -> float:
            node = box.find(tag)
            if node is None or node.text is None:
                raise FormatError(f"falta <{tag}> en <bndbox>")
            return float(node.text)

        # A base 0, que es como lo lee SU dataloader (`- 1` en dota_obb.py:60).
        xmin, ymin = field("xmin") - 1, field("ymin") - 1
        xmax, ymax = field("xmax") - 1, field("ymax") - 1
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        half_w, half_h = (xmax - xmin) / 2, (ymax - ymin) / 2
        theta = math.radians(field("angle"))

        cos_a, sin_a = math.cos(theta), math.sin(theta)
        local = [
            (-half_w, -half_h),
            (+half_w, -half_h),
            (+half_w, +half_h),
            (-half_w, +half_h),
        ]
        points = [
            (cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)
            for lx, ly in local
        ]
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
        points = _to_pixels(quad, size)
        cx = sum(p[0] for p in points) / 4.0
        cy = sum(p[1] for p in points) / 4.0
        # NO se asume que `p0->p1` sea el lado largo. El orden canonico lo ancla
        # ahi *en el espacio normalizado y con el aspecto aplicado*, y en cajas
        # casi cuadradas eso no coincide con el lado largo en pixeles: medido,
        # una anotacion del export tiene ratio 0.951 por p0->p1.
        #
        # La caja saldria geometricamente correcta igualmente -- `(w, h, angulo)`
        # con w < h describe el mismo rectangulo -- pero romperia el INVARIANTE
        # del fork, cuyo generador garantiza w >= h con este mismo intercambio
        # (`if w_o <= h_o`). Un consumidor que se fie de el leeria mal la caja.
        side_a = math.dist(points[0], points[1])
        side_b = math.dist(points[1], points[2])
        if side_a >= side_b:
            long_side, short_side = side_a, side_b
            start, end = points[0], points[1]
        else:
            long_side, short_side = side_b, side_a
            start, end = points[1], points[2]
        degrees = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
        # Su convenio deja el angulo en (-90, 90]: el mismo rectangulo con el
        # lado largo apuntando al otro sentido, que es el mismo rectangulo.
        degrees = (degrees + 90.0) % 180.0 - 90.0
        if degrees <= -90.0:
            degrees += 180.0

        element = ET.Element("object")
        ET.SubElement(element, "name").text = _class_name(class_id, class_names)
        ET.SubElement(element, "difficult").text = "0"
        box = ET.SubElement(element, "bndbox")
        for tag, value in (
            ("xmin", cx - long_side / 2),
            ("ymin", cy - short_side / 2),
            ("xmax", cx + long_side / 2),
            ("ymax", cy + short_side / 2),
        ):
            # Redondeo, no truncado. Su generador usa `int()`, que con valores
            # positivos sesga siempre hacia abajo y desplaza la caja medio pixel.
            #
            # El `+ 1` es la base 1 de VOC, y NO es un adorno: su dataloader
            # hace `int(texto) - 1`. Ver la nota de la clase.
            ET.SubElement(box, tag).text = str(round(value) + 1)
        ET.SubElement(box, "angle").text = f"{degrees:.6f}"
        return Record(class_id=class_id, payload=element)
