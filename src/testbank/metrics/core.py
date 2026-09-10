"""Nucleo geometrico de las metricas. Todo en PIXELES, sin excepciones.

Por que pixeles y no normalizado
--------------------------------
Normalizar divide x por el ancho e y por el alto: un escalado ANISOTROPO. Las
razones de area (IoU, cobertura, contaminacion) sobreviven a cualquier afinidad
y darian igual en los dos espacios, pero el ANGULO y la DISTANCIA POR VERTICE
no. Un billete 2:1 tumbado en una imagen 20:9 cambia de angulo y de ratio al
normalizar, y el ancla canonica llega a saltar de lado.

Mezclar espacios segun la metrica seria pedir el error. La regla es una sola:
aqui dentro todo esta en pixeles, y la conversion ocurre en la frontera.

Nada de operadores compilados: IoU rotado y NMS rotado con shapely. Lento y
suficiente para 500 imagenes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import Polygon
from shapely.ops import unary_union

from testbank.dataio.formats import ImageSize
from testbank.geometry.quad import Quad

#: Angulos que difieren en 180 grados describen el mismo rectangulo.
HALF_TURN = 180.0


@dataclass(frozen=True, slots=True)
class Prediction:
    """Una deteccion. `score` ordena el emparejamiento greedy."""

    quad: Quad
    score: float
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class ImageEval:
    """Todo lo que hace falta para evaluar UNA imagen.

    `ignored` son los quads que el filtro de area relativa dejo fuera. No son
    verdad que haya que detectar ni fondo que penalice: son anotaciones reales
    que decidimos no usar. Ver `matching.py`.
    """

    sample_id: str
    size: ImageSize
    truths: tuple[Quad, ...] = ()
    predictions: tuple[Prediction, ...] = ()
    ignored: tuple[Quad, ...] = ()

    def __post_init__(self) -> None:
        if self.size.width <= 0 or self.size.height <= 0:
            raise ValueError(f"{self.sample_id}: tamano de imagen invalido")


def to_polygon(quad: Quad, size: ImageSize) -> Polygon:
    """Quad normalizado -> poligono en pixeles, saneado."""
    polygon = Polygon(
        [(x * size.width, y * size.height) for x, y in quad.points]
    )
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def iou(a: Polygon, b: Polygon) -> float:
    """IoU rotado. Las razones de area no dependen del espacio, pero el resto
    del modulo si, asi que entra en pixeles como todo lo demas."""
    if not a.intersects(b):
        return 0.0
    intersection = a.intersection(b).area
    if intersection <= 0.0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def angle_error_deg(a: Quad, b: Quad, size: ImageSize) -> float:
    """Error de angulo modulo 180: `min(|d|, 180 - |d|)`.

    Un rectangulo girado 179 grados y otro girado 1 grado son casi el mismo
    rectangulo, no dos que difieren en 178. Sin el modulo, esos casos dominan
    la media y la metrica deja de medir nada.
    """
    delta = abs(a.angle_deg(size.aspect) - b.angle_deg(size.aspect)) % HALF_TURN
    return min(delta, HALF_TURN - delta)


def vertex_distances_px(a: Quad, b: Quad, size: ImageSize) -> list[float]:
    """Distancia por vertice, ya en el orden canonico de cada uno.

    Diagnostico, no criterio de exito: la especificacion dice que la precision
    geometrica exacta no es el objetivo.
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
    """Recorte con margen: escala el rectangulo respecto a su centro.

    `margin` es la fraccion de CADA LADO que se anade en CADA BORDE, asi que
    el lado total queda multiplicado por `1 + 2*margin`. Con margin=0.05 el
    recorte lleva un 5% de holgura por cada lado, no un 5% repartido.

    Se escala en vez de dilatar con `buffer` porque `buffer` redondea las
    esquinas y el recorte tiene que seguir siendo un cuadrilatero para poder
    rectificarlo por homografia.
    """
    if margin < 0:
        raise ValueError(f"el margen no puede ser negativo: {margin}")
    if margin == 0:
        return polygon
    from shapely import affinity

    factor = 1.0 + 2.0 * margin
    return affinity.scale(polygon, xfact=factor, yfact=factor, origin="center")


def union_of(polygons) -> Polygon | None:
    """Union de una lista, o None si esta vacia. Evita el caso especial fuera."""
    valid = [p for p in polygons if p is not None and p.area > 0]
    if not valid:
        return None
    merged = unary_union(valid)
    return merged if merged.area > 0 else None


@dataclass
class PolygonCache:
    """Poligonos en pixeles de una imagen, calculados una vez.

    El emparejamiento greedy y las metricas de recorte recorren los mismos
    quads varias veces, y `Polygon` no es gratis. El bootstrap remuestrea 2000
    veces sobre las MISMAS imagenes, asi que sin cache se recalcularia todo
    dos mil veces.
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
