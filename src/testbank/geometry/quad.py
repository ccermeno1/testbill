"""Quad canonico: 4 vertices normalizados, pivote de todas las conversiones.

N formatos son 2N conversores, no N^2: todo pasa por aqui.

Rejilla diadica
---------------
Las coordenadas se ajustan a multiplos de 2^-SNAP_BITS. Es la unica forma de que
`flip(flip(q)) == q` se cumpla de forma EXACTA, como exige la especificacion:
en float64 `1 - (1 - 0.1)` da 0.09999999999999998, asi que x -> 1-x no es una
involucion. Sobre la rejilla, 1-x es representable sin redondeo y la involucion
es bit a bit. El error introducido es <= 2^-31 normalizado (2e-6 px en una imagen
de 4000 px), seis ordenes de magnitud por debajo de la precision de una anotacion
"rectangulo aproximado". Los ficheros de anotacion en disco no se modifican.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

Point = tuple[float, float]

SNAP_BITS = 30
_GRID = float(2**SNAP_BITS)

# Rango tolerante: hay billetes que cruzan el borde de la imagen y sus vertices
# caen fuera de [0,1] legitimamente.
COORD_MIN = -0.5
COORD_MAX = 1.5

# Dos lados se consideran empatados en longitud si difieren menos de esto en
# relativo. Sin tolerancia, el ancla de un rectangulo casi exacto (lados largos
# que difieren un 0.3%) la elige el ruido y un jitter de 1 px la gira 180 grados.
LONG_SIDE_TIE_TOL = 0.05

# Por debajo de este ratio de lados el ancla del lado mas largo es inestable.
MIN_STABLE_SIDE_RATIO = 1.1

_MIN_AREA = 1e-12

#: Relacion ancho/alto en pixeles de la imagen. Las coordenadas normalizadas
#: dividen x por el ancho e y por el alto, que es un escalado ANISOTROPO: en ese
#: espacio el lado mas largo, el angulo y el ratio no son los geometricos. Un
#: billete 2:1 tumbado en una imagen 16:9 tiene ratio normalizado 1.13, y en 20:9
#: baja de 1 y el ancla salta al lado corto. Por eso toda comparacion de
#: longitudes admite el aspecto de la imagen.
DEFAULT_ASPECT = 1.0


class QuadShapeWarning(UserWarning):
    """El quad es geometricamente degenerado para el anclaje canonico."""


class CoordinateRangeWarning(UserWarning):
    """Vertice fuera de [0,1]. Legitimo si el billete cruza el borde."""


class QuadError(ValueError):
    """El quad no es utilizable."""


def snap(value: float) -> float:
    """Ajusta a la rejilla diadica. Exacto: round() da entero, /2^k es exacto."""
    return round(value * _GRID) / _GRID


@dataclass(frozen=True, slots=True)
class Quad:
    """Cuatro vertices normalizados, ya ajustados a la rejilla.

    No garantiza orden canonico: usa `canonicalize`. La construccion valida
    rango y no-degeneracion pero no reordena, para que `flip` pueda ser una
    involucion exacta sobre la secuencia cruda.
    """

    points: tuple[Point, Point, Point, Point]

    @classmethod
    def from_xy(cls, coords) -> Quad:
        """Construye desde 8 flotantes o 4 pares. Ajusta a rejilla y valida."""
        flat: list[float] = []
        for item in coords:
            if isinstance(item, (tuple, list)):
                flat.extend(float(v) for v in item)
            else:
                flat.append(float(item))
        if len(flat) != 8:
            raise QuadError(f"un quad son 8 coordenadas, se recibieron {len(flat)}")
        for v in flat:
            if not math.isfinite(v):
                raise QuadError(f"coordenada no finita: {v!r}")
            if not (COORD_MIN <= v <= COORD_MAX):
                raise QuadError(
                    f"coordenada {v!r} fuera del rango tolerante [{COORD_MIN}, {COORD_MAX}]"
                )
        if any(not (0.0 <= v <= 1.0) for v in flat):
            warnings.warn(
                "quad con vertices fuera de [0,1]; se acepta (billete que cruza "
                "el borde de la imagen)",
                CoordinateRangeWarning,
                stacklevel=2,
            )
        snapped = [snap(v) for v in flat]
        pts = tuple((snapped[i], snapped[i + 1]) for i in range(0, 8, 2))
        quad = cls(points=pts)  # type: ignore[arg-type]
        quad._reject_degenerate()
        return quad

    # -- geometria basica ---------------------------------------------------

    def flat(self) -> tuple[float, ...]:
        return tuple(c for p in self.points for c in p)

    def centroid(self) -> Point:
        return (
            sum(p[0] for p in self.points) / 4.0,
            sum(p[1] for p in self.points) / 4.0,
        )

    def signed_area(self) -> float:
        """Shoelace. En coordenadas de imagen (y hacia abajo), horario > 0."""
        pts = self.points
        total = 0.0
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % 4]
            total += x0 * y1 - x1 * y0
        return total / 2.0

    def is_clockwise(self) -> bool:
        return self.signed_area() > 0.0

    def edge_lengths(
        self, aspect: float = DEFAULT_ASPECT
    ) -> tuple[float, float, float, float]:
        """Longitud del lado que ABRE cada vertice: L[i] = |p_i -> p_{i+1}|.

        `aspect` es ancho/alto en pixeles. Con el valor por defecto 1.0 se mide en
        el espacio normalizado, que solo coincide con el geometrico si la imagen
        es cuadrada.
        """
        pts = self.points
        out = []
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % 4]
            out.append(math.hypot((x1 - x0) * aspect, y1 - y0))
        return tuple(out)  # type: ignore[return-value]

    def side_ratio(self, aspect: float = DEFAULT_ASPECT) -> float:
        """Lado mayor / lado menor, promediando pares opuestos."""
        lengths = sorted(self.edge_lengths(aspect))
        short = (lengths[0] + lengths[1]) / 2.0
        long_ = (lengths[2] + lengths[3]) / 2.0
        if short <= 0.0:
            return math.inf
        return long_ / short

    def _reject_degenerate(self) -> None:
        if abs(self.signed_area()) < _MIN_AREA:
            raise QuadError(f"quad degenerado, area nula: {self.points}")
        if len(set(self.points)) != 4:
            raise QuadError(f"quad con vertices repetidos: {self.points}")

    def angle_deg(self, aspect: float = DEFAULT_ASPECT) -> float:
        """Orientacion del lado largo p0->p1, en [0,180). Requiere canonico."""
        (x0, y0), (x1, y1) = self.points[0], self.points[1]
        return math.degrees(math.atan2(y1 - y0, (x1 - x0) * aspect)) % 180.0


# -- orden canonico ---------------------------------------------------------


def canonical_order(
    quad: Quad,
    *,
    tie_tol: float = LONG_SIDE_TIE_TOL,
    aspect: float = DEFAULT_ASPECT,
) -> tuple[tuple[int, int, int, int], float]:
    """Permutacion canonica de los indices de entrada, y el margen de la decision.

    Devolver la permutacion por separado permite testear la ESTABILIDAD de la
    decision discreta sin confundirla con el valor de las coordenadas: una
    perturbacion mueve los puntos, y lo que tiene que quedarse quieto es que
    vertice hace de ancla, no su valor.

    El margen es la separacion en la clave de desempate entre el candidato
    elegido y el siguiente. Es la unica magnitud honesta frente a la que medir
    la estabilidad: por debajo de ella la eleccion puede cambiar y eso no es un
    fallo, es la frontera que toda seleccion discreta tiene por fuerza.
    """
    cx, cy = quad.centroid()

    def polar(i: int) -> float:
        x, y = quad.points[i]
        return math.atan2(y - cy, (x - cx) * aspect)

    keyed = sorted(range(4), key=polar)
    for a, b in zip(keyed, keyed[1:] + keyed[:1]):
        if polar(a) == polar(b):
            raise QuadError(
                f"dos vertices colineales con el centroide, orden ambiguo: {quad.points}"
            )

    ordered_pts = [quad.points[i] for i in keyed]
    ordered = Quad(points=tuple(ordered_pts))  # type: ignore[arg-type]

    ratio = ordered.side_ratio(aspect)
    if ratio < MIN_STABLE_SIDE_RATIO:
        warnings.warn(
            f"quad con ratio de lados {ratio:.3f} < {MIN_STABLE_SIDE_RATIO}: "
            "el ancla del lado mas largo es inestable",
            QuadShapeWarning,
            stacklevel=3,
        )

    lengths = ordered.edge_lengths(aspect)
    threshold = max(lengths) * (1.0 - tie_tol)
    candidates = [i for i, L in enumerate(lengths) if L >= threshold]

    def tiebreak(i: int) -> tuple[float, float, float]:
        x, y = ordered_pts[i]
        return (x + y, x, y)

    ranked = sorted(candidates, key=tiebreak)
    start = ranked[0]
    margin = (
        tiebreak(ranked[1])[0] - tiebreak(start)[0] if len(ranked) > 1 else math.inf
    )

    perm = tuple(keyed[(start + i) % 4] for i in range(4))
    return perm, margin  # type: ignore[return-value]


def canonicalize(
    quad: Quad,
    *,
    tie_tol: float = LONG_SIDE_TIE_TOL,
    aspect: float = DEFAULT_ASPECT,
) -> Quad:
    """Orden canonico: horario, arrancando en el vertice que abre el lado mas largo.

    Desempate: menor x+y, luego menor x, luego menor y. La cadena completa hace
    falta porque x+y empata en rectangulos cuya diagonal es perpendicular a (1,1)
    -- el caso a 45 grados, justo el que descarta "la esquina mas cercana al origen".

    No se usa "esquina mas cercana al origen": es discontinua cerca de 45 grados
    y un jitter de 2 px rota las etiquetas 90 grados.
    """
    perm, _ = canonical_order(quad, tie_tol=tie_tol, aspect=aspect)
    return Quad(points=tuple(quad.points[i] for i in perm))  # type: ignore[arg-type]


def is_canonical(
    quad: Quad,
    *,
    tie_tol: float = LONG_SIDE_TIE_TOL,
    aspect: float = DEFAULT_ASPECT,
) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuadShapeWarning)
        return canonicalize(quad, tie_tol=tie_tol, aspect=aspect).points == quad.points


# -- volteo horizontal ------------------------------------------------------


def flip_horizontal(quad: Quad) -> Quad:
    """Reflexion x -> 1-x, conservando la secuencia cruda de vertices.

    NO recanonicaliza: la reflexion invierte el sentido de giro, asi que el
    resultado es antihorario y hay que pasarlo por `canonicalize` para volver a
    la forma canonica. Mantenerlo crudo es lo que hace la involucion exacta.

    Sobre la rejilla diadica 1-x no redondea, y el rango [-0.5, 1.5] es simetrico
    respecto a 0.5, asi que la reflexion no puede sacar un vertice del rango.
    """
    flipped = tuple((1.0 - x, y) for x, y in quad.points)
    return Quad(points=flipped)  # type: ignore[arg-type]
