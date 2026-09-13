"""`clip` tiene que devolver RECTANGULOS dentro del marco.

Por que existe este fichero
---------------------------
`clip` pinzaba cada vertice a [0,1] por su cuenta. Es lo obvio y esta mal: un
rectangulo GIRADO cortado contra un marco recto da un trapecio. Medido sobre los
datos reales, 82 de 679 anotaciones (12%) dejaban de ser rectangulos, y el unico
run registrado hasta entonces entreno asi.

Importa porque el modelo predice `(cx, cy, w, h, angulo)`. Un trapecio como
verdad de referencia es un objetivo inalcanzable: solo se puede aprender como el
rectangulo que menos mal le pega.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon, box

from testbank.dataio.prepare import clip_quad
from testbank.geometry.quad import Quad

# Recortar un billete muy salido lo deja casi cuadrado, y entonces el ancla del
# lado mas largo es inestable -- que es justo lo que avisa `QuadShapeWarning`.
# Aqui es el resultado ESPERADO, no un sintoma: los casos de este fichero son
# deliberadamente extremos. Y `CoordinateRangeWarning` salta al CONSTRUIR los
# quads de entrada, que estan fuera del marco a proposito.
pytestmark = pytest.mark.filterwarnings(
    "ignore::testbank.geometry.quad.QuadShapeWarning",
    "ignore::testbank.geometry.quad.CoordinateRangeWarning",
)


def _quad(cx, cy, half_long, half_short, theta, aspect=1.0):
    """Rectangulo girado construido en el espacio con proporciones reales."""
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    points = [
        (cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)
        for lx, ly in (
            (-half_long, -half_short),
            (half_long, -half_short),
            (half_long, half_short),
            (-half_long, half_short),
        )
    ]
    return Quad.from_xy([(x / aspect, y) for x, y in points])


def _sides(quad, aspect=1.0):
    points = [(x * aspect, y) for x, y in quad.points]
    return [math.dist(points[i], points[(i + 1) % 4]) for i in range(4)]


def _angle(quad, aspect=1.0):
    points = [(x * aspect, y) for x, y in quad.points]
    return math.degrees(
        math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    ) % 180.0


@pytest.mark.parametrize("theta", [0.0, 0.2, 0.5, 0.9, 1.3, -0.4, -1.1])
@pytest.mark.parametrize("cx,cy", [(0.05, 0.5), (0.95, 0.5), (0.5, 0.02), (0.02, 0.04)])
def test_lo_que_sale_siempre_es_un_rectangulo(theta, cx, cy):
    recortado = clip_quad(_quad(cx, cy, 0.30, 0.12, theta))
    a, b, c, d = _sides(recortado)
    assert a == pytest.approx(c, abs=1e-6)
    assert b == pytest.approx(d, abs=1e-6)


@pytest.mark.parametrize("theta", [0.0, 0.2, 0.5, 0.9, 1.3, -0.4, -1.1])
@pytest.mark.parametrize("cx,cy", [(0.05, 0.5), (0.95, 0.5), (0.5, 0.02), (0.02, 0.04)])
def test_lo_que_sale_siempre_cabe_en_el_marco(theta, cx, cy):
    """El motivo por el que `clip` existe: que el entrenador no tire la imagen."""
    for x, y in clip_quad(_quad(cx, cy, 0.30, 0.12, theta)).points:
        assert -1e-9 <= x <= 1.0 + 1e-9
        assert -1e-9 <= y <= 1.0 + 1e-9


def test_un_quad_que_ya_cabe_no_se_toca():
    dentro = _quad(0.5, 0.5, 0.2, 0.1, 0.3)
    salida = clip_quad(dentro)
    for (ax, ay), (bx, by) in zip(dentro.points, salida.points):
        assert ax == pytest.approx(bx, abs=1e-9)
        assert ay == pytest.approx(by, abs=1e-9)


@pytest.mark.parametrize("theta", [0.0, 0.15, 0.6, -0.35])
def test_el_angulo_sobrevive_al_recorte(theta):
    """El angulo de la anotacion es dato; el de un trapecio es un artefacto.

    Si el recorte lo recalculase, meteria ruido en la unica magnitud que se mide
    aparte del IoU.
    """
    original = _quad(0.05, 0.5, 0.30, 0.12, theta)
    assert _angle(clip_quad(original)) == pytest.approx(_angle(original), abs=1e-6)


def test_una_violacion_pequena_no_se_come_el_lado_largo():
    """REGRESION del fallo que no se ve leyendo el codigo, solo midiendo.

    Caso real `Multiple_Euro_154`: un billete casi horizontal que se sale 0.026
    por arriba. Como el eje largo tenia `uy = 0.0315`, la restriccion `y >= 0`
    tambien se podia satisfacer encogiendo el lado LARGO -- y eso hacia: de 0.890
    a 0.064, conservando el 8% del billete. Cumplia todas las restricciones.

    La regla que lo arregla es asignar cada restriccion al eje mas alineado con
    ella. Aqui se ancla: el lado largo casi no se toca, el corto absorbe el
    recorte.
    """
    original = Quad.from_xy(
        [(0.096, -0.026), (0.985, 0.002), (0.974, 0.332), (0.085, 0.304)]
    )
    antes = _sides(original)
    despues = _sides(clip_quad(original))
    largo_antes, largo_despues = max(antes), max(despues)
    assert largo_despues > 0.95 * largo_antes, "el lado largo casi no debe tocarse"

    visible = Polygon(original.points).intersection(box(0.0, 0.0, 1.0, 1.0))
    rectangulo = Polygon(clip_quad(original).points)
    cubierto = rectangulo.intersection(visible).area / visible.area
    assert cubierto > 0.90, f"solo cubre el {cubierto:.1%} del billete visible"


def test_con_aspecto_no_cuadrado_el_rectangulo_lo_es_en_PIXELES():
    """En normalizadas un rectangulo girado es un PARALELOGRAMO.

    Sin pasar el aspecto, esto "arreglaria" la forma en el espacio equivocado:
    saldria rectangular en [0,1]x[0,1] y torcido en la imagen de verdad.
    """
    aspect = 2.5
    original = _quad(0.04 * aspect, 0.5, 0.30, 0.12, 0.4, aspect=aspect)
    recortado = clip_quad(original, aspect=aspect)
    a, b, c, d = _sides(recortado, aspect)
    assert a == pytest.approx(c, abs=1e-6)
    assert b == pytest.approx(d, abs=1e-6)
    # Y en normalizadas NO es un rectangulo, que es el motivo de pasar el
    # aspecto. Se mira el ANGULO entre lados, no su longitud: un paralelogramo
    # tambien tiene los lados opuestos iguales, asi que compararlos no distingue
    # nada. (Lo comprobaba asi y el test pasaba por casualidad.)
    puntos = list(recortado.points)
    ax = puntos[1][0] - puntos[0][0], puntos[1][1] - puntos[0][1]
    bx = puntos[2][0] - puntos[1][0], puntos[2][1] - puntos[1][1]
    coseno = (ax[0] * bx[0] + ax[1] * bx[1]) / (
        math.hypot(*ax) * math.hypot(*bx)
    )
    assert abs(coseno) > 1e-3, "en normalizadas las esquinas no son de 90 grados"


def test_un_quad_casi_entero_fuera_no_revienta():
    """El limite de lo que `Quad` admite: mas alla ni se puede construir.

    `Quad.from_xy` rechaza coordenadas fuera de [-0.5, 1.5], asi que "del todo
    fuera" no es un estado alcanzable. Esto es el caso extremo que si lo es.
    """
    fuera = _quad(-0.18, 0.5, 0.30, 0.12, 0.2)
    salida = clip_quad(fuera)
    assert all(math.isfinite(v) for v in salida.flat())
    for x, y in salida.points:
        assert -1e-9 <= x <= 1.0 + 1e-9
        assert -1e-9 <= y <= 1.0 + 1e-9
