"""Las cinco invariantes del volteo horizontal.

Derivacion (verificada antes de escribir el test, porque es donde es facil
escribir un test que hace fallar una implementacion correcta):

Con q = p0,p1,p2,p3 canonico horario y (p0,p1) lado largo, M invierte el sentido
de giro, asi que el recorrido horario del volteado es el ciclo invertido

    M(p0) -> M(p3) -> M(p2) -> M(p1) -> M(p0)

cuyas aristas son las imagenes de (p3,p0), (p2,p3), (p1,p2), (p0,p1). Las largas
son la segunda y la cuarta, abiertas por M(p3) y M(p1). De ahi la invariante 4:
el ancla del volteado es la imagen especular de un vertice que CIERRA lado largo,
nunca de uno que lo abre.
"""

from __future__ import annotations

import pytest
from conftest import rotated_rects
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from testbank.geometry.quad import (
    Quad,
    canonical_order,
    canonicalize,
    flip_horizontal,
    snap,
)


def mirror(point: tuple[float, float]) -> tuple[float, float]:
    return (1.0 - point[0], point[1])


# -- invariante 1 -----------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=500)
def test_1_flip_es_involucion_exacta(quad: Quad):
    """Igualdad exacta, no aproximada. La rejilla diadica es lo que la permite."""
    assert flip_horizontal(flip_horizontal(quad)).points == quad.points


def test_1_involucion_exacta_en_el_caso_que_rompe_float64():
    """1-(1-0.1) da 0.09999999999999998 sin rejilla."""
    quad = Quad.from_xy([0.1, 0.1, 0.7, 0.1, 0.7, 0.4, 0.1, 0.4])
    assert flip_horizontal(flip_horizontal(quad)).points == quad.points


# -- invariante 2 -----------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=300)
def test_2_conserva_el_conjunto_de_puntos(quad: Quad):
    canon = canonicalize(quad)
    flipped = canonicalize(flip_horizontal(canon))
    assert set(flipped.points) == {mirror(p) for p in canon.points}


# -- invariante 3 -----------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=300)
def test_3_el_volteo_invierte_el_giro_y_la_recanonicalizacion_lo_restaura(quad: Quad):
    canon = canonicalize(quad)
    assert canon.is_clockwise()
    raw_flip = flip_horizontal(canon)
    assert not raw_flip.is_clockwise(), "la reflexion tiene que invertir el sentido"
    assert canonicalize(raw_flip).is_clockwise()


# -- invariante 4 -----------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=500)
def test_4_el_ancla_del_volteado_es_imagen_de_un_vertice_que_cierra_lado_largo(
    quad: Quad,
):
    canon = canonicalize(quad)
    p0, p1, p2, p3 = canon.points
    anchor = canonicalize(flip_horizontal(canon)).points[0]
    assert anchor in {mirror(p1), mirror(p3)}
    assert anchor not in {mirror(p0), mirror(p2)}


def test_4_caso_explicito_a_mano():
    """Rectangulo alineado 0.4 x 0.2. Canonico [A,B,C,D]; volteado ancla en M(p1)."""
    quad = Quad.from_xy([0.1, 0.1, 0.5, 0.1, 0.5, 0.3, 0.1, 0.3])
    canon = canonicalize(quad)
    assert canon.points[0] == (snap(0.1), snap(0.1))
    anchor = canonicalize(flip_horizontal(canon)).points[0]
    assert anchor == mirror(canon.points[1]) == (1.0 - snap(0.5), snap(0.1))


# -- invariante 5 -----------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=300)
def test_5_determinismo_repetido(quad: Quad):
    a = canonicalize(flip_horizontal(canonicalize(quad))).points
    b = canonicalize(flip_horizontal(canonicalize(quad))).points
    assert a == b


@given(rotated_rects(min_ratio=1.6), st.integers(min_value=0, max_value=255))
@settings(max_examples=400)
def test_5_determinismo_bajo_perturbacion_subtolerante(quad: Quad, seed: int):
    """Estabilidad de la DECISION, no de las coordenadas.

    La perturbacion mueve los puntos, asi que comparar valores no dice nada. Lo
    que tiene que quedarse quieto es que vertice hace de ancla. Se condiciona a
    que el margen de desempate supere la perturbacion: por debajo de el la
    eleccion puede cambiar legitimamente, y esa frontera es inevitable.
    """
    canon = canonicalize(quad)
    flipped = flip_horizontal(canon)
    perm, margin = canonical_order(flipped)
    jitter = 1e-5
    assume(margin > 8.0 * jitter)

    moved = []
    for i, (x, y) in enumerate(flipped.points):
        sign = 1.0 if (seed >> i) & 1 else -1.0
        moved.append((x + sign * jitter, y - sign * jitter))
    perturbed_perm, _ = canonical_order(Quad.from_xy(moved))
    assert perturbed_perm == perm


# -- guardas ----------------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=200)
def test_doble_volteo_canonico_vuelve_al_canonico_original(quad: Quad):
    canon = canonicalize(quad)
    there = canonicalize(flip_horizontal(canon))
    back = canonicalize(flip_horizontal(there))
    assert back.points == canon.points


@given(rotated_rects())
@settings(max_examples=200)
def test_el_volteo_conserva_area_y_longitudes(quad: Quad):
    canon = canonicalize(quad)
    flipped = canonicalize(flip_horizontal(canon))
    assert abs(flipped.signed_area()) == pytest.approx(abs(canon.signed_area()))
    assert sorted(flipped.edge_lengths()) == pytest.approx(sorted(canon.edge_lengths()))
