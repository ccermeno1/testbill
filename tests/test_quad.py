from __future__ import annotations

import itertools
import math

import pytest
from conftest import rotated_rect_points, rotated_rects
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from testbank.geometry.quad import (
    LONG_SIDE_TIE_TOL,
    CoordinateRangeWarning,
    Quad,
    QuadError,
    QuadShapeWarning,
    canonical_order,
    canonicalize,
    is_canonical,
    snap,
)


# -- construccion y validacion ---------------------------------------------


def test_rechaza_numero_de_coordenadas_incorrecto():
    with pytest.raises(QuadError, match="8 coordenadas"):
        Quad.from_xy([0.1, 0.1, 0.5, 0.1, 0.5, 0.3])


def test_rechaza_no_finito():
    with pytest.raises(QuadError, match="no finita"):
        Quad.from_xy([0.1, 0.1, float("nan"), 0.1, 0.5, 0.3, 0.1, 0.3])


def test_rechaza_fuera_del_rango_tolerante():
    with pytest.raises(QuadError, match="fuera del rango"):
        Quad.from_xy([0.1, 0.1, 1.9, 0.1, 1.9, 0.3, 0.1, 0.3])


def test_avisa_pero_acepta_fuera_de_0_1():
    """Un billete que cruza el borde tiene vertices fuera de [0,1] legitimamente."""
    with pytest.warns(CoordinateRangeWarning):
        quad = Quad.from_xy([-0.2, 0.1, 0.5, 0.1, 0.5, 0.3, -0.2, 0.3])
    assert quad.points[0][0] == pytest.approx(-0.2)


def test_rechaza_degenerado():
    with pytest.raises(QuadError):
        Quad.from_xy([0.1, 0.1, 0.5, 0.1, 0.9, 0.1, 0.3, 0.1])


def test_avisa_ratio_inestable():
    """Ratio < 1.1: el ancla del lado mas largo la decide el ruido."""
    quad = Quad.from_xy(rotated_rect_points(0.5, 0.5, 0.2, 1.02, 0.3))
    with pytest.warns(QuadShapeWarning, match="inestable"):
        canonicalize(quad)


# -- rejilla diadica --------------------------------------------------------


@given(st.floats(min_value=-0.5, max_value=1.5))
def test_snap_es_idempotente(value: float):
    assert snap(snap(value)) == snap(value)


@given(st.floats(min_value=-0.5, max_value=1.5))
def test_reflexion_exacta_sobre_la_rejilla(value: float):
    """1-x no redondea en la rejilla: es lo que hace exacta la involucion."""
    x = snap(value)
    assert 1.0 - (1.0 - x) == x


def test_el_error_de_rejilla_es_despreciable():
    assert abs(snap(0.123456789) - 0.123456789) < 2.0**-31


# -- orden canonico ---------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=300)
def test_canonico_es_horario(quad: Quad):
    assert canonicalize(quad).is_clockwise()


@given(rotated_rects())
@settings(max_examples=300)
def test_canonico_es_idempotente(quad: Quad):
    once = canonicalize(quad)
    assert canonicalize(once).points == once.points
    assert is_canonical(once)


@given(rotated_rects())
@settings(max_examples=300)
def test_ancla_abre_el_lado_mas_largo(quad: Quad):
    canon = canonicalize(quad)
    lengths = canon.edge_lengths()
    assert lengths[0] >= max(lengths) * (1.0 - LONG_SIDE_TIE_TOL)


@given(rotated_rects())
@settings(max_examples=200)
def test_canonico_no_depende_del_orden_de_entrada(quad: Quad):
    """Las 24 permutaciones de los mismos 4 puntos dan el mismo canonico."""
    reference = canonicalize(quad).points
    for perm in itertools.permutations(quad.points):
        assert canonicalize(Quad(points=perm)).points == reference


@given(rotated_rects())
@settings(max_examples=200)
def test_canonico_conserva_el_conjunto_de_puntos(quad: Quad):
    assert set(canonicalize(quad).points) == set(quad.points)


def test_desempate_a_45_grados_es_determinista():
    """x+y empata cuando la diagonal es perpendicular a (1,1). Cadena completa."""
    quad = Quad.from_xy(rotated_rect_points(0.5, 0.5, 0.2, 2.0, math.pi / 4))
    first = canonicalize(quad).points
    for perm in itertools.permutations(quad.points):
        assert canonicalize(Quad(points=perm)).points == first


# -- determinismo -----------------------------------------------------------


@given(rotated_rects())
@settings(max_examples=200)
def test_determinismo_repetido(quad: Quad):
    assert canonicalize(quad).points == canonicalize(quad).points


@given(rotated_rects(min_ratio=1.6), st.integers(min_value=0, max_value=255))
@settings(max_examples=400)
def test_determinismo_bajo_perturbacion_subtolerante(quad: Quad, seed: int):
    """La forma literal de la especificacion es imposible.

    "Una entrada perturbada por debajo de la tolerancia da la misma salida" no
    puede cumplirse en general: toda seleccion discreta sobre entrada continua
    tiene frontera, y la tolerancia la mueve, no la elimina. Lo que si es cierto
    y es lo que se testea: si la separacion entre candidatos supera el tamano de
    la perturbacion, esta no cambia el ancla.

    Se compara la permutacion, no las coordenadas: la perturbacion mueve los
    puntos por definicion, y lo que debe quedarse quieto es la decision.

    Con ratio >= 1.6 los lados cortos quedan al 62% del largo, muy por encima de
    la tolerancia de empate del 5%, asi que el conjunto de candidatos es estable
    y la unica decision en juego es el desempate por x+y.
    """
    perm, margin = canonical_order(quad)
    jitter = 1e-5
    assume(margin > 8.0 * jitter)

    jittered = []
    for i, (x, y) in enumerate(quad.points):
        sign = 1.0 if (seed >> i) & 1 else -1.0
        jittered.append((x + sign * jitter, y - sign * jitter))
    perturbed_perm, _ = canonical_order(Quad.from_xy(jittered))
    assert perturbed_perm == perm
