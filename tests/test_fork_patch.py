"""El parche que hace arrancar el fork de YOLOX-OBB sin compilar nada.

Lo que se prueba aqui NO es el fork -- es un repositorio ajeno y muerto que ni
siquiera esta clonado en CI. Es el parche: que el sustituto de `polyiou` da el
MISMO IoU que el que usa el resto del proyecto para medir.

Eso es lo unico que podria envenenar la comparacion sin avisar. Si el fork
puntuase con un IoU ligeramente distinto del de la tabla, su fila no seria
comparable con las demas y nada lo delataria.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
from shapely.geometry import Polygon

# Se importa con nombre que NO empiece por `test`: pytest lo recogeria como
# caso de prueba y fallaria pidiendo fixtures llamadas `a` y `b`.
from testbank.metrics.core import iou as referencia_iou

PATCHER = Path(__file__).resolve().parents[1] / "tools" / "patch_yolox_obb_fork.py"


@pytest.fixture(scope="module")
def shim(tmp_path_factory):
    """Carga el texto del shim tal cual lo escribe el parcheador."""
    spec = importlib.util.spec_from_file_location("patcher", PATCHER)
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)

    path = tmp_path_factory.mktemp("shim") / "polyiou.py"
    path.write_text(patcher.POLYIOU_SHIM, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("polyiou_shim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rect(cx, cy, half_w, half_h, theta):
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    flat = []
    for lx, ly in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
        flat += [cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a]
    return flat


def _polygon(flat):
    return Polygon([(flat[i], flat[i + 1]) for i in range(0, 8, 2)])


@pytest.mark.parametrize(
    "theta_a,theta_b,dx",
    [
        (0.0, 0.0, 0.0),      # identicos
        (0.0, 0.0, 30.0),     # solapamiento parcial
        (0.0, 0.0, 500.0),    # disjuntos
        (0.0, 0.4, 0.0),      # girados entre si
        (0.3, -0.9, 12.0),    # el caso general
        (0.0, math.pi / 2, 0.0),
    ],
)
def test_el_shim_da_el_mismo_iou_que_testbank(shim, theta_a, theta_b, dx):
    a = _rect(100.0, 100.0, 40.0, 20.0, theta_a)
    b = _rect(100.0 + dx, 100.0, 40.0, 20.0, theta_b)
    esperado = referencia_iou(_polygon(a), _polygon(b))
    assert shim.iou_poly(shim.VectorDouble(a), shim.VectorDouble(b)) == pytest.approx(
        esperado, abs=1e-12
    )


def test_vector_double_acepta_lo_que_le_pasa_su_nms(shim):
    """Su `py_cpu_nms_poly` construye el vector desde un array de numpy."""
    valores = shim.VectorDouble([1, 2, 3, 4, 5, 6, 7, 8])
    assert valores == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert all(isinstance(v, float) for v in valores)


def test_un_cuadrilatero_con_los_lados_cruzados_no_da_un_iou_absurdo(shim):
    """El original en C++ no se atraganta; shapely sin reparar, si.

    Un poligono en forma de lazo es invalido y su `.area` sale mal, asi que el
    shim hace `buffer(0)` -- lo mismo que `to_polygon` en metrics.
    """
    lazo = [0.0, 0.0, 10.0, 10.0, 10.0, 0.0, 0.0, 10.0]
    valor = shim.iou_poly(shim.VectorDouble(lazo), shim.VectorDouble(lazo))
    assert 0.0 <= valor <= 1.0
    assert not math.isnan(valor)


def test_el_parcheador_rechaza_un_directorio_que_no_es_el_fork(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("patcher", PATCHER)
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    assert patcher.apply(tmp_path) == 2
    assert "no parece un clon" in capsys.readouterr().err


def test_el_descarte_por_envolvente_no_cambia_ningun_resultado(shim):
    """x15 medido. Pero una optimizacion que cambie un solo IoU no vale nada:
    se compara contra shapely a pelo sobre pares aleatorios, tocandose o no."""
    import random

    rng = random.Random(0)
    for _ in range(300):
        a = _rect(rng.uniform(0, 400), rng.uniform(0, 400), 30, 15, rng.uniform(-3, 3))
        b = _rect(rng.uniform(0, 400), rng.uniform(0, 400), 30, 15, rng.uniform(-3, 3))
        esperado = referencia_iou(_polygon(a), _polygon(b))
        assert shim.iou_poly(shim.VectorDouble(a), shim.VectorDouble(b)) == pytest.approx(
            esperado, abs=1e-12
        )
