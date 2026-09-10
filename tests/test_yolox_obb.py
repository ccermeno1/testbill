"""Cabeza OBB propia sobre YOLOX: arquitectura y representacion del angulo."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="el candidato propio necesita torch")

from testbank.models.yolox_obb import (
    STRIDES,
    VARIANTS,
    YoloxObb,
    decode_angle,
    encode_angle,
)

IMAGE = 416


# --- el angulo ------------------------------------------------------------


def test_theta_y_theta_mas_180_dan_la_misma_codificacion():
    """La propiedad que motiva toda la representacion.

    Un rectangulo girado t y otro girado t+180 son el MISMO rectangulo. Con
    `(sin 2t, cos 2t)` caen en el mismo punto del circulo, asi que la
    ambiguedad desaparece por construccion en vez de corregirse despues.
    """
    theta = torch.tensor([0.1, 1.0, 2.5, 3.0])
    a = encode_angle(theta)
    b = encode_angle(theta + math.pi)
    assert torch.allclose(a, b, atol=1e-6)


def test_codificar_y_decodificar_devuelve_el_angulo():
    theta = torch.tensor([0.0, 0.3, 1.2, 2.0, 3.0])
    assert torch.allclose(decode_angle(encode_angle(theta)), theta, atol=1e-6)


def test_el_angulo_decodificado_siempre_cae_en_medio_giro():
    """Mas alla de pi se repite: no hay nada que distinguir ahi."""
    theta = torch.linspace(-10.0, 10.0, 200)
    out = decode_angle(encode_angle(theta))
    assert torch.all(out >= 0.0)
    assert torch.all(out < math.pi + 1e-6)


def test_la_codificacion_es_continua_al_pasar_por_cero():
    """Al cruzar 0/180 la distancia codificada sigue siendo pequena.

    Es el motivo de no regresar theta directamente. Un billete a 179.4 grados y
    otro a 0.6 estan a 1.2 grados de distancia real, pero en theta crudo
    distan 178.8: el modelo recibiria un gradiente enorme por acertar casi
    exactamente. Aqui la distancia codificada es proporcional a la real.
    """
    a, b = math.pi - 0.01, 0.01
    distancia_codificada = float(
        torch.norm(encode_angle(torch.tensor([a])) - encode_angle(torch.tensor([b])))
    )
    distancia_cruda = abs(a - b)

    # 0.02 rad de diferencia real -> 0.04 en el circulo doblado (~2 * 0.02).
    assert distancia_codificada == pytest.approx(0.04, abs=0.005)
    # Y frente a los 3.12 rad que veria una regresion directa sobre theta.
    assert distancia_cruda > 3.0
    assert distancia_codificada < distancia_cruda / 50


def test_angulos_de_verdad_distintos_no_colisionan():
    """El doblado une t con t+180, y NADA mas: si uniera de mas, el modelo no
    podria distinguir un billete tumbado de uno de pie."""
    a = encode_angle(torch.tensor([0.0]))
    b = encode_angle(torch.tensor([math.pi / 2]))
    assert not torch.allclose(a, b, atol=0.1)


# --- la arquitectura ------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_cada_variante_construye_y_corre(variant):
    model = YoloxObb(variant, num_classes=1).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, IMAGE, IMAGE))
    assert len(outputs) == 3


def test_variante_desconocida_dice_cuales_hay():
    with pytest.raises(ValueError, match="variante desconocida"):
        YoloxObb("gigante")


def test_las_formas_siguen_los_strides():
    model = YoloxObb("nano", num_classes=1).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(2, 3, IMAGE, IMAGE))
    for output, stride in zip(outputs, STRIDES):
        side = IMAGE // stride
        assert output.stride == stride
        assert output.distances.shape == (2, 4, side, side)
        assert output.angle.shape == (2, 2, side, side)
        assert output.objectness.shape == (2, 1, side, side)
        assert output.classes.shape == (2, 1, side, side)


def test_nano_cabe_en_un_movil():
    """Menos de un millon de parametros. Es la razon de elegir esta variante."""
    assert YoloxObb("nano", 1).parameter_count() < 1_000_000


def test_las_variantes_crecen_en_orden():
    counts = [YoloxObb(v, 1).parameter_count() for v in ("nano", "tiny", "small")]
    assert counts == sorted(counts)
    # nano tiene que ser MUCHO menor, no un poco: es convolucion separable, no
    # solo menos canales.
    assert counts[1] > 4 * counts[0]


def test_las_distancias_son_positivas():
    """Son distancias al centro de la celda: negativas no significan nada."""
    model = YoloxObb("nano", 1).eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, IMAGE, IMAGE))
    for output in outputs:
        assert torch.all(output.distances >= 0)


def test_al_arrancar_predice_casi_nada():
    """Sin el sesgo inicial, la red predice objeto en todas las celdas y el
    gradiente de miles de falsos positivos domina las primeras iteraciones."""
    model = YoloxObb("nano", 1).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, IMAGE, IMAGE))
    probability = torch.sigmoid(outputs[0].objectness).mean()
    assert probability < 0.05


def test_la_cabeza_comparte_pesos_entre_niveles():
    """Con ~450 imagenes, tres cabezas independientes triplican los parametros
    de la parte que mas facilmente se sobreajusta."""
    model = YoloxObb("nano", 1)
    # Un solo juego de ramas, no uno por nivel: los stems si son por nivel
    # porque cada uno recibe un numero de canales distinto.
    assert len(model.head.stems) == len(STRIDES)
    assert isinstance(model.head.cls_pred, torch.nn.Conv2d)
    assert model.head.cls_pred.out_channels == 1
    assert model.head.angle_pred.out_channels == 2


def test_el_gradiente_llega_a_todas_las_ramas():
    """Si una rama quedara desconectada, entrenaria en silencio sin aprender."""
    model = YoloxObb("nano", 1)
    outputs = model(torch.randn(1, 3, 128, 128))
    loss = sum(
        o.distances.sum() + o.angle.sum() + o.objectness.sum() + o.classes.sum()
        for o in outputs
    )
    loss.backward()
    for name in ("reg_pred", "angle_pred", "obj_pred", "cls_pred"):
        layer = getattr(model.head, name)
        assert layer.weight.grad is not None, name
        assert torch.any(layer.weight.grad != 0), name
