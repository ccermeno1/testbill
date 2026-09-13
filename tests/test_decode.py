"""Decodificacion, NMS rotado y paso a quads canonicos."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="el candidato propio necesita torch")

from testbank.dataio.formats import ImageSize
from testbank.metrics.core import to_polygon
from testbank.models.decode import (
    box_to_polygon,
    boxes_to_quads,
    decode_outputs,
    detections,
    rotated_nms,
)
from testbank.models.yolox_obb import YoloxObb

SIZE = ImageSize(416, 416)


def box(cx, cy, w, h, theta=0.0):
    return torch.tensor([cx, cy, w, h, theta], dtype=torch.float32)


# --- el poligono de una caja ----------------------------------------------


def test_una_caja_alineada_da_el_rectangulo_esperado():
    polygon = box_to_polygon(box(100, 100, 40, 20, 0.0))
    minx, miny, maxx, maxy = polygon.bounds
    assert (minx, miny, maxx, maxy) == pytest.approx((80.0, 90.0, 120.0, 110.0))


def test_el_area_no_cambia_al_girar():
    """Girar no crea ni destruye superficie. Si cambiara, la rotacion estaria
    mal aplicada."""
    recta = box_to_polygon(box(100, 100, 60, 20, 0.0)).area
    girada = box_to_polygon(box(100, 100, 60, 20, 0.9)).area
    assert girada == pytest.approx(recta, rel=1e-6)


def test_girar_90_grados_intercambia_los_lados():
    polygon = box_to_polygon(box(100, 100, 60, 20, math.pi / 2))
    minx, miny, maxx, maxy = polygon.bounds
    assert (maxx - minx) == pytest.approx(20.0, abs=1e-4)
    assert (maxy - miny) == pytest.approx(60.0, abs=1e-4)


# --- NMS rotado -----------------------------------------------------------


def test_dos_detecciones_iguales_se_quedan_en_una():
    boxes = torch.stack([box(100, 100, 60, 30), box(100, 100, 60, 30)])
    kept = rotated_nms(boxes, torch.tensor([0.9, 0.8]))
    assert kept == [0], "sobrevive la de mas confianza"


def test_dos_detecciones_lejanas_sobreviven_las_dos():
    boxes = torch.stack([box(50, 50, 40, 20), box(350, 350, 40, 20)])
    assert len(rotated_nms(boxes, torch.tensor([0.9, 0.8]))) == 2


def test_el_nms_rotado_conserva_billetes_en_abanico():
    """La razon de que todo el proyecto sea OBB.

    Dos billetes alargados cruzados en angulos distintos tienen envolventes
    ALINEADAS que se solapan casi por completo: un NMS estandar suprimiria una
    deteccion verdadera. El rotado ve que los rectangulos apenas se tocan.
    """
    # A +45 y a -45: cruzados, pero con la MISMA envolvente cuadrada. Es el
    # caso que rompe el NMS alineado. Perpendiculares a 0 y 90 no valdrian: sus
    # envolventes son una barra horizontal y otra vertical, que apenas solapan.
    a = box(200, 200, 160, 30, math.pi / 4)
    b = box(200, 200, 160, 30, -math.pi / 4)
    boxes = torch.stack([a, b])

    from testbank.models.assign import enclosing_boxes, pairwise_iou

    iou_alineado = float(pairwise_iou(enclosing_boxes(boxes), enclosing_boxes(boxes))[0, 1])
    assert iou_alineado > 0.99, "las envolventes son el mismo cuadrado"

    # Pero el IoU rotado es bajo, y las dos sobreviven.
    assert len(rotated_nms(boxes, torch.tensor([0.9, 0.85]))) == 2


def test_sin_cajas_no_falla():
    assert rotated_nms(torch.zeros((0, 5)), torch.zeros(0)) == []


def test_se_respeta_el_tope_de_detecciones():
    boxes = torch.stack([box(20 + 40 * i, 20, 20, 10) for i in range(10)])
    scores = torch.linspace(0.9, 0.1, 10)
    assert len(rotated_nms(boxes, scores, max_detections=3)) == 3


def test_el_orden_de_salida_es_por_confianza():
    boxes = torch.stack([box(50, 50, 30, 20), box(300, 300, 30, 20)])
    kept = rotated_nms(boxes, torch.tensor([0.3, 0.95]))
    assert kept[0] == 1


# --- paso a quads ---------------------------------------------------------


def test_los_quads_salen_normalizados():
    quads = boxes_to_quads(torch.stack([box(208, 208, 100, 50)]), SIZE)
    assert len(quads) == 1
    for x, y in quads[0].points:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_el_quad_conserva_el_area_de_la_caja():
    caja = box(208, 208, 100, 50, 0.6)
    quad = boxes_to_quads(torch.stack([caja]), SIZE)[0]
    area_quad = to_polygon(quad, SIZE).area
    assert area_quad == pytest.approx(100 * 50, rel=1e-3)


def test_el_quad_sale_en_orden_canonico():
    """Si no lo estuviera, las metricas compararian vertices desemparejados."""
    from testbank.geometry.quad import is_canonical

    quad = boxes_to_quads(torch.stack([box(208, 208, 120, 60, 0.4)]), SIZE)[0]
    assert is_canonical(quad, aspect=SIZE.width / SIZE.height)


# --- el recorrido completo ------------------------------------------------


def _outputs():
    model = YoloxObb("nano", num_classes=1).eval()
    with torch.no_grad():
        return model(torch.randn(1, 3, SIZE.height, SIZE.width))


def test_decodificar_da_una_caja_por_celda():
    outputs = _outputs()
    boxes, scores = decode_outputs(outputs, SIZE)
    esperado = sum(o.distances.shape[-1] * o.distances.shape[-2] for o in outputs)
    assert boxes.shape == (esperado, 5)
    assert scores.shape == (esperado,)


def test_los_lados_decodificados_son_positivos():
    boxes, _ = decode_outputs(_outputs(), SIZE)
    assert torch.all(boxes[:, 2] >= 0)
    assert torch.all(boxes[:, 3] >= 0)


def test_el_angulo_decodificado_cae_en_medio_giro():
    boxes, _ = decode_outputs(_outputs(), SIZE)
    assert torch.all(boxes[:, 4] >= 0.0)
    assert torch.all(boxes[:, 4] < math.pi + 1e-6)


def test_las_puntuaciones_son_probabilidades():
    _, scores = decode_outputs(_outputs(), SIZE)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_una_red_recien_creada_apenas_detecta_nada():
    """El sesgo inicial deja p(objeto) = 0.01, asi que con umbral 0.25 no
    deberia salir casi nada. Si saliera, el sesgo no se estaria aplicando."""
    found = detections(_outputs(), SIZE, confidence=0.25)
    assert len(found) < 5


def test_detections_devuelve_predictions_usables_por_las_metricas():
    # El sesgo inicial deja objectness y clase en 0.01, y la puntuacion es su
    # producto: 1e-4. El umbral tiene que quedar por debajo de eso.
    found = detections(_outputs(), SIZE, confidence=1e-8, max_detections=10)
    assert found, "con umbral casi cero tiene que salir algo"
    for prediction in found:
        assert 0.0 <= prediction.score <= 1.0
        assert prediction.class_id == 0
        assert len(prediction.quad.points) == 4


def test_las_detecciones_salen_ordenadas_por_confianza():
    found = detections(_outputs(), SIZE, confidence=1e-8, max_detections=20)
    scores = [p.score for p in found]
    assert scores == sorted(scores, reverse=True)


def test_una_prediccion_muy_fuera_del_marco_cuenta_como_falso_positivo():
    """REGRESION. Con el backbone COCO recien cargado y la cabeza sin entrenar,
    la red predijo un vertice en -0.54 normalizado y `Quad` lo rechazo con un
    QuadError: la evaluacion entera de la ejecucion se cayo.

    No se descarta: se emite sin geometria y la metrica la cuenta como falso
    positivo. Descartarla habria sido regalar al modelo un error que cometio.
    """
    from testbank.metrics.core import ImageEval, PolygonCache, Prediction
    from testbank.metrics.matching import Outcome, match_image

    fuera = box(-300, 208, 100, 50)  # centro medio marco a la izquierda de la imagen
    dentro = box(208, 208, 100, 50)
    quads = boxes_to_quads(torch.stack([fuera, dentro]), SIZE)
    assert quads[0] is None and quads[1] is not None

    # En la metrica: la de dentro empareja con la verdad, la de fuera no
    # empareja con nada y queda como falso positivo a su puntuacion.
    truth = quads[1]
    item = ImageEval(
        sample_id="x", size=SIZE, truths=(truth,),
        predictions=(Prediction(quad=None, score=0.9), Prediction(quad=quads[1], score=0.8)),
    )
    matching = match_image(item, PolygonCache.build(item), match_iou=0.5)
    assert [p.prediction_index for p in matching.pairs] == [1]
    outcome = {index: result for index, result, _ in matching.outcomes}
    assert outcome[0] is Outcome.FALSE_POSITIVE, "sin geometria = falso positivo, no descarte"


def test_dibujar_una_prediccion_sin_geometria_no_revienta():
    """REGRESION: el port con DOTA produjo una prediccion sin quad, la metrica
    la conto bien y luego la VISUALIZACION de la ejecucion se cayo con
    `'NoneType' object has no attribute 'points'`."""
    import numpy as np

    from testbank.metrics.core import Prediction
    from testbank.viz.inspect import _draw_prediction

    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    _draw_prediction(canvas, Prediction(quad=None, score=0.5), offset=(0, 0), width=64, height=64)
    assert canvas.sum() == 0
