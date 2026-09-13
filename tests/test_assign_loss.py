"""Asignacion SimOTA y perdidas del candidato propio."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="el candidato propio necesita torch")

from testbank.config import AngleWeightConfig
from testbank.models.assign import (
    Assignment,
    build_anchor_grid,
    enclosing_boxes,
    pairwise_iou,
    points_in_rotated_boxes,
    simota_assign,
)
from testbank.models.loss import angle_loss, compute_losses, iou_loss


def box(cx, cy, w, h, theta=0.0):
    return torch.tensor([[cx, cy, w, h, theta]], dtype=torch.float32)


# --- la rejilla -----------------------------------------------------------


def test_la_rejilla_tiene_una_celda_por_posicion():
    grid = build_anchor_grid([(52, 52), (26, 26), (13, 13)], (8, 16, 32))
    assert len(grid) == 52 * 52 + 26 * 26 + 13 * 13


def test_los_puntos_caen_en_el_CENTRO_de_la_celda():
    """Sin el medio pixel, todas las cajas salen sesgadas media celda: a stride
    32 son 16 pixeles de sesgo sistematico."""
    grid = build_anchor_grid([(2, 2)], (32,))
    assert torch.allclose(grid.centers[0], torch.tensor([16.0, 16.0]))
    assert torch.allclose(grid.centers[-1], torch.tensor([48.0, 48.0]))


# --- geometria ------------------------------------------------------------


def test_el_centro_siempre_esta_dentro():
    points = torch.tensor([[100.0, 100.0]])
    assert bool(points_in_rotated_boxes(points, box(100, 100, 40, 20, 0.7))[0, 0])


def test_un_punto_lejano_esta_fuera():
    points = torch.tensor([[300.0, 300.0]])
    assert not bool(points_in_rotated_boxes(points, box(100, 100, 40, 20))[0, 0])


def test_el_giro_cambia_que_puntos_estan_dentro():
    """Si no lo hiciera, la comprobacion estaria ignorando el angulo."""
    punto = torch.tensor([[100.0, 118.0]])  # arriba del centro
    horizontal = box(100, 100, 60, 20, 0.0)
    vertical = box(100, 100, 60, 20, math.pi / 2)
    assert not bool(points_in_rotated_boxes(punto, horizontal)[0, 0])
    assert bool(points_in_rotated_boxes(punto, vertical)[0, 0])


def test_la_envolvente_de_una_caja_alineada_es_ella_misma():
    out = enclosing_boxes(box(100, 100, 40, 20, 0.0))[0]
    assert torch.allclose(out, torch.tensor([80.0, 90.0, 120.0, 110.0]), atol=1e-4)


def test_la_envolvente_de_una_girada_es_mayor():
    """Es la aproximacion que sostiene el coste, y hay que saber que es holgada."""
    recta = enclosing_boxes(box(100, 100, 60, 20, 0.0))[0]
    girada = enclosing_boxes(box(100, 100, 60, 20, math.pi / 4))[0]
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    assert area(girada) > area(recta)


def test_iou_de_cajas_identicas_es_uno():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert float(pairwise_iou(a, a)) == pytest.approx(1.0)


def test_iou_de_cajas_disjuntas_es_cero():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[50.0, 50.0, 60.0, 60.0]])
    assert float(pairwise_iou(a, b)) == pytest.approx(0.0)


# --- la asignacion --------------------------------------------------------


def _scenario(n_targets=1):
    grid = build_anchor_grid([(16, 16)], (16,))
    targets = torch.cat(
        [box(80 + 100 * i, 128, 60, 30, 0.0) for i in range(n_targets)]
    )
    predicted = torch.cat(
        [grid.centers, torch.full((len(grid), 2), 40.0), torch.zeros(len(grid), 1)],
        dim=1,
    )
    scores = torch.full((len(grid),), 0.5)
    return grid, targets, predicted, scores


def test_una_imagen_sin_billetes_no_asigna_nada():
    """Caso legitimo: el filtro de area puede vaciar una imagen de franjas."""
    grid, _, predicted, scores = _scenario()
    result = simota_assign(predicted, scores, torch.zeros((0, 5)), grid)
    assert result.num_positives == 0
    assert not result.positive.any()


def test_todo_billete_recibe_al_menos_un_positivo():
    """Un billete sin positivos no genera gradiente: es como no anotarlo."""
    grid, targets, predicted, scores = _scenario(n_targets=2)
    result = simota_assign(predicted, scores, targets, grid)
    asignados = set(result.matched[result.positive].tolist())
    assert asignados == {0, 1}


def test_los_positivos_caen_cerca_del_billete():
    grid, targets, predicted, scores = _scenario()
    result = simota_assign(predicted, scores, targets, grid)
    centros = grid.centers[result.positive]
    distancia = (centros - targets[0, :2]).abs().max()
    assert float(distancia) < 100.0


def test_ninguna_celda_sirve_a_dos_billetes():
    """Recibiria dos objetivos distintos y aprenderia el promedio, que no es
    ninguno de los dos."""
    grid, targets, predicted, scores = _scenario(n_targets=2)
    result = simota_assign(predicted, scores, targets, grid)
    assert isinstance(result, Assignment)
    # `matched` es un unico indice por celda: la exclusividad es estructural.
    assert result.matched.shape == (len(grid),)
    assert result.matched[result.positive].min() >= 0


def test_los_positivos_son_una_minoria():
    """Si casi todo fuera positivo, el fondo dejaria de ensenarse."""
    grid, targets, predicted, scores = _scenario()
    result = simota_assign(predicted, scores, targets, grid)
    assert 0 < result.num_positives < len(grid) // 4


# --- las perdidas ---------------------------------------------------------


def test_una_caja_perfecta_no_tiene_perdida_de_caja():
    caja = box(100, 100, 60, 30, 0.0)
    assert float(iou_loss(caja, caja)) == pytest.approx(0.0, abs=1e-5)


def test_una_caja_lejana_tiene_perdida_maxima():
    assert float(iou_loss(box(0, 0, 10, 10), box(500, 500, 10, 10))) == pytest.approx(1.0)


def test_el_angulo_acertado_no_tiene_perdida():
    theta = torch.tensor([0.7])
    predicho = torch.stack((torch.sin(2 * theta), torch.cos(2 * theta)), dim=-1)
    wh = torch.tensor([[60.0, 20.0]])
    loss, _ = angle_loss(predicho, theta, wh)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_acertar_el_angulo_mas_180_tampoco_penaliza():
    """La propiedad que motiva toda la representacion, ahora en la perdida."""
    theta = torch.tensor([0.3])
    predicho_girado = torch.stack(
        (torch.sin(2 * (theta + math.pi)), torch.cos(2 * (theta + math.pi))), dim=-1
    )
    wh = torch.tensor([[60.0, 20.0]])
    loss, _ = angle_loss(predicho_girado, theta, wh)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_el_atenuador_baja_la_perdida_de_una_caja_casi_cuadrada():
    """Es su motivo de ser: no castigar un angulo que no esta definido."""
    theta = torch.tensor([0.0])
    predicho = torch.tensor([[1.0, 0.0]])  # 90 grados de error
    casi_cuadrada = torch.tensor([[50.0, 49.0]])

    con, pesos = angle_loss(predicho, theta, casi_cuadrada)
    sin, _ = angle_loss(predicho, theta, casi_cuadrada, enabled=False)
    # ratio 50/49 = 1.02, apenas un 20% del camino hasta el umbral 1.1
    assert float(pesos[0]) == pytest.approx(0.108, abs=0.01)
    assert float(con) == pytest.approx(0.108, abs=0.01)
    assert float(sin) == pytest.approx(1.0)


def test_el_atenuador_no_toca_una_caja_bien_definida():
    theta = torch.tensor([0.0])
    predicho = torch.tensor([[1.0, 0.0]])
    alargada = torch.tensor([[100.0, 40.0]])
    con, pesos = angle_loss(predicho, theta, alargada)
    sin, _ = angle_loss(predicho, theta, alargada, enabled=False)
    assert float(pesos[0]) == pytest.approx(1.0)
    assert float(con) == pytest.approx(float(sin))


def test_se_normaliza_por_N_y_no_por_la_suma_de_pesos():
    """Dividir por la suma de pesos DESHARIA la atenuacion.

    Con una sola caja de peso 0.1, `(1 * 0.1) / 0.1` vuelve a dar 1.0; y con un
    lote entero de cajas ambiguas el gradiente saldria a plena potencia, que es
    justo lo que el atenuador existe para evitar.
    """
    theta = torch.zeros(2)
    predicho = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    # Una bien definida (peso 1) y otra casi cuadrada (peso ~0.108).
    wh = torch.tensor([[100.0, 40.0], [50.0, 49.0]])
    loss, pesos = angle_loss(predicho, theta, wh)
    esperado = (float(pesos[0]) + float(pesos[1])) / 2
    assert float(loss) == pytest.approx(esperado, abs=1e-4)
    assert float(loss) < 1.0, "atenuar tiene que reducir la magnitud de verdad"


def test_un_lote_entero_de_cajas_ambiguas_apenas_pesa():
    """El caso que la normalizacion por suma de pesos rompia por completo."""
    theta = torch.zeros(4)
    predicho = torch.tensor([[1.0, 0.0]] * 4)
    casi_cuadradas = torch.tensor([[50.0, 49.0]] * 4)
    loss, _ = angle_loss(predicho, theta, casi_cuadradas)
    assert float(loss) < 0.2


def test_sin_positivos_solo_queda_la_perdida_de_fondo():
    n = 32
    vacio = Assignment(
        positive=torch.zeros(n, dtype=torch.bool),
        matched=torch.zeros(n, dtype=torch.long),
        matched_iou=torch.zeros(n),
    )
    terms = compute_losses(
        torch.zeros(n, 5), torch.zeros(n, 2), torch.zeros(n), torch.zeros(n, 1),
        torch.zeros((0, 5)), torch.zeros(0, dtype=torch.long), vacio,
    )
    assert terms.num_positives == 0
    assert float(terms.box) == 0.0
    assert float(terms.objectness) > 0.0
    assert float(terms.total) > 0.0


def test_las_perdidas_se_reportan_por_separado():
    """Un escalar unico oculta cual de los tres terminos esta fallando."""
    grid, targets, predicted, scores = _scenario()
    assignment = simota_assign(predicted, scores, targets, grid)
    terms = compute_losses(
        predicted,
        torch.zeros(len(grid), 2),
        torch.zeros(len(grid)),
        torch.zeros(len(grid), 1),
        targets,
        torch.zeros(1, dtype=torch.long),
        assignment,
        angle_config=AngleWeightConfig(),
    )
    keys = terms.to_dict()
    # `dfl` y `l1` son de otras recetas y aqui valen 0: se reportan siempre
    # para que el registro de toda ejecucion tenga las mismas columnas.
    assert set(keys) == {
        "box", "angle", "objectness", "classes", "dfl", "l1", "total", "num_positives"
    }
    assert keys["dfl"] == 0.0 and keys["l1"] == 0.0
    assert keys["num_positives"] > 0


def test_el_gradiente_fluye_desde_la_perdida_total():
    grid, targets, predicted, scores = _scenario()
    assignment = simota_assign(predicted, scores, targets, grid)
    angulo = torch.zeros(len(grid), 2, requires_grad=True)
    objeto = torch.zeros(len(grid), requires_grad=True)
    terms = compute_losses(
        predicted, angulo, objeto, torch.zeros(len(grid), 1),
        targets, torch.zeros(1, dtype=torch.long), assignment,
    )
    terms.total.backward()
    assert objeto.grad is not None and torch.any(objeto.grad != 0)
