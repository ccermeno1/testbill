"""Las tres recetas de perdida: cada una entera, sobre la misma cabeza propia.

Que se comprueba y por que:

- Que la cabeza que exige cada receta es la que sale, y que un checkpoint la
  lleva dentro: cargar pesos DFL en una cabeza directa reventaria a mitad.
- Que `target_distances` es la inversa EXACTA del decodificador: si no lo
  fuera, la DFL y la L1 tardia aprenderian hacia una caja que no es la verdad.
- Propiedades de TAL y DFL que los papers garantizan.
- Que las tres recetas producen perdidas finitas con gradiente finito sobre
  una red sin entrenar, y que cada una reporta SUS terminos y no los de otra.
"""

from __future__ import annotations

import math

import pytest
import torch

from testbank.config import Config
from testbank.models.assign import AnchorGrid, build_anchor_grid, tal_assign
from testbank.models.decode import decode_outputs
from testbank.models.gaussian import pairwise_probiou
from testbank.models.recipes import (
    RECIPES,
    distribution_focal_loss,
    head_spec_for,
    losses_for_image,
    target_distances,
)
from testbank.models.train import load_model
from testbank.models.yolox_obb import STRIDES, HeadSpec, YoloxObb, decode_angle

torch.manual_seed(0)
SIDE = 128


def _config(recipe: str, **detector) -> Config:
    base = Config()
    loss = base.detector.loss.model_copy(update={"recipe": recipe})
    return base.model_copy(
        update={
            "detector": base.detector.model_copy(
                update={"image_size": SIDE, "epochs": 2, "loss": loss, **detector}
            )
        }
    )


def _single_image_outputs(model):
    outputs = model(torch.randn(1, 3, SIDE, SIDE))
    return outputs


def _targets():
    boxes = torch.tensor(
        [[64.0, 64.0, 60.0, 30.0, 0.4], [30.0, 90.0, 40.0, 20.0, 1.2]]
    )
    return boxes, torch.zeros(2, dtype=torch.long)


# --- la cabeza que exige cada receta ---------------------------------------


def test_cada_receta_fija_su_cabeza():
    assert head_spec_for(_config("own")) == HeadSpec()
    assert head_spec_for(_config("yolox_obb_fork")) == HeadSpec()
    ul = head_spec_for(_config("ultralytics_obb"))
    assert (ul.regression, ul.angle, ul.objectness) == ("dfl", "scalar", False)


def test_la_cabeza_dfl_saca_las_formas_correctas():
    model = YoloxObb("nano", head=HeadSpec(regression="dfl", angle="scalar", objectness=False))
    for o in _single_image_outputs(model):
        assert o.distances.shape[1] == 4
        assert o.distribution.shape[1] == 4 * 16
        assert o.angle.shape[1] == 1
        assert o.objectness is None
        # La esperanza de la distribucion cae en [0, reg_max - 1].
        assert (o.distances >= 0).all() and (o.distances <= 15).all()


def test_la_cabeza_directa_no_cambia():
    model = YoloxObb("nano")
    for o in _single_image_outputs(model):
        assert o.distribution is None and o.objectness is not None
        assert o.angle.shape[1] == 2


def test_el_checkpoint_lleva_la_cabeza_dentro(tmp_path):
    """Cargar unos pesos DFL en una cabeza directa fallaria por tamanos. La
    especificacion de la cabeza viaja con los pesos, no con la config."""
    model = YoloxObb("nano", head=HeadSpec(regression="dfl", angle="scalar", objectness=False))
    path = tmp_path / "w.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "variant": "nano",
            "num_classes": 1,
            "head": model.head_spec.to_dict(),
        },
        path,
    )
    loaded = load_model(path)
    assert loaded.head_spec == model.head_spec


def test_un_checkpoint_viejo_sin_cabeza_carga_como_directa(tmp_path):
    model = YoloxObb("nano")
    path = tmp_path / "w.pt"
    torch.save({"model": model.state_dict(), "variant": "nano", "num_classes": 1}, path)
    assert load_model(path).head_spec == HeadSpec()


# --- angulo escalar ---------------------------------------------------------


def test_el_angulo_escalar_cubre_medio_giro():
    logits = torch.linspace(-12, 12, 200).view(1, 1, 200, 1)
    theta = decode_angle(logits.permute(0, 1, 2, 3).reshape(200, 1), "scalar")
    assert (theta >= 0).all() and (theta < math.pi).all()
    # Sigmoide de -inf a +inf recorre (-pi/4, 3pi/4): modulo pi, todo el rango.
    assert theta.max() - theta.min() > 0.95 * math.pi


# --- target_distances es la inversa del decodificador ------------------------


def test_target_distances_deshace_el_decodificador():
    """Se construyen distancias, se decodifican a cajas, y se vuelve: tiene que
    salir lo mismo. Si no, la DFL aprenderia hacia otra caja."""
    from testbank.models.yolox_obb import HeadOutput, encode_angle

    grid_sizes = [(SIDE // s, SIDE // s) for s in STRIDES]
    grid = build_anchor_grid(grid_sizes, STRIDES, device=torch.device("cpu"))
    outputs = []
    torch.manual_seed(1)
    for (h, w), stride in zip(grid_sizes, STRIDES):
        distances = torch.rand(1, 4, h, w) * 4 + 0.5
        theta = torch.rand(1, h, w) * math.pi
        angle = encode_angle(theta).permute(0, 3, 1, 2)
        outputs.append(
            HeadOutput(
                distances=distances,
                angle=angle,
                objectness=torch.zeros(1, 1, h, w),
                classes=torch.zeros(1, 1, h, w),
                stride=stride,
            )
        )
    from testbank.dataio.formats import ImageSize

    boxes, _ = decode_outputs(outputs, ImageSize(SIDE, SIDE))
    back = target_distances(grid.centers, grid.strides, boxes)
    original = torch.cat([o.distances[0].permute(1, 2, 0).reshape(-1, 4) for o in outputs])
    assert torch.allclose(back, original, atol=1e-4), (back - original).abs().max()


# --- TAL ------------------------------------------------------------------


def _grid_for(side=SIDE) -> AnchorGrid:
    return build_anchor_grid([(side // s, side // s) for s in STRIDES], STRIDES, device=torch.device("cpu"))


def test_tal_solo_elige_celdas_dentro_de_la_caja():
    grid = _grid_for()
    boxes, classes = _targets()
    n = len(grid)
    predicted = torch.cat([grid.centers, torch.full((n, 1), 30.0), torch.full((n, 1), 15.0), torch.zeros(n, 1)], dim=1)
    scores = torch.rand(n, 1)
    a = tal_assign(predicted, scores, boxes, classes, grid, overlap=pairwise_probiou)
    from testbank.models.assign import points_in_rotated_boxes

    inside = points_in_rotated_boxes(grid.centers, boxes)
    assert a.positive.any()
    assert inside[a.positive].any(dim=1).all(), "una positiva fuera de toda caja"


def test_tal_da_a_lo_sumo_topk_por_billete_y_objetivos_suaves_en_cero_uno():
    grid = _grid_for()
    boxes, classes = _targets()
    n = len(grid)
    predicted = torch.cat([grid.centers, torch.full((n, 1), 30.0), torch.full((n, 1), 15.0), torch.zeros(n, 1)], dim=1)
    scores = torch.rand(n, 1)
    a = tal_assign(predicted, scores, boxes, classes, grid, overlap=pairwise_probiou, topk=5)
    for target in range(2):
        assert int(((a.matched == target) & a.positive).sum()) <= 5
    ts = a.target_scores
    assert ts is not None and (ts >= 0).all() and (ts <= 1).all()
    assert (ts[~a.positive] == 0).all(), "el fondo tiene objetivo cero"
    assert (ts[a.positive].sum(dim=1) > 0).all()


def test_tal_sin_billetes_devuelve_todo_fondo():
    grid = _grid_for()
    n = len(grid)
    predicted = torch.cat([grid.centers, torch.ones(n, 2) * 10, torch.zeros(n, 1)], dim=1)
    a = tal_assign(predicted, torch.rand(n, 1), torch.zeros(0, 5), torch.zeros(0, dtype=torch.long), grid, overlap=pairwise_probiou)
    assert not a.positive.any() and (a.target_scores == 0).all()


def test_tal_la_celda_mejor_alineada_recibe_su_mejor_solape():
    """Es la normalizacion de TOOD: max(metrica) -> max(solape) por billete."""
    grid = _grid_for()
    boxes, classes = _targets()
    n = len(grid)
    predicted = torch.cat([grid.centers, torch.full((n, 1), 60.0), torch.full((n, 1), 30.0), torch.full((n, 1), 0.4)], dim=1)
    scores = torch.full((n, 1), 0.9)
    a = tal_assign(predicted, scores, boxes, classes, grid, overlap=pairwise_probiou)
    for target in range(2):
        mine = a.positive & (a.matched == target)
        if not mine.any():
            continue
        assert a.target_scores[mine].max().item() == pytest.approx(a.matched_iou[mine].max().item(), abs=1e-5)


# --- DFL -----------------------------------------------------------------


def test_dfl_es_minima_cuando_la_masa_esta_en_los_bins_correctos():
    reg_max = 16
    target = torch.tensor([[3.25, 7.0, 0.5, 14.9]])
    logits = torch.full((1, 4 * reg_max), -20.0).view(1, 4, reg_max)
    for side, y in enumerate(target[0]):
        lo = int(y.floor())
        logits[0, side, lo] = math.log(float(lo + 1 - y) + 1e-9) + 20
        logits[0, side, min(lo + 1, reg_max - 1)] = math.log(float(y - lo) + 1e-9) + 20
    perfecta = distribution_focal_loss(logits.view(1, -1), target, reg_max)
    peor = distribution_focal_loss(torch.zeros(1, 4 * reg_max), target, reg_max)
    assert perfecta.item() < peor.item()
    assert perfecta.item() >= 0


def test_dfl_clampa_el_objetivo_al_ultimo_bin():
    """Un objetivo mas alla de reg_max-1 no puede representarse: se clampa en
    vez de indexar fuera y reventar."""
    logits = torch.randn(3, 4 * 16)
    loss = distribution_focal_loss(logits, torch.full((3, 4), 40.0), 16)
    assert torch.isfinite(loss).all()


# --- las tres recetas, de punta a punta --------------------------------------


@pytest.mark.parametrize("recipe", RECIPES)
def test_cada_receta_da_perdida_finita_con_gradiente(recipe):
    config = _config(recipe)
    model = YoloxObb("nano", head=head_spec_for(config))
    outputs = _single_image_outputs(model)
    boxes, classes = _targets()
    terms = losses_for_image(outputs, boxes, classes, config, torch.device("cpu"), epoch=1, total_epochs=2)
    assert torch.isfinite(terms.total)
    assert terms.num_positives > 0
    terms.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_cada_receta_reporta_sus_terminos_y_no_los_de_otra():
    boxes, classes = _targets()
    reported = {}
    for recipe in RECIPES:
        config = _config(recipe)
        model = YoloxObb("nano", head=head_spec_for(config))
        terms = losses_for_image(_single_image_outputs(model), boxes, classes, config, torch.device("cpu"), epoch=1, total_epochs=2)
        reported[recipe] = terms.to_dict()
    assert reported["own"]["angle"] > 0 and reported["own"]["dfl"] == 0 and reported["own"]["l1"] == 0
    assert reported["yolox_obb_fork"]["angle"] == 0 and reported["yolox_obb_fork"]["dfl"] == 0
    assert reported["yolox_obb_fork"]["objectness"] > 0
    assert reported["ultralytics_obb"]["dfl"] > 0
    assert reported["ultralytics_obb"]["objectness"] == 0 and reported["ultralytics_obb"]["angle"] == 0


def test_la_l1_del_fork_solo_se_enciende_al_final():
    boxes, classes = _targets()
    config = _config("yolox_obb_fork", epochs=20)
    model = YoloxObb("nano")
    outputs = _single_image_outputs(model)
    early = losses_for_image(outputs, boxes, classes, config, torch.device("cpu"), epoch=0, total_epochs=20)
    late = losses_for_image(outputs, boxes, classes, config, torch.device("cpu"), epoch=19, total_epochs=20)
    assert early.to_dict()["l1"] == 0.0
    assert late.to_dict()["l1"] > 0.0


def test_la_receta_ultralytics_exige_la_cabeza_dfl():
    config = _config("ultralytics_obb")
    model = YoloxObb("nano")  # cabeza directa, a proposito
    boxes, classes = _targets()
    with pytest.raises(ValueError, match="cabeza DFL"):
        losses_for_image(_single_image_outputs(model), boxes, classes, config, torch.device("cpu"))


def test_las_ganancias_por_defecto_son_las_de_cada_original():
    config = Config()
    assert config.detector.loss.fork.box_gain == 5.0
    assert config.detector.loss.fork.tau == 1.0
    ul = config.detector.loss.ultralytics
    assert (ul.box_gain, ul.cls_gain, ul.dfl_gain) == (7.5, 0.5, 1.5)
    assert (ul.reg_max, ul.tal_topk, ul.tal_alpha, ul.tal_beta) == (16, 10, 0.5, 6.0)


def test_la_receta_es_una_literal_y_no_acepta_otra_cosa():
    """Por el constructor, que valida. `model_copy(update=...)` NO valida -- es
    el footgun documentado en `StrictModel` -- y por eso `_config` no sirve
    aqui: aceptaria la cadena y reventaria mas tarde en el despacho."""
    from testbank.config import LossConfig

    with pytest.raises(ValueError):
        LossConfig(recipe="gaussiana_mezclada")


def test_una_receta_colada_sin_validar_revienta_en_el_despacho_con_mensaje():
    boxes, classes = _targets()
    config = _config("gaussiana_mezclada")  # se cuela: model_copy no valida
    with pytest.raises(ValueError, match="receta desconocida"):
        losses_for_image(_single_image_outputs(YoloxObb("nano")), boxes, classes, config, torch.device("cpu"))
