"""El port de DDGRCF/YOLOX_OBB, el IoU de poligonos y los cargadores de pesos.

La comparacion tensor a tensor contra SU modelo construido de verdad se hizo
fuera de la suite (necesita su clon y un stub de sus operadores): 426 claves y
formas identicas, `strict=True` carga, y con sus pesos la salida coincide con
la suya a 3e-5 px. Aqui se ancla lo que se puede anclar sin el clon: el numero
de parametros que salio de esa comparacion, los nombres que su checkpoint
espera, y que los cargadores fallan ruidosamente cuando algo no encaja.
"""

from __future__ import annotations

import math

import pytest
import torch
from shapely.geometry import Polygon

from testbank.models.ddgrcf import DdgrcfYoloxObb
from testbank.models.ddgrcf import load_pretrained as load_ddgrcf
from testbank.models.polygon import box_corners, pairwise_rotated_iou, rotated_iou
from testbank.models.pretrained import load_megvii_yolox, remap_megvii_key
from testbank.models.recipes import regularize_angle_ddgrcf
from testbank.models.yolox_obb import YoloxObb

torch.manual_seed(0)

# --- el port -----------------------------------------------------------------


def test_el_port_tiene_exactamente_los_parametros_de_su_modelo():
    """8.051.797: el numero que dio SU `Model` construido con sus yamls."""
    assert DdgrcfYoloxObb(num_classes=1).parameter_count() == 8_051_797


def test_las_claves_siguen_su_esquema_de_nombres():
    keys = set(DdgrcfYoloxObb(num_classes=1).state_dict())
    assert "model.0.conv.weight" in keys
    assert "model.2.m.0.cv1.conv.weight" in keys
    # Los tallos de las cabezas van SIN Sequential: `round(2 * 0.33) = 1`.
    assert "model.27.conv.weight" in keys and "model.27.0.conv.weight" not in keys
    assert "model.33.cls_preds.0.bias" in keys and "model.33.reg_preds.2.weight" in keys


def test_el_port_saca_tres_niveles_con_su_regresion_y_angulo():
    outputs = DdgrcfYoloxObb(num_classes=1)(torch.zeros(1, 3, 128, 128))
    assert [o.stride for o in outputs] == [8, 16, 32]
    for o in outputs:
        assert o.regression == "yolox" and o.angle_mode == "radians"
        assert o.distances.shape[1] == 4 and o.angle.shape[1] == 1
        assert o.objectness is not None and o.classes.shape[1] == 1


def test_cargar_sus_pesos_de_dota_salta_solo_la_capa_de_clase():
    """Sus pesos tienen 15 clases; el port, una. Todo lo demas, estricto."""
    fifteen = DdgrcfYoloxObb(num_classes=15).state_dict()
    model = DdgrcfYoloxObb(num_classes=1)
    skipped = load_ddgrcf(model, _saved(fifteen))
    assert skipped and all("cls_preds" in k for k in skipped)
    assert torch.equal(model.state_dict()["model.0.conv.weight"], fifteen["model.0.conv.weight"])


def test_un_checkpoint_que_no_cubre_el_port_es_error():
    partial = {k: v for k, v in DdgrcfYoloxObb(num_classes=1).state_dict().items() if "model.1" not in k}
    with pytest.raises(RuntimeError, match="no cubre el port"):
        load_ddgrcf(DdgrcfYoloxObb(num_classes=1), _saved(partial))


def _saved(state, tmp=[]):  # noqa: B006
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "w.pth"
    torch.save({"model": state}, path)
    return path


# --- el IoU exacto en torch ----------------------------------------------------


def _random_boxes(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack(
        [
            torch.rand(n, generator=g) * 300 + 50,
            torch.rand(n, generator=g) * 300 + 50,
            20 + torch.rand(n, generator=g) * 150,
            10 + torch.rand(n, generator=g) * 80,
            (torch.rand(n, generator=g) - 0.5) * math.pi,
        ],
        -1,
    )


def _shapely(a, b):
    out = []
    for x, y in zip(box_corners(a).tolist(), box_corners(b).tolist()):
        pa, pb = Polygon(x), Polygon(y)
        inter = pa.intersection(pb).area
        union = pa.area + pb.area - inter
        out.append(inter / union if union > 0 else 0.0)
    return torch.tensor(out)


def test_el_iou_de_poligonos_coincide_con_shapely():
    a, b = _random_boxes(500, 1), _random_boxes(500, 2)
    b[:, :2] = a[:, :2] + (torch.rand(500, 2) - 0.5) * 80  # que se solapen
    assert torch.allclose(rotated_iou(a, b), _shapely(a, b), atol=1e-5)


def test_casos_limite_del_iou():
    b = _random_boxes(20, 3)
    assert torch.allclose(rotated_iou(b, b), torch.ones(20), atol=1e-5)
    lejos = b.clone()
    lejos[:, 0] += 1000
    assert rotated_iou(b, lejos).max().item() == 0.0
    dentro = b.clone()
    dentro[:, 2:4] *= 0.5
    assert torch.allclose(rotated_iou(b, dentro), torch.full((20,), 0.25), atol=1e-5)


def test_el_iou_es_diferenciable_con_gradiente_finito():
    p = _random_boxes(50, 4).requires_grad_(True)
    (1 - rotated_iou(p, _random_boxes(50, 5))).sum().backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_la_version_por_pares_coincide_con_la_directa():
    a, b = _random_boxes(6, 6), _random_boxes(3, 7)
    pw = pairwise_rotated_iou(a, b)
    for i in range(6):
        for j in range(3):
            assert pw[i, j].item() == pytest.approx(rotated_iou(a[i : i + 1], b[j : j + 1]).item(), abs=1e-6)


# --- su convencion de angulo ---------------------------------------------------


def test_regularizar_el_angulo_deja_el_mismo_rectangulo_en_su_rango():
    boxes = _random_boxes(200, 8)
    reg = regularize_angle_ddgrcf(boxes)
    assert (reg[:, 4] > -math.pi / 4 - 1e-6).all() and (reg[:, 4] <= math.pi / 4 + 1e-6).all()
    assert torch.allclose(rotated_iou(boxes, reg), torch.ones(200), atol=1e-4), "es el mismo rectangulo"


# --- COCO de Megvii en la cabeza propia ----------------------------------------


def test_el_mapeo_de_megvii_cubre_backbone_y_cuello_y_descarta_su_cabeza():
    """Se simula su checkpoint renombrando el nuestro al reves: la ida y vuelta
    tiene que cubrir los 354 tensores de backbone+cuello y ninguno de la cabeza."""
    model = YoloxObb("small")
    from testbank.models.pretrained import _MEGVII_NECK

    inverse = {v: k for k, v in _MEGVII_NECK.items()}
    fake = {}
    for key, value in model.state_dict().items():
        if key.startswith("backbone."):
            fake["backbone.backbone." + key[len("backbone.") :]] = value * 0 + 7.0
        elif key.startswith("neck."):
            module, _, tail = key[len("neck.") :].partition(".")
            fake[f"backbone.{inverse[module]}.{tail}"] = value * 0 + 7.0
    fake["head.cls_preds.0.weight"] = torch.zeros(80, 128, 1, 1)  # su cabeza COCO
    report = load_megvii_yolox(model, _saved(fake))
    assert report["loaded"] == 354 and report["skipped_head"] == 1
    assert (model.state_dict()["neck.p4.conv1.conv.weight"] == 7.0).all()


def test_cargar_la_variante_equivocada_es_error_y_no_medias_tintas():
    small = YoloxObb("small").state_dict()
    fake = {"backbone.backbone." + k[len("backbone.") :]: v for k, v in small.items() if k.startswith("backbone.")}
    with pytest.raises(RuntimeError, match="no encaja"):
        load_megvii_yolox(YoloxObb("nano"), _saved(fake))


def test_una_clave_de_cuello_desconocida_no_se_traga():
    with pytest.raises(KeyError, match="cuello desconocido"):
        remap_megvii_key("backbone.modulo_inventado.conv.weight")
    assert remap_megvii_key("head.obj_preds.0.bias") is None
