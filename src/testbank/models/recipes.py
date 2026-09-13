"""Las tres recetas de perdida de la cabeza propia, cada una ENTERA.

Una receta no es una perdida de caja: es la combinacion de asignador, objetivos,
terminos, normalizacion y ganancias que una red concreta usa. Mezclar la caja de
una con el asignador de otra daria algo que no es ninguna de las dos, y la tabla
lo llamaria por un nombre que no le corresponde. Asi que aqui cada receta es una
funcion cerrada, y `losses_for_image` solo elige cual.

    own              `loss.py`: IoU alineada + angulo atenuado + obj + cls; SimOTA
    yolox_obb_fork   KLD x5 + obj + cls por solape + L1 tardia; SimOTA con KLD
    ultralytics_obb  ProbIoU x7.5 + DFL x1.5 + cls x0.5 con objetivo suave; TAL

Que es fiel y que no, dicho por adelantado
------------------------------------------
**Fork.** Su codigo es Apache-2.0 y se ha leido entero. Se reproduce su
`get_losses`: mismos terminos, mismas ganancias, misma normalizacion por numero
de positivos, mismo objetivo de clase (one-hot por solape), mismo SimOTA con la
KLD como coste. Dos desviaciones, las dos de cabeza y no de perdida: ellos
regresan el angulo en grados directamente y aqui se usa `(sin 2t, cos 2t)`; y su
L1 tardia va sobre `(dx, dy, log w, log h)` mientras aqui va sobre las cuatro
distancias, que es la regresion cruda de ESTA cabeza. La normalizacion es por
imagen y luego media del lote, en vez de por lote entero: con uno o dos
billetes por foto la diferencia es de ponderacion entre imagenes, no de forma.

**Ultralytics.** Su codigo es AGPL y NO se ha leido. La composicion -- que
terminos, que ganancias, que asignador -- sale de su documentacion publica; las
formulas salen de los papers (ProbIoU: Llerena 2021; DFL: Li 2020; TAL: Feng
2021). Es una reproduccion de la RECETA descrita, no una copia. Si en su codigo
hubiera un detalle no documentado, aqui no esta.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from testbank.config import Config
from testbank.models.assign import (
    AnchorGrid,
    build_anchor_grid,
    simota_assign,
    tal_assign,
)
from testbank.models.decode import decode_outputs
from testbank.models.gaussian import (
    kld_loss,
    pairwise_kld_loss,
    pairwise_probiou,
    probiou,
)
from testbank.models.loss import LossTerms, compute_losses
from testbank.models.yolox_obb import STRIDES, HeadSpec

RECIPES = ("own", "yolox_obb_fork", "ultralytics_obb")


def head_spec_for(config: Config) -> HeadSpec:
    """La cabeza que exige cada receta. No es elegible por separado."""
    recipe = config.detector.loss.recipe
    if recipe == "ultralytics_obb":
        return HeadSpec(
            regression="dfl",
            reg_max=config.detector.loss.ultralytics.reg_max,
            angle="scalar",
            objectness=False,
        )
    return HeadSpec()


# --- utilidades comunes ----------------------------------------------------


def _flatten(outputs):
    """Los tres niveles en una lista de celdas. Todo (N, ·)."""
    angle = torch.cat([o.angle[0].permute(1, 2, 0).reshape(-1, o.angle.shape[1]) for o in outputs])
    classes = torch.cat(
        [o.classes[0].permute(1, 2, 0).reshape(-1, o.classes.shape[1]) for o in outputs]
    )
    objectness = (
        torch.cat([o.objectness[0].reshape(-1) for o in outputs])
        if outputs[0].objectness is not None
        else None
    )
    distances = torch.cat([o.distances[0].permute(1, 2, 0).reshape(-1, 4) for o in outputs])
    distribution = (
        torch.cat(
            [
                o.distribution[0].permute(1, 2, 0).reshape(-1, o.distribution.shape[1])
                for o in outputs
            ]
        )
        if outputs[0].distribution is not None
        else None
    )
    return angle, classes, objectness, distances, distribution


def _grid(outputs, device) -> AnchorGrid:
    return build_anchor_grid(
        [tuple(o.distances.shape[-2:]) for o in outputs], STRIDES, device=device
    )


def target_distances(
    centers: torch.Tensor, strides: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    """`(l, t, r, b)` en unidades de stride desde cada celda a SU caja girada.

    Es la inversa exacta de `decode_level`: el desplazamiento del centro de la
    caja respecto a la celda se lleva al marco de la caja, y ahi

        l = w/2 - u    r = w/2 + u    t = h/2 - v    b = h/2 + v

    Sirve de objetivo tanto para la L1 tardia del fork como para la DFL de
    Ultralytics. Puede salir negativo si la celda esta fuera de la caja; los
    asignadores no eligen esas celdas, pero se clampa igual por si acaso.
    """
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    dx, dy = cx - centers[:, 0], cy - centers[:, 1]
    cos, sin = torch.cos(theta), torch.sin(theta)
    u = dx * cos + dy * sin
    v = -dx * sin + dy * cos
    ltrb = torch.stack((w / 2 - u, h / 2 - v, w / 2 + u, h / 2 + v), dim=-1)
    return (ltrb / strides[:, None]).clamp(min=0.0)


# --- receta propia ---------------------------------------------------------


def _own(outputs, boxes, classes, config: Config, device) -> LossTerms:
    predicted_boxes, scores = decode_outputs(outputs, _image_size(config))
    angle, cls, obj, _, _ = _flatten(outputs)
    grid = _grid(outputs, device)
    assignment = simota_assign(predicted_boxes.detach(), scores.detach(), boxes, grid)
    return compute_losses(
        predicted_boxes,
        angle,
        obj,
        cls,
        boxes,
        classes,
        assignment,
        angle_config=config.detector.loss.angle_weight,
    )


# --- receta del fork -------------------------------------------------------


def _fork(outputs, boxes, classes, config: Config, device, *, epoch, total_epochs) -> LossTerms:
    """`get_losses` de buzhidaoshenme/YOLOX-OBB (Apache-2.0), sobre nuestra cabeza.

    Del fork, literal: la KLD como coste del SimOTA (x3.0) y como perdida de
    caja (x5.0); `cls_target = one_hot * solape`, con solape `= 1 - kld_loss`;
    todo `sum / num_fg`; y la L1 en las ultimas epocas.
    """
    options = config.detector.loss.fork
    predicted_boxes, scores = decode_outputs(outputs, _image_size(config))
    _, cls, obj, distances, _ = _flatten(outputs)
    grid = _grid(outputs, device)

    def overlap(p, t):
        return 1.0 - pairwise_kld_loss(p, t)

    assignment = simota_assign(
        predicted_boxes.detach(),
        scores.detach(),
        boxes,
        grid,
        iou_weight=3.0,
        overlap=overlap,
        overlap_cost="one_minus",
    )
    positive = assignment.positive
    num_fg = max(assignment.num_positives, 1)
    zero = torch.zeros((), device=device)

    objectness = F.binary_cross_entropy_with_logits(
        obj, positive.to(obj.dtype), reduction="sum"
    ) / num_fg
    if not positive.any():
        return LossTerms(zero, zero, objectness, zero, objectness, 0, dfl=zero, l1=zero)

    matched = assignment.matched[positive]
    boxes_p, boxes_t = predicted_boxes[positive], boxes[matched]

    box = kld_loss(boxes_p, boxes_t, tau=options.tau).sum() / num_fg
    cls_target = (
        F.one_hot(classes[matched], cls.shape[-1]).to(cls.dtype)
        * assignment.matched_iou[positive, None]
    )
    class_loss = F.binary_cross_entropy_with_logits(
        cls[positive], cls_target, reduction="sum"
    ) / num_fg

    l1 = zero
    if options.l1_last_epochs > 0 and epoch >= total_epochs - options.l1_last_epochs:
        target = target_distances(grid.centers[positive], grid.strides[positive], boxes_t)
        l1 = F.l1_loss(distances[positive], target, reduction="sum") / num_fg

    total = options.box_gain * box + objectness + class_loss + l1
    return LossTerms(
        box=box,
        angle=zero,
        objectness=objectness,
        classes=class_loss,
        total=total,
        num_positives=assignment.num_positives,
        dfl=zero,
        l1=l1,
    )


# --- receta de Ultralytics -------------------------------------------------


def distribution_focal_loss(
    logits: torch.Tensor, target: torch.Tensor, reg_max: int
) -> torch.Tensor:
    """DFL (Li et al., 2020). `logits` (P, 4*reg_max), `target` (P, 4). -> (P,).

    Cada distancia objetivo `y` cae entre dos bins enteros `yl <= y < yr`; la
    perdida es la entropia cruzada contra los dos, ponderada por cercania:

        DFL = -(yr - y) log p(yl) - (y - yl) log p(yr)

    Asi la distribucion aprende a concentrar masa alrededor del valor real en
    vez de solo acertar su esperanza. Media sobre las cuatro distancias.
    """
    positives = logits.shape[0]
    logits = logits.view(positives, 4, reg_max)
    target = target.clamp(min=0.0, max=reg_max - 1 - 0.01)
    left = target.floor().long()
    right = left + 1
    weight_left = right.to(target.dtype) - target
    weight_right = 1.0 - weight_left
    log_probabilities = logits.log_softmax(dim=-1)
    loss = -(
        log_probabilities.gather(-1, left[..., None]).squeeze(-1) * weight_left
        + log_probabilities.gather(-1, right[..., None]).squeeze(-1) * weight_right
    )
    return loss.mean(dim=-1)


def _ultralytics(outputs, boxes, classes, config: Config, device) -> LossTerms:
    """La receta v8-OBB, reconstruida desde su documentacion y los papers.

    Sin objectness: la BCE de clase se calcula sobre TODAS las celdas contra el
    objetivo suave del TAL, que vale cero en el fondo y la calidad de
    localizacion normalizada en los positivos. Caja y DFL se ponderan por ese
    mismo objetivo. Todo se divide por su suma, que es la normalizacion que
    Ultralytics documenta (`target_scores_sum`).
    """
    options = config.detector.loss.ultralytics
    predicted_boxes, _ = decode_outputs(outputs, _image_size(config))
    _, cls, _, _, distribution = _flatten(outputs)
    if distribution is None:
        raise ValueError(
            "la receta ultralytics_obb exige la cabeza DFL; construye el modelo "
            "con head_spec_for(config)"
        )
    grid = _grid(outputs, device)
    assignment = tal_assign(
        predicted_boxes.detach(),
        torch.sigmoid(cls).detach(),
        boxes,
        classes,
        grid,
        overlap=pairwise_probiou,
        topk=options.tal_topk,
        alpha=options.tal_alpha,
        beta=options.tal_beta,
    )
    target_scores = assignment.target_scores
    assert target_scores is not None
    scores_sum = target_scores.sum().clamp(min=1.0)
    zero = torch.zeros((), device=device)

    class_loss = F.binary_cross_entropy_with_logits(
        cls, target_scores, reduction="sum"
    ) / scores_sum

    positive = assignment.positive
    if not positive.any():
        total = options.cls_gain * class_loss
        return LossTerms(zero, zero, zero, class_loss, total, 0, dfl=zero, l1=zero)

    matched = assignment.matched[positive]
    boxes_p, boxes_t = predicted_boxes[positive], boxes[matched]
    weight = target_scores.sum(dim=-1)[positive]

    box = ((1.0 - probiou(boxes_p, boxes_t)) * weight).sum() / scores_sum
    target = target_distances(grid.centers[positive], grid.strides[positive], boxes_t)
    dfl = (
        distribution_focal_loss(distribution[positive], target, options.reg_max) * weight
    ).sum() / scores_sum

    total = options.box_gain * box + options.cls_gain * class_loss + options.dfl_gain * dfl
    return LossTerms(
        box=box,
        angle=zero,
        objectness=zero,
        classes=class_loss,
        total=total,
        num_positives=assignment.num_positives,
        dfl=dfl,
        l1=zero,
    )


# --- despacho --------------------------------------------------------------


def losses_for_image(
    outputs, boxes, classes, config: Config, device, *, epoch: int = 0, total_epochs: int = 1
) -> LossTerms:
    recipe = config.detector.loss.recipe
    if recipe == "own":
        return _own(outputs, boxes, classes, config, device)
    if recipe == "yolox_obb_fork":
        return _fork(
            outputs, boxes, classes, config, device, epoch=epoch, total_epochs=total_epochs
        )
    if recipe == "ultralytics_obb":
        return _ultralytics(outputs, boxes, classes, config, device)
    raise ValueError(f"receta desconocida: {recipe!r}; hay {RECIPES}")


def _image_size(config: Config):
    from testbank.dataio.formats import ImageSize

    side = config.detector.image_size
    return ImageSize(side, side)


__all__ = [
    "RECIPES",
    "distribution_focal_loss",
    "head_spec_for",
    "losses_for_image",
    "target_distances",
]

