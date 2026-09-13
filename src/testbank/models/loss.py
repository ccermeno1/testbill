"""Perdidas del candidato propio.

Se descompone en tres, en vez de optimizar el IoU rotado directamente:

    caja      IoU alineada sobre la envolvente
    angulo    coseno sobre (sin 2t, cos 2t), ATENUADO por el ratio
    presencia BCE para objectness y clase

Por que descomponer
-------------------
Optimizar el IoU rotado de verdad exigiria construir e intersectar poligonos en
cada iteracion, y eso no cabe en el bucle (ver `assign.py`). La descomposicion
es una aproximacion: no optimiza exactamente lo que luego se mide.

Se asume a conciencia, y es medible. La metrica de evaluacion SIGUE siendo el
IoU rotado por shapely, asi que si la descomposicion asigna mal el compromiso
entre forma y giro, el numero final lo dice. Entrenar con una aproximacion y
medir con lo bueno es el orden correcto; al reves seria enganarse.

El atenuador de angulo
----------------------
Es lo unico no estandar. Cuando la caja verdadera es casi cuadrada su angulo no
esta definido -- `(w, h, t)` y `(h, w, t+90)` son el mismo rectangulo -- asi que
la perdida de angulo se rebaja. Parametrizable y apagable para poder medir
cuanto aporta. Ver `angle_weight.py` y el README.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from testbank.models.angle_weight import angle_weight, side_ratio_px
from testbank.models.assign import Assignment, enclosing_boxes, pairwise_iou


@dataclass(frozen=True, slots=True)
class LossTerms:
    """Cada termino por separado. Juntarlos en un escalar oculta cual falla."""

    box: torch.Tensor
    angle: torch.Tensor
    objectness: torch.Tensor
    classes: torch.Tensor
    total: torch.Tensor
    num_positives: int
    #: Solo la receta de Ultralytics. Cero en las demas, para que el registro
    #: de cada ejecucion tenga siempre las mismas columnas.
    dfl: torch.Tensor | None = None
    #: Solo la receta del fork, y solo en sus ultimas epocas.
    l1: torch.Tensor | None = None

    def to_dict(self) -> dict:
        """Solo para registrar. `detach` a proposito: convertir a float un
        tensor todavia enganchado al grafo avisa, y arrastrar el grafo a un
        diccionario de informes es como se filtran las fugas de memoria."""
        def value(t):
            return 0.0 if t is None else t.detach().item()

        return {
            "box": value(self.box),
            "angle": value(self.angle),
            "objectness": value(self.objectness),
            "classes": value(self.classes),
            "dfl": value(self.dfl),
            "l1": value(self.l1),
            "total": value(self.total),
            "num_positives": self.num_positives,
        }


def iou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """`1 - IoU` sobre las envolventes alineadas.

    IoU y no L1 sobre las coordenadas: L1 trata igual un error de 5 pixeles en
    un billete pequeno y en uno grande, cuando en el primero es fatal y en el
    segundo irrelevante. El IoU es relativo al tamano por construccion.
    """
    if predicted.numel() == 0:
        return predicted.new_zeros(())
    boxes_p = enclosing_boxes(predicted)
    boxes_t = enclosing_boxes(target)
    iou = pairwise_iou(boxes_p, boxes_t).diagonal()
    return (1.0 - iou).mean()


def angle_loss(
    predicted_angle: torch.Tensor,
    target_theta: torch.Tensor,
    target_wh: torch.Tensor,
    *,
    enabled: bool = True,
    ratio_threshold: float = 1.1,
    min_weight: float = 0.0,
    decay: str = "smoothstep",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve `(perdida, pesos)`. Los pesos salen para poder registrarlos.

    `predicted_angle` es (P, 2) con `(sin 2t, cos 2t)` SIN normalizar: la red
    saca dos numeros libres. Se normalizan aqui para que la perdida mida solo
    direccion; la magnitud no significa nada y dejarla suelta le daria al
    modelo una via de bajar la perdida sin acertar el angulo.
    """
    if predicted_angle.numel() == 0:
        zero = predicted_angle.new_zeros(())
        return zero, predicted_angle.new_zeros((0,))

    predicted = F.normalize(predicted_angle, dim=-1, eps=1e-6)
    target = torch.stack(
        (torch.sin(2 * target_theta), torch.cos(2 * target_theta)), dim=-1
    )
    # 1 - coseno: cero cuando apuntan igual, 2 en el peor caso. Continuo en todo
    # el circulo, que es el motivo de haber doblado el angulo.
    per_box = 1.0 - (predicted * target).sum(dim=-1)

    weights = angle_weight(
        side_ratio_px(target_wh[:, 0], target_wh[:, 1]),
        enabled=enabled,
        ratio_threshold=ratio_threshold,
        min_weight=min_weight,
        decay=decay,
    )
    # Se divide por el NUMERO de cajas, no por la suma de pesos.
    #
    # Dividir por la suma deshace la atenuacion: con una sola caja de peso 0.1,
    # `(1 * 0.1) / 0.1` vuelve a dar 1.0, y con un lote entero de cajas ambiguas
    # el gradiente sale a plena potencia -- justo lo que el atenuador existe
    # para evitar. Normalizando por N, atenuar reduce de verdad la magnitud.
    #
    # Que eso baje la perdida total no falsea nada: los candidatos se comparan
    # por las METRICAS de evaluacion, no por el valor de una perdida, y comparar
    # perdidas entre funciones de perdida distintas no significa nada de todos
    # modos.
    return (per_box * weights).sum() / per_box.numel(), weights


def compute_losses(
    predicted_boxes: torch.Tensor,
    predicted_angle: torch.Tensor,
    predicted_objectness: torch.Tensor,
    predicted_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    assignment: Assignment,
    *,
    angle_config=None,
    box_gain: float = 5.0,
    angle_gain: float = 1.0,
    obj_gain: float = 1.0,
    cls_gain: float = 1.0,
) -> LossTerms:
    """Junta los tres terminos. Todo en pixeles."""
    positive = assignment.positive
    matched = assignment.matched[positive]
    device = predicted_objectness.device

    # Objectness: TODAS las celdas participan. Es la unica senal que ensena que
    # el fondo es fondo, y con ~3500 celdas y 2 billetes es casi toda la senal.
    obj_target = positive.to(predicted_objectness.dtype)
    objectness = F.binary_cross_entropy_with_logits(
        predicted_objectness, obj_target, reduction="mean"
    )

    if not positive.any():
        zero = torch.zeros((), device=device)
        return LossTerms(zero, zero, objectness * obj_gain, zero, objectness * obj_gain, 0)

    boxes_p = predicted_boxes[positive]
    boxes_t = target_boxes[matched]

    box = iou_loss(boxes_p, boxes_t)

    options = {} if angle_config is None else {
        "enabled": angle_config.enabled,
        "ratio_threshold": angle_config.ratio_threshold,
        "min_weight": angle_config.min_weight,
        "decay": angle_config.decay,
    }
    angle, _ = angle_loss(
        predicted_angle[positive], boxes_t[:, 4], boxes_t[:, 2:4], **options
    )

    classes = F.binary_cross_entropy_with_logits(
        predicted_classes[positive],
        F.one_hot(
            target_classes[matched], predicted_classes.shape[-1]
        ).to(predicted_classes.dtype),
        reduction="mean",
    )

    total = (
        box_gain * box + angle_gain * angle + obj_gain * objectness + cls_gain * classes
    )
    return LossTerms(
        box=box,
        angle=angle,
        objectness=objectness,
        classes=classes,
        total=total,
        num_positives=assignment.num_positives,
    )


__all__ = ["LossTerms", "angle_loss", "compute_losses", "iou_loss"]
