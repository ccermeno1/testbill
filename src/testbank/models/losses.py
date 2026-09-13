"""Las cuatro recetas de perdida, cada una entera, y lo que comparten.

Las tres recetas de perdida de la cabeza propia, cada una ENTERA.

Una receta no es una perdida de caja: es la combinacion de asignador, objetivos,
terminos, normalizacion y ganancias que una red concreta usa. Mezclar la caja de
una con el asignador de otra daria algo que no es ninguna de las dos, y la tabla
lo llamaria por un nombre que no le corresponde. Asi que aqui cada receta es una
funcion cerrada, y `losses_for_image` solo elige cual.

    own              `losses.py`: IoU alineada + angulo atenuado + obj + cls; SimOTA
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

=== La receta propia, en detalle ===
Perdidas del candidato propio.

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
cuanto aporta. Ver `losses.py` y el README.

=== El atenuador de angulo de la receta propia ===
Peso de la perdida de angulo segun lo cuadrada que sea la caja.

Por que hace falta
------------------
La representacion `(cx, cy, w, h, theta)` es ambigua cuando `w ~ h`: la caja
`(w, h, t)` y la caja `(h, w, t+90)` son el MISMO rectangulo. La codificacion
`(sin 2t, cos 2t)` resuelve que `t` y `t+180` sean el mismo, pero NO esto: manda
las dos versiones a puntos opuestos del circulo, asi que el modelo recibiria dos
objetivos contradictorios para la misma caja.

Medido sobre el export: 145 de 762 anotaciones (19%) tienen ratio < 1.1.

Por que atenuar y no arreglarlo
-------------------------------
Porque en el caso ambiguo el angulo DA IGUAL para lo que nos importa. Un
rectangulo casi cuadrado recortado con 5 grados de error tapa practicamente lo
mismo, y la politica de anotacion ya dice que un rectangulo aproximado basta.
Castigar al modelo por no acertar algo que ni esta bien definido ni cambia el
resultado es gastar capacidad en ruido.

La alternativa seria la representacion gaussiana, que absorbe la ambiguedad de
forma natural. Descartada a proposito: se aleja del IoU rotado por shapely con
el que medimos, y esa trazabilidad pesa mas que la elegancia de la formulacion.

Todo parametrizable
-------------------
Umbral, forma y suelo van en `detector.loss.angle_weight`, con `enabled` para
apagarlo. La pregunta "cuanto aporta esto" se responde entrenando con y sin, no
razonando. Ver `AngleWeightConfig`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from testbank.config import Config
from testbank.models.assign import (
    AnchorGrid,
    Assignment,
    build_anchor_grid,
    enclosing_boxes,
    pairwise_iou,
    simota_assign,
    tal_assign,
)
from testbank.models.decode import decode_outputs
from testbank.models.overlap import (
    kld_loss,
    pairwise_kld_loss,
    pairwise_probiou,
    pairwise_rotated_iou,
    probiou,
    rotated_iou,
)
from testbank.models.yolox_obb import STRIDES, HeadSpec, YoloxObb

# --------------------------------------------------------------------------
# Atenuador de angulo (receta propia)
# --------------------------------------------------------------------------

#: Formas de subida entre el cuadrado perfecto y el umbral. La clave es lo que
#: acepta `decay` en la config; anadir una es anadir una entrada aqui.
DECAYS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # Sube recto. La mas simple de interpretar: el peso es la fraccion del
    # camino recorrido hacia el umbral.
    "linear": lambda t: t,
    # Arranca despacio y frena al final. Deja casi sin peso la franja mas
    # ambigua en vez de subir desde el primer momento.
    "smoothstep": lambda t: t * t * (3.0 - 2.0 * t),
    # Intermedia: arranca despacio pero no frena.
    "quadratic": lambda t: t * t,
    # Escalon. Sirve de referencia para medir si la transicion suave aporta
    # algo frente a cortar por lo sano.
    "step": lambda t: (t >= 1.0).to(t.dtype),
}


def angle_weight(
    ratio: torch.Tensor,
    *,
    enabled: bool = True,
    ratio_threshold: float = 1.1,
    min_weight: float = 0.0,
    decay: str = "smoothstep",
) -> torch.Tensor:
    """Peso en `[min_weight, 1]` para la perdida de angulo de cada caja.

    `ratio` es lado largo / lado corto de la caja VERDADERA, en pixeles. En
    pixeles y no normalizado: normalizar escala x e y por factores distintos, y
    el ratio dejaria de ser el geometrico -- el mismo error que ya mordio dos
    veces en este proyecto.

    Con `enabled=False` devuelve unos: es la rama de control del experimento, y
    tiene que costar exactamente lo mismo escribirla que la otra.
    """
    if not enabled:
        return torch.ones_like(ratio)
    if ratio_threshold <= 1.0:
        raise ValueError(
            f"ratio_threshold tiene que ser > 1, se recibio {ratio_threshold}"
        )
    try:
        shape = DECAYS[decay]
    except KeyError:
        raise ValueError(
            f"forma de decaimiento desconocida {decay!r}; hay {sorted(DECAYS)}"
        ) from None

    # Un ratio por debajo de 1 no existe: es el lado largo entre el corto. Si
    # llega, es que alguien los ha intercambiado, y truncar en 1 evita pesos
    # negativos sin ocultar el problema (el peso saldria minimo, no absurdo).
    progress = ((ratio.clamp(min=1.0) - 1.0) / (ratio_threshold - 1.0)).clamp(0.0, 1.0)
    return min_weight + (1.0 - min_weight) * shape(progress)


def side_ratio_px(width: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """Lado largo / lado corto, sin asumir cual de los dos es cual.

    El orden de `w` y `h` es justo lo que la ambiguedad vuelve arbitrario, asi
    que el ratio no puede depender de el.
    """
    long_side = torch.maximum(width, height)
    short_side = torch.minimum(width, height)
    return long_side / short_side.clamp(min=1e-6)


# --------------------------------------------------------------------------
# Receta propia: terminos
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Las cuatro recetas y el despacho
# --------------------------------------------------------------------------

RECIPES = ("own", "yolox_obb_fork", "ultralytics_obb", "ddgrcf")


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
    if recipe == "ddgrcf":
        from testbank.models.ddgrcf import DdgrcfYoloxObb

        return DdgrcfYoloxObb.HEAD_SPEC
    return HeadSpec()


def build_model(config: Config):
    """El modelo que exige la receta. `ddgrcf` es una RED distinta, no solo una
    cabeza: el port de su yaml, para que sus pesos de DOTA carguen."""
    if config.detector.loss.recipe == "ddgrcf":
        from testbank.models.ddgrcf import DdgrcfYoloxObb

        return DdgrcfYoloxObb(num_classes=1)
    return YoloxObb(config.detector.variant, num_classes=1, head=head_spec_for(config))


def architecture_of(model) -> str:
    """Lo que va al checkpoint para reconstruir la red al cargar."""
    return "ddgrcf" if model.__class__.__name__ == "DdgrcfYoloxObb" else "yolox_obb"


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


# --- receta de DDGRCF ------------------------------------------------------


def regularize_angle_ddgrcf(boxes: torch.Tensor) -> torch.Tensor:
    """Su `mintheta_obb`: el angulo en `(-pi/4, pi/4]`, intercambiando w y h.

    `(w, h, t)` y `(h, w, t + pi/2)` son el mismo rectangulo; su convenio elige
    la representacion de |angulo| menor. Su red predice el angulo crudo, asi que
    el objetivo tiene que estar en ese rango o no seria alcanzable con gradiente
    pequeno.
    """

    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    theta1 = (theta + math.pi / 2) % math.pi - math.pi / 2
    theta2 = (theta + math.pi) % math.pi - math.pi / 2
    swap = theta1.abs() >= theta2.abs()
    return torch.stack(
        (
            cx,
            cy,
            torch.where(swap, h, w),
            torch.where(swap, w, h),
            torch.where(swap, theta2, theta1),
        ),
        dim=-1,
    )


def _ddgrcf(outputs, boxes, classes, config: Config, device, *, epoch, total_epochs) -> LossTerms:
    """`get_losses` de DDGRCF/YOLOX_OBB (Apache-2.0), sobre el port de su red.

    Literal de su `detectx.py` + `obbdetectx.py` + yaml de perdidas: SimOTA con
    coste `-log(IoU)` x3 e IoU EXACTO; caja `1 - IoU` x5; obj y cls BCE con
    objetivo `one_hot * IoU`; todo `sum / num_fg`; L1 extra sobre la regresion
    cruda en las ultimas epocas.

    Una desviacion, dicha: su L1 extra deja el objetivo del ANGULO a cero (su
    `get_reg_l1_target` rellena 4 de 5 componentes), lo que empuja el angulo
    crudo hacia 0 en las ultimas epocas. Aqui el objetivo es el angulo real. Es
    casi seguro un descuido suyo, no una decision, y replicarlo seria copiar el
    fallo con el nombre de fidelidad.
    """
    options = config.detector.loss.ddgrcf
    boxes = regularize_angle_ddgrcf(boxes)
    predicted_boxes, scores = decode_outputs(outputs, _image_size(config))
    angle, cls, obj, raw, _ = _flatten(outputs)
    grid = _grid(outputs, device)

    assignment = simota_assign(
        predicted_boxes.detach(),
        scores.detach(),
        boxes,
        grid,
        iou_weight=3.0,
        overlap=pairwise_rotated_iou,
        overlap_cost="neg_log",
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

    box = (1.0 - rotated_iou(boxes_p, boxes_t)).sum() / num_fg
    cls_target = (
        F.one_hot(classes[matched], cls.shape[-1]).to(cls.dtype)
        * assignment.matched_iou[positive, None]
    )
    class_loss = F.binary_cross_entropy_with_logits(
        cls[positive], cls_target, reduction="sum"
    ) / num_fg

    l1 = zero
    if options.l1_last_epochs > 0 and epoch >= total_epochs - options.l1_last_epochs:
        # Su `get_reg_l1_target`: (cx/stride - i, cy/stride - j, log w/stride,
        # log h/stride), con (i, j) la esquina de la celda. Nuestra rejilla
        # guarda el CENTRO de la celda, de ahi el medio stride.
        stride = grid.strides[positive]
        corner = (grid.centers[positive] - stride[:, None] / 2) / stride[:, None]
        target = torch.stack(
            (
                boxes_t[:, 0] / stride - corner[:, 0],
                boxes_t[:, 1] / stride - corner[:, 1],
                torch.log(boxes_t[:, 2] / stride + 1e-8),
                torch.log(boxes_t[:, 3] / stride + 1e-8),
                boxes_t[:, 4],
            ),
            dim=-1,
        )
        raw_all = torch.cat((raw[positive], angle[positive]), dim=-1)
        l1 = F.l1_loss(raw_all, target, reduction="sum") / num_fg

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
    if recipe == "ddgrcf":
        return _ddgrcf(
            outputs, boxes, classes, config, device, epoch=epoch, total_epochs=total_epochs
        )
    raise ValueError(f"receta desconocida: {recipe!r}; hay {RECIPES}")


def _image_size(config: Config):
    from testbank.dataio.formats import ImageSize

    side = config.detector.image_size
    return ImageSize(side, side)


__all__ = ['DECAYS', 'RECIPES', 'LossTerms', 'angle_loss', 'angle_weight', 'architecture_of', 'build_model', 'compute_losses', 'distribution_focal_loss', 'head_spec_for', 'iou_loss', 'losses_for_image', 'regularize_angle_ddgrcf', 'side_ratio_px', 'target_distances']
