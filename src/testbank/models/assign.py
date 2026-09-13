"""SimOTA: que celda se hace responsable de que billete.

Un detector sin anclas tiene ~3500 celdas para 416x416 y, en nuestras imagenes,
uno o dos billetes. Casi todo es fondo. Decidir MAL que celdas son positivas es
lo que mas cuesta en un detector de una etapa: elegir pocas y no aprende,
elegir muchas y el fondo domina.

SimOTA lo resuelve en dos pasos: filtrar candidatos geometricamente, y entre
esos elegir por coste, con un numero de positivos que se ADAPTA a cada billete
en vez de ser fijo. Un billete bien definido se lleva mas celdas que uno dudoso.

Por que aqui no se usa el IoU rotado de shapely
-----------------------------------------------
Porque no cabe en el bucle. Nuestro IoU rotado construye poligonos y los
intersecta; para evaluar 100 imagenes una vez es perfecto, pero aqui haria falta
`candidatos x billetes` intersecciones por imagen y por iteracion. Medido en el
proyecto: shapely tarda milisegundos por par.

Asi que el coste de asignacion usa una APROXIMACION en torch: IoU de las cajas
alineadas envolventes mas la distancia entre centros. Es un proxy, y se declara
como tal. Dos cosas lo hacen aceptable:

1. El filtro de candidatos SI es exacto: mira si el centro de la celda cae
   dentro del rectangulo GIRADO, y eso es barato en torch.
2. La metrica de verdad sigue siendo el IoU rotado por shapely en evaluacion.
   Si el proxy asigna mal, el numero final lo delata. Entrenamos con proxy y
   medimos con lo bueno, que es el orden correcto de las dos cosas.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

#: Radio, en celdas, de la region central que tambien se acepta como candidata.
#: Un billete muy fino puede no contener el centro de NINGUNA celda en los
#: niveles gruesos; sin esta ventana se quedaria sin positivos y sin aprender.
CENTER_RADIUS = 2.5

#: Cuantos de los mejores IoU se suman para decidir cuantos positivos lleva cada
#: billete. Es el `dynamic k` de SimOTA.
TOP_CANDIDATES = 10

#: Coste que se suma a lo que no es candidato geometrico. Grande, pero finito:
#: con infinito la seleccion por `topk` propaga NaN si un billete se queda sin
#: candidatos, y preferimos un positivo malo a un gradiente envenenado.
BLOCKED_COST = 1e5


@dataclass(frozen=True, slots=True)
class AnchorGrid:
    """Centros de celda de todos los niveles, ya en pixeles de la imagen."""

    #: (N, 2) -- centro de cada celda.
    centers: torch.Tensor
    #: (N,) -- reduccion espacial del nivel al que pertenece cada celda.
    strides: torch.Tensor

    def __len__(self) -> int:
        return self.centers.shape[0]


def build_anchor_grid(
    sizes: list[tuple[int, int]],
    strides: tuple[int, ...],
    device: torch.device | None = None,
) -> AnchorGrid:
    """Un punto por celda, en el CENTRO de la celda, no en su esquina.

    El medio pixel importa: sin el, todas las cajas salen sesgadas media celda
    hacia arriba y hacia la izquierda, que a stride 32 son 16 pixeles.
    """
    centers, all_strides = [], []
    for (height, width), stride in zip(sizes, strides):
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        grid = torch.stack(((xs + 0.5) * stride, (ys + 0.5) * stride), dim=-1)
        centers.append(grid.reshape(-1, 2))
        all_strides.append(
            torch.full((height * width,), float(stride), device=device)
        )
    return AnchorGrid(torch.cat(centers), torch.cat(all_strides))


def points_in_rotated_boxes(
    points: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    """(N, M) -- si el punto n cae dentro del rectangulo girado m.

    Exacto y barato: se lleva cada punto al sistema de referencia de la caja
    girando -theta, y ahi la pregunta es una comparacion con w/2 y h/2. No hace
    falta construir ningun poligono.

    `boxes` es (M, 5) con `cx, cy, w, h, theta`.
    """
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    offset = points[:, None, :] - torch.stack((cx, cy), dim=-1)[None, :, :]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    # Rotacion inversa: girar el punto -theta alrededor del centro de la caja.
    local_x = offset[..., 0] * cos_t + offset[..., 1] * sin_t
    local_y = -offset[..., 0] * sin_t + offset[..., 1] * cos_t
    return (local_x.abs() <= w / 2) & (local_y.abs() <= h / 2)


def points_near_centers(
    points: torch.Tensor, boxes: torch.Tensor, strides: torch.Tensor
) -> torch.Tensor:
    """Ventana cuadrada alrededor del centro, medida en celdas del nivel.

    En celdas y no en pixeles a proposito: un nivel grueso necesita una ventana
    fisicamente mayor para tener el mismo numero de candidatos.
    """
    centers = boxes[:, :2]
    radius = (CENTER_RADIUS * strides)[:, None]
    delta = (points[:, None, :] - centers[None, :, :]).abs()
    return (delta[..., 0] <= radius) & (delta[..., 1] <= radius)


def enclosing_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """(M, 4) -- envolvente alineada al eje `x1, y1, x2, y2` de cada caja girada.

    Es la aproximacion que sostiene el coste. Para una caja alineada es exacta;
    para una girada 45 grados es generosa. El sesgo esta acotado y va siempre en
    la misma direccion, que es lo que lo hace usable como coste RELATIVO.
    """
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    cos_t, sin_t = torch.cos(theta).abs(), torch.sin(theta).abs()
    half_w = (w * cos_t + h * sin_t) / 2
    half_h = (w * sin_t + h * cos_t) / 2
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N, M) -- IoU de cajas alineadas `x1, y1, x2, y2`."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    left_top = torch.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


@dataclass(frozen=True, slots=True)
class Assignment:
    """Resultado: que celda va con que billete."""

    #: (N,) bool -- celdas positivas.
    positive: torch.Tensor
    #: (N,) long -- indice del billete asignado. Solo vale donde `positive`.
    matched: torch.Tensor
    #: (N,) -- IoU de la pareja, para ponderar la perdida de objectness.
    matched_iou: torch.Tensor
    #: (N, C) -- solo TAL: objetivo SUAVE de clasificacion por celda, con la
    #: metrica de alineacion normalizada. Con SimOTA es None y el objetivo es
    #: duro (one-hot, o one-hot por IoU en la receta del fork).
    target_scores: torch.Tensor | None = None

    @property
    def num_positives(self) -> int:
        return int(self.positive.sum())


#: (N, 5) x (M, 5) -> (N, M) de "parecido" en [0, 1]: 1 es la misma caja.
OverlapFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def enclosing_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """El solape por defecto: IoU de las envolventes alineadas."""
    return pairwise_iou(enclosing_boxes(predicted), enclosing_boxes(target))


def simota_assign(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    grid: AnchorGrid,
    *,
    iou_weight: float = 3.0,
    overlap: OverlapFn = enclosing_iou,
    overlap_cost: str = "neg_log",
) -> Assignment:
    """Asigna celdas a billetes. Todo en pixeles.

    `predicted_boxes` y `target_boxes` son (·, 5) con `cx, cy, w, h, theta`.
    `predicted_scores` es (N,) con la confianza ya en probabilidad.

    `overlap` y `overlap_cost` existen para reproducir la receta del fork de
    YOLOX-OBB SIN tocar la propia:

        propia   overlap = IoU de envolventes,   coste = -log(overlap)     (YOLOX)
        fork     overlap = 1 - kld_loss,         coste = 1 - overlap       (= kld_loss)

    El `dynamic k` usa `overlap` en los dos casos, que es lo que hace el fork
    (`pair_wise_iou_approximate = 1 - kld_loss`).

    El coste de clase es `-log(score)`. En el fork es una BCE contra el one-hot
    de la clase; con UNA clase, `BCE(p, 1) = -log(p)` y es lo mismo. Con mas
    clases dejaria de serlo, y este proyecto tiene una.
    """
    n_points = len(grid)
    device = grid.centers.device
    empty = Assignment(
        positive=torch.zeros(n_points, dtype=torch.bool, device=device),
        matched=torch.zeros(n_points, dtype=torch.long, device=device),
        matched_iou=torch.zeros(n_points, device=device),
    )
    if target_boxes.numel() == 0:
        # Imagen sin billetes: todo es fondo y no hay nada que asignar. Es un
        # caso legitimo, no un error -- el filtro de area puede dejar vacia una
        # imagen que solo tenia franjas.
        return empty

    inside = points_in_rotated_boxes(grid.centers, target_boxes)
    near = points_near_centers(grid.centers, target_boxes, grid.strides)
    candidate = inside | near
    if not candidate.any():
        return empty

    iou = overlap(predicted_boxes, target_boxes)
    if overlap_cost == "neg_log":
        # El log del IoU castiga con dureza creciente el solape malo, que es lo
        # que se quiere -- entre 0.8 y 0.9 la diferencia importa poco; entre
        # 0.1 y 0.2, mucho.
        pair_cost = -torch.log(iou.clamp(min=1e-8))
    elif overlap_cost == "one_minus":
        pair_cost = 1.0 - iou
    else:
        raise ValueError(f"overlap_cost desconocido: {overlap_cost!r}")
    cost = (
        -torch.log(predicted_scores[:, None].clamp(min=1e-8))
        + iou_weight * pair_cost
        + (~candidate) * BLOCKED_COST
    )

    matching = _dynamic_k_matching(cost, iou, candidate)
    positive = matching.any(dim=1)
    matched = matching.float().argmax(dim=1)
    return Assignment(
        positive=positive,
        matched=matched,
        matched_iou=iou.gather(1, matched[:, None]).squeeze(1) * positive,
    )


def _dynamic_k_matching(
    cost: torch.Tensor, iou: torch.Tensor, candidate: torch.Tensor
) -> torch.Tensor:
    """(N, M) bool -- la parte "OTA" de SimOTA.

    Cada billete se lleva `k` celdas, y `k` sale de la suma de sus mejores IoU:
    un billete que el modelo ya localiza bien recibe mas positivos que uno que
    apenas encuentra. Un `k` fijo trataria igual al caso facil y al dificil.
    """
    n_points, n_targets = cost.shape
    matching = torch.zeros_like(cost, dtype=torch.bool)

    top = min(TOP_CANDIDATES, n_points)
    top_iou, _ = torch.topk(iou * candidate, top, dim=0)
    # Al menos uno: un billete sin ningun positivo no genera gradiente y es
    # como si no estuviera anotado.
    dynamic_k = top_iou.sum(dim=0).int().clamp(min=1)

    for target in range(n_targets):
        k = int(dynamic_k[target])
        _, indices = torch.topk(cost[:, target], k, largest=False)
        matching[indices, target] = True

    # Una celda no puede servir a dos billetes: se queda con el mas barato. Sin
    # esto, la celda recibiria dos objetivos distintos y aprenderia el promedio,
    # que no es ninguno de los dos.
    conflicts = matching.sum(dim=1) > 1
    if conflicts.any():
        best = cost[conflicts].argmin(dim=1)
        matching[conflicts] = False
        matching[conflicts, best] = True
    return matching


def tal_assign(
    predicted_boxes: torch.Tensor,
    predicted_class_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    grid: AnchorGrid,
    *,
    overlap: OverlapFn,
    topk: int = 10,
    alpha: float = 0.5,
    beta: float = 6.0,
) -> Assignment:
    """Task-Aligned Assigner (Feng et al., "TOOD", ICCV 2021). Todo en pixeles.

    Es el asignador de la receta de Ultralytics, implementado desde el paper:
    su codigo es AGPL y no se ha leido. Los valores por defecto (`topk=10`,
    `alpha=0.5`, `beta=6.0`) son los que Ultralytics DOCUMENTA para su
    `TaskAlignedAssigner`; no salen del paper, que usa alpha=1.

    Como decide:

    1. **Metrica de alineacion** por par celda-billete:
       `t = score^alpha * overlap^beta`, con `score` la probabilidad de la clase
       correcta y `overlap` el ProbIoU. Premia a la vez clasificar bien y
       localizar bien, que es la idea de TOOD: que las celdas positivas sean
       las buenas en las DOS tareas y no en una.
    2. **Candidatas**: solo celdas cuyo centro cae dentro del rectangulo girado.
    3. **Top-k** por billete, por metrica.
    4. **Conflictos**: una celda pedida por dos billetes se queda con el de mas
       solape.
    5. **Objetivo suave**: `t` normalizada por billete a `[0, max overlap]` y
       repartida por clase. Es lo que hace que la BCE de clase y el peso de la
       perdida de caja lleven la calidad de la localizacion dentro.

    `predicted_class_scores` es (N, C) en probabilidad.
    """
    n_points = len(grid)
    n_classes = predicted_class_scores.shape[-1]
    device = grid.centers.device
    empty = Assignment(
        positive=torch.zeros(n_points, dtype=torch.bool, device=device),
        matched=torch.zeros(n_points, dtype=torch.long, device=device),
        matched_iou=torch.zeros(n_points, device=device),
        target_scores=torch.zeros(n_points, n_classes, device=device),
    )
    if target_boxes.numel() == 0:
        return empty

    overlaps = overlap(predicted_boxes, target_boxes).clamp(min=0.0)  # (N, M)
    score_of_class = predicted_class_scores[:, target_classes]  # (N, M)
    metric = score_of_class.clamp(min=0.0) ** alpha * overlaps**beta

    inside = points_in_rotated_boxes(grid.centers, target_boxes)  # (N, M)
    if not inside.any():
        return empty

    k = min(topk, n_points)
    masked = torch.where(inside, metric, torch.zeros_like(metric))
    top_values, top_indices = torch.topk(masked, k, dim=0)  # (k, M)
    matching = torch.zeros_like(inside)
    # Solo cuentan las top-k con metrica > 0: un billete con menos de k
    # candidatas no debe llevarse celdas de fuera de su caja rellenando.
    valid = top_values > 0
    for column in range(matching.shape[1]):
        matching[top_indices[valid[:, column], column], column] = True

    conflicts = matching.sum(dim=1) > 1
    if conflicts.any():
        best = overlaps[conflicts].argmax(dim=1)
        matching[conflicts] = False
        matching[conflicts, best] = True

    positive = matching.any(dim=1)
    matched = matching.float().argmax(dim=1)
    if not positive.any():
        return empty

    # Normalizacion por billete: la celda mejor alineada de cada uno recibe
    # exactamente su mejor solape como objetivo, y el resto en proporcion.
    aligned = metric * matching
    max_metric = aligned.max(dim=0, keepdim=True).values
    max_overlap = (overlaps * matching).max(dim=0, keepdim=True).values
    normalized = aligned * max_overlap / max_metric.clamp(min=1e-9)  # (N, M)
    per_cell = normalized.gather(1, matched[:, None]).squeeze(1) * positive
    target_scores = torch.zeros(n_points, n_classes, device=device)
    target_scores[positive, target_classes[matched[positive]]] = per_cell[positive]

    return Assignment(
        positive=positive,
        matched=matched,
        matched_iou=overlaps.gather(1, matched[:, None]).squeeze(1) * positive,
        target_scores=target_scores,
    )


__all__ = [
    "BLOCKED_COST",
    "CENTER_RADIUS",
    "TOP_CANDIDATES",
    "AnchorGrid",
    "Assignment",
    "OverlapFn",
    "build_anchor_grid",
    "enclosing_boxes",
    "enclosing_iou",
    "pairwise_iou",
    "points_in_rotated_boxes",
    "points_near_centers",
    "simota_assign",
    "tal_assign",
]
