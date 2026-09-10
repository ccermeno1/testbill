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

    @property
    def num_positives(self) -> int:
        return int(self.positive.sum())


def simota_assign(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    grid: AnchorGrid,
    *,
    iou_weight: float = 3.0,
) -> Assignment:
    """Asigna celdas a billetes. Todo en pixeles.

    `predicted_boxes` y `target_boxes` son (·, 5) con `cx, cy, w, h, theta`.
    `predicted_scores` es (N,) con la confianza ya en probabilidad.
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

    iou = pairwise_iou(enclosing_boxes(predicted_boxes), enclosing_boxes(target_boxes))
    # Coste: penaliza poca confianza y poco solape. El log del IoU castiga con
    # dureza creciente el solape malo, que es lo que se quiere -- entre 0.8 y
    # 0.9 la diferencia importa poco; entre 0.1 y 0.2, mucho.
    cost = (
        -torch.log(predicted_scores[:, None].clamp(min=1e-8))
        + iou_weight * -torch.log(iou.clamp(min=1e-8))
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


__all__ = [
    "BLOCKED_COST",
    "CENTER_RADIUS",
    "TOP_CANDIDATES",
    "AnchorGrid",
    "Assignment",
    "build_anchor_grid",
    "enclosing_boxes",
    "pairwise_iou",
    "points_in_rotated_boxes",
    "points_near_centers",
    "simota_assign",
]
