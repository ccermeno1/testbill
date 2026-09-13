"""IoU EXACTO entre rectangulos girados, en torch puro y diferenciable.

Es lo que sustituye al operador compilado (`box_iou_rotated` / `convex`) de
DDGRCF/YOLOX_OBB, que lo usa en dos sitios: el coste del SimOTA y la perdida de
caja (PolyIoU). Sin esto el port no reproduce su receta; con las gaussianas la
cambiaria.

Como se calcula
---------------
Para dos cuadrilateros convexos, la interseccion es un poligono convexo cuyos
vertices son de tres tipos: esquinas del primero dentro del segundo, esquinas
del segundo dentro del primero, y cruces de aristas. Se recogen los 24
candidatos (4 + 4 + 16) con una mascara de validez, se ordenan por angulo
alrededor de su centro y se aplica la formula del cordon (shoelace) enmascarada.
Como mucho 8 son validos.

Es diferenciable respecto a las coordenadas: las esquinas y los cruces son
funciones lisas de las cajas; solo el ORDEN se calcula sin gradiente, y el orden
es constante a trozos, asi que no lo necesita. Es el mismo planteamiento que la
"Rotated IoU" de Lanxiao Li (MIT), reescrito aqui desde la geometria.

Coste: para el asignador son `celdas x billetes` pares por imagen (3549 x ~2),
todo vectorizado. Medido, es del orden de milisegundos: no es el "no cabe en el
bucle" de shapely, que va par a par en Python.
"""

from __future__ import annotations

import torch

_EPS = 1e-8


def box_corners(boxes: torch.Tensor) -> torch.Tensor:
    """`(N, 5)` cx, cy, w, h, theta -> `(N, 4, 2)` esquinas en sentido horario
    (en coordenadas de imagen, con y hacia abajo)."""
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    dx = torch.stack((-w, w, w, -w), dim=-1) / 2
    dy = torch.stack((-h, -h, h, h), dim=-1) / 2
    x = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    y = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    return torch.stack((x, y), dim=-1)


def _cross(o, a, b):
    """Producto vectorial 2D de (a - o) x (b - o). Signo = de que lado esta b."""
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (
        b[..., 0] - o[..., 0]
    )


def _edge_intersections(c1: torch.Tensor, c2: torch.Tensor):
    """Cruces entre las 4 aristas de cada caja: `(N, 16, 2)` y mascara `(N, 16)`."""
    a = c1  # (N, 4, 2)
    b = torch.roll(c1, -1, dims=1)
    c = c2
    d = torch.roll(c2, -1, dims=1)
    # Todas las combinaciones arista_i de 1 x arista_j de 2.
    a = a[:, :, None, :].expand(-1, 4, 4, -1)
    b = b[:, :, None, :].expand(-1, 4, 4, -1)
    c = c[:, None, :, :].expand(-1, 4, 4, -1)
    d = d[:, None, :, :].expand(-1, 4, 4, -1)
    # Parametros t (sobre ab) y u (sobre cd) del cruce de las rectas.
    ab = b - a
    cd = d - c
    ac = c - a
    denominator = ab[..., 0] * cd[..., 1] - ab[..., 1] * cd[..., 0]
    parallel = denominator.abs() < _EPS
    safe = torch.where(parallel, torch.ones_like(denominator), denominator)
    t = (ac[..., 0] * cd[..., 1] - ac[..., 1] * cd[..., 0]) / safe
    u = (ac[..., 0] * ab[..., 1] - ac[..., 1] * ab[..., 0]) / safe
    valid = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    points = a + t[..., None] * ab
    return points.reshape(-1, 16, 2), valid.reshape(-1, 16)


def _points_inside(points: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
    """`(N, P)`: si cada punto cae dentro del cuadrilatero convexo `corners`.

    Dentro = al mismo lado de las cuatro aristas. Vale para ambas orientaciones
    porque se compara el signo entre aristas, no contra un sentido fijo.
    """
    a = corners[:, None, :, :]  # (N, 1, 4, 2)
    b = torch.roll(corners, -1, dims=1)[:, None, :, :]
    p = points[:, :, None, :]  # (N, P, 1, 2)
    side = _cross(a, b, p)  # (N, P, 4)
    return (side >= -_EPS).all(dim=-1) | (side <= _EPS).all(dim=-1)


def intersection_area(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Area de la interseccion de cada par `(N, 5)` x `(N, 5)`. Diferenciable."""
    c1, c2 = box_corners(boxes_a), box_corners(boxes_b)
    crossings, crossing_valid = _edge_intersections(c1, c2)
    inside_1 = _points_inside(c1, c2)  # esquinas de A dentro de B
    inside_2 = _points_inside(c2, c1)

    vertices = torch.cat((c1, c2, crossings), dim=1)  # (N, 24, 2)
    valid = torch.cat((inside_1, inside_2, crossing_valid), dim=1)  # (N, 24)
    count = valid.sum(dim=1, keepdim=True).clamp(min=1)

    # Orden angular alrededor del centro de los validos. Sin gradiente: el
    # orden es constante a trozos, y lo que se deriva son las coordenadas.
    with torch.no_grad():
        mask = valid[..., None].to(vertices.dtype)
        center = (vertices * mask).sum(dim=1, keepdim=True) / count[..., None]
        angles = torch.atan2(vertices[..., 1] - center[..., 1], vertices[..., 0] - center[..., 0])
        angles = torch.where(valid, angles, torch.full_like(angles, 10.0))  # invalidos al final
        order = angles.argsort(dim=1)
    sorted_vertices = torch.gather(vertices, 1, order[..., None].expand(-1, -1, 2))
    sorted_valid = torch.gather(valid, 1, order)

    # Shoelace enmascarado: cada valido con el siguiente valido, y el ultimo
    # valido cierra contra el primero.
    n = sorted_valid.sum(dim=1)  # (N,)
    index = torch.arange(24, device=vertices.device)[None, :]
    next_index = torch.where(index + 1 < n[:, None], index + 1, torch.zeros_like(index))
    next_vertices = torch.gather(sorted_vertices, 1, next_index[..., None].expand(-1, -1, 2))
    cross = (
        sorted_vertices[..., 0] * next_vertices[..., 1]
        - next_vertices[..., 0] * sorted_vertices[..., 1]
    )
    area = 0.5 * (cross * sorted_valid.to(cross.dtype)).sum(dim=1).abs()
    return torch.where(n >= 3, area, torch.zeros_like(area))


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return boxes[:, 2] * boxes[:, 3]


def rotated_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """IoU exacto por pares `(N,)`, diferenciable. Es el `PolyIoU` de DDGRCF."""
    inter = intersection_area(boxes_a, boxes_b)
    union = box_area(boxes_a) + box_area(boxes_b) - inter
    return (inter / (union + eps)).clamp(min=0.0, max=1.0)


def pairwise_rotated_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`(N, M)` de IoU exacto. Para el asignador; se llama sin gradiente."""
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return a.new_zeros((n, m))
    left = a[:, None, :].expand(n, m, 5).reshape(-1, 5)
    right = b[None, :, :].expand(n, m, 5).reshape(-1, 5)
    return rotated_iou(left, right).view(n, m)


__all__ = ["box_corners", "intersection_area", "pairwise_rotated_iou", "rotated_iou"]
