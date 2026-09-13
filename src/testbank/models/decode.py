"""De la salida cruda de la cabeza a quads canonicos, con NMS rotado.

Aqui SI se usa shapely, al reves que en el entrenamiento. No es incoherencia:
son dos regimenes distintos. En el bucle de entrenamiento habria que intersectar
poligonos en cada iteracion durante horas; en inferencia se hace una vez por
imagen sobre las pocas cajas que sobreviven al umbral de confianza. La
especificacion pide NMS rotado con shapely y en inferencia el coste es asumible.

Por que el NMS tiene que ser rotado
-----------------------------------
Es el motivo de que todo el proyecto sea OBB. Dos billetes en abanico, alargados
y en angulos distintos, tienen envolventes alineadas que se solapan casi por
completo: un NMS estandar suprimiria una deteccion verdadera. El rotado ve que
los rectangulos apenas se tocan y conserva las dos.

El marco de las distancias
--------------------------
`l, t, r, b` se predicen en el sistema de referencia de la CAJA, no en el de la
imagen. Tienen que serlo: si fueran ejes de imagen, `w = l + r` no seria el
ancho del billete sino el de su envolvente, y el angulo no querria decir nada.
Asi que el desplazamiento del centro se gira por theta antes de sumarlo.
"""

from __future__ import annotations

import math
import warnings

import torch
from shapely.geometry import Polygon

from testbank.dataio.formats import ImageSize
from testbank.geometry.quad import (
    CoordinateRangeWarning,
    Quad,
    QuadShapeWarning,
    canonicalize,
)
from testbank.metrics.core import Prediction
from testbank.models.assign import AnchorGrid, build_anchor_grid
from testbank.models.yolox_obb import decode_angle

#: Por debajo de esto no se emite deteccion. Bajo a proposito: la curva
#: precision-recall necesita la cola. Ver `MetricsConfig.report_confidence`.
DEFAULT_CONF = 0.01

#: IoU rotado por encima del cual dos detecciones se consideran la misma.
DEFAULT_NMS_IOU = 0.5

#: Tope de detecciones por imagen tras el NMS. Con uno o dos billetes por foto,
#: 300 es holgadisimo; existe para que una red sin entrenar no cuelgue el NMS.
MAX_DETECTIONS = 300


def decode_level(output, grid_slice: AnchorGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Un nivel -> `(cajas (N,5) en pixeles, puntuaciones (N,))`."""
    batch = output.distances.shape[0]
    if batch != 1:
        raise ValueError("decode_level espera una imagen; usa decode_outputs")

    # (1, C, H, W) -> (N, C)
    def flat(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[0].permute(1, 2, 0).reshape(-1, tensor.shape[1])

    distances = flat(output.distances) * output.stride
    angle = flat(output.angle)
    classes = torch.sigmoid(flat(output.classes))
    # Sin rama de objectness (cabeza al estilo v8), la clase lleva la presencia
    # dentro: su objetivo de entrenamiento ya es la calidad de la localizacion.
    objectness = (
        torch.sigmoid(flat(output.objectness))[:, 0]
        if output.objectness is not None
        else torch.ones(classes.shape[0], device=classes.device)
    )

    left, top, right, bottom = distances.unbind(dim=-1)
    width = left + right
    height = top + bottom
    # `decode_angle` indexa la dim 1, asi que un (N, 2) le vale igual que el
    # (B, 2, H, W) de la cabeza. Una sola implementacion para los dos usos.
    theta = decode_angle(angle)

    # El desplazamiento del centro va en el marco de la CAJA, asi que se gira
    # antes de sumarlo al centro de la celda.
    local_x = (right - left) / 2
    local_y = (bottom - top) / 2
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    centers = grid_slice.centers
    cx = centers[:, 0] + local_x * cos_t - local_y * sin_t
    cy = centers[:, 1] + local_x * sin_t + local_y * cos_t

    boxes = torch.stack((cx, cy, width, height, theta), dim=-1)
    # La puntuacion combina "hay algo" con "es un billete": una celda segura de
    # que hay objeto pero insegura de la clase no debe puntuar alto.
    scores = objectness * classes.max(dim=-1).values
    return boxes, scores


def decode_outputs(outputs, size: ImageSize) -> tuple[torch.Tensor, torch.Tensor]:
    """Todos los niveles juntos, en pixeles de la imagen."""
    sizes = [tuple(o.distances.shape[-2:]) for o in outputs]
    strides = tuple(o.stride for o in outputs)
    grid = build_anchor_grid(sizes, strides, device=outputs[0].distances.device)

    boxes, scores, start = [], [], 0
    for output, (height, width) in zip(outputs, sizes):
        count = height * width
        piece = AnchorGrid(
            grid.centers[start : start + count], grid.strides[start : start + count]
        )
        level_boxes, level_scores = decode_level(output, piece)
        boxes.append(level_boxes)
        scores.append(level_scores)
        start += count
    return torch.cat(boxes), torch.cat(scores)


def box_to_polygon(box: torch.Tensor) -> Polygon:
    """`cx, cy, w, h, theta` -> poligono en pixeles."""
    cx, cy, w, h, theta = (float(v) for v in box)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return Polygon(
        [
            (cx + x * cos_t - y * sin_t, cy + x * sin_t + y * cos_t)
            for x, y in corners
        ]
    )


def rotated_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    iou_threshold: float = DEFAULT_NMS_IOU,
    max_detections: int = MAX_DETECTIONS,
) -> list[int]:
    """Supresion por IoU ROTADO. Devuelve los indices que sobreviven.

    El orden es por confianza descendente, como manda: la deteccion mas segura
    se queda y suprime a las que se le parecen.
    """
    if boxes.numel() == 0:
        return []
    order = torch.argsort(scores, descending=True).tolist()
    polygons: dict[int, Polygon] = {}

    def polygon(index: int) -> Polygon:
        if index not in polygons:
            poly = box_to_polygon(boxes[index])
            polygons[index] = poly if poly.is_valid else poly.buffer(0)
        return polygons[index]

    kept: list[int] = []
    for index in order:
        if len(kept) >= max_detections:
            break
        current = polygon(index)
        if current.area <= 0:
            continue
        suppressed = False
        for other in kept:
            previous = polygon(other)
            intersection = current.intersection(previous).area
            if intersection <= 0:
                continue
            union = current.area + previous.area - intersection
            if union > 0 and intersection / union > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(index)
    return kept


def boxes_to_quads(boxes: torch.Tensor, size: ImageSize) -> list[Quad]:
    """Pixeles -> quads NORMALIZADOS y canonicos, que es lo que consume todo.

    La canonicalizacion se hace con el aspecto de la imagen: sin el, el "lado
    mas largo" que ancla el orden no es el geometrico. Es el mismo error que ha
    mordido dos veces en este proyecto.
    """
    aspect = size.width / size.height
    quads = []
    # Los avisos de `quad.py` existen para juzgar ANOTACIONES: un vertice fuera
    # del marco o un ancla inestable son cosas que mirar y quizas corregir en
    # Roboflow. Una prediccion no es una anotacion. Un modelo a medio entrenar
    # emite miles de cajas casi cuadradas y fuera de la imagen, y dejarlas
    # avisar convertiria cada inferencia en un muro de texto -- y con la
    # configuracion de pytest, en un error.
    #
    # No se pierde la senal: para predicciones el equivalente es la distancia
    # por vertice del informe de metricas, que ya sale como diagnostico.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuadShapeWarning)
        warnings.simplefilter("ignore", CoordinateRangeWarning)
        for box in boxes:
            polygon = box_to_polygon(box)
            points = [
                (x / size.width, y / size.height)
                for x, y in list(polygon.exterior.coords)[:4]
            ]
            quads.append(canonicalize(Quad.from_xy(points), aspect=aspect))
    return quads


def detections(
    outputs,
    size: ImageSize,
    *,
    confidence: float = DEFAULT_CONF,
    iou_threshold: float = DEFAULT_NMS_IOU,
    max_detections: int = MAX_DETECTIONS,
) -> list[Prediction]:
    """Salida de la cabeza -> `Prediction` listas para las metricas."""
    boxes, scores = decode_outputs(outputs, size)
    keep_mask = scores >= confidence
    boxes, scores = boxes[keep_mask], scores[keep_mask]
    if boxes.numel() == 0:
        return []

    kept = rotated_nms(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    if not kept:
        return []
    selected = boxes[kept]
    quads = boxes_to_quads(selected, size)
    return [
        Prediction(quad=quad, score=float(scores[index]), class_id=0)
        for quad, index in zip(quads, kept)
    ]


__all__ = [
    "DEFAULT_CONF",
    "DEFAULT_NMS_IOU",
    "MAX_DETECTIONS",
    "box_to_polygon",
    "boxes_to_quads",
    "decode_level",
    "decode_outputs",
    "detections",
    "rotated_nms",
]
