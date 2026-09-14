"""From the raw head output to canonical quads, with rotated NMS.

Here shapely IS used, unlike in training. It is not an inconsistency: they are
two different regimes. In the training loop polygons would have to be
intersected on every iteration for hours; in inference it is done once per
image over the few boxes that survive the confidence threshold. The
specification asks for rotated NMS with shapely and in inference the cost is
affordable.

Why the NMS has to be rotated
-----------------------------
It is the reason the whole project is OBB. Two fanned banknotes, elongated and
at different angles, have axis-aligned envelopes that overlap almost entirely:
a standard NMS would suppress a true detection. The rotated one sees that the
rectangles barely touch and keeps both.

The frame of the distances
--------------------------
`l, t, r, b` are predicted in the reference frame of the BOX, not of the image.
They have to be: if they were image axes, `w = l + r` would not be the width of
the banknote but of its envelope, and the angle would mean nothing. So the
center offset is rotated by theta before being added.
"""

from __future__ import annotations

import math
import warnings

import torch
from shapely.geometry import Polygon

from testbank.dataio.image_sizes import ImageSize
from testbank.geometry.quad import (
    COORD_MAX,
    COORD_MIN,
    CoordinateRangeWarning,
    Quad,
    QuadShapeWarning,
    canonicalize,
)
from testbank.models.assign import AnchorGrid, build_anchor_grid
from testbank.models.yolox_obb import decode_angle
from testbank.prediction import Prediction

#: Below this no detection is emitted. Low on purpose: the precision-recall
#: curve needs the tail. See `MetricsConfig.report_confidence`.
DEFAULT_CONF = 0.01

#: Rotated IoU above which two detections are considered the same one.
DEFAULT_NMS_IOU = 0.5

#: Cap on detections per image after NMS. With one or two banknotes per photo,
#: 300 is very generous; it exists so an untrained network does not hang the NMS.
MAX_DETECTIONS = 300


def decode_level(output, grid_slice: AnchorGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """One level -> `(boxes (N,5) in pixels, scores (N,))`."""
    batch = output.distances.shape[0]
    if batch != 1:
        raise ValueError("decode_level expects one image; use decode_outputs")

    # (1, C, H, W) -> (N, C)
    def flat(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[0].permute(1, 2, 0).reshape(-1, tensor.shape[1])

    distances = flat(output.distances) * output.stride
    angle = flat(output.angle)
    classes = torch.sigmoid(flat(output.classes))
    # Without an objectness branch (v8-style head), the class carries presence
    # inside: its training target is already the localization quality.
    objectness = (
        torch.sigmoid(flat(output.objectness))[:, 0]
        if output.objectness is not None
        else torch.ones(classes.shape[0], device=classes.device)
    )

    # `decode_angle` indexes dim 1, so an (N, 2) works for it just like the
    # (B, 2, H, W) of the head. A single implementation for both uses.
    theta = decode_angle(angle, output.angle_mode)
    centers = grid_slice.centers

    if output.regression == "yolox":
        # The YOLOX original, used by the DDGRCF port: `(dx, dy)` in cells from
        # the cell corner and `(log w, log h)` in strides. `distances` already
        # comes multiplied by the stride above, so `dx * stride` is the offset
        # in pixels; the cell center of our grid is half a stride past the
        # corner.
        raw = flat(output.distances)
        stride = output.stride
        cx = centers[:, 0] - stride / 2 + raw[:, 0] * stride
        cy = centers[:, 1] - stride / 2 + raw[:, 1] * stride
        width = torch.exp(raw[:, 2]) * stride
        height = torch.exp(raw[:, 3]) * stride
    else:
        left, top, right, bottom = distances.unbind(dim=-1)
        width = left + right
        height = top + bottom
        # The center offset lives in the frame of the BOX, so it is rotated
        # before being added to the cell center.
        local_x = (right - left) / 2
        local_y = (bottom - top) / 2
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        cx = centers[:, 0] + local_x * cos_t - local_y * sin_t
        cy = centers[:, 1] + local_x * sin_t + local_y * cos_t

    boxes = torch.stack((cx, cy, width, height, theta), dim=-1)
    # The score combines "there is something" with "it is a banknote": a cell
    # sure there is an object but unsure of the class must not score high.
    scores = objectness * classes.max(dim=-1).values
    return boxes, scores


def decode_outputs(outputs, size: ImageSize) -> tuple[torch.Tensor, torch.Tensor]:
    """All levels together, in image pixels."""
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
    """`cx, cy, w, h, theta` -> polygon in pixels."""
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
    """Suppression by ROTATED IoU. Returns the surviving indices.

    The order is by descending confidence, as it should be: the most confident
    detection stays and suppresses the ones that resemble it.
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


def boxes_to_quads(boxes: torch.Tensor, size: ImageSize) -> list[Quad | None]:
    """Pixels -> NORMALIZED, canonical quads, which is what everything consumes.

    Canonicalization is done with the image aspect: without it, the "longest
    side" that anchors the order is not the geometric one. It is the same
    mistake that has bitten twice in this project.
    """
    aspect = size.width / size.height
    quads = []
    # The warnings in `quad.py` exist to judge ANNOTATIONS: a vertex outside
    # the frame or an unstable anchor are things to look at and maybe fix in
    # Roboflow. A prediction is not an annotation. A half-trained model emits
    # thousands of nearly square boxes outside the image, and letting them warn
    # would turn every inference into a wall of text -- and with the pytest
    # configuration, into an error.
    #
    # The signal is not lost: for predictions the equivalent is the per-vertex
    # distance of the metrics report, which already comes out as a diagnostic.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuadShapeWarning)
        warnings.simplefilter("ignore", CoordinateRangeWarning)
        for box in boxes:
            polygon = box_to_polygon(box)
            points = [
                (x / size.width, y / size.height)
                for x, y in list(polygon.exterior.coords)[:4]
            ]
            # `Quad` admits vertices up to half a frame outside the image, which
            # is what an ANNOTATION of a banknote crossing the border can have.
            # A prediction can go much further -- a network with the backbone
            # freshly loaded and the head untrained does -- and before that blew
            # up the whole evaluation with a QuadError. None is returned and the
            # prediction keeps existing WITHOUT geometry: the metric counts it
            # as a false positive, which is what it is. The reference frameworks
            # do the equivalent: they do not discard.
            if not all(COORD_MIN <= v <= COORD_MAX for xy in points for v in xy):
                quads.append(None)
                continue
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
    """Head output -> `Prediction`s ready for the metrics."""
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
    # A None quad (box more than half a frame outside the image) is emitted
    # anyway: it is a false positive the model committed and the metric counts it.
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
