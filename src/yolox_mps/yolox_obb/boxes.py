"""Rotated box utilities (``cx, cy, w, h, angle_rad``; angle clockwise in image coords)."""
import math

import cv2
import numpy as np
import torch

PI = math.pi


def regularize_le90(boxes: torch.Tensor) -> torch.Tensor:
    """Unique representation: w >= h and angle in [-pi/2, pi/2) (RotatedBoxes.regularize_boxes)."""
    x, y, w, h, t = boxes.unbind(-1)
    swap = w <= h  # mmrotate keeps (w, h, t) when w > h, otherwise swaps and adds pi/2
    w_ = torch.where(swap, h, w)
    h_ = torch.where(swap, w, h)
    t = torch.where(swap, t + PI / 2, t)
    t = (t + PI / 2) % PI - PI / 2
    return torch.stack([x, y, w_, h_, t], -1)


def regularize_mintheta(boxes: torch.Tensor) -> torch.Tensor:
    """YOLOX_OBB target representation (``mintheta_obb``): of the two equivalent
    ``(w, h, t)`` / ``(h, w, t + pi/2)`` forms keep the one with the smallest ``|t|``,
    so the angle lies in [-pi/4, pi/4] and w / h follow the box, not the long side."""
    x, y, w, h, t = boxes.unbind(-1)
    t1 = (t + PI / 2) % PI - PI / 2
    t2 = (t + PI) % PI - PI / 2  # t + pi/2 wrapped to [-pi/2, pi/2)
    keep = t1.abs() < t2.abs()
    return torch.stack([x, y, torch.where(keep, w, h), torch.where(keep, h, w), torch.where(keep, t1, t2)], -1)


def rbox2poly(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 5) -> (..., 8) corner polygon x1 y1 ... x4 y4 (rbox2qbox)."""
    ctr, w, h, t = torch.split(boxes, (2, 1, 1, 1), dim=-1)
    cos, sin = torch.cos(t), torch.sin(t)
    v1 = torch.cat([w / 2 * cos, w / 2 * sin], -1)
    v2 = torch.cat([-h / 2 * sin, h / 2 * cos], -1)
    return torch.cat([ctr + v1 + v2, ctr + v1 - v2, ctr - v1 - v2, ctr - v1 + v2], -1)


def poly2rbox(polys: np.ndarray) -> np.ndarray:
    """(n, 8) polygons in pixels -> (n, 5) float32 boxes via cv2.minAreaRect, then le90."""
    out = np.zeros((len(polys), 5), dtype=np.float32)
    for i, p in enumerate(np.asarray(polys, dtype=np.float32).reshape(-1, 4, 2)):
        (x, y), (w, h), a = cv2.minAreaRect(p)
        out[i] = (x, y, w, h, a / 180 * PI)
    if len(out):
        out = regularize_le90(torch.from_numpy(out)).numpy()
    return out


def points_in_rboxes(points: torch.Tensor, boxes: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """(m, 2) points x (n, 5) boxes -> (m, n) bool, strictly inside (OBBDetectX.get_in_boxes_info)."""
    ctr, wh, t = torch.split(boxes[None], [2, 2, 1], dim=-1)  # (1, n, .)
    cos, sin = torch.cos(t[..., 0]), torch.sin(t[..., 0])
    off = points[:, None, :] - ctr  # (m, n, 2)
    ox = cos * off[..., 0] + sin * off[..., 1]
    oy = -sin * off[..., 0] + cos * off[..., 1]
    w, h = wh[..., 0], wh[..., 1]
    return (ox <= w / 2 - eps) & (ox >= -w / 2 + eps) & (oy <= h / 2 - eps) & (oy >= -h / 2 + eps)


# ---------------------------------------------------------------- numpy augmentation helpers

def flip_rboxes(boxes: np.ndarray, img_shape, direction: str) -> np.ndarray:
    h, w = img_shape[:2]
    b = boxes.copy()
    if direction == 'horizontal':
        b[:, 0] = w - b[:, 0]
        b[:, 4] = -b[:, 4]
    elif direction == 'vertical':
        b[:, 1] = h - b[:, 1]
        b[:, 4] = -b[:, 4]
    elif direction == 'diagonal':
        b[:, 0] = w - b[:, 0]
        b[:, 1] = h - b[:, 1]
    else:
        raise ValueError(direction)
    return b


def rotate_rboxes(boxes: np.ndarray, center, angle_deg: float) -> np.ndarray:
    """Rotate boxes clockwise by ``angle_deg`` around ``center`` (matches cv2 image rotation)."""
    if len(boxes) == 0:
        return boxes
    m = cv2.getRotationMatrix2D(center, -angle_deg, 1.0).astype(np.float32)  # (2, 3)
    b = boxes.copy()
    pts = np.concatenate([b[:, :2], np.ones((len(b), 1), np.float32)], 1)
    b[:, :2] = pts @ m.T
    b[:, 4] += angle_deg / 180 * PI
    return b
