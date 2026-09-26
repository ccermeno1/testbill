"""Drawing: oriented boxes with their probability, and heatmap overlays."""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Detections, corners

# one colour per model (BGR), readable on photos
PALETTE = [(0, 200, 0), (255, 140, 0), (0, 90, 255), (200, 0, 200), (0, 200, 255), (255, 60, 60)]


def draw_detections(bgr: np.ndarray, det: Detections, class_names: list[str], color=(0, 200, 0),
                    highlight: int | None = None) -> np.ndarray:
    img = bgr.copy()
    k = max(1.0, min(img.shape[:2]) / 500)
    thick, font = max(2, round(2 * k)), 0.55 * k
    polys = corners(det.boxes).round().astype(np.int32) if len(det) else []
    for i, (poly, s, l) in enumerate(zip(polys, det.scores, det.labels)):
        c = (0, 255, 255) if i == highlight else color
        cv2.polylines(img, [poly], True, c, thick + (2 if i == highlight else 0), cv2.LINE_AA)
        # marks the first corner, so the orientation of the box is visible too
        cv2.circle(img, tuple(int(v) for v in poly[0]), thick + 2, c, -1, cv2.LINE_AA)
        name = class_names[l] if l < len(class_names) else str(l)
        text = f"#{i + 1} {name} {s:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, max(1, thick // 2))
        x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
        x = min(max(x, 0), max(img.shape[1] - tw - 4, 0))
        y = max(y, th + base + 4)
        cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 4, y), c, -1)
        cv2.putText(img, text, (x + 2, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0),
                    max(1, thick // 2), cv2.LINE_AA)
    return img


def overlay_heatmap(bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
