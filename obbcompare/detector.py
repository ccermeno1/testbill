"""ONNX inference with the same pre/post-processing for every model.

Pre-processing (the three branches agree on it, only colour / pad value change):
resize keeping the aspect ratio so the long side is ``size`` (INTER_AREA when shrinking,
INTER_LINEAR when enlarging), paste at the top-left of a canvas filled with ``pad_value``.
The canvas is the graph's input size when it is static, otherwise the resized image padded
up to a multiple of ``pad_multiple``.

Outputs: either decoded ``boxes (1, N, 5)`` + ``scores (1, N, C)`` / ``(1, C, N)`` (YOLOX-OBB,
PP-YOLOE-R), or the raw head maps cls / reg / angle per level (RTMDet-R of
``feature/rt_refactor``), which are decoded here exactly as its ``onnx_example.py`` does.

Post-processing: best class per prior, score threshold, class-aware greedy rotated NMS,
divide cx, cy, w, h by the resize scale. The index of the prior each detection comes from
is kept: Grad-CAM differentiates exactly that score.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from .config import ModelConfig


@dataclass
class RawOutput:
    boxes: np.ndarray    # (N, 5) input pixels
    scores: np.ndarray   # (N, C)
    scale: float
    input_hw: tuple[int, int]
    ms: float            # time of session.run


@dataclass
class Detections:
    boxes: np.ndarray    # (k, 5) cx, cy, w, h, angle_rad in ORIGINAL pixels
    scores: np.ndarray   # (k,)
    labels: np.ndarray   # (k,)
    prior_idx: np.ndarray  # (k,) row of the network output each one comes from

    def __len__(self) -> int:
        return len(self.scores)


class OnnxDetector:

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(cfg.onnx_path), opts, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        self.input_name = cfg.input_name or inputs[0].name
        shape = next(i.shape for i in inputs if i.name == self.input_name)
        h, w = shape[-2:]
        self.static_hw = (h, w) if isinstance(h, int) and isinstance(w, int) else None
        if self.static_hw:
            # a static graph decides the size, whatever the metadata says
            self.size = max(self.static_hw)
        else:
            self.size = cfg.size
        if cfg.raw_outputs:
            r = cfg.raw_outputs
            self.levels = [(r[i], r[i + 1], r[i + 2]) for i in range(0, len(r), 3)]
            # which outputs carry the scores and which the geometry (read by explain.py)
            self.score_names = [lv[0] for lv in self.levels]
            self.box_names = [n for lv in self.levels for n in lv[1:]]
        else:
            self.levels = []
            boxes, scores = self._output_names()
            self.score_names, self.box_names = [scores], [boxes]
        self.warmup()

    def _output_names(self) -> tuple[str, str]:
        outs = self.session.get_outputs()
        names = [o.name for o in outs]
        boxes = self.cfg.boxes_output if self.cfg.boxes_output in names else None
        if boxes is None:
            boxes = next(o.name for o in outs if o.shape[-1] == 5)
        scores = self.cfg.scores_output if self.cfg.scores_output in names else None
        if scores is None:
            scores = next(n for n in names if n != boxes)
        return boxes, scores

    # ------------------------------------------------------------------ pre-processing
    def preprocess(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """BGR uint8 photo -> (1, 3, H, W) float32 input tensor and the resize scale."""
        cfg = self.cfg
        h0, w0 = bgr.shape[:2]
        scale = self.size / max(h0, w0)
        img = bgr
        if scale != 1.0:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (max(1, round(w0 * scale)), max(1, round(h0 * scale))), interpolation=interp)
        if cfg.color == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rh, rw = img.shape[:2]
        if self.static_hw:
            ch, cw = self.static_hw
        else:
            m = cfg.pad_multiple
            ch, cw = int(np.ceil(rh / m) * m), int(np.ceil(rw / m) * m)
        canvas = np.full((ch, cw, 3), cfg.pad_value, np.float32)
        canvas[:rh, :rw] = img
        if cfg.mean is not None:
            canvas = (canvas - np.asarray(cfg.mean, np.float32)) / np.asarray(cfg.std, np.float32)
        return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), scale

    # ------------------------------------------------------------------ inference
    def warmup(self) -> None:
        hw = self.static_hw or (self.size, self.size)
        x = np.zeros((1, 3, *hw), np.float32)
        out = self.session.run(None, {self.input_name: x})
        if not self.levels:
            names = [o.name for o in self.session.get_outputs()]
            self.scores_nc(out[names.index(self.score_names[0])])  # fixes cfg.scores_layout (Grad-CAM reads it)

    def infer(self, bgr: np.ndarray) -> RawOutput:
        x, scale = self.preprocess(bgr)
        t0 = time.perf_counter()
        if self.levels:
            maps = self.session.run(self.cfg.raw_outputs, {self.input_name: x})
            ms = (time.perf_counter() - t0) * 1000
            boxes, scores = decode_raw(maps, self.cfg.strides)
            return RawOutput(boxes, scores, scale, x.shape[-2:], ms)
        boxes, scores = self.session.run(self.box_names + self.score_names, {self.input_name: x})
        ms = (time.perf_counter() - t0) * 1000
        return RawOutput(boxes[0], self.scores_nc(scores)[0], scale, x.shape[-2:], ms)

    def scores_nc(self, scores: np.ndarray) -> np.ndarray:
        """(1, N, C) or (1, C, N) -> (1, N, C)."""
        layout = self.cfg.scores_layout
        if layout is None:
            n = self.session.get_outputs()
            n_boxes = next(o.shape[1] for o in n if o.name == self.box_names[0])
            if isinstance(n_boxes, int) and scores.shape[1] != n_boxes and scores.shape[2] == n_boxes:
                layout = "CN"
            elif not isinstance(n_boxes, int) and scores.shape[1] < scores.shape[2]:
                layout = "CN"  # dynamic shapes: there are far more priors than classes
            else:
                layout = "NC"
            self.cfg.scores_layout = layout
        return scores.transpose(0, 2, 1) if layout == "CN" else scores


def decode_raw(maps: list[np.ndarray], strides: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """RTMDet-R head maps (cls, reg, angle per level) -> boxes (N, 5), scores (N, C) over every
    cell, level after level, row-major (the order Grad-CAM uses to find a detection's score).

    reg = distances left/top/right/bottom already multiplied by the stride; the prior is the
    cell corner (x * stride, y * stride); angle in le90."""
    boxes, scores = [], []
    for (cls, reg, ang), stride in zip(zip(maps[0::3], maps[1::3], maps[2::3]), strides):
        c, h, w = cls.shape[1:]
        scores.append(1 / (1 + np.exp(-cls[0].reshape(c, -1).T)))
        left, top, right, bottom = reg[0].reshape(4, -1)
        angle = ang[0].reshape(-1)
        yy, xx = np.divmod(np.arange(h * w), w)
        cos, sin = np.cos(angle), np.sin(angle)
        dx, dy = (right - left) / 2, (bottom - top) / 2
        cx = xx * stride + cos * dx - sin * dy
        cy = yy * stride + sin * dx + cos * dy
        angle = (angle + np.pi / 2) % np.pi - np.pi / 2
        boxes.append(np.stack([cx, cy, left + right, top + bottom, angle], 1))
    return np.concatenate(boxes).astype(np.float32), np.concatenate(scores).astype(np.float32)


# ---------------------------------------------------------------------- post-processing
def _cv_rect(b: np.ndarray):
    return (float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(np.degrees(b[4]))


def rotated_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one (5,) box with (n, 5) boxes."""
    ious = np.zeros(len(others), np.float32)
    # boxes whose circumscribed circles do not touch cannot overlap
    r = np.hypot(box[2], box[3]) / 2
    ro = np.hypot(others[:, 2], others[:, 3]) / 2
    near = np.hypot(others[:, 0] - box[0], others[:, 1] - box[1]) < r + ro
    area = box[2] * box[3]
    rect = _cv_rect(box)
    for i in np.nonzero(near)[0]:
        o = others[i]
        kind, pts = cv2.rotatedRectangleIntersection(rect, _cv_rect(o))
        if kind != cv2.INTERSECT_NONE and pts is not None and len(pts) >= 3:
            inter = cv2.contourArea(cv2.convexHull(pts))
            ious[i] = inter / max(area + o[2] * o[3] - inter, 1e-9)
    return ious


def nms_rotated(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        i, order = order[0], order[1:]
        keep.append(i)
        if len(order):
            order = order[rotated_iou(boxes[i], boxes[order]) <= iou_thr]
    return np.asarray(keep, np.int64)


def postprocess(raw: RawOutput, score_thr: float, nms_iou: float, max_det: int = 100,
                nms_pre: int = 1000) -> Detections:
    labels = raw.scores.argmax(1)
    scores = raw.scores.max(1)
    idx = np.nonzero(scores > score_thr)[0]
    if len(idx) > nms_pre:
        idx = idx[np.argsort(-scores[idx])[:nms_pre]]
    boxes, scores, labels = raw.boxes[idx], scores[idx], labels[idx]
    kept = []
    for c in np.unique(labels):
        sel = np.nonzero(labels == c)[0]
        kept.append(sel[nms_rotated(boxes[sel], scores[sel], nms_iou)])
    kept = np.concatenate(kept) if kept else np.zeros(0, np.int64)
    kept = kept[np.argsort(-scores[kept], kind="stable")][:max_det]
    out = boxes[kept].astype(np.float32).copy()
    out[:, :4] /= raw.scale
    return Detections(out, scores[kept].astype(np.float32), labels[kept], idx[kept])


def corners(boxes: np.ndarray) -> np.ndarray:
    """(n, 5) -> (n, 4, 2); angle in radians, clockwise with y down (same in the 3 repos)."""
    cx, cy, w, h, a = boxes.T
    cos, sin = np.cos(a), np.sin(a)
    v1 = np.stack([w / 2 * cos, w / 2 * sin], 1)
    v2 = np.stack([-h / 2 * sin, h / 2 * cos], 1)
    c = np.stack([cx, cy], 1)
    return np.stack([c + v1 + v2, c + v1 - v2, c - v1 - v2, c - v1 + v2], 1)
