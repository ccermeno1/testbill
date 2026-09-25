#!/usr/bin/env python
"""Preprocessing, NMS and post-processing outside the network, for an exported ONNX model.

The exported graph handles the forward pass and the box decode, since both depend only on
shapes. What it does NOT cover is the score filter and the NMS loop, which depend on values.
Those are the ~40 lines in this file, and they are exactly what a client (mobile, C++, ...)
has to reproduce to get the same results.

Run it to check the numbers against the PyTorch model::

    python src/ppyoloer_mps/onnx_example.py <image> \
        --onnx models/ppyoloe_r/ppyoloe_r_s_banknotes_640.onnx \
        --checkpoint models/ppyoloe_r/ppyoloe_r_s_banknotes_torch.pt
"""
from __future__ import annotations

import argparse
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0


# --------------------------------------------------------------------------- pre-proceso
def preprocess(path: str, img_size: int = 640):
    """Image -> (1, 3, H, W) float32 RGB in 0..255, plus the scale that was applied.

    Normalisation lives INSIDE the graph, so this only resizes and pads. Downscaling uses
    INTER_AREA: bilinear aliases on large photos and that costs a noticeable amount of mAP.
    """
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    scale = min(img_size / min(h0, w0), img_size / max(h0, w0))
    rh, rw = int(scale * h0 + 0.5), int(scale * w0 + 0.5)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (rw, rh), interpolation=interp)
    # zero padding up to a multiple of 32 (the network is fully convolutional)
    ph, pw = int(np.ceil(rh / 32) * 32), int(np.ceil(rw / 32) * 32)
    canvas = np.zeros((ph, pw, 3), np.uint8)
    canvas[:rh, :rw] = img
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    return x, rw / w0, rh / h0


# --------------------------------------------------------------------------- post-proceso
def rbox2poly(boxes: np.ndarray) -> np.ndarray:
    """(n, 5) -> (n, 8). Angle in radians, y pointing down."""
    cx, cy, w, h, a = boxes.T
    cos, sin = np.cos(a), np.sin(a)
    # local corners in PP-YOLOE-R order, rotated by [[cos, sin], [-sin, cos]]:
    #   x = cx + dx*cos - dy*sin ;  y = cy + dx*sin + dy*cos
    dx = np.array([0.5, 0.5, -0.5, -0.5])[:, None] * w
    dy = np.array([-0.5, 0.5, 0.5, -0.5])[:, None] * h
    x = cx + dx * cos - dy * sin
    y = cy + dx * sin + dy * cos
    return np.stack([x, y], -1).transpose(1, 0, 2).reshape(-1, 8)


def rotated_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact rotated IoU (a: (n,5), b: (m,5)) -> (n, m), by clipping convex polygons."""
    pa, pb = rbox2poly(a).reshape(-1, 4, 2), rbox2poly(b).reshape(-1, 4, 2)
    area_a = np.abs(a[:, 2] * a[:, 3])
    area_b = np.abs(b[:, 2] * b[:, 3])
    out = np.zeros((len(a), len(b)), np.float32)
    for i, poly_a in enumerate(pa):
        for j, poly_b in enumerate(pb):
            inter, _ = cv2.intersectConvexConvex(poly_a.astype(np.float32), poly_b.astype(np.float32))
            union = area_a[i] + area_b[j] - inter
            out[i, j] = inter / union if union > 0 else 0.0
    return out


def rotated_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS over rotated boxes. Returns indices, ordered by score."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = rotated_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return keep


def postprocess(scores: np.ndarray, boxes: np.ndarray, scale_x: float, scale_y: float,
                score_thr: float = 0.05, nms_iou: float = 0.1,
                nms_pre: int = 2000, max_per_img: int = 2000):
    """Graph output -> detections in ORIGINAL image coordinates.

    scores (1, C, L), boxes (1, L, 5). Returns (rboxes, polys, scores, labels).
    """
    scores, boxes = scores[0], boxes[0]           # (C, L), (L, 5)
    keep_boxes, keep_scores, keep_labels = [], [], []
    for cls in range(scores.shape[0]):
        sc = scores[cls]
        sel = np.nonzero(sc >= score_thr)[0]
        if sel.size == 0:
            continue
        if sel.size > nms_pre:                     # nms_pre caps the cost of NMS
            sel = sel[np.argsort(-sc[sel])[:nms_pre]]
        idx = rotated_nms(boxes[sel], sc[sel], nms_iou)
        keep_boxes.append(boxes[sel][idx])
        keep_scores.append(sc[sel][idx])
        keep_labels.append(np.full(len(idx), cls, np.int64))
    if not keep_scores:
        return np.zeros((0, 5), np.float32), np.zeros((0, 8), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)

    rb = np.concatenate(keep_boxes)
    sc = np.concatenate(keep_scores)
    lb = np.concatenate(keep_labels)
    order = np.argsort(-sc)[:max_per_img]          # max_per_img caps detections after NMS
    rb, sc, lb = rb[order], sc[order], lb[order]
    # back to the original image: undo the resize scale. Padding sits at the top-left, so
    # there is no offset to correct
    rb = np.stack([rb[:, 0] / scale_x, rb[:, 1] / scale_y,
                   rb[:, 2] / scale_x, rb[:, 3] / scale_y, rb[:, 4]], axis=1)
    return rb, rbox2poly(rb), sc, lb


# --------------------------------------------------------------------------- comprobacion
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--checkpoint", default=None, help=".pt checkpoint to cross-check against")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.1)
    args = ap.parse_args()

    import onnxruntime as ort

    x, sx, sy = preprocess(args.image, args.img_size)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    scores, boxes = sess.run(None, {sess.get_inputs()[0].name: x})
    rb, polys, sc, lb = postprocess(scores, boxes, sx, sy, score_thr=args.conf, nms_iou=args.nms_iou)

    print(f"ONNX: {len(sc)} detections (conf >= {args.conf}, NMS IoU {args.nms_iou})")
    for i in range(min(5, len(sc))):
        print(f"  score {sc[i]:.4f}  rbox {np.round(rb[i], 1)}")

    if args.checkpoint:
        import torch

        from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r, load_checkpoint
        from ppyoloer_mps.ppyoloe_obb.ops import batched_postprocess

        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})
        model = build_ppyoloe_r(num_classes=meta.get("num_classes", 1), size=meta.get("size", "s"))
        load_checkpoint(model, args.checkpoint)
        model.eval()
        xt = torch.from_numpy((x - IMAGENET_MEAN.reshape(1, 3, 1, 1)) / IMAGENET_STD.reshape(1, 3, 1, 1)).float()
        with torch.no_grad():
            s_t, b_t = model(xt)
        det = batched_postprocess(
            s_t, b_t, score_threshold=args.conf, nms_threshold=args.nms_iou,
            scale_factor=torch.tensor([[sy, sx]], dtype=torch.float32),
        )[0]
        rb_t = det["rboxes"].numpy()
        print(f"PyTorch: {len(det['scores'])} detections")
        if len(rb_t) == len(rb) and len(rb):
            print(f"  max box difference: {np.abs(np.sort(rb_t, 0) - np.sort(rb, 0)).max():.4f} px")
            print(f"  max score difference: {np.abs(np.sort(det['scores'].numpy()) - np.sort(sc)).max():.6f}")
        elif len(rb_t) != len(rb):
            print("  WARNING: ONNX and PyTorch returned a different number of detections")


if __name__ == "__main__":
    main()
