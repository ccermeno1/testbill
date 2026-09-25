"""Preprocessing + decode + rotated NMS outside the network, for an exported model.

``model.forward()`` exports to ONNX cleanly (it is only convolutions); ``predict()`` does not,
because the score filter and the NMS loop depend on the values, not just on the shapes. So the
graph gives you the raw head outputs and you do these ~40 lines yourself. Run this file to check
the numbers against ``RTMDetR.predict``::

    python onnx_example.py <image> [checkpoint]
"""
import os.path as osp
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from rtmdet_obb.ops import batched_nms_rotated  # noqa: E402

STRIDES = (8, 16, 32)  # one per head level


def preprocess(path, img_size=800):
    """BGR image -> (1, 3, S, S) float32 in 0..255 plus the scale used.

    The mean/std normalisation lives *inside* the network, so the exported graph expects raw
    0..255 BGR values. Downscaling uses INTER_AREA: plain bilinear aliases on big photos.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h0, w0 = img.shape[:2]
    scale = img_size / max(h0, w0)
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (round(w0 * scale), round(h0 * scale)), interpolation=interp)
    canvas = np.full((img_size, img_size, 3), 114, np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img
    return canvas.transpose(2, 0, 1)[None].astype(np.float32), scale


def decode(cls_scores, bbox_preds, angle_preds, score_thr=0.5):
    """Head outputs (lists of numpy arrays, one per level) -> boxes (n, 5), scores, labels.

    Per cell the head predicts 4 distances to the edges (already multiplied by the stride) and
    an angle; the box centre is the cell centre displaced by half the difference of the
    opposite distances, rotated by that angle.
    """
    boxes, scores, labels = [], [], []
    for cls, reg, ang, stride in zip(cls_scores, bbox_preds, angle_preds, STRIDES):
        num_classes, h, w = cls.shape[1:]
        prob = 1 / (1 + np.exp(-cls[0]))                       # sigmoid -> (C, H, W)
        keep = prob.max(0) >= score_thr                        # (H, W)
        if not keep.any():
            continue
        yy, xx = np.nonzero(keep)
        left, top, right, bottom = reg[0][:, yy, xx]           # distances, in pixels
        angle = ang[0][0, yy, xx]
        cos, sin = np.cos(angle), np.sin(angle)
        dx, dy = (right - left) / 2, (bottom - top) / 2        # offset in the box frame
        cx = xx * stride + cos * dx - sin * dy
        cy = yy * stride + sin * dx + cos * dy
        boxes.append(np.stack([cx, cy, left + right, top + bottom, angle], 1))
        scores.append(prob[:, yy, xx].max(0))
        labels.append(prob[:, yy, xx].argmax(0))
    if not boxes:
        return np.zeros((0, 5), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(labels)


def postprocess(cls_scores, bbox_preds, angle_preds, scale, score_thr=0.5, nms_iou=0.3):
    """Decode, rotated NMS, and back to the original image coordinates."""
    boxes, scores, labels = decode(cls_scores, bbox_preds, angle_preds, score_thr)
    if len(boxes) == 0:
        return boxes, scores, labels
    b = torch.from_numpy(boxes.astype(np.float32))
    _, keep = batched_nms_rotated(b, torch.from_numpy(scores), torch.from_numpy(labels), nms_iou)
    keep = keep.numpy()
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    boxes[:, :4] /= scale                                      # undo the letterbox scaling
    return boxes, scores, labels


def main():
    from rtmdet_obb import RTMDetR, load_state_dict_file
    image = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/rtmdet_r_tiny_banknotes_C_strongaug.pth'
    model = RTMDetR(num_classes=1)
    model.load_state_dict(load_state_dict_file(ckpt))
    model.eval()

    x, scale = preprocess(image)
    with torch.no_grad():
        cls_scores, bbox_preds, angle_preds = model(torch.from_numpy(x))
    boxes, scores, labels = postprocess(
        [c.numpy() for c in cls_scores], [b.numpy() for b in bbox_preds], [a.numpy() for a in angle_preds],
        scale)

    # reference: the same thing done by the package
    with torch.no_grad():
        ref = model.predict(torch.from_numpy(x), score_thr=0.5, nms_iou=0.3, max_per_img=100)[0]
    rb = ref['boxes'].numpy().copy()
    rb[:, :4] /= scale
    print(f'{osp.basename(image)}: {len(boxes)} detections (reference: {len(rb)})')
    if len(boxes) == len(rb) and len(boxes):
        order, rorder = np.argsort(-scores), np.argsort(-ref['scores'].numpy())
        print(f'  max box difference {np.abs(boxes[order] - rb[rorder]).max():.4f} px, '
              f'max score difference {np.abs(scores[order] - ref["scores"].numpy()[rorder]).max():.2e}')
    for b, s in zip(boxes, scores):
        print(f'  cx={b[0]:7.1f} cy={b[1]:7.1f} w={b[2]:6.1f} h={b[3]:6.1f} angle={np.degrees(b[4]):6.1f}deg score={s:.3f}')


if __name__ == '__main__':
    main()
