"""Reference deployment of an exported YOLOX-OBB model: onnxruntime + numpy + OpenCV, no PyTorch.

This is the code a client has to reproduce: ``letterbox`` before the network,
``postprocess`` after it (score threshold, rotated NMS, back to original pixels).
The input size and class names are read from the ``.metadata.json`` next to the model.

    python src/yolox_mps/onnx_example.py model_640.onnx photo.jpg other.heic --out onnx_vis
    python src/yolox_mps/onnx_example.py model_640.onnx photos/ --check best_epoch_100.pth
"""
import argparse
import glob
import json
import os
import os.path as osp

import cv2
import numpy as np
import onnxruntime as ort

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.heic', '.heif')


def read_image(path: str) -> np.ndarray:
    """BGR uint8 with the EXIF orientation applied (cv2 does it for JPEG; HEIC through pillow-heif)."""
    if path.lower().endswith(('.heic', '.heif')):
        import pillow_heif
        from PIL import Image, ImageOps
        pillow_heif.register_heif_opener()
        return cv2.cvtColor(np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB')), cv2.COLOR_RGB2BGR)
    return cv2.imread(path, cv2.IMREAD_COLOR)


def letterbox(img: np.ndarray, size: int):
    """Resize keeping the aspect ratio (long side = size), paste top-left on a 114 canvas.
    Returns the (1, 3, size, size) float32 BGR 0..255 tensor and the scale."""
    h0, w0 = img.shape[:2]
    scale = size / max(h0, w0)
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (round(w0 * scale), round(h0 * scale)), interpolation=interp)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img
    return canvas.transpose(2, 0, 1)[None].astype(np.float32), scale


def rotated_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one (5,) box with (n, 5) boxes; angles in radians, clockwise."""
    def rect(b):
        return (float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(np.degrees(b[4]))
    area = box[2] * box[3]
    ious = np.zeros(len(others), np.float32)
    for i, o in enumerate(others):
        kind, pts = cv2.rotatedRectangleIntersection(rect(box), rect(o))
        if kind != cv2.INTERSECT_NONE and pts is not None:
            inter = cv2.contourArea(cv2.convexHull(pts))
            ious[i] = inter / max(area + o[2] * o[3] - inter, 1e-9)
    return ious


def nms_rotated(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy rotated NMS, returns the kept indices sorted by score."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i, order = order[0], order[1:]
        keep.append(i)
        if len(order):
            order = order[rotated_iou(boxes[i], boxes[order]) <= iou_thr]
    return np.array(keep, np.int64)


def postprocess(boxes: np.ndarray, scores: np.ndarray, scale: float, score_thr=0.5, nms_iou=0.3,
                max_det=100, nms_pre=1000):
    """Network outputs of one image ``(N, 5)``, ``(N, C)`` -> boxes in original pixels, scores, labels."""
    labels = scores.argmax(1)
    scores = scores.max(1)
    keep = scores > score_thr
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if len(scores) > nms_pre:
        top = np.argsort(-scores)[:nms_pre]
        boxes, scores, labels = boxes[top], scores[top], labels[top]
    kept = []
    for c in np.unique(labels):  # class-aware NMS
        idx = np.nonzero(labels == c)[0]
        kept.append(idx[nms_rotated(boxes[idx], scores[idx], nms_iou)])
    kept = np.concatenate(kept) if kept else np.zeros(0, np.int64)
    kept = kept[np.argsort(-scores[kept])][:max_det]
    boxes = boxes[kept].copy()
    boxes[:, :4] /= scale
    return boxes, scores[kept], labels[kept]


def corners(boxes: np.ndarray) -> np.ndarray:
    """(n, 5) -> (n, 4, 2) polygon corners."""
    cx, cy, w, h, a = boxes.T
    cos, sin = np.cos(a), np.sin(a)
    v1 = np.stack([w / 2 * cos, w / 2 * sin], 1)
    v2 = np.stack([-h / 2 * sin, h / 2 * cos], 1)
    c = np.stack([cx, cy], 1)
    return np.stack([c + v1 + v2, c + v1 - v2, c - v1 - v2, c - v1 + v2], 1)


class Detector:

    def __init__(self, onnx_path: str):
        with open(osp.splitext(onnx_path)[0] + '.metadata.json', encoding='utf-8') as f:
            self.meta = json.load(f)
        self.size = self.meta['input']['shape'][-1]
        self.class_names = self.meta['class_names']
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    def __call__(self, img: np.ndarray, score_thr=None, nms_iou=None):
        rec = self.meta['recommended']
        x, scale = letterbox(img, self.size)
        boxes, scores = self.session.run(None, {self.meta['input']['name']: x})
        return postprocess(boxes[0], scores[0], scale, rec['score_thr'] if score_thr is None else score_thr,
                           rec['nms_iou'] if nms_iou is None else nms_iou, rec['max_detections'])


def check_against_pytorch(det: Detector, img: np.ndarray, checkpoint: str, score_thr: float, nms_iou: float):
    """Same image through ``OBBDetector.predict`` (the training code): the detections must match."""
    import sys
    import torch
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from yolox_obb import build_model, checkpoint_arch, load_state_dict_file
    model = build_model(checkpoint_arch(checkpoint), len(det.class_names))
    model.load_state_dict(load_state_dict_file(checkpoint))
    model.eval()
    x, scale = letterbox(img, det.size)
    ref = model.predict(torch.from_numpy(x), score_thr=score_thr, nms_iou=nms_iou, max_per_img=100)[0]
    rb = ref['boxes'].numpy().copy()
    rb[:, :4] /= scale
    ob, os_, _ = det(img, score_thr, nms_iou)
    if len(ob) != len(rb):
        return f'MISMATCH: {len(ob)} ONNX detections vs {len(rb)} PyTorch'
    if not len(ob):
        return 'ok (no detections in both)'
    o, r = np.argsort(-os_), np.argsort(-ref['scores'].numpy())
    return (f'ok: max box diff {np.abs(ob[o, :4] - rb[r, :4]).max():.3f} px, '
            f'max score diff {np.abs(os_[o] - ref["scores"].numpy()[r]).max():.1e}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('onnx')
    p.add_argument('images', nargs='+', help='image files or folders')
    p.add_argument('--score-thr', type=float, help='default: recommended value of the metadata')
    p.add_argument('--nms-iou', type=float, help='default: recommended value of the metadata')
    p.add_argument('--out', help='folder for the annotated images')
    p.add_argument('--check', metavar='CHECKPOINT', help='compare every image with the PyTorch model')
    args = p.parse_args()

    det = Detector(args.onnx)
    files = []
    for path in args.images:
        files += (sorted(f for f in glob.glob(osp.join(path, '*')) if f.lower().endswith(IMG_EXTS))
                  if osp.isdir(path) else [path])
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    rec = det.meta['recommended']
    score_thr = rec['score_thr'] if args.score_thr is None else args.score_thr
    nms_iou = rec['nms_iou'] if args.nms_iou is None else args.nms_iou
    for path in files:
        img = read_image(path)
        boxes, scores, labels = det(img, score_thr, nms_iou)
        print(f'{osp.basename(path)}: {len(boxes)} detections')
        for b, s, l in zip(boxes, scores, labels):
            print(f'  {det.class_names[l]} {s:.3f}  cx={b[0]:.1f} cy={b[1]:.1f} w={b[2]:.1f} h={b[3]:.1f} '
                  f'angle={np.degrees(b[4]):.1f}deg')
        if args.check:
            print('  check vs PyTorch: ' + check_against_pytorch(det, img, args.check, score_thr, nms_iou))
        if args.out:
            k = max(1, min(img.shape[:2]) // 400)
            for poly, s, l in zip(corners(boxes).astype(np.int32), scores, labels):
                cv2.polylines(img, [poly], True, (0, 255, 0), 2 * k, cv2.LINE_AA)
                cv2.putText(img, f'{det.class_names[l]} {s:.2f}', tuple(int(v) for v in poly.min(0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (0, 255, 0), k, cv2.LINE_AA)
            cv2.imwrite(osp.join(args.out, osp.splitext(osp.basename(path))[0] + '.jpg'), img)


if __name__ == '__main__':
    main()
