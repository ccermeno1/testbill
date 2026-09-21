"""Run a trained RTMDet-R on a folder of images (or one image).

Writes one visualisation per image and ``predictions.json`` with, per image, the
detected oriented boxes ``[cx, cy, w, h, angle_rad]`` (original pixel coords), the
4-corner polygon and the score - same format as ``tools/infer.py`` of the mmrotate project.

    python infer.py work_dirs/rtmdet_tiny_banknotes/best_epoch_XX.pth photos/ --out work_dirs/infer --score-thr 0.5
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from rtmdet_obb import RTMDetR, load_state_dict_file  # noqa: E402
from rtmdet_obb.boxes import rbox2poly  # noqa: E402
from rtmdet_obb.data import IMG_EXTS, letterbox_image  # noqa: E402
from rtmdet_obb.engine import pick_device  # noqa: E402


def imread_any(path):
    """cv2.imread plus HEIC/HEIF through pillow-heif when available (EXIF orientation applied)."""
    if path.lower().endswith(('.heic', '.heif')):
        from PIL import Image, ImageOps
        import pillow_heif
        pillow_heif.register_heif_opener()
        im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return cv2.imread(path, cv2.IMREAD_COLOR)


def draw(img, boxes, scores, labels, class_names, thickness=2, font_scale=0.6, color=(0, 255, 0)):
    polys = rbox2poly(boxes).reshape(-1, 4, 2).numpy().astype(np.int32)
    for poly, s, l in zip(polys, scores.tolist(), labels.tolist()):
        cv2.polylines(img, [poly], True, color, thickness, cv2.LINE_AA)
        x, y = poly[:, 0].min(), poly[:, 1].min()
        cv2.putText(img, f'{class_names[l]} {s:.2f}', (int(x), max(int(y) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, max(1, thickness // 2), cv2.LINE_AA)
    return img


def load_gt(img_path, gt_dir, class_names, w, h):
    """Ground truth for one image: YOLO-OBB (``<cls> x1..y4`` normalised) or DOTA (pixels + class name)."""
    from rtmdet_obb.boxes import poly2rbox
    stem = osp.splitext(osp.basename(img_path))[0]
    path = osp.join(gt_dir, stem + '.txt')
    polys, labels = [], []
    if osp.exists(path):
        for raw in open(path):
            tok = raw.split()
            if len(tok) >= 9 and tok[8] in class_names:  # DOTA
                polys.append(np.array(tok[:8], np.float32))
                labels.append(class_names.index(tok[8]))
            elif len(tok) >= 9:  # YOLO-OBB
                c = np.clip(np.array(tok[1:9], np.float32), 0, 1)
                c[0::2] *= w
                c[1::2] *= h
                polys.append(c)
                labels.append(int(tok[0]))
    if not polys:
        return torch.zeros((0, 5)), torch.zeros((0,), dtype=torch.long)
    return torch.from_numpy(poly2rbox(np.stack(polys))), torch.tensor(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('images', help='image file or folder')
    p.add_argument('--out', default='work_dirs/infer')
    p.add_argument('--img-size', type=int, default=640, help='long side at inference (1024 helps on large photos)')
    p.add_argument('--score-thr', type=float, default=0.3)
    p.add_argument('--nms-iou', type=float, default=0.5)
    p.add_argument('--classes', default='euro_banknote', help='comma separated class names')
    p.add_argument('--size', default='tiny', choices=['tiny', 's', 'm', 'l'])
    p.add_argument('--device', default='auto')
    p.add_argument('--gt', help='folder with ground-truth .txt per image (YOLO-OBB labels/ or DOTA annfiles/); drawn in red')
    args = p.parse_args()

    class_names = args.classes.split(',')
    device = pick_device(args.device)
    model = RTMDetR(num_classes=len(class_names), size=args.size)
    model.load_state_dict(load_state_dict_file(args.checkpoint))
    model.to(device).eval()

    if osp.isdir(args.images):
        files = sorted(f for f in glob.glob(osp.join(args.images, '*')) if f.lower().endswith(IMG_EXTS + ('.heic', '.heif')))
    else:
        files = [args.images]
    os.makedirs(args.out, exist_ok=True)

    predictions = {}
    for path in files:
        img = imread_any(path)
        tensor, scale = letterbox_image(img, args.img_size)
        with torch.no_grad():
            res = model.predict(tensor[None].to(device), score_thr=args.score_thr, nms_iou=args.nms_iou, max_per_img=100)[0]
        boxes = res['boxes'].cpu()
        boxes[:, :4] /= scale  # back to original pixels
        scores, labels = res['scores'].cpu(), res['labels'].cpu()
        polys = rbox2poly(boxes)
        predictions[osp.basename(path)] = [
            dict(label=int(l), class_name=class_names[int(l)], score=float(s), obb=[float(v) for v in b],
                 polygon=[[float(poly[i]), float(poly[i + 1])] for i in range(0, 8, 2)])
            for b, s, l, poly in zip(boxes, scores, labels, polys)]
        k = max(1, min(img.shape[:2]) // 400)
        vis = img.copy()
        if args.gt:
            gb, gl = load_gt(path, args.gt, class_names, img.shape[1], img.shape[0])
            vis = draw(vis, gb, torch.ones(len(gb)), gl, ['GT'] * len(class_names), thickness=2 * k,
                       font_scale=0.6 * k, color=(0, 0, 255))
        vis = draw(vis, boxes, scores, labels, class_names, thickness=2 * k, font_scale=0.6 * k)
        cv2.imwrite(osp.join(args.out, osp.splitext(osp.basename(path))[0] + '.jpg'), vis)
        print(f'{osp.basename(path)}: {len(scores)} detections')

    with open(osp.join(args.out, 'predictions.json'), 'w') as f:
        json.dump(predictions, f, indent=2)
    n = sum(len(v) for v in predictions.values())
    print(f'{len(files)} images, {n} detections >= {args.score_thr} written to {args.out}')


if __name__ == '__main__':
    main()
