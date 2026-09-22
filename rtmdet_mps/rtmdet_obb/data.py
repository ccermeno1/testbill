"""YOLOv8-OBB dataset (Roboflow export) -> tensors, with the RTMDet-R DOTA augmentations.

Label line: ``<cls> x1 y1 x2 y2 x3 y3 x4 y4`` normalised to [0, 1].

Train pipeline (mmrotate ``dota_rr.py``): resize keep-ratio to ``img_size`` ->
random flip (h / v / diagonal, p=0.75) -> random rotation (p=0.5, +-180 deg) ->
pad to ``img_size`` x ``img_size`` with 114. Images stay BGR (cv2) like the DOTA
checkpoint expects.
"""
import glob
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# OpenCV's own thread pool fights the DataLoader workers (and crashes them on macOS);
# mmrotate sets opencv_num_threads=0 for the same reason.
cv2.setNumThreads(0)

from .boxes import flip_rboxes, poly2rbox, rotate_rboxes

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def base_id(stem: str) -> str:
    """Strip the Roboflow hash: 'x_jpg.rf.ABC' -> 'x_jpg'."""
    return stem.split('.rf.')[0]


def read_split(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def index_yolo_export(root: str) -> Dict[str, dict]:
    """Map base id -> dict(img, label) over the train/valid/test folders of a Roboflow export."""
    index = {}
    for sub in ('train', 'valid', 'test'):
        for img in sorted(glob.glob(osp.join(root, sub, 'images', '*'))):
            if not img.lower().endswith(IMG_EXTS):
                continue
            stem = osp.splitext(osp.basename(img))[0]
            lbl = osp.join(root, sub, 'labels', stem + '.txt')
            bid = base_id(stem)
            if bid in index:
                raise RuntimeError(f'duplicate base id {bid}: {img} and {index[bid]["img"]}')
            index[bid] = dict(img=img, label=lbl if osp.exists(lbl) else None, split=sub)
    return index


def read_yolo_obb(label_path: Optional[str], width: int, height: int):
    """Return (n, 5) le90 boxes in pixels and (n,) labels."""
    if label_path is None or not osp.exists(label_path):
        return np.zeros((0, 5), np.float32), np.zeros((0,), np.int64)
    polys, labels = [], []
    with open(label_path) as f:
        for raw in f:
            tok = raw.split()
            if len(tok) < 9:
                continue
            coords = np.clip(np.array(tok[1:9], dtype=np.float32), 0.0, 1.0)
            coords[0::2] *= width
            coords[1::2] *= height
            polys.append(coords)
            labels.append(int(tok[0]))
    if not polys:
        return np.zeros((0, 5), np.float32), np.zeros((0,), np.int64)
    return poly2rbox(np.stack(polys)), np.array(labels, dtype=np.int64)


class StrongAug:
    """RTMDet "aug" recipe (mmrotate ``rotated_rtmdet_l-100e-aug-dota``):
    mosaic of 4 images on a 2x canvas -> random resize -> random rotate -> random crop
    -> HSV jitter -> random flip -> pad -> mixup. ``stage2=True`` is the light pipeline
    used for the last epochs (resize 0.9-1.1, rotate, flip, pad only).

    As in mmdet, mosaic and mixup take their extra samples from caches of recent ones
    (``CachedMosaic`` / ``CachedMixUp``), so each ``__getitem__`` decodes a single image."""

    def __init__(self, mosaic_prob=1.0, mixup_prob=0.5, resize_range=(0.5, 1.5), stage2_resize_range=(0.9, 1.1),
                 rotate_prob=0.5, rotate_range=180.0, flip_prob=0.5, hsv=(5, 30, 30), stage2=False,
                 mosaic_max_cached=40, mixup_max_cached=20):
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.resize_range = tuple(resize_range)
        self.stage2_resize_range = tuple(stage2_resize_range)
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.flip_prob = flip_prob
        self.hsv = tuple(hsv)
        self.stage2 = stage2
        self.mosaic_max_cached = mosaic_max_cached
        self.mixup_max_cached = mixup_max_cached


class YoloObbDataset(Dataset):
    """
    Args:
        root: Roboflow export folder (contains train/valid/test).
        split: 'train' | 'valid' | 'test' - uses the export's own folders, or, if
            ``split_dir`` is given, the ids listed in ``<split_dir>/<split>.txt``
            (matched by base name, hash-insensitive).
        img_size: long side after resize; the image is padded to a square of this size.
        train: apply augmentations.
        extra_dirs: flat folders (``<dir>/images``, ``<dir>/labels``) appended to the
            split, e.g. an offline-augmented copy of the training set.
    """

    def __init__(self, root: str, split: str, img_size: int = 640, train: bool = False,
                 split_dir: Optional[str] = None, flip_prob: float = 0.75, rotate_prob: float = 0.5,
                 rotate_range: float = 180.0, filter_empty: bool = True, extra_dirs: Sequence[str] = (),
                 strong_aug: Optional[StrongAug] = None):
        self.root = root
        self.strong_aug = strong_aug
        self._mosaic_cache = []  # recently loaded samples, for the mosaic
        self._mixup_cache = []   # recently augmented samples, for the mixup
        self.img_size = img_size
        self.train = train
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        index = index_yolo_export(root)
        if split_dir:
            ids = [base_id(i) for i in read_split(osp.join(split_dir, f'{split}.txt'))]
            missing = [i for i in ids if i not in index]
            if missing:
                raise FileNotFoundError(f'{len(missing)} ids of {split}.txt not in {root}: {missing[:3]}')
            self.items = [dict(id=i, **index[i]) for i in ids]
        else:
            self.items = [dict(id=k, **v) for k, v in index.items() if v['split'] == split]
        for d in extra_dirs:
            n0 = len(self.items)
            for img in sorted(glob.glob(osp.join(d, 'images', '*'))):
                if not img.lower().endswith(IMG_EXTS):
                    continue
                stem = osp.splitext(osp.basename(img))[0]
                lbl = osp.join(d, 'labels', stem + '.txt')
                self.items.append(dict(id=stem, img=img, label=lbl if osp.exists(lbl) else None, split='extra'))
            if len(self.items) == n0:
                raise RuntimeError(f'no images found in {d}/images')
        if filter_empty and train:
            kept = []
            for it in self.items:
                if it['label'] and os.path.getsize(it['label']) > 0:
                    kept.append(it)
            self.items = kept
        if not self.items:
            raise RuntimeError(f'no images for split {split!r} in {root}')

    def __len__(self):
        return len(self.items)

    def load(self, i: int):
        it = self.items[i]
        img = cv2.imread(it['img'], cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(it['img'])
        h, w = img.shape[:2]
        boxes, labels = read_yolo_obb(it['label'], w, h)
        return img, boxes, labels

    def __getitem__(self, i: int):
        if self.train and self.strong_aug is not None:
            return self._getitem_strong(i)
        img, boxes, labels = self.load(i)
        h0, w0 = img.shape[:2]
        scale = self.img_size / max(h0, w0)
        if scale != 1.0:
            img = cv2.resize(img, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_LINEAR)
            boxes[:, :4] *= scale
        h, w = img.shape[:2]

        if self.train:
            r = np.random.rand()
            if r < self.flip_prob:
                direction = ['horizontal', 'vertical', 'diagonal'][int(r / self.flip_prob * 3)]
                img = _flip_img(img, direction)
                boxes = flip_rboxes(boxes, (h, w), direction)
            if np.random.rand() < self.rotate_prob:
                angle = self.rotate_range * (2 * np.random.rand() - 1)
                center = ((w - 1) * 0.5, (h - 1) * 0.5)
                m = cv2.getRotationMatrix2D(center, -angle, 1.0)
                img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
                boxes = rotate_rboxes(boxes, center, angle)
                inside = (boxes[:, 0] >= 0) & (boxes[:, 0] < w) & (boxes[:, 1] >= 0) & (boxes[:, 1] < h)
                boxes, labels = boxes[inside], labels[inside]

        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        canvas[:h, :w] = img
        return dict(
            image=torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))),
            boxes=torch.from_numpy(boxes.astype(np.float32)),
            labels=torch.from_numpy(labels.astype(np.int64)),
            scale=scale, id=self.items[i]['id'], path=self.items[i]['img'], ori_shape=(h0, w0))


    # ------------------------------------------------------------------ strong augmentation
    def _load_scaled(self, i: int):
        """Image + boxes resized keep-ratio so the long side is img_size."""
        img, boxes, labels = self.load(i)
        h0, w0 = img.shape[:2]
        scale = self.img_size / max(h0, w0)
        if scale != 1.0:
            img = cv2.resize(img, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_LINEAR)
            boxes[:, :4] *= scale
        return img, boxes, labels, (h0, w0), scale

    def _cache_push(self, cache, item, max_cached):
        """mmdet's cache policy: append, then drop a random entry when over capacity."""
        cache.append(item)
        if len(cache) > max_cached:
            cache.pop(np.random.randint(len(cache)))

    def _mosaic(self, sample):
        """mmdet CachedMosaic: the sample plus 3 from the cache, around a random centre."""
        s = self.img_size
        canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        cx = int(np.random.uniform(0.5, 1.5) * s)
        cy = int(np.random.uniform(0.5, 1.5) * s)
        others = [self._mosaic_cache[np.random.randint(len(self._mosaic_cache))] for _ in range(3)]
        all_boxes, all_labels = [], []
        for loc, (img, boxes, labels) in zip(('top_left', 'top_right', 'bottom_left', 'bottom_right'),
                                             [sample] + others):
            h, w = img.shape[:2]
            if loc == 'top_left':
                x1, y1, x2, y2 = max(cx - w, 0), max(cy - h, 0), cx, cy
                crop = (w - (x2 - x1), h - (y2 - y1), w, h)
            elif loc == 'top_right':
                x1, y1, x2, y2 = cx, max(cy - h, 0), min(cx + w, 2 * s), cy
                crop = (0, h - (y2 - y1), min(w, x2 - x1), h)
            elif loc == 'bottom_left':
                x1, y1, x2, y2 = max(cx - w, 0), cy, cx, min(2 * s, cy + h)
                crop = (w - (x2 - x1), 0, w, min(y2 - y1, h))
            else:
                x1, y1, x2, y2 = cx, cy, min(cx + w, 2 * s), min(2 * s, cy + h)
                crop = (0, 0, min(w, x2 - x1), min(y2 - y1, h))
            canvas[y1:y2, x1:x2] = img[crop[1]:crop[3], crop[0]:crop[2]]
            boxes = boxes.copy()
            boxes[:, 0] += x1 - crop[0]
            boxes[:, 1] += y1 - crop[1]
            all_boxes.append(boxes)
            all_labels.append(labels)
        boxes = np.concatenate(all_boxes)
        labels = np.concatenate(all_labels)
        inside = (boxes[:, 0] >= 0) & (boxes[:, 0] < 2 * s) & (boxes[:, 1] >= 0) & (boxes[:, 1] < 2 * s)
        return canvas, boxes[inside], labels[inside]

    @staticmethod
    def _random_crop(img, boxes, labels, size):
        """Random size x size crop keeping boxes whose centre is inside; retried so that at
        least one box survives when the image had any (mmdet RandomCrop, allow_negative_crop=False)."""
        h, w = img.shape[:2]
        ch, cw = min(size, h), min(size, w)
        ox = oy = 0
        for attempt in range(10):
            if attempt < 9 or len(boxes) == 0:
                ox = np.random.randint(0, w - cw + 1)
                oy = np.random.randint(0, h - ch + 1)
            else:  # last try: centre the crop on a random box
                k = np.random.randint(len(boxes))
                ox = int(np.clip(boxes[k, 0] - cw / 2, 0, w - cw))
                oy = int(np.clip(boxes[k, 1] - ch / 2, 0, h - ch))
            b = boxes.copy()
            b[:, 0] -= ox
            b[:, 1] -= oy
            inside = (b[:, 0] >= 0) & (b[:, 0] < cw) & (b[:, 1] >= 0) & (b[:, 1] < ch)
            if len(boxes) == 0 or inside.any():
                break
        return img[oy:oy + ch, ox:ox + cw], b[inside], labels[inside]

    @staticmethod
    def _hsv_aug(img, deltas):
        """mmdet YOLOXHSVRandomAug."""
        gains = np.random.uniform(-1, 1, 3) * np.array(deltas)
        gains *= np.random.randint(0, 2, 3)
        gains = gains.astype(np.int16)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + gains[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + gains[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + gains[2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _strong(self, i: int):
        a = self.strong_aug
        s = self.img_size
        img, boxes, labels, _, _ = self._load_scaled(i)
        if not a.stage2:
            self._cache_push(self._mosaic_cache, (img, boxes, labels), a.mosaic_max_cached)
        # mmdet applies the mosaic only once the cache holds more than 4 samples
        if not a.stage2 and len(self._mosaic_cache) > 4 and np.random.rand() < a.mosaic_prob:
            img, boxes, labels = self._mosaic((img, boxes, labels))
            lo, hi = a.resize_range
        else:
            img, boxes, labels = img.copy(), boxes.copy(), labels.copy()
            lo, hi = a.stage2_resize_range if a.stage2 else a.resize_range
        f = np.random.uniform(lo, hi)
        if f != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_LINEAR)
            boxes[:, :4] *= f
        h, w = img.shape[:2]
        if np.random.rand() < a.rotate_prob:
            angle = a.rotate_range * (2 * np.random.rand() - 1)
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
            m = cv2.getRotationMatrix2D(center, -angle, 1.0)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            boxes = rotate_rboxes(boxes, center, angle)
            inside = (boxes[:, 0] >= 0) & (boxes[:, 0] < w) & (boxes[:, 1] >= 0) & (boxes[:, 1] < h)
            boxes, labels = boxes[inside], labels[inside]
        if not a.stage2:
            img, boxes, labels = self._random_crop(img, boxes, labels, s)
            img = self._hsv_aug(img, a.hsv)
        elif max(img.shape[:2]) > s:
            # stage 2 has no crop: make sure the (possibly 1.1x) image fits the canvas
            img, boxes, labels = self._random_crop(img, boxes, labels, s)
        h, w = img.shape[:2]
        r = np.random.rand()
        if r < a.flip_prob:
            direction = ['horizontal', 'vertical', 'diagonal'][int(r / a.flip_prob * 3)]
            img = _flip_img(img, direction)
            boxes = flip_rboxes(boxes, (h, w), direction)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        canvas[:h, :w] = img
        if not a.stage2:
            self._cache_push(self._mixup_cache, (canvas, boxes, labels), a.mixup_max_cached)
            if len(self._mixup_cache) > 4 and np.random.rand() < a.mixup_prob:
                # CachedMixUp (ratio 1.0): blend 50/50 with an earlier augmented sample
                other, oboxes, olabels = self._mixup_cache[np.random.randint(len(self._mixup_cache))]
                canvas = ((canvas.astype(np.float32) + other.astype(np.float32)) * 0.5).astype(np.uint8)
                boxes = np.concatenate([boxes, oboxes])
                labels = np.concatenate([labels, olabels])
        return canvas, boxes, labels

    def _getitem_strong(self, i: int):
        canvas, boxes, labels = self._strong(i)
        return dict(
            image=torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))),
            boxes=torch.from_numpy(boxes.astype(np.float32)),
            labels=torch.from_numpy(labels.astype(np.int64)),
            scale=1.0, id=self.items[i]['id'], path=self.items[i]['img'], ori_shape=(self.img_size, self.img_size))


class DotaObbDataset(YoloObbDataset):
    """Folder in DOTA layout: ``images/*.jpg`` + ``annfiles/*.txt`` with lines
    ``x1 y1 ... x4 y4 <class_name> <difficulty>`` in pixels (e.g. ``data/billetesprueba``)."""

    def __init__(self, root: str, class_names: Sequence[str], img_size: int = 640, train: bool = False, **kw):
        self.class_names = list(class_names)
        self.img_size = img_size
        self.train = train
        self.flip_prob = kw.get('flip_prob', 0.75)
        self.rotate_prob = kw.get('rotate_prob', 0.5)
        self.rotate_range = kw.get('rotate_range', 180.0)
        self.items = []
        for img in sorted(glob.glob(osp.join(root, 'images', '*'))):
            if img.lower().endswith(IMG_EXTS):
                stem = osp.splitext(osp.basename(img))[0]
                lbl = osp.join(root, 'annfiles', stem + '.txt')
                self.items.append(dict(id=stem, img=img, label=lbl if osp.exists(lbl) else None, split='dota'))
        if not self.items:
            raise RuntimeError(f'no images in {root}/images')

    def load(self, i: int):
        it = self.items[i]
        img = cv2.imread(it['img'], cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(it['img'])
        polys, labels = [], []
        if it['label']:
            with open(it['label']) as f:
                for raw in f:
                    tok = raw.split()
                    if len(tok) < 9:
                        continue
                    polys.append(np.array(tok[:8], dtype=np.float32))
                    labels.append(self.class_names.index(tok[8]))
        if polys:
            return img, poly2rbox(np.stack(polys)), np.array(labels, dtype=np.int64)
        return img, np.zeros((0, 5), np.float32), np.zeros((0,), np.int64)


def _flip_img(img, direction):
    if direction == 'horizontal':
        return np.ascontiguousarray(img[:, ::-1])
    if direction == 'vertical':
        return np.ascontiguousarray(img[::-1])
    return np.ascontiguousarray(img[::-1, ::-1])


def collate(batch: Sequence[dict]) -> dict:
    return dict(
        images=torch.stack([b['image'] for b in batch]),
        boxes=[b['boxes'] for b in batch],
        labels=[b['labels'] for b in batch],
        meta=[{k: b[k] for k in ('scale', 'id', 'path', 'ori_shape')} for b in batch])


def letterbox_image(img: np.ndarray, img_size: int):
    """Inference preprocessing for one BGR image: resize keep-ratio + pad to square. Returns (tensor, scale)."""
    h0, w0 = img.shape[:2]
    scale = img_size / max(h0, w0)
    if scale != 1.0:
        img = cv2.resize(img, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    canvas[:h, :w] = img
    return torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))), scale
