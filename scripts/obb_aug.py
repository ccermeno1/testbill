"""
Augmentaciones geometricas para OBB (poligonos de 4 puntos) al estilo Ultralytics, para
PaddleDetection. Los operadores estandar de ppdet (Mosaic, RandomExpand, RandomCrop...) no
propagan `gt_poly`, asi que aqui se implementan como transformaciones afines de los poligonos.

Operadores (se registran en ppdet.data.transform al importar este modulo):
  RMosaic        mosaico 2x2 + escala/traslacion aleatoria, salida input_size x input_size.
                 Recibe la lista [muestra, 4 aleatorias] que entrega el dataset cuando el
                 TrainReader tiene `mosaic_epoch: N` (epoca < N). Con una sola muestra no hace nada.
  RRandomAffine  escala +-scale y traslacion +-translate sobre una imagen (fase sin mosaico).

Los poligonos parcialmente fuera del lienzo se conservan sin recortar (como en el dataset)
si al menos `min_visible` de su area queda dentro; si no, se descartan.

Uso: importar este modulo antes de crear el TrainReader (scripts/train.py lo hace).
"""
from collections.abc import Sequence

import cv2
import numpy as np
from shapely.geometry import Polygon, box as shapely_box

import ppdet.data.transform as T
from ppdet.data.transform.operators import BaseOperator

__all__ = ["RMosaic", "RRandomAffine"]


def _affine_polys(polys, M):
    if len(polys) == 0:
        return np.zeros((0, 8), dtype=np.float32)
    pts = np.asarray(polys, dtype=np.float64).reshape(-1, 4, 2)
    pts = np.concatenate([pts, np.ones((pts.shape[0], 4, 1))], axis=-1) @ M.T
    return pts.reshape(-1, 8).astype(np.float32)


def _visible_mask(polys, w, h, min_visible):
    canvas = shapely_box(0, 0, w, h)
    keep = []
    for p in polys:
        pg = Polygon(p.reshape(4, 2))
        if not pg.is_valid or pg.area <= 1:
            keep.append(False)
            continue
        keep.append(pg.intersection(canvas).area / pg.area >= min_visible)
    return np.array(keep, dtype=bool)


def _poly2bbox(polys):
    if len(polys) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xs, ys = polys[:, 0::2], polys[:, 1::2]
    return np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1).astype(np.float32)


def _set_geometry(sample, image, polys, keep):
    sample["image"] = image
    sample["gt_poly"] = polys[keep]
    sample["gt_bbox"] = _poly2bbox(sample["gt_poly"])
    for k in ("gt_class", "is_crowd", "difficult", "gt_score"):
        if k in sample and len(sample[k]) == len(keep):
            sample[k] = sample[k][keep]
    h, w = image.shape[:2]
    sample["im_shape"] = np.array([h, w], dtype=np.float32)
    sample["h"], sample["w"] = h, w
    return sample


def _scale_translate_matrix(cx, cy, s, tx, ty):
    """Escala s alrededor de (cx, cy) y luego traslada el centro a (tx, ty)."""
    return np.array([[s, 0, tx - s * cx], [0, s, ty - s * cy]], dtype=np.float64)


class RMosaic(BaseOperator):
    def __init__(self, input_size=640, scale=0.2, translate=0.05, prob=1.0,
                 min_visible=0.3, fill_value=114):
        super().__init__()
        self.S = int(input_size)
        self.scale = float(scale)
        self.translate = float(translate)
        self.prob = float(prob)
        self.min_visible = float(min_visible)
        self.fill = (fill_value,) * 3

    def __call__(self, sample, context=None):
        if not isinstance(sample, Sequence):
            return sample
        if np.random.uniform() >= self.prob:
            return sample[0]
        return self.apply_mosaic(list(sample[:4]))

    def apply_mosaic(self, samples):
        S = self.S
        canvas = np.full((2 * S, 2 * S, 3), self.fill[0], dtype=np.uint8)
        xc = int(np.random.uniform(0.5 * S, 1.5 * S))
        yc = int(np.random.uniform(0.5 * S, 1.5 * S))
        polys, classes, crowd = [], [], []
        for i, s in enumerate(samples):
            img = s["image"]
            h0, w0 = img.shape[:2]
            r = S / max(h0, w0)
            if r != 1:
                img = cv2.resize(img, (int(round(w0 * r)), int(round(h0 * r))), interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]
            if i == 0:    # arriba-izquierda
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # arriba-derecha
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * S), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # abajo-izquierda
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * S, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:         # abajo-derecha
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * S), min(2 * S, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b
            p = np.asarray(s.get("gt_poly", np.zeros((0, 8))), dtype=np.float32).reshape(-1, 8)
            if len(p):
                p = p * r
                p[:, 0::2] += padw
                p[:, 1::2] += padh
                polys.append(p)
                classes.append(np.asarray(s["gt_class"]).reshape(-1, 1))
                crowd.append(np.asarray(s.get("is_crowd", np.zeros((len(p), 1), dtype=np.int32))).reshape(-1, 1))
        polys = np.concatenate(polys) if polys else np.zeros((0, 8), dtype=np.float32)
        classes = np.concatenate(classes) if classes else np.zeros((0, 1), dtype=np.int32)
        crowd = np.concatenate(crowd) if crowd else np.zeros((0, 1), dtype=np.int32)

        # escala/traslacion aleatoria del lienzo 2S y recorte central a S x S
        sc = np.random.uniform(1 - self.scale, 1 + self.scale)
        tx = S / 2 + np.random.uniform(-self.translate, self.translate) * S
        ty = S / 2 + np.random.uniform(-self.translate, self.translate) * S
        M = _scale_translate_matrix(S, S, sc, tx, ty)
        out = cv2.warpAffine(canvas, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=self.fill)
        polys = _affine_polys(polys, M)
        keep = _visible_mask(polys, S, S, self.min_visible)

        sample = samples[0]
        sample["gt_class"] = classes.astype(np.int32)
        sample["is_crowd"] = crowd.astype(np.int32)
        for k in ("difficult", "gt_score"):
            sample.pop(k, None)
        return _set_geometry(sample, out, polys, keep)


class RRandomAffine(BaseOperator):
    def __init__(self, scale=0.2, translate=0.05, prob=1.0, min_visible=0.3, fill_value=114,
                 start_epoch=0):
        """start_epoch: no aplicar antes de esa epoca (para no sumar escala a la del mosaico)."""
        super().__init__()
        self.start_epoch = int(start_epoch)
        self.scale = float(scale)
        self.translate = float(translate)
        self.prob = float(prob)
        self.min_visible = float(min_visible)
        self.fill = (fill_value,) * 3

    def apply(self, sample, context=None):
        if sample.get("curr_epoch", 0) < self.start_epoch or np.random.uniform() >= self.prob:
            return sample
        img = sample["image"]
        h, w = img.shape[:2]
        sc = np.random.uniform(1 - self.scale, 1 + self.scale)
        tx = w / 2 + np.random.uniform(-self.translate, self.translate) * w
        ty = h / 2 + np.random.uniform(-self.translate, self.translate) * h
        M = _scale_translate_matrix(w / 2, h / 2, sc, tx, ty)
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=self.fill)
        polys = _affine_polys(np.asarray(sample.get("gt_poly", np.zeros((0, 8))), dtype=np.float32).reshape(-1, 8), M)
        keep = _visible_mask(polys, w, h, self.min_visible)
        return _set_geometry(sample, out, polys, keep)


for _cls in (RMosaic, RRandomAffine):
    setattr(T, _cls.__name__, _cls)
    if _cls.__name__ not in T.__all__:
        T.__all__.append(_cls.__name__)
