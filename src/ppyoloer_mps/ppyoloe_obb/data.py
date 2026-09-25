"""
Datos: lectura de los JSON COCO con poligonos, preproceso de inferencia y augmentacion
de entrenamiento (mosaico, afin, HSV, flip, rotaciones), todo en numpy/OpenCV.

El preproceso reproduce exactamente el de PaddleDetection:
  Decode      cv2.imdecode (BGR) -> RGB
  Resize      keep_ratio: escala = min(lado_menor_obj/lado_menor, lado_mayor_obj/lado_mayor)
              redondeo de tamano con int(escala*lado + 0.5)
  Normalize   /255 y luego (x - mean) / std con estadisticas de ImageNet
  Permute     HWC -> CHW
  PadBatch    relleno con ceros hasta multiplo de 32
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:  # HEIC opcional (fotos de iPhone)
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pillow_heif = None

__all__ = [
    "MEAN",
    "STD",
    "AugmentConfig",
    "ObbDataset",
    "collate",
    "load_image",
    "preprocess_image",
    "resize_keep_ratio",
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")


def load_image(path: str) -> np.ndarray:
    """Lee una imagen a RGB uint8. Soporta HEIC si pillow_heif esta instalado."""
    if path.lower().endswith((".heic", ".heif")):
        if pillow_heif is None:
            raise RuntimeError(f"hace falta pillow-heif para leer {path}")
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    data = np.fromfile(path, dtype=np.uint8)  # soporta rutas con acentos en Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"no se pudo leer la imagen {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_keep_ratio(img: np.ndarray, target: int, interp: int = cv2.INTER_AREA):
    """Escala manteniendo proporcion, igual que `Resize(keep_ratio=True)` de ppdet.

    Devuelve (imagen, scale_x, scale_y).
    """
    h, w = img.shape[:2]
    scale = min(target / min(h, w), target / max(h, w))
    resize_h = int(scale * h + 0.5)
    resize_w = int(scale * w + 0.5)
    scale_y = resize_h / h
    scale_x = resize_w / w
    out = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    return out, scale_x, scale_y


def normalize_chw(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def pad_to_stride(x: np.ndarray, stride: int = 32) -> np.ndarray:
    _, h, w = x.shape
    ph = int(math.ceil(h / stride) * stride)
    pw = int(math.ceil(w / stride) * stride)
    if ph == h and pw == w:
        return x
    out = np.zeros((x.shape[0], ph, pw), dtype=x.dtype)
    out[:, :h, :w] = x
    return out


def preprocess_image(path_or_img, img_size: int = 640, interp: int = cv2.INTER_AREA):
    """Imagen -> (tensor (1, 3, H, W), scale_factor (2,) = (scale_y, scale_x))."""
    img = load_image(path_or_img) if isinstance(path_or_img, str) else path_or_img
    resized, sx, sy = resize_keep_ratio(img, img_size, interp)
    x = pad_to_stride(normalize_chw(resized))
    return torch.from_numpy(x).unsqueeze(0), np.array([sy, sx], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Augmentacion
# --------------------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """Augmentacion equivalente a la usada en el entrenamiento con Paddle."""

    img_size: int = 640
    mosaic_prob: float = 1.0
    mosaic_scale: float = 0.2      # escala aleatoria +-20 % dentro del mosaico
    mosaic_translate: float = 0.05
    affine_scale: float = 0.2      # fase sin mosaico
    affine_translate: float = 0.05
    hsv_h: float = 5.0 / 180.0     # tono, fraccion de vuelta (5 grados)
    hsv_s: tuple[float, float] = (0.4, 1.6)
    hsv_v: tuple[float, float] = (0.6, 1.4)
    contrast: tuple[float, float] = (0.8, 1.2)
    flip_prob: float = 0.5
    rot90_angles: tuple[int, ...] = (0, 90, 180, -90)
    extra_rot_angles: tuple[int, ...] = (30, 60)
    extra_rot_prob: float = 0.5
    min_visible: float = 0.1
    fill_value: int = 114


def _poly_area(poly: np.ndarray) -> np.ndarray:
    x, y = poly[..., 0::2], poly[..., 1::2]
    return 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1)
    )


def _clip_visible(polys: np.ndarray, w: int, h: int, min_visible: float) -> np.ndarray:
    """Mascara de poligonos cuya area visible dentro del lienzo supera `min_visible`."""
    if len(polys) == 0:
        return np.zeros(0, dtype=bool)
    keep = np.zeros(len(polys), dtype=bool)
    canvas = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    for i, p in enumerate(polys):
        pts = p.reshape(4, 2).astype(np.float32)
        area = _poly_area(p)
        if area <= 1:
            continue
        inter, _ = cv2.intersectConvexConvex(pts, canvas)
        keep[i] = (inter / area) >= min_visible
    return keep


def _affine_polys(polys: np.ndarray, m: np.ndarray) -> np.ndarray:
    if len(polys) == 0:
        return polys.reshape(0, 8)
    pts = polys.reshape(-1, 4, 2)
    ones = np.ones((*pts.shape[:2], 1), dtype=pts.dtype)
    out = np.concatenate([pts, ones], axis=-1) @ m.T
    return out.reshape(-1, 8).astype(np.float32)


def _scale_translate_matrix(cx, cy, s, tx, ty) -> np.ndarray:
    return np.array([[s, 0, tx - s * cx], [0, s, ty - s * cy]], dtype=np.float64)


def _hsv_jitter(img: np.ndarray, cfg: AugmentConfig, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    if rng.random() < 0.5:
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hsv_h, cfg.hsv_h) * 180.0) % 180.0
    if rng.random() < 0.5:
        hsv[..., 1] *= rng.uniform(*cfg.hsv_s)
    if rng.random() < 0.5:
        hsv[..., 2] *= rng.uniform(*cfg.hsv_v)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.3:
        f = rng.uniform(*cfg.contrast)
        out = np.clip((out.astype(np.float32) - 128.0) * f + 128.0, 0, 255).astype(np.uint8)
    return out


def _rotate(img: np.ndarray, polys: np.ndarray, angle: float, fill: int):
    """Rota manteniendo el lienzo (se recorta lo que sale); los poligonos se transforman."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(fill,) * 3)
    return out, _affine_polys(polys, m)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
@dataclass
class Sample:
    image_path: str
    polys: np.ndarray  # (N, 8) en pixeles de la imagen original
    labels: np.ndarray  # (N,)
    image_id: int
    width: int
    height: int


class ObbDataset(Dataset):
    """Lee un JSON COCO cuyas anotaciones llevan `segmentation` con un poligono de 4 puntos."""

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        img_size: int = 640,
        train: bool = False,
        augment: AugmentConfig | None = None,
        mosaic_epochs: int = 0,
        interp_train: int = cv2.INTER_CUBIC,
        interp_eval: int = cv2.INTER_AREA,
        seed: int = 0,
    ):
        self.image_dir = image_dir
        self.img_size = img_size
        self.train = train
        self.cfg = augment or AugmentConfig(img_size=img_size)
        self.mosaic_epochs = mosaic_epochs
        self.interp_train = interp_train
        self.interp_eval = interp_eval
        self.epoch = 0
        self.seed = seed

        with open(annotation_file, encoding="utf-8") as f:
            coco = json.load(f)
        self.classes = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
        cat2idx = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}
        anns: dict[int, list[tuple[np.ndarray, int]]] = {}
        for a in coco["annotations"]:
            poly = np.asarray(a["segmentation"][0], dtype=np.float32)
            anns.setdefault(a["image_id"], []).append((poly, cat2idx[a["category_id"]]))
        self.samples: list[Sample] = []
        for im in coco["images"]:
            items = anns.get(im["id"], [])
            polys = np.stack([p for p, _ in items]) if items else np.zeros((0, 8), np.float32)
            labels = np.array([c for _, c in items], dtype=np.int64) if items else np.zeros(0, np.int64)
            self.samples.append(
                Sample(
                    image_path=os.path.join(image_dir, im["file_name"]),
                    polys=polys,
                    labels=labels,
                    image_id=im["id"],
                    width=im["width"],
                    height=im["height"],
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    # -- carga basica -------------------------------------------------------------
    def _load(self, idx: int):
        s = self.samples[idx]
        img = load_image(s.image_path)
        return img, s.polys.copy(), s.labels.copy()

    # -- mosaico ------------------------------------------------------------------
    def _mosaic(self, idx: int, rng: random.Random):
        size = self.img_size
        canvas = np.full((2 * size, 2 * size, 3), self.cfg.fill_value, dtype=np.uint8)
        xc = int(rng.uniform(0.5 * size, 1.5 * size))
        yc = int(rng.uniform(0.5 * size, 1.5 * size))
        indices = [idx] + [rng.randrange(len(self.samples)) for _ in range(3)]
        polys_all, labels_all = [], []
        for i, index in enumerate(indices):
            img, polys, labels = self._load(index)
            h0, w0 = img.shape[:2]
            r = size / max(h0, w0)
            if r != 1:
                img = cv2.resize(img, (int(round(w0 * r)), int(round(h0 * r))), interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * size), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * size, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * size), min(2 * size, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            if len(polys):
                p = polys * r
                p[:, 0::2] += x1a - x1b
                p[:, 1::2] += y1a - y1b
                polys_all.append(p)
                labels_all.append(labels)
        polys = np.concatenate(polys_all) if polys_all else np.zeros((0, 8), np.float32)
        labels = np.concatenate(labels_all) if labels_all else np.zeros(0, np.int64)

        s = rng.uniform(1 - self.cfg.mosaic_scale, 1 + self.cfg.mosaic_scale)
        tx = size / 2 + rng.uniform(-self.cfg.mosaic_translate, self.cfg.mosaic_translate) * size
        ty = size / 2 + rng.uniform(-self.cfg.mosaic_translate, self.cfg.mosaic_translate) * size
        m = _scale_translate_matrix(size, size, s, tx, ty)
        out = cv2.warpAffine(
            canvas, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=(self.cfg.fill_value,) * 3
        )
        polys = _affine_polys(polys, m)
        keep = _clip_visible(polys, size, size, self.cfg.min_visible)
        return out, polys[keep], labels[keep]

    def _random_affine(self, img, polys, rng: random.Random):
        h, w = img.shape[:2]
        s = rng.uniform(1 - self.cfg.affine_scale, 1 + self.cfg.affine_scale)
        tx = w / 2 + rng.uniform(-self.cfg.affine_translate, self.cfg.affine_translate) * w
        ty = h / 2 + rng.uniform(-self.cfg.affine_translate, self.cfg.affine_translate) * h
        m = _scale_translate_matrix(w / 2, h / 2, s, tx, ty)
        out = cv2.warpAffine(
            img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(self.cfg.fill_value,) * 3
        )
        return out, _affine_polys(polys, m)

    # -- item ---------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.train:
            s = self.samples[idx]
            img = load_image(s.image_path)
            resized, sx, sy = resize_keep_ratio(img, self.img_size, self.interp_eval)
            polys = s.polys.copy()
            if len(polys):
                polys[:, 0::2] *= sx
                polys[:, 1::2] *= sy
            return {
                "image": torch.from_numpy(normalize_chw(resized)),
                "polys": torch.from_numpy(polys),
                "labels": torch.from_numpy(s.labels.copy()),
                "image_id": s.image_id,
                "scale_factor": torch.tensor([sy, sx], dtype=torch.float32),
                "image_path": s.image_path,
            }

        rng = random.Random((self.seed * 1_000_003 + self.epoch * 9973 + idx))
        use_mosaic = self.epoch < self.mosaic_epochs and rng.random() < self.cfg.mosaic_prob
        if use_mosaic:
            img, polys, labels = self._mosaic(idx, rng)
        else:
            img, polys, labels = self._load(idx)
            img, polys = self._random_affine(img, polys, rng)

        img = _hsv_jitter(img, self.cfg, rng)

        if rng.random() < self.cfg.flip_prob:
            img = img[:, ::-1].copy()
            if len(polys):
                polys[:, 0::2] = img.shape[1] - polys[:, 0::2] - 1

        angle = rng.choice(self.cfg.rot90_angles)
        if angle:
            img, polys = _rotate(img, polys, angle, self.cfg.fill_value)
        if rng.random() < self.cfg.extra_rot_prob:
            img, polys = _rotate(img, polys, rng.choice(self.cfg.extra_rot_angles), self.cfg.fill_value)

        if len(polys):
            keep = _clip_visible(polys, img.shape[1], img.shape[0], self.cfg.min_visible)
            polys, labels = polys[keep], labels[keep]

        resized, sx, sy = resize_keep_ratio(img, self.img_size, self.interp_train)
        if len(polys):
            polys = polys.copy()
            polys[:, 0::2] *= sx
            polys[:, 1::2] *= sy

        return {
            "image": torch.from_numpy(normalize_chw(resized)),
            "polys": torch.from_numpy(np.ascontiguousarray(polys, dtype=np.float32)),
            "labels": torch.from_numpy(labels),
            "image_id": self.samples[idx].image_id,
            "scale_factor": torch.tensor([sy, sx], dtype=torch.float32),
            "image_path": self.samples[idx].image_path,
        }


def collate(batch: Sequence[dict[str, Any]], stride: int = 32) -> dict[str, Any]:
    """Apila imagenes con relleno a multiplo de `stride` y rellena las cajas a la longitud maxima."""
    heights = [b["image"].shape[1] for b in batch]
    widths = [b["image"].shape[2] for b in batch]
    ph = int(math.ceil(max(heights) / stride) * stride)
    pw = int(math.ceil(max(widths) / stride) * stride)
    images = torch.zeros(len(batch), 3, ph, pw)
    for i, b in enumerate(batch):
        img = b["image"]
        images[i, :, : img.shape[1], : img.shape[2]] = img

    max_gt = max(1, max(int(b["polys"].shape[0]) for b in batch))
    polys = torch.zeros(len(batch), max_gt, 8)
    labels = torch.zeros(len(batch), max_gt, 1, dtype=torch.long)
    pad_mask = torch.zeros(len(batch), max_gt, 1)
    for i, b in enumerate(batch):
        n = int(b["polys"].shape[0])
        if n:
            polys[i, :n] = b["polys"]
            labels[i, :n, 0] = b["labels"]
            pad_mask[i, :n, 0] = 1.0
    return {
        "image": images,
        "gt_poly": polys,
        "gt_class": labels,
        "pad_gt_mask": pad_mask,
        "image_id": [b["image_id"] for b in batch],
        "scale_factor": torch.stack([b["scale_factor"] for b in batch]),
        "image_path": [b["image_path"] for b in batch],
    }
