"""Fixtures y constructores de datasets sinteticos compartidos por los tests."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r, rbox2poly


@pytest.fixture(scope="session")
def model():
    """PP-YOLOE-R-s con pesos aleatorios, compartido por los tests que solo miran formas."""
    torch.manual_seed(0)
    return build_ppyoloe_r(num_classes=1, size="s").eval()


def shapely_iou(a, b) -> float:
    """IoU de referencia calculada con Shapely, independiente de nuestra implementacion."""
    from shapely.geometry import Polygon

    pa = Polygon(rbox2poly(torch.tensor(a)).reshape(4, 2).tolist())
    pb = Polygon(rbox2poly(torch.tensor(b)).reshape(4, 2).tolist())
    inter = pa.intersection(pb).area
    return inter / (pa.area + pb.area - inter)


def _write_coco(ann_dir, images, annotations, name="train.json"):
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "obj"}]}
    path = ann_dir / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return str(path)


@pytest.fixture
def make_dataset(tmp_path):
    """Dataset de ruido con una caja fija por imagen. Devuelve (anotaciones, carpeta)."""
    import cv2

    def _make(n_images: int = 4):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        images, annotations = [], []
        for i in range(n_images):
            cv2.imwrite(str(img_dir / f"im{i}.jpg"), rng.integers(0, 255, (128, 160, 3), dtype=np.uint8))
            images.append({"id": i, "file_name": f"im{i}.jpg", "width": 160, "height": 128})
            poly = [40.0, 30.0, 100.0, 30.0, 100.0, 70.0, 40.0, 70.0]
            annotations.append({"id": i, "image_id": i, "category_id": 1, "segmentation": [poly],
                                "bbox": [40, 30, 60, 40], "area": 2400})
        return _write_coco(ann_dir, images, annotations), str(img_dir)

    return _make


@pytest.fixture
def make_learnable_dataset(tmp_path):
    """Un rectangulo claro sobre fondo oscuro: sintetico pero aprendible."""
    import cv2

    def _make(n_images: int = 4, size=(96, 128)):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        h, w = size
        x1, y1, x2, y2 = 30, 24, 90, 66
        images, annotations = [], []
        for i in range(n_images):
            img = np.full((h, w, 3), 30, np.uint8)
            cv2.rectangle(img, (x1, y1), (x2, y2), (220, 200, 60), -1)
            cv2.imwrite(str(img_dir / f"im{i}.jpg"), img)
            images.append({"id": i, "file_name": f"im{i}.jpg", "width": w, "height": h})
            poly = [float(x1), float(y1), float(x2), float(y1), float(x2), float(y2), float(x1), float(y2)]
            annotations.append({"id": i, "image_id": i, "category_id": 1, "segmentation": [poly],
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1)})
        return _write_coco(ann_dir, images, annotations), str(img_dir)

    return _make


@pytest.fixture
def write_coco(tmp_path):
    """Escribe un COCO a medida; util para casos limite de anotaciones."""
    import cv2

    def _make(polys, img_size=(128, 128)):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        h, w = img_size
        cv2.imwrite(str(img_dir / "a.jpg"), np.zeros((h, w, 3), np.uint8))
        annotations = []
        for i, poly in enumerate(polys):
            xs, ys = poly[0::2], poly[1::2]
            annotations.append({"id": i + 1, "image_id": 0, "category_id": 1, "segmentation": [poly],
                                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                                "area": (max(xs) - min(xs)) * (max(ys) - min(ys))})
        images = [{"id": 0, "file_name": "a.jpg", "width": w, "height": h}]
        return _write_coco(ann_dir, images, annotations), str(img_dir)

    return _make
