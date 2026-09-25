"""
Dataset, preproceso y augmentacion.

Varios de estos tests fijan detalles que se descubrieron comparando contra PaddleDetection y
que, al faltar, degradaban el entrenamiento sin dar ningun error visible.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ppyoloer_mps.ppyoloe_obb.boxes import poly2rbox
from ppyoloer_mps.ppyoloe_obb.data import ObbDataset, collate, resize_keep_ratio


def test_resize_keep_ratio_matches_paddle_rule():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, sx, sy = resize_keep_ratio(img, 640)
    assert out.shape[:2] == (320, 640)
    assert pytest.approx(sx, abs=1e-6) == 3.2
    assert pytest.approx(sy, abs=1e-6) == 3.2


def test_dataset_eval_and_collate(make_dataset):
    ann, img_dir = make_dataset()
    ds = ObbDataset(ann, img_dir, img_size=320, train=False)
    assert len(ds) == 4 and ds.num_classes == 1
    batch = collate([ds[0], ds[1]])
    assert batch["image"].shape[0] == 2
    assert batch["image"].shape[2] % 32 == 0 and batch["image"].shape[3] % 32 == 0
    assert batch["gt_poly"].shape[:2] == (2, 1)
    assert float(batch["pad_gt_mask"].sum()) == 2.0


def test_dataset_train_augmentation_keeps_boxes(make_dataset):
    ann, img_dir = make_dataset()
    ds = ObbDataset(ann, img_dir, img_size=320, train=True, mosaic_epochs=1)
    ds.set_epoch(0)
    item = ds[0]
    assert item["image"].shape[1:] == (320, 320)
    ds.set_epoch(5)  # ya sin mosaico
    assert ds[0]["image"].ndim == 3


def test_dataset_filters_degenerate_boxes(write_coco):
    """Cajas con lado menor < 2 px se descartan, como Poly2RBox de ppdet: si llegan a la
    perdida, ProbIoU hace log(0) y el entrenamiento se va a NaN."""
    thin = [10.0, 10.0, 90.0, 10.0, 90.0, 10.5, 10.0, 10.5]   # 0.5 px de alto
    ok = [10.0, 40.0, 90.0, 40.0, 90.0, 80.0, 10.0, 80.0]
    ann, img_dir = write_coco([thin, ok])
    ds = ObbDataset(ann, img_dir, img_size=128, train=True, mosaic_epochs=0)
    ds.set_epoch(0)
    for _ in range(10):
        polys = ds[0]["polys"]
        if polys.numel():
            rb = poly2rbox(polys)
            assert float(torch.minimum(rb[:, 2], rb[:, 3]).min()) >= 2.0


def test_dataset_clips_polygons_to_canvas(write_coco):
    """Los vertices se recortan al lienzo, como RResize de ppdet.

    Sin este recorte se entrena al modelo con la extension completa de objetos que solo se ven
    en parte, lo que infla las cajas predichas y hunde el mAP75.
    """
    ann, img_dir = write_coco([[60.0, 60.0, 200.0, 60.0, 200.0, 190.0, 60.0, 190.0]])
    ds = ObbDataset(ann, img_dir, img_size=128, train=True, mosaic_epochs=0)
    ds.set_epoch(0)
    for _ in range(10):
        item = ds[0]
        if item["polys"].numel():
            h, w = item["image"].shape[1:]
            assert float(item["polys"][:, 0::2].min()) >= -1e-3
            assert float(item["polys"][:, 0::2].max()) <= w + 1e-3
            assert float(item["polys"][:, 1::2].max()) <= h + 1e-3


def test_rotation_shrinks_instead_of_cropping(make_dataset):
    """Las rotaciones usan auto_bound: el contenido rotado cabe entero en el lienzo."""
    from ppyoloer_mps.ppyoloe_obb.data import _rotate

    img = np.full((100, 100, 3), 255, np.uint8)
    poly = np.array([[10.0, 10.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0]], np.float32)
    out, rotated = _rotate(img, poly, 45.0, 0)
    assert out.shape == img.shape                      # el lienzo no crece
    assert rotated[:, 0::2].min() > -1 and rotated[:, 0::2].max() < 101


def test_dataset_is_deterministic_per_epoch(make_dataset):
    ann, img_dir = make_dataset()
    ds = ObbDataset(ann, img_dir, img_size=320, train=True, mosaic_epochs=1, seed=7)
    ds.set_epoch(3)
    a = ds[2]["image"].clone()
    ds.set_epoch(3)
    assert torch.allclose(a, ds[2]["image"])
