"""
Tests de runtime del port a PyTorch. No necesitan Paddle ni el dataset completo:
comprueban geometria, NMS, modelo, perdidas y el ciclo de entrenamiento con datos sinteticos.

    uv run pytest
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r
from ppyoloer_mps.ppyoloe_obb.boxes import (
    box2corners,
    poly2rbox,
    probiou,
    rbox2poly,
    rotated_iou,
)
from ppyoloer_mps.ppyoloe_obb.data import ObbDataset, collate, resize_keep_ratio
from ppyoloer_mps.ppyoloe_obb.engine import evaluate_model, resolve_device, train_one_epoch
from ppyoloer_mps.ppyoloe_obb.losses import PPYOLOERLoss
from ppyoloer_mps.ppyoloe_obb.ops import batched_postprocess, rotated_nms


def shapely_iou(a, b):
    """IoU de referencia con un calculo independiente (formula del cordon sobre el recorte)."""
    from shapely.geometry import Polygon

    pa = Polygon(rbox2poly(torch.tensor(a)).reshape(4, 2).tolist())
    pb = Polygon(rbox2poly(torch.tensor(b)).reshape(4, 2).tolist())
    inter = pa.intersection(pb).area
    return inter / (pa.area + pb.area - inter)


# --------------------------------------------------------------------------- geometria
def test_rbox_poly_roundtrip():
    rb = torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.3]])
    back = poly2rbox(rbox2poly(rb))
    assert torch.allclose(back[:, :2], rb[:, :2], atol=1e-3)
    assert torch.allclose(back[:, 2:4], rb[:, 2:4], atol=1e-3)
    assert abs(float(back[0, 4]) - 0.3) < 1e-3


def test_box2corners_is_rectangle():
    corners = box2corners(torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]]))[0]
    sides = (corners - torch.roll(corners, -1, dims=0)).norm(dim=-1)
    assert pytest.approx(float(sides[0]), abs=1e-4) == 2.0
    assert pytest.approx(float(sides[1]), abs=1e-4) == 4.0


def test_rotated_iou_known_cases():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    assert pytest.approx(float(rotated_iou(a, a)[0, 0]), abs=1e-5) == 1.0
    disjoint = torch.tensor([[100.0, 100.0, 10.0, 10.0, 0.0]])
    assert float(rotated_iou(a, disjoint)[0, 0]) == pytest.approx(0.0, abs=1e-6)
    inside = torch.tensor([[0.0, 0.0, 5.0, 5.0, 0.0]])
    assert pytest.approx(float(rotated_iou(a, inside)[0, 0]), abs=1e-4) == 0.25
    rot90 = torch.tensor([[0.0, 0.0, 10.0, 10.0, math.pi / 2]])
    assert pytest.approx(float(rotated_iou(a, rot90)[0, 0]), abs=1e-4) == 1.0


def test_rotated_iou_matches_reference():
    pytest.importorskip("shapely")
    rng = np.random.default_rng(0)
    boxes1 = rng.uniform([0, 0, 5, 5, -math.pi], [200, 200, 80, 80, math.pi], (6, 5)).astype(np.float32)
    boxes2 = rng.uniform([0, 0, 5, 5, -math.pi], [200, 200, 80, 80, math.pi], (7, 5)).astype(np.float32)
    got = rotated_iou(torch.tensor(boxes1), torch.tensor(boxes2)).numpy()
    ref = np.array([[shapely_iou(a, b) for b in boxes2] for a in boxes1])
    assert np.abs(got - ref).max() < 1e-4


def test_probiou_finite_for_degenerate_boxes():
    """Una caja degenerada (lado ~0) no debe producir NaN: hundia el entrenamiento."""
    boxes = torch.tensor(
        [
            [10.0, 10.0, 0.0, 8.0, 0.0],      # ancho nulo
            [10.0, 10.0, 20.0, 1e-6, 0.3],    # alto casi nulo
            [0.0, 0.0, 0.0, 0.0, 0.0],        # totalmente degenerada
        ]
    )
    target = torch.tensor([[10.0, 10.0, 20.0, 8.0, 0.0]] * 3)
    loss = probiou(boxes, target)
    assert torch.isfinite(loss).all(), loss


def test_dataset_filters_degenerate_boxes(tmp_path):
    """El dataset debe descartar cajas con lado menor < 2 px, como Poly2RBox de ppdet."""
    import cv2

    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "a.jpg"), np.zeros((128, 128, 3), np.uint8))
    thin = [10.0, 10.0, 90.0, 10.0, 90.0, 10.5, 10.0, 10.5]   # 0.5 px de alto
    ok = [10.0, 40.0, 90.0, 40.0, 90.0, 80.0, 10.0, 80.0]
    coco = {
        "images": [{"id": 0, "file_name": "a.jpg", "width": 128, "height": 128}],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [thin], "bbox": [10, 10, 80, 1], "area": 40},
            {"id": 2, "image_id": 0, "category_id": 1, "segmentation": [ok], "bbox": [10, 40, 80, 40], "area": 3200},
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    with open(ann_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(coco, f)
    ds = ObbDataset(str(ann_dir / "train.json"), str(img_dir), img_size=128, train=True, mosaic_epochs=0)
    ds.set_epoch(0)
    for _ in range(10):
        polys = ds[0]["polys"]
        if polys.numel():
            rb = poly2rbox(polys)
            assert float(torch.minimum(rb[:, 2], rb[:, 3]).min()) >= 2.0


def test_dataset_clips_polygons_to_canvas(tmp_path):
    """Los vertices deben recortarse al lienzo, como hace RResize de ppdet.

    Sin este recorte se entrena al modelo con la extension completa de objetos que solo se
    ven en parte, lo que infla las cajas predichas y hunde el mAP75.
    """
    import cv2

    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "a.jpg"), np.zeros((128, 128, 3), np.uint8))
    # caja que se sale por la derecha y por abajo
    poly = [60.0, 60.0, 200.0, 60.0, 200.0, 190.0, 60.0, 190.0]
    coco = {
        "images": [{"id": 0, "file_name": "a.jpg", "width": 128, "height": 128}],
        "annotations": [{"id": 1, "image_id": 0, "category_id": 1, "segmentation": [poly],
                         "bbox": [60, 60, 140, 130], "area": 18200}],
        "categories": [{"id": 1, "name": "obj"}],
    }
    with open(ann_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(coco, f)
    ds = ObbDataset(str(ann_dir / "train.json"), str(img_dir), img_size=128, train=True, mosaic_epochs=0)
    ds.set_epoch(0)
    for _ in range(10):
        item = ds[0]
        if item["polys"].numel():
            h, w = item["image"].shape[1:]
            assert float(item["polys"][:, 0::2].min()) >= -1e-3
            assert float(item["polys"][:, 1::2].max()) <= h + 1e-3
            assert float(item["polys"][:, 0::2].max()) <= w + 1e-3


def test_probiou_zero_for_identical_boxes():
    box = torch.tensor([[10.0, 10.0, 20.0, 8.0, 0.4]])
    assert float(probiou(box, box)) < 0.05


# --------------------------------------------------------------------------- NMS
def test_rotated_nms_suppresses_overlaps():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 0.0],
            [0.5, 0.5, 10.0, 10.0, 0.0],  # casi la misma -> se suprime
            [100.0, 100.0, 10.0, 10.0, 0.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = rotated_nms(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2]


def test_postprocess_rescales_to_original():
    scores = torch.zeros(1, 1, 2)
    scores[0, 0, 0] = 0.9
    rboxes = torch.tensor([[[100.0, 100.0, 40.0, 20.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]]])
    scale = torch.tensor([[0.5, 0.5]])  # (scale_y, scale_x)
    det = batched_postprocess(scores, rboxes, score_threshold=0.5, scale_factor=scale)[0]
    assert len(det) == 1
    assert pytest.approx(float(det["rboxes"][0, 0]), abs=1e-4) == 200.0
    assert pytest.approx(float(det["rboxes"][0, 2]), abs=1e-4) == 80.0


# --------------------------------------------------------------------------- modelo
@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_ppyoloe_r(num_classes=1, size="s").eval()


def test_model_eval_shapes(model):
    with torch.no_grad():
        scores, rboxes = model(torch.zeros(1, 3, 320, 320))
    expected = (320 // 32) ** 2 + (320 // 16) ** 2 + (320 // 8) ** 2
    assert scores.shape == (1, 1, expected)
    assert rboxes.shape == (1, expected, 5)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0


def test_model_train_shapes(model):
    model.train()
    try:
        outs = model(torch.zeros(2, 3, 320, 320))
        cls, dist, angle, anchors, num_list, strides = outs
        n = (320 // 32) ** 2 + (320 // 16) ** 2 + (320 // 8) ** 2
        assert cls.shape == (2, n, 1)
        assert dist.shape == (2, n, 4)
        assert angle.shape == (2, n, 91)
        assert anchors.shape == (1, n, 2)
        assert sum(num_list) == n
        assert strides.shape == (1, n, 1)
    finally:
        model.eval()


def test_model_accepts_non_square_input(model):
    with torch.no_grad():
        scores, rboxes = model(torch.zeros(1, 3, 320, 640))
    n = (320 // 32) * (640 // 32) + (320 // 16) * (640 // 16) + (320 // 8) * (640 // 8)
    assert scores.shape[-1] == n and rboxes.shape[1] == n


# --------------------------------------------------------------------------- datos
def _make_dataset(tmp_path, n_images=4):
    import cv2

    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    rng = np.random.default_rng(0)
    for i in range(n_images):
        arr = rng.integers(0, 255, (128, 160, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"im{i}.jpg"), arr)
        images.append({"id": i, "file_name": f"im{i}.jpg", "width": 160, "height": 128})
        poly = [40.0, 30.0, 100.0, 30.0, 100.0, 70.0, 40.0, 70.0]
        annotations.append(
            {"id": i, "image_id": i, "category_id": 1, "segmentation": [poly], "bbox": [40, 30, 60, 40], "area": 2400}
        )
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "obj"}]}
    with open(ann_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return str(ann_dir / "train.json"), str(img_dir)


def test_resize_keep_ratio_matches_paddle_rule():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, sx, sy = resize_keep_ratio(img, 640)
    assert out.shape[:2] == (320, 640)
    assert pytest.approx(sx, abs=1e-6) == 3.2
    assert pytest.approx(sy, abs=1e-6) == 3.2


def test_dataset_eval_and_collate(tmp_path):
    ann, img_dir = _make_dataset(tmp_path)
    ds = ObbDataset(ann, img_dir, img_size=320, train=False)
    assert len(ds) == 4 and ds.num_classes == 1
    batch = collate([ds[0], ds[1]])
    assert batch["image"].shape[0] == 2
    assert batch["image"].shape[2] % 32 == 0 and batch["image"].shape[3] % 32 == 0
    assert batch["gt_poly"].shape[:2] == (2, 1)
    assert float(batch["pad_gt_mask"].sum()) == 2.0


def test_dataset_train_augmentation_keeps_boxes(tmp_path):
    ann, img_dir = _make_dataset(tmp_path)
    ds = ObbDataset(ann, img_dir, img_size=320, train=True, mosaic_epochs=1)
    ds.set_epoch(0)
    item = ds[0]
    assert item["image"].shape[1:] == (320, 320)
    assert item["polys"].shape[1] == 8 if item["polys"].numel() else True
    ds.set_epoch(5)  # ya sin mosaico
    item = ds[0]
    assert item["image"].ndim == 3


# --------------------------------------------------------------------------- perdida y bucle
def _make_learnable_dataset(tmp_path, n_images=4, size=(96, 128)):
    """Un rectangulo claro sobre fondo oscuro: sintetico pero aprendible."""
    import cv2

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
        annotations.append(
            {"id": i, "image_id": i, "category_id": 1, "segmentation": [poly],
             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1)}
        )
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "obj"}]}
    with open(ann_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return str(ann_dir / "train.json"), str(img_dir)


def test_training_can_overfit(tmp_path):
    """El bucle completo (asignador + perdidas + optimizador) debe aprender un caso trivial.

    La perdida sube durante las primeras epocas: con asignacion task-aligned desde pesos
    aleatorios casi no hay positivos al principio. Lo que se comprueba es que converge.
    """
    torch.manual_seed(0)
    ann, img_dir = _make_learnable_dataset(tmp_path)
    ds = ObbDataset(ann, img_dir, img_size=128, train=False)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate)
    model = build_ppyoloe_r(num_classes=1, size="s")
    loss_fn = PPYOLOERLoss(num_classes=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    device = torch.device("cpu")
    losses = []
    for epoch in range(90):
        losses.append(train_one_epoch(model, loss_fn, loader, opt, device, epoch, log_interval=0)["loss"])
    assert all(math.isfinite(v) for v in losses), "la perdida ha divergido"
    assert losses[-1] < max(losses) / 2, f"no converge: {losses[0]:.3f} -> {losses[-1]:.3f} (max {max(losses):.3f})"
    res = evaluate_model(model, loader, device, name="overfit", logger=lambda *_: None)
    assert res["mAP50"] > 0.5, f"no ha aprendido el caso trivial: mAP50={res['mAP50']:.3f}"


def test_evaluate_model_runs(tmp_path):
    ann, img_dir = _make_dataset(tmp_path)
    ds = ObbDataset(ann, img_dir, img_size=320, train=False)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    model = build_ppyoloe_r(num_classes=1, size="s")
    res = evaluate_model(model, loader, torch.device("cpu"), name="t", logger=lambda *_: None)
    assert res["n_gt"] == 4
    assert 0.0 <= res["mAP50"] <= 1.0


# --------------------------------------------------------------------------- dispositivo
def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


@pytest.mark.parametrize("device_name", ["cuda", "mps"])
def test_forward_on_accelerator(device_name, model):
    """Se salta si el acelerador no esta disponible; en el Mac cubre MPS."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA no disponible")
    if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS no disponible")
    device = torch.device(device_name)
    m = build_ppyoloe_r(num_classes=1, size="s").to(device).eval()
    with torch.no_grad():
        scores, rboxes = m(torch.zeros(1, 3, 320, 320, device=device))
    assert scores.device.type == device_name
    assert torch.isfinite(rboxes).all()
