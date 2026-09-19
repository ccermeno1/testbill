"""Tests for COCO polygon → le90 rbox conversion and DOTA export."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from oriented_det.data.coco_obb import (
    annotation_from_polygon_coords,
    export_coco_json_to_dota,
    min_area_rect_corners,
    parse_coco_instances,
)
from oriented_det.data.dota import DOTADataset


def test_annotation_from_quad_keeps_corners():
    ann = annotation_from_polygon_coords(
        [10.0, 10.0, 90.0, 10.0, 90.0, 50.0, 10.0, 50.0],
        "ship",
    )
    assert ann.class_name == "ship"
    assert abs(ann.rbox.cx - 50.0) < 1e-3
    assert abs(ann.rbox.cy - 30.0) < 1e-3
    assert abs(ann.rbox.area - 80.0 * 40.0) < 1e-2


def test_min_area_rect_on_ngon_recovers_rectangle():
    # Axis-aligned 100×40 rectangle with extra edge midpoints (n-gon).
    coords = [0.0, 0.0, 50.0, 0.0, 100.0, 0.0, 100.0, 40.0, 50.0, 40.0, 0.0, 40.0]
    corners = min_area_rect_corners(
        [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    )
    ann = annotation_from_polygon_coords(coords, "ship")
    assert len(corners) == 8
    assert abs(ann.rbox.area - 100.0 * 40.0) < 1.0
    assert abs(ann.rbox.cx - 50.0) < 1.0
    assert abs(ann.rbox.cy - 20.0) < 1.0


def _write_mini_coco(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir(parents=True)
    Image.new("L", (80, 60), color=40).save(images / "a.png")
    Image.new("RGB", (80, 60), color=(10, 20, 30)).save(images / "b.png")
    payload = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 80, "height": 60},
            {"id": 2, "file_name": "b.png", "width": 80, "height": 60},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[10.0, 10.0, 70.0, 10.0, 70.0, 40.0, 10.0, 40.0]],
                "bbox": [10.0, 10.0, 60.0, 30.0],
                "iscrowd": 0,
                "area": 1800.0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "segmentation": [
                    [5.0, 5.0, 25.0, 5.0, 40.0, 5.0, 40.0, 25.0, 25.0, 25.0, 5.0, 25.0]
                ],
                "bbox": [5.0, 5.0, 35.0, 20.0],
                "iscrowd": 0,
                "area": 700.0,
            },
        ],
        "categories": [{"id": 1, "name": "ship"}],
    }
    ann_file = root / "instances.json"
    ann_file.write_text(json.dumps(payload), encoding="utf-8")
    return ann_file, images


def test_parse_coco_instances_quad_and_ngon(tmp_path: Path):
    ann_file, _images = _write_mini_coco(tmp_path)
    records = parse_coco_instances(ann_file)
    assert len(records) == 2
    by_name = {r.file_name: r for r in records}
    assert len(by_name["a.png"].annotations) == 1
    assert by_name["a.png"].annotations[0].class_name == "ship"
    assert abs(by_name["a.png"].annotations[0].rbox.area - 60.0 * 30.0) < 1e-2
    assert len(by_name["b.png"].annotations) == 1
    assert by_name["b.png"].annotations[0].rbox.area > 0


def test_parse_coco_skips_crowd_and_rle(tmp_path: Path):
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 20, "height": 20}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": {"counts": [1, 2, 3], "size": [20, 20]},
                "iscrowd": 1,
                "bbox": [0, 0, 10, 10],
            }
        ],
        "categories": [{"id": 1, "name": "ship"}],
    }
    ann_file = tmp_path / "c.json"
    ann_file.write_text(json.dumps(payload), encoding="utf-8")
    records = parse_coco_instances(ann_file)
    assert len(records) == 1
    assert records[0].annotations == tuple()


def test_export_coco_json_to_dota_round_trip(tmp_path: Path):
    ann_file, images = _write_mini_coco(tmp_path / "coco")
    out = tmp_path / "dota"
    n = export_coco_json_to_dota(ann_file, images, out, split="train")
    assert n == 2
    exported = list((out / "train" / "images").glob("*.png"))
    assert len(exported) == 2
    with Image.open(out / "train" / "images" / "a.png") as img:
        assert img.mode == "RGB"
        assert img.size == (80, 60)
    dota = DOTADataset(
        root_dir=out / "train",
        split="train",
        label_dir=out / "train" / "labels",
        image_dir=out / "train" / "images",
        difficult_strategy="keep",
    )
    assert len(dota) == 2
    sample = next(s for s in dota if s.image_path.stem == "a")
    assert len(sample.annotations) == 1
    assert sample.annotations[0].class_name == "ship"
