import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("PIL")

from yolox_obb.data import (  # noqa: E402
    StrongAug,
    YoloObbDataset,
    letterbox_image,
    read_split,
    resize_image,
)


def _write_sample(root, split="valid", name="sample.jpg"):
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image = np.full((20, 40, 3), 80, np.uint8)
    cv2.imwrite(str(image_dir / name), image)
    # One square in normalized YOLO-OBB format.
    (label_dir / (name.removesuffix(".jpg") + ".txt")).write_text(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"
    )


def test_dataset_split_and_sample_from_temporary_export(tmp_path):
    _write_sample(tmp_path)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "valid.txt").write_text("sample\n# comments are ignored\n")

    ds = YoloObbDataset(str(tmp_path), "valid", img_size=32, split_dir=str(split_dir))
    item = ds[0]
    assert len(ds) == 1
    assert tuple(item["image"].shape) == (3, 32, 32)
    assert tuple(item["boxes"].shape) == (1, 5)
    assert torch.isfinite(item["boxes"]).all()
    assert read_split(str(split_dir / "valid.txt")) == ["sample"]


def test_resize_and_letterbox_keep_bgr_shape_and_padding():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, :, 0] = 17
    resized = resize_image(image, 20, 10)
    assert resized.shape == (10, 20, 3)
    tensor, scale = letterbox_image(image, 32)
    assert tuple(tensor.shape) == (3, 32, 32)
    assert scale == pytest.approx(0.8)
    assert int(tensor[1, 31, 31]) == 114
    assert int(tensor[0, 0, 0]) == 17


def test_strong_aug_stage2_returns_finite_shapes(tmp_path):
    _write_sample(tmp_path)
    ds = YoloObbDataset(
        str(tmp_path), "valid", img_size=32, train=True,
        filter_empty=True,
        strong_aug=StrongAug(
            mosaic_prob=0, mixup_prob=0, rotate_prob=0, flip_prob=0,
            stage2=True, stage2_resize_range=(1, 1),
        ),
    )
    item = ds[0]
    assert tuple(item["image"].shape) == (3, 32, 32)
    assert item["boxes"].shape[1] == 5
    assert torch.isfinite(item["boxes"]).all()
    assert torch.isfinite(item["image"].float()).all()
