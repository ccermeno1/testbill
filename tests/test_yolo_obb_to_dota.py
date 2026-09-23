"""tools/yolo_obb_to_dota.py: YOLO-OBB → DOTA layout, EXIF baking, class handling."""

import importlib
import sys
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
conv = importlib.import_module("yolo_obb_to_dota")


def _write_yolo(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(str(v) for v in r) for r in rows) + "\n", encoding="utf-8")


def _parse_dota(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        out.append(([float(v) for v in parts[:8]], parts[8], int(parts[9])))
    return out


def test_ultralytics_layout_with_names_and_negative(tmp_path):
    src = tmp_path / "yolo"
    (src / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(src / "images" / "train" / "a.png")
    Image.new("RGB", (50, 50)).save(src / "images" / "train" / "neg.PNG")
    _write_yolo(src / "labels" / "train" / "a.txt", [[1, 0.1, 0.2, 0.5, 0.2, 0.5, 0.6, 0.1, 0.6]])

    assert conv.main(["--src", str(src), "--dst", str(tmp_path / "out"), "--names", "5 eur", "10eur"]) == 0
    out = tmp_path / "out" / "train"
    assert (out / "images" / "a.png").is_file()
    assert (out / "images" / "neg.png").is_file()  # lower-case extension
    (coords, cls, diff), = _parse_dota(out / "labels" / "a.txt")
    assert coords == pytest.approx([20, 20, 100, 20, 100, 60, 20, 60])
    assert cls == "10eur" and diff == 0
    assert (out / "labels" / "neg.txt").read_text() == ""


def test_roboflow_layout_single_class_and_valid_alias(tmp_path):
    src = tmp_path / "rf"
    for split in ("train", "valid"):
        (src / split / "images").mkdir(parents=True)
        Image.new("RGB", (100, 100)).save(src / split / "images" / "x.jpg")
        _write_yolo(src / split / "labels" / "x.txt", [[3, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]])
    assert conv.main(["--src", str(src), "--dst", str(tmp_path / "o"), "--single-class", "banknote"]) == 0
    for split in ("train", "val"):
        (_, cls, _), = _parse_dota(tmp_path / "o" / split / "labels" / "x.txt")
        assert cls == "banknote"


def test_exif_orientation_is_baked_and_labels_follow_displayed_image(tmp_path):
    src = tmp_path / "yolo"
    img_dir = src / "images" / "train"
    img_dir.mkdir(parents=True)
    # Stored 300x100 landscape; EXIF orientation 6 → displayed 100x300 portrait.
    im = Image.new("RGB", (300, 100))
    exif = Image.Exif()
    exif[0x0112] = 6
    im.save(img_dir / "phone.JPG", exif=exif.tobytes())
    # Label drawn on the displayed (portrait) photo.
    _write_yolo(src / "labels" / "train" / "phone.txt", [[0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.5]])

    assert conv.main(["--src", str(src), "--dst", str(tmp_path / "out"), "--single-class", "b"]) == 0
    out_img = tmp_path / "out" / "train" / "images" / "phone.jpg"
    with Image.open(out_img) as im2:
        assert im2.size == (100, 300)
        assert im2.getexif().get(0x0112) in (None, 1)
    (coords, _, _), = _parse_dota(tmp_path / "out" / "train" / "labels" / "phone.txt")
    assert coords == pytest.approx([0, 0, 100, 0, 100, 150, 0, 150])


def test_max_side_downscales_labels_consistently(tmp_path):
    src = tmp_path / "yolo"
    (src / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (400, 200)).save(src / "images" / "train" / "big.png")
    _write_yolo(src / "labels" / "train" / "big.txt", [[0, 0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75]])
    conv.main(["--src", str(src), "--dst", str(tmp_path / "o"), "--single-class", "b", "--max-side", "200"])
    with Image.open(tmp_path / "o" / "train" / "images" / "big.png") as im:
        assert im.size == (200, 100)
    (coords, _, _), = _parse_dota(tmp_path / "o" / "train" / "labels" / "big.txt")
    assert coords == pytest.approx([50, 25, 150, 25, 150, 75, 50, 75])


def test_bad_label_line_fails_with_location(tmp_path):
    src = tmp_path / "yolo"
    (src / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(src / "images" / "train" / "a.png")
    _write_yolo(src / "labels" / "train" / "a.txt", [[0, 0.5, 0.5, 0.2, 0.2]])  # YOLO HBB, not OBB
    with pytest.raises(SystemExit, match=r"a\.txt:1"):
        conv.main(["--src", str(src), "--dst", str(tmp_path / "o")])


def test_converted_labels_load_with_dota_dataset(tmp_path):
    from oriented_det.data.dota import DOTADataset

    src = tmp_path / "yolo"
    (src / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (200, 200)).save(src / "images" / "train" / "a.jpg")
    _write_yolo(src / "labels" / "train" / "a.txt", [[0, 0.3, 0.2, 0.8, 0.4, 0.7, 0.65, 0.2, 0.45]])
    conv.main(["--src", str(src), "--dst", str(tmp_path / "o"), "--single-class", "banknote"])
    root = tmp_path / "o" / "train"
    ds = DOTADataset(root, label_dir=root / "labels", image_dir=root / "images")
    assert len(ds) == 1
    assert ds.get_class_names() == ["banknote"]
    assert ds[0].annotations[0].class_name == "banknote"
