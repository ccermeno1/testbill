"""Exporters: file tree, contents and format compatibility."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from conftest import rotated_rect_points
from PIL import Image

from testbank.config import Config
from testbank.data.discover import Sample
from testbank.dataio import export as export_mod
from testbank.dataio.export import (
    Exporter,
    ExportError,
    export,
    exporters,
    get,
    register,
)
from testbank.dataio.formats import ImageSize, Record
from testbank.dataio.formats import get as get_format
from testbank.geometry.quad import Quad, canonicalize

SIZE = (200, 160)


def _quad(cx=0.5, cy=0.5, half_long=0.15, ratio=2.0, theta=0.0) -> Quad:
    """Rotated rectangle built in PIXELS, like the real banknotes."""
    points = rotated_rect_points(
        cx * SIZE[0], cy * SIZE[1], half_long * SIZE[0], ratio, theta
    )
    return canonicalize(
        Quad.from_xy([(x / SIZE[0], y / SIZE[1]) for x, y in points]),
        aspect=SIZE[0] / SIZE[1],
    )


def _write(directory: Path, sample_id: str, quads) -> Sample:
    images = directory / "images"
    labels = directory / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image_path = images / f"{sample_id}.jpg"
    Image.new("RGB", SIZE, (20, 20, 20)).save(image_path)
    writer = get_format("obb_yolo")
    label_path = labels / f"{sample_id}.txt"
    label_path.write_text(
        "\n".join(f"0 {writer.from_quad(q).payload}" for q in quads) + "\n",
        encoding="utf-8",
    )
    return Sample(sample_id=sample_id, image_path=image_path, label_path=label_path)


@pytest.fixture()
def config(tmp_path) -> Config:
    base = Config()
    return base.model_copy(
        update={"data": base.data.model_copy(update={"derived_dir": tmp_path / "d"})}
    )


@pytest.fixture()
def samples(tmp_path):
    quads = (_quad(0.3, 0.5, theta=0.4), _quad(0.7, 0.5))
    return {
        "train": [_write(tmp_path / "src", f"t{i}", quads) for i in range(3)],
        "valid": [_write(tmp_path / "src", f"v{i}", quads) for i in range(2)],
    }


# --- the registry ---------------------------------------------------------


def test_all_exporters_are_there():
    assert exporters() == ["coco", "dota"]


@pytest.mark.parametrize("name", ["coco", "dota"])
def test_they_satisfy_the_protocol(name):
    assert isinstance(get(name), Exporter)


def test_unknown_exporter_says_which_exist():
    with pytest.raises(ExportError, match="unknown"):
        get("does_not_exist")


def test_registering_a_repeated_name_is_an_error():
    with pytest.raises(ExportError, match="duplicate"):

        @register
        class Other:
            name = "dota"
            annotation_format = "dota"


# --- DOTA -----------------------------------------------------------------


def test_dota_writes_the_expected_tree(tmp_path, config, samples):
    result = export("dota", samples, config, out_dir=tmp_path / "out")
    for split, count in (("train", 3), ("valid", 2)):
        assert len(list((result.root / split / "images").glob("*.jpg"))) == count
        assert len(list((result.root / split / "labelTxt").glob("*.txt"))) == count


def test_dota_writes_pixels_and_category(tmp_path, config, samples):
    result = export("dota", samples, config, out_dir=tmp_path / "out")
    line = (result.root / "train" / "labelTxt" / "t0.txt").read_text().splitlines()[0]
    fields = line.split()
    assert len(fields) == 10
    assert fields[8] == "euro_banknote"
    assert fields[9] == "0"
    # Pixels, not normalized: with 200x160 the values go well past 1.
    assert max(float(v) for v in fields[:8]) > 2.0


def test_dota_round_trips(tmp_path, config, samples):
    """Lossless: what is written has to rebuild the original quad."""
    original = _quad(0.3, 0.5, theta=0.4)
    result = export("dota", samples, config, out_dir=tmp_path / "out")
    line = (result.root / "train" / "labelTxt" / "t0.txt").read_text().splitlines()[0]

    size = ImageSize(*SIZE)
    back = get_format("dota").to_quad(Record(0, line), size=size)
    error = max(
        math.dist((a[0] * SIZE[0], a[1] * SIZE[1]), (b[0] * SIZE[0], b[1] * SIZE[1]))
        for a, b in zip(original.points, back.points)
    )
    assert error < 1e-3


# --- COCO -----------------------------------------------------------------


def test_coco_writes_one_json_per_split(tmp_path, config, samples):
    result = export("coco", samples, config, out_dir=tmp_path / "out")
    for split, count in (("train", 3), ("valid", 2)):
        data = json.loads((result.root / split / "annotations.json").read_text())
        assert len(data["images"]) == count
        assert len(data["annotations"]) == count * 2
        assert data["categories"] == [{"id": 0, "name": "euro_banknote"}]
    assert result.entry_point.name == "annotations.json"


def test_coco_numbers_without_repeating(tmp_path, config, samples):
    data = json.loads(
        (
            export("coco", samples, config, out_dir=tmp_path / "out").root
            / "train"
            / "annotations.json"
        ).read_text()
    )
    ids = [a["id"] for a in data["annotations"]]
    assert len(ids) == len(set(ids))
    assert {a["image_id"] for a in data["annotations"]} == {1, 2, 3}


def test_coco_warns_that_it_loses_the_orientation(tmp_path, config, samples):
    data = json.loads(
        (
            export("coco", samples, config, out_dir=tmp_path / "out").root
            / "train"
            / "annotations.json"
        ).read_text()
    )
    assert "loses the orientation" in data["info"]["note"]
    assert get_format("bbox_coco").lossy is True


# --- what is shared -------------------------------------------------------


@pytest.mark.parametrize("name", ["dota", "coco"])
def test_all_respect_the_area_filter(tmp_path, name):
    """If an exporter skipped the filter, that candidate would train on a
    different truth than the others and the table would compare them as equals."""
    large = _quad(0.5, 0.5, half_long=0.30)
    strip = _quad(0.5, 0.5, half_long=0.30, ratio=30.0)
    sample = _write(tmp_path / "src", "a", [large, strip])

    base = Config()
    config = base.model_copy(
        update={"data": base.data.model_copy(update={"derived_dir": tmp_path / "d"})}
    )
    result = export(name, {"train": [sample]}, config, out_dir=tmp_path / "out")
    assert result.report.dropped == 1

    if name == "dota":
        lines = (result.root / "train" / "labelTxt" / "a.txt").read_text().splitlines()
        assert len(lines) == 1
    else:
        data = json.loads((result.root / "train" / "annotations.json").read_text())
        assert len(data["annotations"]) == 1


def test_the_report_says_which_border_policy_was_used(tmp_path, config, samples):
    result = export("dota", samples, config, out_dir=tmp_path / "out")
    assert "border=clip" in result.report.describe()
    assert result.report.counts == {"train": 3, "valid": 2}


def test_exporting_twice_does_not_accumulate(tmp_path, config, samples):
    export("dota", samples, config, out_dir=tmp_path / "out")
    result = export("dota", samples, config, out_dir=tmp_path / "out")
    assert len(list((result.root / "train" / "labelTxt").glob("*.txt"))) == 3


def test_adding_an_exporter_does_not_touch_export(tmp_path, config, samples):
    """The registry is the only thing that decides: `export` knows no name."""

    @register
    class Counter:
        name = "_test_counter"
        annotation_format = "obb_yolo"
        requires_rectangles = False

        def write(self, prepared, root, *, class_names):
            target = root / "count.txt"
            target.write_text(
                str(sum(len(v) for v in prepared.values())), encoding="utf-8"
            )
            return target

    try:
        result = export(
            "_test_counter", samples, config, out_dir=tmp_path / "out"
        )
        assert result.entry_point.read_text() == "5"
    finally:
        del export_mod.REGISTRY["_test_counter"]
