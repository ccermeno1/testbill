"""Offline augmentation: copies of the train split on disk, reviewable and
reproducible, refused when they come from another split."""

from __future__ import annotations

import json
import random

import cv2
import numpy as np
import pytest
from conftest import rotated_rect_points
from PIL import Image

from testbank.config import Config
from testbank.data.augmented import (
    CONTACT_SHEET,
    SUFFIX,
    AugmentedDataError,
    augment_sample,
    describe,
    list_versions,
    load_augmented,
    load_manifest,
    materialize_augmented,
    next_version,
    resolve_version,
)
from testbank.data.discover import Sample
from testbank.dataio.formats import get as get_format
from testbank.dataio.obb_yolo import read_label_file
from testbank.geometry.quad import Quad, canonicalize, is_canonical

SPLIT = {"version": "v1", "mode": "adopt", "digest": "abc123", "counts": {"train": 3}}


def _quad(cx, cy, half_long=0.25, ratio=2.0, theta=0.4) -> Quad:
    return canonicalize(Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta)))


def _write(directory, sample_id, quads, size=(96, 96), table=(30, 90, 200)) -> Sample:
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "labels").mkdir(parents=True, exist_ok=True)
    w, h = size
    image = np.full((h, w, 3), table, dtype=np.uint8)
    for q in quads:
        pts = np.array([[x * w, y * h] for x, y in q.points], dtype=np.int32)
        cv2.fillPoly(image, [pts], (255, 255, 255))
    Image.fromarray(image[..., ::-1]).save(directory / "images" / f"{sample_id}.jpg", quality=95)
    fmt = get_format("obb_yolo")
    (directory / "labels" / f"{sample_id}.txt").write_text(
        "".join(f"0 {fmt.from_quad(q).payload}\n" for q in quads), encoding="utf-8"
    )
    return Sample(sample_id, directory / "images" / f"{sample_id}.jpg", directory / "labels" / f"{sample_id}.txt")


@pytest.fixture()
def train(tmp_path):
    src = tmp_path / "src"
    return [
        _write(src, "a", [_quad(0.5, 0.5)]),
        _write(src, "b", [_quad(0.3, 0.4, half_long=0.15), _quad(0.7, 0.6, half_long=0.15, theta=1.2)]),
        _write(src, "c", [_quad(0.5, 0.5, half_long=0.3, ratio=3.0)], size=(120, 80)),
    ]


def _config(tmp_path, **recipe) -> Config:
    base = Config()
    return base.model_copy(update={
        "data": base.data.model_copy(update={"augmented_dir": tmp_path / "aug"}),
        "offline_augment": base.offline_augment.model_copy(update={"copies": 2, **recipe}),
    })


# --- what gets written ----------------------------------------------------


def test_every_photo_gets_its_copies_with_labels_that_parse_and_sit_inside(train, tmp_path):
    out = materialize_augmented(train, _config(tmp_path), split=SPLIT)
    assert out == tmp_path / "aug" / "v1"
    manifest = load_manifest(out)
    assert manifest["sources"] == 3 and manifest["copies"] == 6
    assert manifest["split"] == SPLIT
    assert manifest["recipe"]["degrees"] == 180.0 and manifest["recipe"]["copies"] == 2
    assert manifest["boxes"] == {"source": 4, "augmented": 8}, "keep_whole: nothing dropped"
    assert manifest["empty"] == []
    for entry in manifest["files"]:
        assert entry["id"].startswith(entry["source"] + SUFFIX)
        image = cv2.imread(str(out / "images" / f"{entry['id']}.jpg"))
        assert image is not None and image.shape[0] == image.shape[1], "square canvas"
        labels = read_label_file(out / "labels" / f"{entry['id']}.txt")
        assert len(labels.annotations) == entry["boxes"]
        for a in labels.annotations:
            assert a.class_id == 0 and is_canonical(a.quad)
            assert all(0.0 <= v <= 1.0 for xy in a.quad.points for v in xy)
    assert (out / CONTACT_SHEET).is_file()
    sheet = cv2.imread(str(out / CONTACT_SHEET))
    assert sheet.shape[1] == 6 * 256


def test_the_label_is_on_the_banknote(train, tmp_path):
    """The copy's white pixels must be where its label says: the quad
    follows the pixels through the offline warp too."""
    out = materialize_augmented(train, _config(tmp_path, degrees=180.0, shear=2.0), split=SPLIT)
    for entry in load_manifest(out)["files"]:
        if entry["source"] != "a":
            continue
        image = cv2.imread(str(out / "images" / f"{entry['id']}.jpg"))
        side = image.shape[0]
        painted = image[..., 0] > 127
        for a in read_label_file(out / "labels" / f"{entry['id']}.txt").annotations:
            pts = np.array([[x * side * 256, y * side * 256] for x, y in a.quad.points]).astype(np.int32)
            mask = np.zeros((side, side), np.uint8)
            cv2.fillPoly(mask, [pts], 1, shift=8)
            inter = np.logical_and(painted, mask).sum()
            union = np.logical_or(painted, mask).sum()
            assert inter / union > 0.85


def test_the_canvas_takes_the_photos_own_background(train, tmp_path):
    out = materialize_augmented(train, _config(tmp_path, fill="border"), split=SPLIT)
    entry = next(e for e in load_manifest(out)["files"] if e["source"] == "c")
    image = cv2.imread(str(out / "images" / f"{entry['id']}.jpg"))
    # `c` is 120x80: padded to a 120 square, the padding must be table blue, not gray.
    corner = image[2, 2].astype(int)
    assert abs(corner[0] - 30) < 12 and abs(corner[2] - 200) < 12, corner


# --- reproducible, versioned ----------------------------------------------


def test_the_same_seed_gives_byte_identical_copies_whatever_the_order(train, tmp_path):
    first = materialize_augmented(train, _config(tmp_path), split=SPLIT)
    second = materialize_augmented(list(reversed(train)), _config(tmp_path), split=SPLIT)
    assert (first.name, second.name) == ("v1", "v2")
    assert load_manifest(first)["digest"] == load_manifest(second)["digest"]
    for entry in load_manifest(first)["files"]:
        a = (first / "labels" / f"{entry['id']}.txt").read_bytes()
        b = (second / "labels" / f"{entry['id']}.txt").read_bytes()
        assert a == b
    third = materialize_augmented(train, _config(tmp_path, seed=1), split=SPLIT)
    assert load_manifest(third)["digest"] != load_manifest(first)["digest"]


def test_versions_are_never_overwritten_and_resolve_to_the_latest(train, tmp_path):
    materialize_augmented(train, _config(tmp_path), split=SPLIT)
    with pytest.raises(AugmentedDataError, match="never overwritten"):
        materialize_augmented(train, _config(tmp_path), split=SPLIT, version="v1")
    materialize_augmented(train, _config(tmp_path), split=SPLIT)
    assert list_versions(tmp_path / "aug") == ["v1", "v2"]
    assert next_version(tmp_path / "aug") == "v3"
    assert resolve_version(tmp_path / "aug").name == "v2"
    assert resolve_version(tmp_path / "aug", "v1").name == "v1"
    with pytest.raises(AugmentedDataError, match="does not exist"):
        resolve_version(tmp_path / "aug", "v9")
    with pytest.raises(AugmentedDataError, match="make-augmented"):
        resolve_version(tmp_path / "nowhere")


# --- into training ----------------------------------------------------------


def test_the_copies_load_as_samples_and_another_split_is_refused(train, tmp_path):
    out = materialize_augmented(train, _config(tmp_path), split=SPLIT)
    samples, manifest = load_augmented(out, split=SPLIT)
    assert len(samples) == 6
    assert {s.sample_id.split(SUFFIX)[0] for s in samples} == {"a", "b", "c"}
    assert all(s.image_path.is_file() and s.label_path.is_file() for s in samples)
    assert describe(manifest)["version"] == "v1" and describe(manifest)["copies"] == 6
    other = {**SPLIT, "version": "v2", "digest": "zzz"}
    with pytest.raises(AugmentedDataError, match="could include valid or test"):
        load_augmented(out, split=other)
    # Same version name but a different digest is another split too.
    with pytest.raises(AugmentedDataError):
        load_augmented(out, split={**SPLIT, "digest": "other"})


def test_augment_sample_writes_copies_numbered_from_one(train, tmp_path):
    outputs = augment_sample(train[0], _config(tmp_path).offline_augment, output_dir=tmp_path / "o")
    assert [o.sample_id for o in outputs] == [f"a{SUFFIX}1", f"a{SUFFIX}2"]
    assert all(o.n_boxes == 1 for o in outputs)


def test_the_run_records_the_augmentation(tmp_path):
    from testbank.experiment.run import ExperimentRun

    run = ExperimentRun.create(
        "x", Config(), runs_dir=tmp_path / "runs", split=SPLIT,
        augmented={"version": "v1", "digest": "d", "copies": 6, "sources": 3, "recipe": {}},
    )
    record = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))
    assert record["augmented"]["version"] == "v1" and record["augmented"]["copies"] == 6


def test_the_cli_knows_both_sides():
    from testbank.cli import build_parser

    args = build_parser().parse_args(["make-augmented", "--copies", "4", "--degrees", "45"])
    assert (args.command, args.copies, args.degrees, args.version) == ("make-augmented", 4, 45.0, None)
    args = build_parser().parse_args(["train", "yolox-obb-nano", "--augmented", "v2", "--augment"])
    assert args.augmented == "v2" and args.augment is True


def test_the_online_recipe_is_back_to_ultralytics_geometry():
    """Rotation, shear and perspective live offline now; on the fly the
    defaults are theirs again, so `--augment` on top adds only mosaic,
    scale, HSV and flips."""
    cfg = Config()
    online, offline = cfg.detector.augment, cfg.offline_augment
    assert (online.degrees, online.shear, online.perspective) == (0.0, 0.0, 0.0)
    assert online.keep_whole is False and online.fill == "gray"
    assert (offline.degrees, offline.shear, offline.perspective) == (180.0, 2.0, 0.0002)
    assert offline.keep_whole is True and offline.fill == "border" and offline.scale == 0.0
    assert random.Random(offline.seed) is not None
