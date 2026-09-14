"""Offline augmentation: geometric copies of the train split, on disk.

`testbank make-augmented` reads the TRAIN split of one split version and
writes, for every photo, `copies` rotated / sheared / perspective-warped
copies with their labels, into `data/augmented/vN/` -- versioned like the
splits, never overwritten, with a manifest that says which split, which
seed, which recipe, and a contact sheet to look at before training on it.
`testbank train --augmented vN` adds those copies to the training set.

Why offline, when the loop already augments on the fly
------------------------------------------------------
Because the rotation is the part that decides whether the model learns the
banknotes that are not horizontal or vertical, and it is the part with
geometry that can go wrong (a label that drifts off its banknote, a banknote
cut or collapsed). Written to disk it can be REVIEWED: the sheet, the
labels, any file. And it is reproducible: the copies of a photo depend only
on the seed and the photo's id, not on the order of generation. The on-the-
fly recipe (mosaic, scale, HSV, flips -- Ultralytics') still runs on top
with `--augment`, on originals and copies alike.

Leakage
-------
Only the train split is read, and the manifest records the split version and
its digest. `train --augmented` refuses a version made from another split:
a copy of a photo that is in `valid` of the split being trained on would be
seen in training, and the validation number would be a lie.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from testbank.config import Config, OfflineAugmentConfig
from testbank.data.discover import Sample
from testbank.dataio.formats import get as get_format
from testbank.dataio.obb_yolo import read_label_file
from testbank.geometry.quad import CoordinateRangeWarning, QuadShapeWarning
from testbank.models.augment import (
    FILL,
    _corners_px,
    background_colour,
    random_perspective,
)

MANIFEST_NAME = "manifest.json"
CONTACT_SHEET = "_contact_sheet.png"
_VERSION = re.compile(r"^v(\d+)$")
#: `<source id>__aug<k>`: the source is readable from the name, and `__aug`
#: cannot collide with a Roboflow export id.
SUFFIX = "__aug"


class AugmentedDataError(RuntimeError):
    pass


def list_versions(container: str | Path) -> list[str]:
    container = Path(container)
    if not container.is_dir():
        return []
    found = []
    for child in container.iterdir():
        match = _VERSION.match(child.name)
        if match and child.is_dir() and (child / MANIFEST_NAME).is_file():
            found.append((int(match.group(1)), child.name))
    return [name for _, name in sorted(found)]


def next_version(container: str | Path) -> str:
    versions = list_versions(container)
    if not versions:
        return "v1"
    return f"v{int(_VERSION.match(versions[-1]).group(1)) + 1}"


def resolve_version(container: str | Path, version: str | None = None) -> Path:
    container = Path(container)
    versions = list_versions(container)
    if not versions:
        raise AugmentedDataError(
            f"no augmented data in {container}. Run `testbank make-augmented` first"
        )
    if version is None:
        return container / versions[-1]
    if version not in versions:
        raise AugmentedDataError(
            f"augmented version {version!r} does not exist in {container}; available: {versions}"
        )
    return container / version


def _rng_for(seed: int, sample_id: str) -> random.Random:
    """One generator per photo, from the seed and the id: the copies of a
    photo do not depend on which photos came before it."""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _square(image: np.ndarray, fill: tuple[int, int, int]) -> np.ndarray:
    """The photo centred on a square canvas of its longer side. Most photos
    are square already (489 of 502); the rest keep their aspect here, and
    the training resize distorts originals and copies alike."""
    h, w = image.shape[:2]
    if h == w:
        return image
    side = max(h, w)
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y : y + h, x : x + w] = image
    return canvas


@dataclass(frozen=True, slots=True)
class AugmentedOutput:
    source_id: str
    sample_id: str
    image_path: Path
    label_path: Path
    n_boxes: int


def augment_sample(
    sample: Sample,
    recipe: OfflineAugmentConfig,
    *,
    output_dir: Path,
) -> list[AugmentedOutput]:
    """`recipe.copies` warped copies of one photo, written as `<id>__augK`."""
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise AugmentedDataError(f"could not read {sample.image_path}")
    labels = read_label_file(sample.label_path)
    class_ids = [a.class_id for a in labels.annotations]
    fill = background_colour(image) if recipe.fill == "border" else (FILL, FILL, FILL)
    h, w = image.shape[:2]
    # Corners in the square canvas's pixels: the quads are normalized over
    # the photo, the canvas is the photo plus padding.
    side = max(h, w)
    offset = np.array([(side - w) / 2.0, (side - h) / 2.0])
    corners = [
        np.array([[x * w, y * h] for x, y in a.quad.points]) + offset for a in labels.annotations
    ]
    canvas = _square(image, fill)
    rng = _rng_for(recipe.seed, sample.sample_id)
    writer = get_format("obb_yolo")
    outputs = []
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuadShapeWarning)
        warnings.simplefilter("ignore", CoordinateRangeWarning)
        for k in range(1, recipe.copies + 1):
            warped, quads = random_perspective(
                canvas, corners, rng, out_side=side, degrees=recipe.degrees,
                scale=recipe.scale, translate=recipe.translate, shear=recipe.shear,
                perspective=recipe.perspective, min_visible=recipe.min_visible,
                keep_whole=recipe.keep_whole, fill=fill,
            )
            sample_id = f"{sample.sample_id}{SUFFIX}{k}"
            image_path = output_dir / "images" / f"{sample_id}.jpg"
            label_path = output_dir / "labels" / f"{sample_id}.txt"
            if not cv2.imwrite(str(image_path), warped, [cv2.IMWRITE_JPEG_QUALITY, recipe.jpeg_quality]):
                raise AugmentedDataError(f"could not write {image_path}")
            # `_survivors` keeps the order of the boxes it keeps, but drops
            # some: the class of a surviving box is found by its corners.
            lines = []
            for quad in quads:
                class_id = _class_of(quad, corners, class_ids, side)
                lines.append(f"{class_id} {writer.from_quad(quad, class_id=class_id).payload}")
            label_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            outputs.append(AugmentedOutput(sample.sample_id, sample_id, image_path, label_path, len(quads)))
    return outputs


def _class_of(quad, source_corners, class_ids, side) -> int:
    """All banknotes are class 0 today; kept general by nearest source box
    centre, in case the export ever carries several classes."""
    if len(set(class_ids)) <= 1:
        return class_ids[0] if class_ids else 0
    centre = _corners_px(quad, side).mean(axis=0)
    distances = [np.linalg.norm(pts.mean(axis=0) - centre) for pts in source_corners]
    return class_ids[int(np.argmin(distances))]


def materialize_augmented(
    samples,
    config: Config,
    *,
    split: dict,
    container: str | Path | None = None,
    version: str | None = None,
    log=print,
) -> Path:
    """Writes a new version with the copies of `samples` (the train split)
    and its manifest and contact sheet. Refuses to overwrite a version."""
    recipe = config.offline_augment
    container = Path(container or config.data.augmented_dir)
    version = version or next_version(container)
    if not _VERSION.match(version):
        raise AugmentedDataError(f"a version is `vN`, got {version!r}")
    output_dir = container / version
    if output_dir.exists():
        raise AugmentedDataError(
            f"{output_dir} already exists: versions are never overwritten, make a new one"
        )
    samples = list(samples)
    outputs: list[AugmentedOutput] = []
    for index, sample in enumerate(samples, 1):
        outputs.extend(augment_sample(sample, recipe, output_dir=output_dir))
        if log and (index % 50 == 0 or index == len(samples)):
            log(f"  {index}/{len(samples)} photos -> {len(outputs)} copies", flush=True)
    # Sorted by id: the manifest and the digest do not depend on the order
    # the photos were given in.
    outputs.sort(key=lambda o: o.sample_id)
    empty = [o.sample_id for o in outputs if o.n_boxes == 0]
    manifest = {
        "version": version,
        "split": dict(split),
        "recipe": recipe.model_dump(),
        "sources": len(samples),
        "copies": len(outputs),
        "boxes": {
            "source": sum(len(read_label_file(s.label_path).annotations) for s in samples),
            "augmented": sum(o.n_boxes for o in outputs),
        },
        "empty": empty,
        "digest": _digest(outputs),
        "files": [
            {"source": o.source_id, "id": o.sample_id, "boxes": o.n_boxes} for o in outputs
        ],
    }
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    contact_sheet(outputs, output_dir / CONTACT_SHEET, seed=recipe.seed)
    return output_dir


def _digest(outputs: list[AugmentedOutput]) -> str:
    """Over the labels and the image bytes: two versions with the same
    digest are the same data, whatever their number."""
    h = hashlib.sha256()
    for o in outputs:
        h.update(o.sample_id.encode())
        h.update(o.label_path.read_bytes())
        h.update(o.image_path.read_bytes())
    return h.hexdigest()


def contact_sheet(
    outputs: list[AugmentedOutput], path: Path, *, seed: int, count: int = 24, tile: int = 256
) -> Path:
    """Up to `count` copies, a fixed random pick, with their boxes drawn: the
    thing to look at before training."""
    picked = sorted(outputs, key=lambda o: o.sample_id)
    picked = random.Random(seed).sample(picked, min(count, len(picked)))
    tiles = []
    for o in picked:
        image = cv2.imread(str(o.image_path), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        for a in read_label_file(o.label_path).annotations:
            pts = np.array([[x * tile, y * tile] for x, y in a.quad.points], dtype=np.int32)
            cv2.polylines(image, [pts], True, (40, 190, 255), 2, cv2.LINE_AA)
            cv2.circle(image, tuple(pts[0]), 4, (40, 190, 255), -1, cv2.LINE_AA)
        cv2.putText(image, o.sample_id[-14:], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    if not tiles:
        tiles = [np.full((tile, tile, 3), FILL, dtype=np.uint8)]
    columns = 6
    while len(tiles) % columns:
        tiles.append(np.full((tile, tile, 3), FILL, dtype=np.uint8))
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def load_manifest(directory: str | Path) -> dict:
    return json.loads((Path(directory) / MANIFEST_NAME).read_text(encoding="utf-8"))


def load_augmented(directory: str | Path, *, split: dict | None = None) -> tuple[list[Sample], dict]:
    """The copies of a version as `Sample`s, to add to the train split.

    `split` is the description of the split about to be trained on: the
    version has to have been made from the same one, or it is refused.
    """
    directory = Path(directory)
    manifest = load_manifest(directory)
    if split is not None:
        made_from = manifest.get("split") or {}
        same = made_from.get("version") == split.get("version") and (
            made_from.get("digest") == split.get("digest")
        )
        if not same:
            raise AugmentedDataError(
                f"{directory.name} was made from split {made_from.get('version')!r} "
                f"(digest {str(made_from.get('digest'))[:12]}...), not from "
                f"{split.get('version')!r}: its copies could include valid or test "
                "photos of this split. Make a new version with `make-augmented`."
            )
    samples = []
    for entry in manifest["files"]:
        samples.append(
            Sample(
                sample_id=entry["id"],
                image_path=directory / "images" / f"{entry['id']}.jpg",
                label_path=directory / "labels" / f"{entry['id']}.txt",
            )
        )
    return samples, manifest


def describe(manifest: dict) -> dict:
    """What goes into `run.json["augmented"]`."""
    return {
        "version": manifest["version"],
        "digest": manifest["digest"],
        "copies": manifest["copies"],
        "sources": manifest["sources"],
        "recipe": manifest["recipe"],
    }


__all__ = [
    "CONTACT_SHEET",
    "MANIFEST_NAME",
    "SUFFIX",
    "AugmentedDataError",
    "AugmentedOutput",
    "augment_sample",
    "contact_sheet",
    "describe",
    "list_versions",
    "load_augmented",
    "load_manifest",
    "materialize_augmented",
    "next_version",
    "resolve_version",
]
