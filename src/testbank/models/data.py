"""Training dataset of the own candidate.

It reads through the same door as everything else: `dataio/prepare.prepare`,
which applies the relative area filter and the policy for banknotes crossing
the border. If this loaded the files on its own, the own candidate would train
on a different truth than Ultralytics and the table would compare them as
equals.

About the resizing
------------------
It rescales directly to `image_size x image_size`, without letterbox. It might
look careless, but it is what is consistent with these data: 489 of the 502
images ALREADY arrive at 416x416 from the source, deformed to a square by the
export. The real aspect of the banknotes was lost before we touched anything,
so adding letterbox now recovers nothing -- it only adds black bands and a
second geometry to explain. See the README note about the 19% of nearly square
boxes.

If at some point the originals are re-uploaded without resizing, this has to
be revisited: there the letterbox would be worth it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from testbank.config import Config
from testbank.dataio.prepare import PreparedSample, prepare
from testbank.geometry.quad import Quad


def image_to_input(image_bgr: np.ndarray) -> torch.Tensor:
    """OpenCV image (BGR, uint8) -> `(3, H, W)` tensor fed to the network.

    **Raw BGR in 0-255, not normalized.** It is the convention of YOLOX (Megvii)
    and of DDGRCF, and therefore the one the loaded pretrained weights expect:
    Megvii's COCO in the own head and DDGRCF's DOTA in its port. Before, RGB in
    [0, 1] was given -- swapped channels and a scale 255 times smaller -- and
    the pretraining arrived wrecked at the first layer: measured, the port with
    DOTA gave mAP 0.000 and maximum scores of 0.004 after one epoch.

    For training from scratch it does not matter (BatchNorm absorbs the scale),
    so the convention is fixed here, in a single place, and used by training
    and inference. Changing it in only one of them would invalidate every saved
    weight.
    """
    return torch.from_numpy(np.ascontiguousarray(image_bgr.transpose(2, 0, 1))).float()


def quad_to_box(quad: Quad, width: int, height: int) -> tuple[float, ...]:
    """Normalized quad -> `cx, cy, w, h, theta` in PIXELS.

    The canonical quad anchors `p0->p1` on the longest side, so `w` always
    comes out as the long side and `theta` as its orientation. That is not a
    coincidence being exploited: it is the reason the canonical order exists.
    """
    points = [(x * width, y * height) for x, y in quad.points]
    cx = sum(p[0] for p in points) / 4
    cy = sum(p[1] for p in points) / 4
    long_side = math.dist(points[0], points[1])
    short_side = math.dist(points[1], points[2])
    theta = math.atan2(
        points[1][1] - points[0][1], points[1][0] - points[0][0]
    ) % math.pi
    return cx, cy, long_side, short_side, theta


@dataclass(frozen=True, slots=True)
class Batch:
    """One batch. Boxes go in a list because every image has a different
    number of them and stacking would require padding with garbage that one
    must then remember to ignore."""

    images: torch.Tensor
    boxes: list[torch.Tensor]
    classes: list[torch.Tensor]
    sample_ids: list[str]

    def __len__(self) -> int:
        return self.images.shape[0]


class BanknoteDataset(Dataset):
    """Images and oriented boxes, already filtered and at the input size.

    `augment` (an `AugmentConfig` with `enabled`) applies the training-time
    transforms of `models/augment.py` AFTER the resize to the square input,
    so the 90-degree rotations are exact. Only the `train` dataset gets one:
    `valid` and `test` are measured on the real photos.
    """

    def __init__(
        self,
        prepared: list[PreparedSample],
        image_size: int,
        *,
        augment=None,
        seed: int = 0,
        epochs: int = 1,
    ) -> None:
        self.items = list(prepared)
        self.image_size = image_size
        self.augment = augment if augment is not None and augment.enabled else None
        # Own generator, drawn in sampler order: deterministic with
        # `num_workers = 0`, and independent of torch's global RNG.
        self._rng = random.Random(seed)
        self.epochs = epochs
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """The loop says which epoch it is: the mosaic switches off for the
        last `close_mosaic` epochs, as in Ultralytics."""
        self.epoch = epoch

    @property
    def mosaic_active(self) -> bool:
        a = self.augment
        return a is not None and a.mosaic > 0 and self.epoch < self.epochs - a.close_mosaic

    def _read_square(self, item: PreparedSample) -> np.ndarray:
        image = cv2.imread(str(item.sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"could not read {item.sample.image_path}")
        return cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = self._read_square(item)
        quads = list(item.quads)
        if self.augment is not None:
            from testbank.models.augment import augment

            a = self.augment

            def others():
                other = self.items[self._rng.randrange(len(self.items))]
                return self._read_square(other), list(other.quads)

            image, quads = augment(
                image, quads, self._rng,
                others=others if self.mosaic_active else None,
                mosaic=a.mosaic, scale=a.scale, translate=a.translate,
                hsv_h=a.hsv_h, hsv_s=a.hsv_s, hsv_v=a.hsv_v,
                flip_h=a.flip_horizontal, flip_v=a.flip_vertical, rotations=a.rotations,
                min_visible=a.min_visible,
            )
        tensor = image_to_input(image)

        # Quads are normalized, so the boxes are computed directly at the input
        # scale: no rescaling afterwards.
        boxes = [quad_to_box(quad, self.image_size, self.image_size) for quad in quads]
        return (
            tensor,
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5),
            torch.zeros(len(boxes), dtype=torch.long),
            item.sample_id,
        )


def collate(entries) -> Batch:
    images, boxes, classes, ids = zip(*entries)
    return Batch(
        images=torch.stack(images),
        boxes=list(boxes),
        classes=list(classes),
        sample_ids=list(ids),
    )


def build_datasets(
    samples_by_split: dict[str, list], config: Config
) -> dict[str, BanknoteDataset]:
    """One dataset per split, sharing filter and border policy."""
    prepared, _ = prepare(samples_by_split, config)
    return {
        split: BanknoteDataset(
            items,
            config.detector.image_size,
            augment=config.detector.augment if split == "train" else None,
            seed=config.metrics.seed,
            epochs=config.detector.epochs,
        )
        for split, items in prepared.items()
    }


__all__ = ["BanknoteDataset", "Batch", "build_datasets", "collate", "image_to_input", "quad_to_box"]
