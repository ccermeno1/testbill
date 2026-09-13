"""View of the dataset in the format Ultralytics reads.

Ultralytics, MMDetection and the YOLOX forks know nothing about `SplitLoader`
or the area filter: they read files from a directory tree. If we handed them
the data directory as is, they would train with TWO inconsistencies:

1. **The wrong split.** Ours is frozen in `splits/*.txt`, and the Roboflow
   tree can change. Everything reads from the txt files, never from the tree.

2. **Annotations that are no longer in force.** The relative area filter is
   applied on load, so the metrics evaluate against the filtered truth, but the
   trainer would read the unfiltered source files. Training with one truth and
   measuring against another makes the figures mean nothing.

So a derived view is written under `data/derived/`. **The source files are not
touched**: this is a working copy, regenerable and disposable.

The filter and the border policy live in `dataio/prepare.py`, shared with the
DOTA and COCO exporters. Repeating them here would have let them drift, and
then two candidates would train on different truths while the table compares
them as if they were the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from testbank.config import Config, OutOfBoundsPolicy
from testbank.dataio.formats import DEFAULT_CLASS_NAMES
from testbank.dataio.formats import get as get_format
from testbank.dataio.prepare import (
    PreparationReport,
    clip_quad,
    pad_geometry,
    pad_image,
    pad_quad,
    place_image,
    prepare,
)

DERIVED_DIRNAME = "ultralytics"

__all__ = [
    "DERIVED_DIRNAME",
    "MaterializedDataset",
    "clip_quad",
    "materialize",
    "pad_geometry",
    "pad_image",
    "pad_quad",
]


@dataclass(frozen=True, slots=True)
class MaterializedDataset:
    root: Path
    data_yaml: Path
    counts: dict[str, int]
    #: Annotations the filter left out and that therefore were NOT written.
    dropped: int
    #: Policy applied to banknotes crossing the border, and how many needed
    #: it. With `keep` this number is the number of images Ultralytics will
    #: discard, so the loss is recorded instead of being invisible.
    out_of_bounds: str = OutOfBoundsPolicy.CLIP.value
    adjusted: int = 0
    #: Only with `pad`: quads that were still outside AFTER padding and had
    #: to be clipped. A high number says `pad_fraction` falls short.
    clipped_after_pad: int = 0
    pad_fraction: float = 0.0

    @classmethod
    def from_report(
        cls, root: Path, data_yaml: Path, report: PreparationReport
    ) -> MaterializedDataset:
        return cls(
            root=root,
            data_yaml=data_yaml,
            counts=dict(report.counts),
            dropped=report.dropped,
            out_of_bounds=report.policy,
            adjusted=report.adjusted,
            clipped_after_pad=report.clipped_after_pad,
            pad_fraction=report.pad_fraction,
        )

    def describe(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        extra = ""
        if self.out_of_bounds == OutOfBoundsPolicy.PAD.value:
            extra = (
                f", pad={self.pad_fraction:.0%}, "
                f"{self.clipped_after_pad} clipped anyway"
            )
        return (
            f"{self.root} ({counts}; {self.dropped} annotations filtered; "
            f"border={self.out_of_bounds}, {self.adjusted} annotations outside "
            f"the frame{extra})"
        )


def materialize(
    samples_by_split: dict[str, list],
    config: Config | None = None,
    *,
    out_dir: Path | None = None,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    overwrite: bool = True,
) -> MaterializedDataset:
    """Writes `images/` and `labels/` per split, with the truth already filtered."""
    import shutil

    config = config or Config()
    root = Path(out_dir or (config.data.derived_dir / DERIVED_DIRNAME))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    prepared, report = prepare(samples_by_split, config)
    writer = get_format("obb_yolo")

    for split, items in prepared.items():
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            place_image(item, images_dir / item.sample.image_path.name)
            lines = [
                f"{class_id} " + writer.from_quad(quad).payload
                for quad, class_id in zip(item.quads, item.class_ids)
            ]
            (labels_dir / f"{item.sample_id}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": "train/images",
                # Ultralytics uses the `val` key; the directory is called
                # `valid` because that is the name Roboflow exports. They are
                # different things.
                "val": "valid/images",
                "names": dict(enumerate(class_names)),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return MaterializedDataset.from_report(root, data_yaml, report)
