"""Dataset exporters, registered by decorator.

The converters in `formats.py` translate ONE annotation. This writes the file
tree each trainer expects to find: where the images go, where the labels, and
under which names.

Who reads what:

    dota   RTMDet-R (and BboxToolkit): one .txt per image in labelTxt/
    coco   diagnostics and third-party tools: a single annotations.json

Both go through `dataio/prepare.prepare`, so they share the area filter and
the border policy with the Ultralytics view. That is what keeps two candidates
from training on different truths while the table compares them as equals.

There used to be two more (`voc_xml` and the VOC variant of the YOLOX-OBB fork);
they left in the refactor with the candidates that consumed them. The lesson
they left: two things can look like the same VOC format and not be -- one puts
`<robndbox>` elements, the other an `<angle>` INSIDE a plain `<bndbox>` whose
xmin/xmax are really w and h around the center. Before plugging an exporter
into a concrete tool, read ITS parser.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from testbank.config import Config
from testbank.dataio.formats import DEFAULT_CLASS_NAMES, ImageSize
from testbank.dataio.formats import get as get_format
from testbank.dataio.prepare import (
    PreparationReport,
    PreparedSample,
    place_image,
    prepare,
)


class ExportError(ValueError):
    """The requested view could not be written."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    name: str
    root: Path
    report: PreparationReport
    #: What to hand the trainer: a yaml, a json or a directory.
    entry_point: Path

    def describe(self) -> str:
        return f"{self.name} -> {self.entry_point} ({self.report.describe()})"


@runtime_checkable
class Exporter(Protocol):
    name: ClassVar[str]
    #: Format from `formats.py` used to serialize each annotation.
    annotation_format: ClassVar[str]

    def write(
        self,
        prepared: dict[str, list[PreparedSample]],
        root: Path,
        *,
        class_names: tuple[str, ...],
    ) -> Path: ...


REGISTRY: dict[str, Exporter] = {}


def register(cls: type) -> type:
    if cls.name in REGISTRY:
        raise ExportError(f"duplicate exporter in the registry: {cls.name!r}")
    REGISTRY[cls.name] = cls()
    return cls


def get(name: str) -> Exporter:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ExportError(
            f"unknown exporter: {name!r}; registered: {sorted(REGISTRY)}"
        ) from None


def exporters() -> list[str]:
    return sorted(REGISTRY)


# --- formats ---------------------------------------------------------------


@register
class DotaExporter:
    """`images/` + `labelTxt/`, one .txt per image. The DOTA convention."""

    name = "dota"
    annotation_format = "dota"

    def write(self, prepared, root, *, class_names) -> Path:
        writer = get_format(self.annotation_format)
        for split, items in prepared.items():
            images = root / split / "images"
            labels = root / split / "labelTxt"
            labels.mkdir(parents=True, exist_ok=True)
            for item in items:
                place_image(item, images / item.sample.image_path.name)
                lines = [
                    writer.from_quad(
                        quad, class_id=class_id, size=item.size, class_names=class_names
                    ).payload
                    for quad, class_id in zip(item.quads, item.class_ids)
                ]
                (labels / f"{item.sample_id}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
        return root


@register
class CocoExporter:
    """One `annotations.json` per split. LOSSY: it loses the orientation.

    Exported anyway because it is what many inspection tools consume, but a
    candidate trained from here cannot predict rotated boxes: `bbox_coco` is
    marked `lossy` in the format registry for exactly this reason.
    """

    name = "coco"
    annotation_format = "bbox_coco"

    def write(self, prepared, root, *, class_names) -> Path:
        writer = get_format(self.annotation_format)
        entry_point = root
        for split, items in prepared.items():
            images_dir = root / split / "images"
            payload = {
                "info": {
                    "description": "Exported by testbank",
                    "note": (
                        "bbox_coco loses the orientation: the boxes are the "
                        "axis-aligned envelope of the banknote."
                    ),
                },
                "categories": [
                    {"id": i, "name": name} for i, name in enumerate(class_names)
                ],
                "images": [],
                "annotations": [],
            }
            annotation_id = 1
            for image_id, item in enumerate(items, start=1):
                place_image(item, images_dir / item.sample.image_path.name)
                payload["images"].append(
                    {
                        "id": image_id,
                        "file_name": item.sample.image_path.name,
                        "width": item.size.width,
                        "height": item.size.height,
                    }
                )
                for quad, class_id in zip(item.quads, item.class_ids):
                    record = writer.from_quad(
                        quad, class_id=class_id, size=item.size, class_names=class_names
                    )
                    entry = dict(record.payload)
                    entry["id"] = annotation_id
                    entry["image_id"] = image_id
                    payload["annotations"].append(entry)
                    annotation_id += 1

            target = root / split / "annotations.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            entry_point = target
        return entry_point


def export(
    name: str,
    samples_by_split: dict[str, list],
    config: Config | None = None,
    *,
    out_dir: Path | None = None,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    overwrite: bool = True,
) -> ExportResult:
    """Write the full view in the requested format."""
    config = config or Config()
    exporter = get(name)
    root = Path(out_dir or (config.data.derived_dir / name))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    prepared, report = prepare(samples_by_split, config)
    entry_point = exporter.write(prepared, root, class_names=class_names)
    return ExportResult(
        name=name, root=root, report=report, entry_point=entry_point
    )


__all__ = [
    "REGISTRY",
    "ExportError",
    "ExportResult",
    "Exporter",
    "ImageSize",
    "export",
    "exporters",
    "get",
    "register",
]
