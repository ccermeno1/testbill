"""HRSID native COCO loader (single-class ship) yielding DOTASample objects.

Wei et al. 2020: 5,604 800×800 chips, 16,951 ships, official **65/35 train/test**
(MS COCO polygons). There is **no official val** — recipes evaluate on test.

Layouts discovered (first match wins):

- ``annotations/train2017.json`` + ``images/`` (or ``images/train2017/``)
- MMRotate-style ``trainsplit/`` / ``testsplit/`` with a JSON inside
- wrapping ``HRSID/`` or ``HRSID_JPG/``

``dataset.format: hrsid`` trains whole-image at keep-ratio 800. Optional
``odet coco-to-dota --data-root`` writes DOTA folders. Polygons are OBBs only.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from .coco_obb import find_image_file, parse_coco_instances, write_dota_split
from .dota import DOTAAnnotation, DOTASample

HRSID_CLASSES: list[str] = ["ship"]
HRSID_CLASS_SET = frozenset(HRSID_CLASSES)
HRSID_SPLIT_NAMES = frozenset(
    {
        "train",
        "val",
        "test",
        "trainsplit",
        "valsplit",
        "testsplit",
        "train2017",
        "val2017",
        "test2017",
    }
)

_SPLIT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "train": ("train2017", "trainsplit", "train"),
    "val": ("val2017", "valsplit", "val", "test2017", "testsplit", "test"),
    "test": ("test2017", "testsplit", "test"),
    "trainsplit": ("trainsplit", "train2017", "train"),
    "valsplit": ("valsplit", "val2017", "val", "testsplit", "test2017", "test"),
    "testsplit": ("testsplit", "test2017", "test"),
    "train2017": ("train2017", "trainsplit", "train"),
    "val2017": ("val2017", "valsplit", "val"),
    "test2017": ("test2017", "testsplit", "test"),
}


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError("PIL/Pillow is required.")


def _looks_like_hrsid(root: Path) -> bool:
    if not root.is_dir():
        return False
    ann = root / "annotations"
    if ann.is_dir():
        jsons = [
            p
            for p in ann.iterdir()
            if p.is_file() and p.suffix.lower() == ".json" and not p.name.startswith("._")
        ]
        if jsons:
            return True
    if (root / "train2017.json").is_file() or (root / "trainsplit").is_dir():
        return True
    return False


def resolve_hrsid_root(data_root: str | Path) -> Path:
    """Return the directory that contains HRSID COCO JSON + images."""
    root = Path(data_root)
    candidates = [root]
    for name in ("HRSID", "hrsid", "HRSID_JPG", "HRSID-JPG"):
        candidates.append(root / name)
    nested = root / "HRSID" / "HRSID_JPG"
    candidates.append(nested)
    for cand in candidates:
        if _looks_like_hrsid(cand):
            return cand
    raise FileNotFoundError(
        f"HRSID root not found under {root}. Expected annotations/train2017.json "
        f"(or trainsplit/) next to images/."
    )


def resolve_hrsid_imageset_split(dataset_cfg, role: str) -> str:
    """Map train-loop role to an HRSID split name.

    Defaults: train → train, val → test (no official val).
    """
    name = (role or "").strip().lower()
    if name == "train":
        override = getattr(dataset_cfg, "train_split", None)
        return str(override).strip() if override else "train"
    if name == "val":
        override = getattr(dataset_cfg, "val_split", None)
        return str(override).strip() if override else "test"
    if name in HRSID_SPLIT_NAMES:
        return name
    raise ValueError(
        f"Unsupported HRSID split {role!r}. Expected train, val, test, trainsplit, "
        f"valsplit, or testsplit."
    )


def _json_candidates(root: Path, split: str) -> List[Path]:
    aliases = _SPLIT_ALIASES.get(split, (split,))
    paths: List[Path] = []
    for alias in aliases:
        paths.extend(
            [
                root / "annotations" / f"{alias}.json",
                root / alias / f"{alias}.json",
                root / alias / "annotations.json",
                root / f"{alias}.json",
                root / alias / f"{alias}2017.json",
            ]
        )
    # Unique while preserving order.
    seen: set[Path] = set()
    ordered: List[Path] = []
    for path in paths:
        resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _image_dirs_for(root: Path, json_path: Path, split: str) -> List[Path]:
    aliases = _SPLIT_ALIASES.get(split, (split,))
    dirs: List[Path] = [
        json_path.parent,
        json_path.parent / "images",
        root / "images",
        root / "JPEGImages",
    ]
    for alias in aliases:
        dirs.extend(
            [
                root / "images" / alias,
                root / alias / "images",
                root / alias,
            ]
        )
    seen: set[Path] = set()
    ordered: List[Path] = []
    for directory in dirs:
        resolved = directory.resolve() if directory.exists() else directory
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(directory)
    return ordered


def discover_hrsid_json(root: Path, split: str) -> Path:
    for path in _json_candidates(root, split):
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"HRSID COCO JSON not found for split {split!r} under {root}. "
        f"Tried annotations/train2017.json, trainsplit/, testsplit/, …"
    )


class _Record:
    __slots__ = ("image_path", "annotations", "width", "height")

    def __init__(
        self,
        image_path: Path,
        annotations: Tuple[DOTAAnnotation, ...],
        width: int,
        height: int,
    ):
        self.image_path = image_path
        self.annotations = annotations
        self.width = width
        self.height = height


def discover_hrsid_records(root: Path, split: str) -> List[_Record]:
    json_path = discover_hrsid_json(root, split)
    image_dirs = _image_dirs_for(root, json_path, split)
    records: List[_Record] = []
    for item in parse_coco_instances(json_path):
        image_path = find_image_file(image_dirs, item.file_name)
        if image_path is None:
            warnings.warn(f"HRSID image missing for COCO file_name={item.file_name!r}", UserWarning)
            continue
        records.append(_Record(image_path, item.annotations, item.width, item.height))
    if not records:
        raise FileNotFoundError(
            f"HRSID JSON {json_path} has no resolvable images under {image_dirs}"
        )
    return records


class HRSIDDataset:
    """HRSID dataset yielding ``DOTASample`` (single-class ``ship``)."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str = "train",
        difficult_strategy: str = "drop",
        filter_empty_gt: bool = False,
        allowed_classes: Optional[Sequence[str]] = None,
        ignore_labels: Optional[Sequence[str]] = None,
        lookalike_labels: Optional[Sequence[str]] = None,
    ):
        from .lookalike import resolve_lookalike_label_set

        ds = (difficult_strategy or "drop").strip().lower()
        if ds not in {"drop", "ignore", "keep"}:
            raise ValueError(
                f"Invalid difficult_strategy={difficult_strategy!r}; expected 'drop', 'ignore', or 'keep'."
            )
        split_name = split.strip().lower()
        if split_name not in HRSID_SPLIT_NAMES:
            raise ValueError(
                f"Unsupported HRSID split {split!r}. Expected train, val, test, "
                f"trainsplit, valsplit, or testsplit."
            )

        self.root = resolve_hrsid_root(data_root)
        self.split = split_name
        self.difficult_strategy = ds
        self._drop_difficult = ds == "drop"
        self.filter_empty_gt = bool(filter_empty_gt)
        self.allowed_classes = list(allowed_classes) if allowed_classes is not None else None
        self.ignore_labels = list(ignore_labels) if ignore_labels else None
        self.lookalike_labels = list(lookalike_labels) if lookalike_labels else None
        self._lookalike_set = resolve_lookalike_label_set(self.lookalike_labels)

        discovered = discover_hrsid_records(self.root, self.split)
        self._records_discovered_count = len(discovered)
        kept: List[_Record] = []
        for record in discovered:
            if self.filter_empty_gt and self._effective_gt_count(record) == 0:
                continue
            kept.append(record)
        self._records = kept
        self._annotation_files = [r.image_path for r in self._records]
        self._empty_gt_filtered_count = self._records_discovered_count - len(self._records)

    @property
    def tiles_discovered_count(self) -> int:
        return self._records_discovered_count

    @property
    def annotation_files_discovered_count(self) -> int:
        return self._records_discovered_count

    @property
    def empty_gt_filtered_count(self) -> int:
        return self._empty_gt_filtered_count

    def _effective_gt_count(self, record: _Record) -> int:
        if not record.annotations:
            return 0
        sample = DOTASample(
            image_path=record.image_path,
            width=0,
            height=0,
            annotations=record.annotations,
        )
        sample = sample.filter_by_class(
            allowed_classes=self.allowed_classes,
            ignore_labels=self.ignore_labels,
            drop_difficult=self._drop_difficult,
            lookalike_labels=self.lookalike_labels,
        )
        return len(sample.annotations)

    def _load_size(self, record: _Record) -> Tuple[int, int]:
        if record.width > 0 and record.height > 0:
            return record.width, record.height
        _require_pillow()
        with Image.open(record.image_path) as img:
            return img.size

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> DOTASample:
        if idx < 0 or idx >= len(self._records):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        record = self._records[idx]
        width, height = self._load_size(record)
        sample = DOTASample(
            image_path=record.image_path,
            width=width,
            height=height,
            annotations=record.annotations,
        )
        if self.allowed_classes is not None or self.ignore_labels is not None or self._drop_difficult:
            sample = sample.filter_by_class(
                allowed_classes=self.allowed_classes,
                ignore_labels=self.ignore_labels,
                drop_difficult=self._drop_difficult,
                lookalike_labels=self.lookalike_labels,
            )
        return sample

    def __iter__(self) -> Iterator[DOTASample]:
        for idx in range(len(self)):
            yield self[idx]

    def get_class_names(self) -> List[str]:
        return list(HRSID_CLASSES)


def format_hrsid_empty_gt_filter_log(dataset: HRSIDDataset, *, split: str) -> str:
    discovered = dataset.tiles_discovered_count
    filtered = dataset.empty_gt_filtered_count
    kept = discovered - filtered
    return (
        f"  {split}: filter_empty_gt dropped {filtered} / {discovered} images "
        f"({kept} kept)"
    )


def export_hrsid_to_dota(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    splits: Sequence[str] = ("train", "test"),
    difficult_strategy: str = "keep",
    same_folder: bool = False,
    image_format: str = "original",
    jpeg_quality: int = 95,
) -> Dict[str, int]:
    """Write HRSID splits as DOTA-format images + ``.txt`` folders."""
    counts: Dict[str, int] = {}
    for split in splits:
        dataset = HRSIDDataset(
            data_root,
            split=split,
            difficult_strategy=difficult_strategy,
            filter_empty_gt=False,
        )
        counts[split] = write_dota_split(
            dataset,
            output_dir,
            split,
            same_folder=same_folder,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )
    return counts


__all__ = [
    "HRSID_CLASSES",
    "HRSID_CLASS_SET",
    "HRSID_SPLIT_NAMES",
    "HRSIDDataset",
    "export_hrsid_to_dota",
    "format_hrsid_empty_gt_filter_log",
    "resolve_hrsid_imageset_split",
    "resolve_hrsid_root",
]
