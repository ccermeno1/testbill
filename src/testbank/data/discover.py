"""Detection of the data directory structure.

Mode 1 (adopt): train/valid/test with images/ and labels/ already exist. It is
what Roboflow exports, and the split is adopted as is.
Mode 2 (create): that structure does not exist; the split must be generated.

This module only DISCOVERS. Assigning samples to splits is the exclusive job of
splits.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

CANONICAL_SPLITS = ("train", "valid", "test")

# Roboflow uses "valid". "val" is the Ultralytics name and shows up in
# hand-edited exports: it is accepted as an alias, but noted.
_VALID_ALIASES = ("valid", "val")


class LayoutMode(str, Enum):
    ADOPT = "adopt"
    CREATE = "create"


class LayoutError(RuntimeError):
    """The data directory structure is unusable."""


@dataclass(frozen=True, slots=True)
class Sample:
    """An image and its label file. `sample_id` is the name without extension."""

    sample_id: str
    image_path: Path
    label_path: Path

    @property
    def directory_key(self) -> str:
        return self.image_path.parent.name


@dataclass(frozen=True, slots=True)
class Layout:
    root: Path
    mode: LayoutMode
    #: In adopt mode: split -> samples. In create mode: {"__all__": samples}.
    groups: dict[str, tuple[Sample, ...]]
    notes: tuple[str, ...] = ()

    @property
    def all_samples(self) -> tuple[Sample, ...]:
        out: list[Sample] = []
        for key in sorted(self.groups):
            out.extend(self.groups[key])
        return tuple(out)


def _pair_directory(images_dir: Path, labels_dir: Path) -> tuple[Sample, ...]:
    """Pair images with labels. Any orphan is fatal.

    A missing label is NOT treated as "image with no banknotes": that is how a
    training run gets poisoned silently. Roboflow writes an empty .txt when
    there are no objects, so a missing file is a real anomaly.
    """
    images = {
        p.stem: p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {
        p.stem: p for p in sorted(labels_dir.iterdir()) if p.is_file() and p.suffix == ".txt"
    }

    missing_labels = sorted(set(images) - set(labels))
    if missing_labels:
        raise LayoutError(
            f"{len(missing_labels)} images without a label file in "
            f"{labels_dir}: {missing_labels[:10]}"
            + (" ..." if len(missing_labels) > 10 else "")
        )
    orphan_labels = sorted(set(labels) - set(images))
    if orphan_labels:
        raise LayoutError(
            f"{len(orphan_labels)} labels without an image in {images_dir}: "
            f"{orphan_labels[:10]}" + (" ..." if len(orphan_labels) > 10 else "")
        )

    return tuple(
        Sample(sample_id=stem, image_path=images[stem], label_path=labels[stem])
        for stem in sorted(images)
    )


def _resolve_split_dir(root: Path, split: str) -> tuple[Path | None, str | None]:
    """Return the split directory and a note if an alias was used."""
    names = _VALID_ALIASES if split == "valid" else (split,)
    found = [name for name in names if (root / name).is_dir()]
    if len(found) > 1:
        raise LayoutError(
            f"{found} exist at the same time in {root}; ambiguous. Roboflow uses "
            "'valid'; remove or rename the other one before continuing"
        )
    if not found:
        return None, None
    name = found[0]
    note = None
    if name != split:
        note = (
            f"found '{name}/' and adopting it as split '{split}'. Roboflow "
            "exports 'valid'; check that the directory is the expected one"
        )
    return root / name, note


def detect_layout(root: str | Path) -> Layout:
    """Decide between adopt and create mode by inspecting the directory."""
    root = Path(root)
    if not root.is_dir():
        raise LayoutError(f"data directory does not exist: {root}")

    notes: list[str] = []
    split_dirs: dict[str, Path] = {}
    for split in CANONICAL_SPLITS:
        path, note = _resolve_split_dir(root, split)
        if path is not None and (path / "images").is_dir() and (path / "labels").is_dir():
            split_dirs[split] = path
            if note:
                notes.append(note)

    if "train" in split_dirs and "valid" in split_dirs:
        if "test" not in split_dirs:
            notes.append(
                "no 'test' split in the export; train/valid are adopted and test "
                "stays empty. The evaluate-test command will have nothing to evaluate"
            )
        groups = {
            split: _pair_directory(path / "images", path / "labels")
            for split, path in split_dirs.items()
        }
        return Layout(
            root=root, mode=LayoutMode.ADOPT, groups=groups, notes=tuple(notes)
        )

    if split_dirs:
        raise LayoutError(
            f"half-built structure in {root}: found {sorted(split_dirs)} but adopt "
            "mode requires at least 'train' and 'valid', each with images/ and "
            "labels/. Fix the structure or remove it entirely so the split is "
            "generated in create mode"
        )

    samples = _discover_flat(root)
    return Layout(
        root=root,
        mode=LayoutMode.CREATE,
        groups={"__all__": samples},
        notes=tuple(notes),
    )


def _discover_flat(root: Path) -> tuple[Sample, ...]:
    """Create mode: images/ + labels/ at the root, or images and .txt side by side."""
    if (root / "images").is_dir() and (root / "labels").is_dir():
        return _pair_directory(root / "images", root / "labels")

    images = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise LayoutError(
            f"no image found under {root}. Expected either train/valid/test with "
            "images/ and labels/, or images/ and labels/, or images with their "
            ".txt next to them"
        )

    by_stem: dict[str, Path] = {}
    for path in images:
        if path.stem in by_stem:
            raise LayoutError(
                f"duplicate image name '{path.stem}': {by_stem[path.stem]} and "
                f"{path}. The sample identifier is the name without extension and "
                "must be unique"
            )
        by_stem[path.stem] = path

    samples: list[Sample] = []
    missing: list[str] = []
    for stem, image_path in sorted(by_stem.items()):
        label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            missing.append(stem)
            continue
        samples.append(
            Sample(sample_id=stem, image_path=image_path, label_path=label_path)
        )
    if missing:
        raise LayoutError(
            f"{len(missing)} images without a .txt next to them: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    return tuple(samples)
