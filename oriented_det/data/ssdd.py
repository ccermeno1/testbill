"""SSDD native loader (single-class ship) yielding DOTASample objects.

Official last-digit split (Zhang et al., 2021): image file numbers whose last
digit is **1 or 9** are test; the rest are train (~928 / ~232). Dumps vary:

1. VOC XML with ``robndbox`` (cx/cy/w/h/angle in **degrees**) or 4-corner tags
2. COCO instances JSON + images
3. DOTA ``images/`` + ``labelTxt`` / ``labels``

``dataset.format: ssdd`` trains whole-image (chips ~190–688 px). Optional
``odet ssdd-to-dota`` writes DOTA folders. Polygons are converted to OBBs only;
there is no instance-seg training path.
"""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from ..geometry import RBox, transforms
from .coco_obb import (
    dota_label_dir_for_images,
    find_image_file,
    iter_image_files,
    parse_coco_instances,
    parse_dota_label_txt,
    write_dota_split,
)
from .dota import DOTAAnnotation, DOTASample

SSDD_CLASSES: list[str] = ["ship"]
SSDD_CLASS_SET = frozenset(SSDD_CLASSES)
SSDD_SPLIT_NAMES = frozenset({"train", "val", "test", "inshore", "offshore"})


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError("PIL/Pillow is required.")


def _xml_text(node: Optional[ET.Element]) -> str:
    if node is None or node.text is None:
        return ""
    return str(node.text).strip()


def _first_child_text(parent: ET.Element, names: Sequence[str]) -> str:
    for name in names:
        text = _xml_text(parent.find(name))
        if text:
            return text
        lowered = name.lower()
        for child in parent:
            if child.tag.lower() == lowered and (child.text or "").strip():
                return str(child.text).strip()
    return ""


def ssdd_official_is_test(stem: str) -> bool:
    """True when the last digit of the file number is 1 or 9 (official test)."""
    digits = "".join(ch for ch in Path(stem).stem if ch.isdigit())
    if not digits:
        return False
    return digits[-1] in {"1", "9"}


def ssdd_mbox_to_annotation(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_deg: float,
    *,
    difficult: int = 0,
    class_name: str = "ship",
) -> DOTAAnnotation:
    """VOC ``robndbox`` (angle in **degrees**) → le90 ``DOTAAnnotation``."""
    raw = RBox(cx, cy, width, height, math.radians(float(angle_deg)))
    polygon = raw.to_polygon()
    rbox = transforms.polygon_to_rbox(polygon)
    return DOTAAnnotation(
        class_name=class_name,
        difficult=int(difficult),
        polygon=polygon,
        rbox=rbox,
    )


def _parse_xml_corners(obj: ET.Element) -> Optional[Tuple[float, ...]]:
    keys = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")
    for parent in (obj, obj.find("rotated_bndbox"), obj.find("robndbox"), obj.find("polygon")):
        if parent is None:
            continue
        values: List[float] = []
        ok = True
        for key in keys:
            text = _first_child_text(parent, (key,))
            if not text:
                ok = False
                break
            try:
                values.append(float(text))
            except ValueError:
                ok = False
                break
        if ok and len(values) == 8:
            return tuple(values)
    return None


def _parse_xml_robndbox(obj: ET.Element) -> Optional[DOTAAnnotation]:
    box = obj.find("robndbox")
    parent = box if box is not None else obj
    cx_t = _first_child_text(parent, ("cx", "mbox_cx", "rotated_bbox_cx"))
    cy_t = _first_child_text(parent, ("cy", "mbox_cy", "rotated_bbox_cy"))
    w_t = _first_child_text(parent, ("w", "width", "mbox_w", "rotated_bbox_w"))
    h_t = _first_child_text(parent, ("h", "height", "mbox_h", "rotated_bbox_h"))
    ang_t = _first_child_text(parent, ("angle", "ang", "mbox_ang", "rotated_bbox_theta", "theta"))
    if not (cx_t and cy_t and w_t and h_t):
        return None
    try:
        cx, cy, w, h = float(cx_t), float(cy_t), float(w_t), float(h_t)
        ang = float(ang_t) if ang_t else 0.0
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    class_name = _first_child_text(obj, ("name", "n", "category")) or "ship"
    difficult_raw = _first_child_text(obj, ("difficult",)) or "0"
    try:
        difficult = int(float(difficult_raw))
    except ValueError:
        difficult = 0
    try:
        return ssdd_mbox_to_annotation(cx, cy, w, h, ang, difficult=difficult, class_name=class_name)
    except ValueError:
        return None


def _parse_xml_hbb(obj: ET.Element) -> Optional[DOTAAnnotation]:
    box = obj.find("bndbox")
    if box is None:
        return None
    try:
        xmin = float(_first_child_text(box, ("xmin", "x0")))
        ymin = float(_first_child_text(box, ("ymin", "y0")))
        xmax = float(_first_child_text(box, ("xmax", "x1")))
        ymax = float(_first_child_text(box, ("ymax", "y1")))
    except ValueError:
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    class_name = _first_child_text(obj, ("name", "n", "category")) or "ship"
    difficult_raw = _first_child_text(obj, ("difficult",)) or "0"
    try:
        difficult = int(float(difficult_raw))
    except ValueError:
        difficult = 0
    try:
        return DOTAAnnotation.from_corners(
            (xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax),
            class_name,
            difficult=difficult,
        )
    except ValueError:
        return None


def parse_ssdd_xml(path: str | Path) -> Tuple[DOTAAnnotation, ...]:
    """Parse one SSDD VOC XML (robndbox / 4-corners / HBB fallback)."""
    xml_path = Path(path)
    try:
        root = ET.fromstring(xml_path.read_bytes())
    except ET.ParseError:
        try:
            root = ET.fromstring(xml_path.read_text(encoding="utf-8", errors="ignore"))
        except ET.ParseError as exc:
            raise ValueError(f"Could not parse SSDD XML {xml_path}: {exc}") from exc
    objects = list(root.findall("object"))
    if not objects:
        objects = list(root.iter("object"))
    annotations: List[DOTAAnnotation] = []
    for obj in objects:
        corners = _parse_xml_corners(obj)
        if corners is not None:
            class_name = _first_child_text(obj, ("name", "n", "category")) or "ship"
            difficult_raw = _first_child_text(obj, ("difficult",)) or "0"
            try:
                difficult = int(float(difficult_raw))
            except ValueError:
                difficult = 0
            try:
                annotations.append(DOTAAnnotation.from_corners(corners, class_name, difficult=difficult))
                continue
            except ValueError:
                pass
        rob = _parse_xml_robndbox(obj)
        if rob is not None:
            annotations.append(rob)
            continue
        hbb = _parse_xml_hbb(obj)
        if hbb is not None:
            annotations.append(hbb)
    return tuple(annotations)


def _looks_like_ssdd(root: Path) -> bool:
    if not root.is_dir():
        return False
    from .hrsid import _looks_like_hrsid

    if _looks_like_hrsid(root):
        return False
    if (root / "JPEGImages").is_dir() and (root / "Annotations").is_dir():
        return True
    if (root / "JPEGImages_train").is_dir() and (root / "Annotations_train").is_dir():
        return True
    if (root / "train").is_dir() or (root / "images" / "train").is_dir():
        return True
    anns = root / "annotations"
    if anns.is_dir() and any(
        p.suffix.lower() == ".json" and not p.name.startswith("._") for p in anns.iterdir()
    ):
        return True
    return any(p.suffix.lower() == ".json" and not p.name.startswith("._") for p in root.glob("*.json"))


def resolve_ssdd_root(data_root: str | Path) -> Path:
    """Return the directory that contains SSDD images/annotations.

    Official-SSDD-OPEN unpacks to ``RBox_SSDD/voc_style`` (rotated VOC XML).
    """
    root = Path(data_root)
    nested = [
        root,
        root / "voc_style",
        root / "RBox_SSDD",
        root / "RBox_SSDD" / "voc_style",
        root / "Official-SSDD-OPEN",
        root / "Official-SSDD-OPEN" / "RBox_SSDD" / "voc_style",
        root / "Official-SSDD",
        root / "Official-SSDD" / "RBox_SSDD" / "voc_style",
        root / "SSDD",
        root / "SSDD" / "RBox_SSDD" / "voc_style",
        root / "ssdd",
        root / "RBox_SSDD",
        root / "SSDD_coco",
        root / "RSSDD",
    ]
    seen: set[Path] = set()
    for cand in nested:
        if cand in seen:
            continue
        seen.add(cand)
        if _looks_like_ssdd(cand):
            return cand
    raise FileNotFoundError(
        f"SSDD root not found under {root}. Expected Official-SSDD-OPEN/RBox_SSDD/voc_style "
        f"(JPEGImages + Annotations), train/ + test/, or a COCO JSON next to images/."
    )


def resolve_ssdd_imageset_split(dataset_cfg, role: str) -> str:
    """Map train-loop role to an SSDD split name.

    Defaults: train → train, val → test (official last-digit test is held-out).
    """
    name = (role or "").strip().lower()
    if name == "train":
        override = getattr(dataset_cfg, "train_split", None)
        return str(override).strip() if override else "train"
    if name == "val":
        override = getattr(dataset_cfg, "val_split", None)
        return str(override).strip() if override else "test"
    if name in SSDD_SPLIT_NAMES:
        return name
    raise ValueError(
        f"Unsupported SSDD split {role!r}. Expected train, val, test, inshore, or offshore."
    )


def _split_search_dirs(root: Path, split: str) -> List[Path]:
    name = split.strip().lower()
    dirs: List[Path] = []
    if name == "train":
        dirs.extend([root / "train", root / "Train", root / "trainsplit"])
    elif name == "test":
        dirs.extend([root / "test" / "all", root / "test", root / "Test", root / "testsplit"])
    elif name in {"inshore", "offshore"}:
        dirs.extend(
            [
                root / "test" / name,
                root / name,
                root / "test" / name.capitalize(),
            ]
        )
    elif name == "val":
        dirs.extend([root / "val", root / "Val", root / "valsplit"])
    else:
        dirs.append(root / name)
    return dirs


def _voc_split_dirs(root: Path, split: str) -> Optional[Tuple[Path, Path]]:
    """Official-SSDD JPEGImages_{split} + Annotations_{split} pairs."""
    name = split.strip().lower()
    pairs = {
        "train": ("JPEGImages_train", "Annotations_train"),
        "test": ("JPEGImages_test", "Annotations_test"),
        "inshore": ("JPEGImages_test_inshore", "Annotations_test_inshore"),
        "offshore": ("JPEGImages_test_offshore", "Annotations_test_offshore"),
    }
    if name not in pairs:
        return None
    image_name, xml_name = pairs[name]
    image_dir = root / image_name
    xml_dir = root / xml_name
    if image_dir.is_dir() and xml_dir.is_dir():
        return image_dir, xml_dir
    return None


def _find_coco_json(directory: Path, split: str) -> Optional[Path]:
    names = (
        f"{split}.json",
        f"{split}2017.json",
        "annotations.json",
        "instances.json",
        f"instances_{split}.json",
        f"instances_{split}2017.json",
    )
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
        nested = directory / "annotations" / name
        if nested.is_file():
            return nested
    jsons = sorted(directory.glob("*.json"))
    if len(jsons) == 1:
        return jsons[0]
    return None


def _image_dirs_for(directory: Path) -> List[Path]:
    dirs: List[Path] = []
    for name in ("images", "JPEGImages", "JPEGImages_test", "all", "img", "IMG"):
        candidate = directory / name
        if candidate.is_dir():
            dirs.append(candidate)
    dirs.append(directory)
    return dirs


def _xml_dir_for(directory: Path) -> Optional[Path]:
    for name in ("Annotations", "annotations", "xmls", "labelXml", "XML"):
        candidate = directory / name
        if candidate.is_dir():
            return candidate
    xmls = list(directory.glob("*.xml"))
    if xmls:
        return directory
    return None


def _load_size(image_path: Path) -> Tuple[int, int]:
    _require_pillow()
    with Image.open(image_path) as img:
        return img.size


class _Record:
    __slots__ = ("image_path", "annotations")

    def __init__(self, image_path: Path, annotations: Tuple[DOTAAnnotation, ...]):
        self.image_path = image_path
        self.annotations = annotations


def _records_from_coco(json_path: Path, image_dirs: Sequence[Path]) -> List[_Record]:
    records: List[_Record] = []
    for item in parse_coco_instances(json_path):
        image_path = find_image_file(image_dirs, item.file_name)
        if image_path is None:
            warnings.warn(f"SSDD image missing for COCO file_name={item.file_name!r}", UserWarning)
            continue
        records.append(_Record(image_path, item.annotations))
    return records


def _records_from_xml(image_dir: Path, xml_dir: Path) -> List[_Record]:
    records: List[_Record] = []
    for image_path in iter_image_files(image_dir):
        xml_path = xml_dir / f"{image_path.stem}.xml"
        if not xml_path.is_file():
            warnings.warn(f"SSDD XML missing for {image_path.name}: {xml_path}", UserWarning)
            anns: Tuple[DOTAAnnotation, ...] = tuple()
        else:
            anns = parse_ssdd_xml(xml_path)
        records.append(_Record(image_path, anns))
    return records


def _records_from_dota(image_dir: Path) -> List[_Record]:
    label_dir = dota_label_dir_for_images(image_dir)
    records: List[_Record] = []
    for image_path in iter_image_files(image_dir):
        if label_dir is None:
            anns: Tuple[DOTAAnnotation, ...] = tuple()
        else:
            anns = parse_dota_label_txt(label_dir / f"{image_path.stem}.txt")
        records.append(_Record(image_path, anns))
    return records


def _filter_official_split(records: Sequence[_Record], split: str) -> List[_Record]:
    name = split.strip().lower()
    if name in {"inshore", "offshore"}:
        return list(records)
    if name == "test":
        return [r for r in records if ssdd_official_is_test(r.image_path.stem)]
    if name in {"train", "val"}:
        return [r for r in records if not ssdd_official_is_test(r.image_path.stem)]
    return list(records)


def discover_ssdd_records(root: Path, split: str) -> List[_Record]:
    """Discover ``(image, annotations)`` for one split. Layout order: COCO, XML, DOTA, 1/9 VOC."""
    split_name = split.strip().lower()
    voc_pair = _voc_split_dirs(root, split_name)
    if voc_pair is not None:
        found = _records_from_xml(*voc_pair)
        if found:
            return found
    for split_dir in _split_search_dirs(root, split_name):
        if not split_dir.is_dir():
            continue
        json_path = _find_coco_json(split_dir, split_name)
        if json_path is not None:
            found = _records_from_coco(json_path, _image_dirs_for(split_dir))
            if found:
                return found
        xml_dir = _xml_dir_for(split_dir)
        image_dirs = [d for d in _image_dirs_for(split_dir) if d != split_dir or iter_image_files(d)]
        if xml_dir is not None and image_dirs:
            found = _records_from_xml(image_dirs[0], xml_dir)
            if found:
                return found
        for image_dir in _image_dirs_for(split_dir):
            label_dir = dota_label_dir_for_images(image_dir)
            if label_dir is not None and iter_image_files(image_dir):
                found = _records_from_dota(image_dir)
                if found:
                    return found

    # Flat VOC / COCO at the dataset root with official 1/9 filtering.
    root_json = _find_coco_json(root, split_name)
    if root_json is not None:
        found = _records_from_coco(root_json, _image_dirs_for(root))
        if found:
            return _filter_official_split(found, split_name)

    voc_images = None
    for name in ("JPEGImages", "images", "JPEGImages_train", "JPEGImages_test"):
        candidate = root / name
        if candidate.is_dir() and iter_image_files(candidate):
            voc_images = candidate
            break
    voc_xml = None
    for name in ("Annotations", "annotations", "Annotations_train", "Annotations_test"):
        candidate = root / name
        if candidate.is_dir():
            voc_xml = candidate
            break
    if voc_images is not None and voc_xml is not None:
        found = _records_from_xml(voc_images, voc_xml)
        if found:
            return _filter_official_split(found, split_name)

    dota_images = root / "images"
    if dota_images.is_dir() and dota_label_dir_for_images(dota_images) is not None:
        found = _records_from_dota(dota_images)
        if found:
            return _filter_official_split(found, split_name)

    raise FileNotFoundError(
        f"No SSDD {split_name!r} images found under {root}. "
        f"Expected train/test folders, JPEGImages+Annotations, COCO JSON, or DOTA labelTxt."
    )


class SSDDDataset:
    """SSDD dataset yielding ``DOTASample`` (single-class ``ship``)."""

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
        if split_name not in SSDD_SPLIT_NAMES:
            raise ValueError(
                f"Unsupported SSDD split {split!r}. Expected train, val, test, inshore, or offshore."
            )

        self.root = resolve_ssdd_root(data_root)
        self.split = split_name
        self.difficult_strategy = ds
        self._drop_difficult = ds == "drop"
        self.filter_empty_gt = bool(filter_empty_gt)
        self.allowed_classes = list(allowed_classes) if allowed_classes is not None else None
        self.ignore_labels = list(ignore_labels) if ignore_labels else None
        self.lookalike_labels = list(lookalike_labels) if lookalike_labels else None
        self._lookalike_set = resolve_lookalike_label_set(self.lookalike_labels)

        discovered = discover_ssdd_records(self.root, self.split)
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

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> DOTASample:
        if idx < 0 or idx >= len(self._records):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        record = self._records[idx]
        width, height = _load_size(record.image_path)
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
        return list(SSDD_CLASSES)


def format_ssdd_empty_gt_filter_log(dataset: SSDDDataset, *, split: str) -> str:
    discovered = dataset.tiles_discovered_count
    filtered = dataset.empty_gt_filtered_count
    kept = discovered - filtered
    return (
        f"  {split}: filter_empty_gt dropped {filtered} / {discovered} images "
        f"({kept} kept)"
    )


def export_ssdd_to_dota(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    splits: Sequence[str] = ("train", "test"),
    difficult_strategy: str = "keep",
    same_folder: bool = False,
    image_format: str = "original",
    jpeg_quality: int = 95,
) -> Dict[str, int]:
    """Write SSDD splits as DOTA-format images + ``.txt`` folders."""
    counts: Dict[str, int] = {}
    for split in splits:
        dataset = SSDDDataset(
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
    "SSDD_CLASSES",
    "SSDD_CLASS_SET",
    "SSDD_SPLIT_NAMES",
    "SSDDDataset",
    "export_ssdd_to_dota",
    "format_ssdd_empty_gt_filter_log",
    "parse_ssdd_xml",
    "resolve_ssdd_imageset_split",
    "resolve_ssdd_root",
    "ssdd_mbox_to_annotation",
    "ssdd_official_is_test",
]
