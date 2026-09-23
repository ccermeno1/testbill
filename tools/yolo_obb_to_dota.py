#!/usr/bin/env python3
"""Convert a YOLO-OBB dataset (Ultralytics format) to the DOTA layout used by oriented_det.

YOLO OBB label line (normalized to [0, 1]):
    class_index x1 y1 x2 y2 x3 y3 x4 y4

Accepted input layouts (split names ``train`` / ``val`` / ``valid`` / ``test``):
    A) <src>/images/<split>/*.jpg  +  <src>/labels/<split>/*.txt   (Ultralytics)
    B) <src>/<split>/images/*.jpg  +  <src>/<split>/labels/*.txt   (Roboflow export)

Output (one DOTA tile root per split, ``valid`` → ``val``):
    <dst>/<split>/images/<stem>.jpg|png
    <dst>/<split>/labels/<stem>.txt      "x1 y1 x2 y2 x3 y3 x4 y4 <class> 0" (pixels)

Images are re-saved with EXIF orientation applied (oriented_det reads pixels with
PIL and ignores EXIF; labeling tools show the rotated photo) and lower-case
extensions (macOS file systems are case-sensitive for the loader's ``*.jpg`` glob).
Images without a label file, or with an empty one, become negatives (empty label).

Examples:
    python tools/yolo_obb_to_dota.py --src data/billetes_yolo --dst data/billetes_dota \\
        --single-class banknote --max-side 1280
    python tools/yolo_obb_to_dota.py --src data/ds --dst data/ds_dota --names 5eur 10eur 20eur
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: pip install pillow")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


def _clean_class_name(name: str) -> str:
    """DOTA lines are whitespace-separated: keep class names token-safe."""
    cleaned = re.sub(r"\s+", "-", str(name).strip())
    if not cleaned:
        raise ValueError(f"Empty class name {name!r}")
    return cleaned


def read_yaml_names(src: Path) -> Optional[List[str]]:
    """Class names from ``data.yaml`` / ``dataset.yaml`` (``names:`` list or dict)."""
    for fname in ("data.yaml", "dataset.yaml", "data.yml"):
        path = src / fname
        if not path.is_file():
            continue
        try:
            import yaml  # type: ignore
        except ImportError:
            print(f"  note: found {path.name} but PyYAML is not installed; use --names.")
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
        if isinstance(names, list):
            return [str(n) for n in names]
    return None


def find_splits(src: Path) -> Dict[str, Tuple[Path, Path]]:
    """Map output split name → (image_dir, label_dir)."""
    found: Dict[str, Tuple[Path, Path]] = {}
    for raw, out in SPLIT_ALIASES.items():
        layout_a = (src / "images" / raw, src / "labels" / raw)
        layout_b = (src / raw / "images", src / raw / "labels")
        for img_dir, lbl_dir in (layout_a, layout_b):
            if img_dir.is_dir():
                if out in found:
                    raise SystemExit(f"Split {out!r} found twice (e.g. {img_dir}); keep only one.")
                found[out] = (img_dir, lbl_dir)
                break
    return found


def parse_yolo_obb_line(line: str) -> Optional[Tuple[int, List[float]]]:
    parts = line.split()
    if not parts:
        return None
    if len(parts) != 9:
        raise ValueError(
            f"Expected 9 fields 'cls x1 y1 x2 y2 x3 y3 x4 y4', got {len(parts)}: {line!r}"
        )
    return int(float(parts[0])), [float(v) for v in parts[1:]]


def convert_image(src_path: Path, dst_dir: Path, max_side: Optional[int]) -> Tuple[Path, int, int, float]:
    """Save EXIF-corrected (and optionally downscaled) image. Returns (path, W, H, scale)."""
    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im)
        w0, h0 = im.size
        scale = 1.0
        if max_side and max(w0, h0) > max_side:
            scale = max_side / float(max(w0, h0))
            im = im.resize((max(1, round(w0 * scale)), max(1, round(h0 * scale))), Image.BILINEAR)
        ext = src_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            dst = dst_dir / f"{src_path.stem}.jpg"
            im.convert("RGB").save(dst, quality=95)
        else:
            dst = dst_dir / f"{src_path.stem}.png"
            im.convert("RGB").save(dst)
        w, h = im.size
    # YOLO coords are normalized to the displayed (EXIF-rotated) image, i.e. (w, h) here.
    return dst, w, h, scale


def convert_split(
    img_dir: Path,
    lbl_dir: Path,
    dst_root: Path,
    names: Optional[Sequence[str]],
    single_class: Optional[str],
    max_side: Optional[int],
    min_side_px: float,
) -> Dict[str, int]:
    out_img = dst_root / "images"
    out_lbl = dst_root / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "objects": 0, "negatives": 0, "dropped_small": 0}
    per_class: Dict[str, int] = {}
    stems_seen: Dict[str, Path] = {}
    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    for img_path in images:
        if img_path.stem in stems_seen:
            raise SystemExit(
                f"Two images share the stem {img_path.stem!r}: {stems_seen[img_path.stem].name} "
                f"and {img_path.name}. Rename one."
            )
        stems_seen[img_path.stem] = img_path
        _, w, h, _ = convert_image(img_path, out_img, max_side)

        lines_out: List[str] = []
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if label_path.is_file():
            for ln, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    parsed = parse_yolo_obb_line(raw.strip())
                except ValueError as exc:
                    raise SystemExit(f"{label_path}:{ln}: {exc}") from exc
                if parsed is None:
                    continue
                cls_idx, coords = parsed
                if single_class:
                    cls_name = single_class
                elif names:
                    if not 0 <= cls_idx < len(names):
                        raise SystemExit(f"{label_path}:{ln}: class {cls_idx} not in names {list(names)}")
                    cls_name = names[cls_idx]
                else:
                    cls_name = f"class{cls_idx}"
                cls_name = _clean_class_name(cls_name)
                pts = [
                    (min(max(coords[i], 0.0), 1.0) * w, min(max(coords[i + 1], 0.0), 1.0) * h)
                    for i in range(0, 8, 2)
                ]
                side_a = ((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2) ** 0.5
                side_b = ((pts[1][0] - pts[2][0]) ** 2 + (pts[1][1] - pts[2][1]) ** 2) ** 0.5
                if min(side_a, side_b) < min_side_px:
                    stats["dropped_small"] += 1
                    continue
                flat = " ".join(f"{v:.2f}" for xy in pts for v in xy)
                lines_out.append(f"{flat} {cls_name} 0")
                per_class[cls_name] = per_class.get(cls_name, 0) + 1

        (out_lbl / f"{img_path.stem}.txt").write_text(
            "\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8"
        )
        stats["images"] += 1
        stats["objects"] += len(lines_out)
        if not lines_out:
            stats["negatives"] += 1
    stats.update({f"class:{k}": v for k, v in sorted(per_class.items())})
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="YOLO-OBB dataset root")
    parser.add_argument("--dst", type=Path, required=True, help="Output root (one DOTA folder per split)")
    parser.add_argument("--names", nargs="+", default=None, help="Class names by YOLO index (default: data.yaml)")
    parser.add_argument(
        "--single-class",
        default=None,
        metavar="NAME",
        help="Collapse every YOLO class into one class NAME (e.g. banknote)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="Downscale images whose long side exceeds this (px). Speeds up data loading; "
        "use ≥ the training target_size.",
    )
    parser.add_argument(
        "--min-side-px",
        type=float,
        default=2.0,
        help="Drop boxes whose short side is below this many pixels after conversion",
    )
    args = parser.parse_args(argv)

    src = args.src.resolve()
    splits = find_splits(src)
    if not splits:
        print(f"No splits found under {src} (expected images/<split> or <split>/images).", file=sys.stderr)
        return 1
    names = args.names or read_yaml_names(src)
    print(f"Source: {src}")
    print(f"Splits: {', '.join(f'{k} <- {v[0]}' for k, v in splits.items())}")
    print(f"Classes: {args.single_class + ' (single class)' if args.single_class else names}")

    summary = {}
    for split, (img_dir, lbl_dir) in splits.items():
        stats = convert_split(
            img_dir,
            lbl_dir,
            args.dst / split,
            names,
            args.single_class,
            args.max_side,
            args.min_side_px,
        )
        summary[split] = stats
        print(f"  {split}: {json.dumps(stats)}")
    print(f"Done. DOTA roots: {', '.join(str((args.dst / s).resolve()) for s in splits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
