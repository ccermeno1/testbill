#!/usr/bin/env python3
"""Export SSDD (VOC XML / COCO / DOTA) to DOTA-format image + label folders.

Native training uses ``dataset.format: ssdd`` and does not require this
conversion. Use this tool when you want DOTA loaders. XML ``robndbox`` angles
are degrees. Grayscale SAR chips are written as RGB.

Usage:
    odet ssdd-to-dota --data-root /path/to/data/Official-SSDD-OPEN --output-dir /path/to/data/SSDD-dota
    python tools/ssdd_to_dota.py --data-root /path/to/data/Official-SSDD-OPEN --output-dir /tmp/ssdd_dota --splits train,test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oriented_det.data.ssdd import SSDD_SPLIT_NAMES, export_ssdd_to_dota


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export SSDD XML/COCO/DOTA splits to DOTA image + .txt folders."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="SSDD root (train/test folders, JPEGImages+Annotations, or COCO JSON).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory; one subdirectory per split.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated split names (default: train,test).",
    )
    parser.add_argument(
        "--difficult-strategy",
        type=str,
        default="keep",
        choices=("drop", "ignore", "keep"),
        help="How to handle difficult=1 objects in exported labels.",
    )
    parser.add_argument(
        "--same-folder",
        action="store_true",
        help="Write images and .txt labels in the same split directory.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="original",
        choices=("original", "jpg", "png"),
        help="Export raster format (original copies JPEG/PNG; grayscale becomes RGB).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality when converting (default 95).",
    )
    args = parser.parse_args(argv)

    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in SSDD_SPLIT_NAMES]
    if unknown:
        parser.error(f"Unknown split(s) {unknown}; expected {sorted(SSDD_SPLIT_NAMES)}")

    counts = export_ssdd_to_dota(
        args.data_root,
        args.output_dir,
        splits=splits,
        difficult_strategy=args.difficult_strategy,
        same_folder=args.same_folder,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"Wrote DOTA export under {args.output_dir}")
    for split, n in counts.items():
        print(f"  {split}: {n} image(s)")


if __name__ == "__main__":
    main()
