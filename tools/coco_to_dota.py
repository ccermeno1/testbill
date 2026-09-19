#!/usr/bin/env python3
"""Export COCO instance polygons to DOTA-format image + label folders.

Native SSDD / HRSID training uses ``dataset.format: ssdd`` / ``hrsid`` and does
not require this conversion. Use this tool when you want DOTA loaders, or to
turn any COCO polygon JSON into ``format: dota``.

n-gon masks become a min-area rectangle (OBB only — not instance segmentation).

Usage:
    odet coco-to-dota --ann-file train2017.json --image-dir images --output-dir out --output-split train
    odet coco-to-dota --data-root /path/to/data/HRSID_JPG --output-dir /path/to/data/HRSID-dota --splits train,test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oriented_det.data.coco_obb import export_coco_json_to_dota
from oriented_det.data.hrsid import HRSID_SPLIT_NAMES, export_hrsid_to_dota, resolve_hrsid_root
from oriented_det.data.ssdd import SSDD_SPLIT_NAMES, export_ssdd_to_dota, resolve_ssdd_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export COCO instance polygons (or an HRSID/SSDD dump) to DOTA folders."
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=None,
        help="COCO instances JSON (use with --image-dir).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing images listed in --ann-file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="HRSID or SSDD dataset root (mutually exclusive with --ann-file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory; one subdirectory per split.",
    )
    parser.add_argument(
        "--output-split",
        type=str,
        default="train",
        help="Split folder name when exporting a single --ann-file (default: train).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated splits for --data-root (default: train,test).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        choices=("auto", "hrsid", "ssdd"),
        help="Dataset layout when using --data-root (default: auto).",
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

    if args.ann_file is not None and args.data_root is not None:
        parser.error("Use either --ann-file/--image-dir or --data-root, not both.")
    if args.ann_file is None and args.data_root is None:
        parser.error("Provide --ann-file (with --image-dir) or --data-root.")

    if args.ann_file is not None:
        if args.image_dir is None:
            parser.error("--image-dir is required with --ann-file.")
        n = export_coco_json_to_dota(
            args.ann_file,
            args.image_dir,
            args.output_dir,
            split=args.output_split.strip() or "train",
            same_folder=args.same_folder,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )
        print(f"Wrote DOTA export under {args.output_dir} / {args.output_split}: {n} image(s)")
        return

    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    layout = args.layout
    if layout == "auto":
        try:
            resolve_hrsid_root(args.data_root)
            layout = "hrsid"
        except FileNotFoundError:
            try:
                resolve_ssdd_root(args.data_root)
                layout = "ssdd"
            except FileNotFoundError:
                parser.error(
                    f"Could not detect HRSID or SSDD under {args.data_root}. "
                    f"Pass --layout hrsid|ssdd, or use --ann-file for a generic COCO JSON."
                )

    if layout == "hrsid":
        unknown = [s for s in splits if s not in HRSID_SPLIT_NAMES]
        if unknown:
            parser.error(f"Unknown HRSID split(s) {unknown}; expected {sorted(HRSID_SPLIT_NAMES)}")
        counts = export_hrsid_to_dota(
            args.data_root,
            args.output_dir,
            splits=splits,
            difficult_strategy=args.difficult_strategy,
            same_folder=args.same_folder,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )
    else:
        unknown = [s for s in splits if s not in SSDD_SPLIT_NAMES]
        if unknown:
            parser.error(f"Unknown SSDD split(s) {unknown}; expected {sorted(SSDD_SPLIT_NAMES)}")
        counts = export_ssdd_to_dota(
            args.data_root,
            args.output_dir,
            splits=splits,
            difficult_strategy=args.difficult_strategy,
            same_folder=args.same_folder,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )
    print(f"Wrote {layout.upper()} DOTA export under {args.output_dir}")
    for split, n in counts.items():
        print(f"  {split}: {n} image(s)")


if __name__ == "__main__":
    main()
