#!/usr/bin/env python
"""
Convert any Roboflow YOLOv8-OBB export (whatever train/valid/test folders it has) into an
evaluation/inference dataset: images re-encoded to JPG (HEIC supported), COCO with polygons
and DOTA labelTxt. Meant for extra test sets; use prepare_dataset.py for the main one, which
carries the v1 splits.

  python convert_yolo_obb.py --src "billetesprueba 2.yolov8-obb" --out data/billetesprueba

Output: <out>/images/<stem>.jpg, <out>/labelTxt/<stem>.txt, <out>/annotations/all.json
(plus one <split>.json per folder found), <out>/classes.txt
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from glob import glob

from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_dataset import poly_area, read_class_names  # noqa: E402

RX_ROBOFLOW = re.compile(r"_(jpe?g|png|heic|heif|bmp|webp)\.rf\.[A-Za-z0-9]+$", re.IGNORECASE)
IMG_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp")


def stem_of(path):
    return RX_ROBOFLOW.sub("", os.path.splitext(os.path.basename(path))[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--min-area", type=float, default=4.0)
    ap.add_argument("--merge_class", default=None, help="merge every kept class into a single one with this name")
    ap.add_argument("--drop_classes", nargs="*", default=[], help="classes to ignore; their boxes are not exported")
    ap.add_argument("--exclude", default=None, help="JSON holding an 'exclude_dup' list of image paths to skip")
    ap.add_argument("--max_single_area", type=float, default=None,
                    help="skip images holding a single box larger than this fraction of the image")
    ap.add_argument("--skip_empty", action="store_true", help="skip images left with no boxes after class filtering")
    args = ap.parse_args()
    excluded = set()
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as f:
            excluded = {os.path.normpath(os.path.abspath(p.replace("\\", "/"))) for p in json.load(f)["exclude_dup"]}

    src_names = read_class_names(os.path.join(args.src, "data.yaml"))
    if not src_names:
        sys.exit("could not read the class names from data.yaml")
    # source class -> target class index (None means drop it)
    if args.merge_class:
        class_names = [args.merge_class]
        cls_map = {i: (None if n in args.drop_classes else 0) for i, n in enumerate(src_names)}
    else:
        class_names = [n for n in src_names if n not in args.drop_classes]
        cls_map = {i: (None if n in args.drop_classes else class_names.index(n)) for i, n in enumerate(src_names)}
    for d in ("images", "labelTxt", "annotations"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]

    splits = [s for s in ("train", "valid", "test") if os.path.isdir(os.path.join(args.src, s, "images"))]
    coco = {s: {"images": [], "annotations": [], "categories": categories} for s in splits + ["all"]}
    seen = {}
    stats = Counter()
    img_id = ann_id = 0
    for split in splits:
        for src_img in sorted(glob(os.path.join(args.src, split, "images", "*"))):
            if not src_img.lower().endswith(IMG_EXT):
                continue
            if os.path.normpath(os.path.abspath(src_img)) in excluded:
                stats["skipped_duplicate"] += 1
                continue
            stem = stem_of(src_img)
            if stem in seen:
                sys.exit(f"duplicate stem: {stem} ({seen[stem]} and {src_img})")
            seen[stem] = src_img
            img_id += 1
            dst = os.path.join(args.out, "images", f"{stem}.jpg")
            with Image.open(src_img) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                W, H = im.size
            lbl = os.path.join(args.src, split, "labels", os.path.splitext(os.path.basename(src_img))[0] + ".txt")
            rows = [r.split() for r in open(lbl, encoding="utf-8") if r.strip()] if os.path.exists(lbl) else []
            if not os.path.exists(lbl):
                stats["missing_label"] += 1
            dota = []
            anns = []
            for r in rows:
                if len(r) != 9:
                    stats["invalid_lines"] += 1
                    continue
                cls = cls_map.get(int(r[0]))
                if cls is None:
                    stats["boxes_class_dropped"] += 1
                    continue
                norm = [float(x) for x in r[1:]]
                poly = [norm[i] * (W if i % 2 == 0 else H) for i in range(8)]
                area = poly_area(poly)
                xs, ys = poly[0::2], poly[1::2]
                bw, bh = max(xs) - min(xs), max(ys) - min(ys)
                if area < args.min_area or bw <= 1 or bh <= 1:
                    stats["boxes_dropped"] += 1
                    continue
                ann_id += 1
                anns.append({"id": ann_id, "image_id": img_id, "category_id": cls + 1,
                             "segmentation": [[round(v, 2) for v in poly]],
                             "bbox": [round(min(xs), 2), round(min(ys), 2), round(bw, 2), round(bh, 2)],
                             "area": round(area, 2), "iscrowd": 0, "ignore": 0})
                dota.append(" ".join(f"{v:.1f}" for v in poly) + f" {class_names[cls]} 0")
                stats["boxes"] += 1
            if args.skip_empty and not anns:
                stats["skipped_no_boxes"] += 1
                img_id -= 1
                continue
            if args.max_single_area and len(anns) == 1 and anns[0]["area"] / (W * H) > args.max_single_area:
                stats["skipped_closeup"] += 1
                stats["boxes"] -= 1
                img_id -= 1
                ann_id -= 1
                continue
            if not os.path.exists(dst):
                with Image.open(src_img) as im:
                    ImageOps.exif_transpose(im).convert("RGB").save(dst, quality=args.quality)
            with open(os.path.join(args.out, "labelTxt", f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(dota) + ("\n" if dota else ""))
            info = {"id": img_id, "file_name": f"{stem}.jpg", "width": W, "height": H,
                    "original_file_name": os.path.basename(src_img), "roboflow_split": split}
            for key in (split, "all"):
                coco[key]["images"].append(info)
                coco[key]["annotations"].extend(anns)
            stats["images"] += 1

    for key, c in coco.items():
        with open(os.path.join(args.out, "annotations", f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(c, f, indent=1)
    print(f"classes: {class_names}")
    print(", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print(f"written to: {os.path.abspath(args.out)}  (annotations: {', '.join(coco)})")


if __name__ == "__main__":
    main()
