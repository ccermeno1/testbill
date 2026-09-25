#!/usr/bin/env python
"""
Fold a folder of offline augmented copies (augmented/images + augmented/labels in YOLOv8-OBB
format, ids `<source_id>__augN`) into data/banknotes_obb, writing
annotations/train_plus_aug.json as train.json plus those copies.

Every copy is checked to come from a TRAIN image (matched by stem, hash ignored). If one
comes from valid or test the script aborts, since that would leak.

  python add_augmented.py --src augmented --dataset data/banknotes_obb
"""
import argparse
import json
import os
import re
import shutil
import sys
from glob import glob

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_dataset import poly_area  # noqa: E402

RX = re.compile(r"^(?P<src>.+?)(?:_jpg\.rf\.[A-Za-z0-9]+)?__(?P<tag>aug\d+)$", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="augmented")
    ap.add_argument("--dataset", default="dataset/banknotes_obb")
    ap.add_argument("--out_name", default="train_plus_aug")
    ap.add_argument("--min-area", type=float, default=4.0)
    args = ap.parse_args()

    ann_dir = os.path.join(args.dataset, "annotations")
    splits = {}
    for s in ("train", "valid", "test"):
        with open(os.path.join(ann_dir, f"{s}.json"), encoding="utf-8") as f:
            for im in json.load(f)["images"]:
                splits[im["file_name"][:-4]] = s
    with open(os.path.join(ann_dir, "train.json"), encoding="utf-8") as f:
        train = json.load(f)
    img_id = max(im["id"] for im in train["images"])
    ann_id = max(a["id"] for a in train["annotations"])
    n_src_train = len(train["images"])
    n_src_ann = len(train["annotations"])

    imgs = sorted({p for p in glob(os.path.join(args.src, "images", "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))})
    if not imgs:
        sys.exit(f"no images in {args.src}/images")
    bad, n_boxes, sources = [], 0, set()
    for p in imgs:
        base = os.path.splitext(os.path.basename(p))[0]
        m = RX.match(base)
        if not m:
            sys.exit(f"unrecognised name, expected <source>__augN: {base}")
        src_stem, tag = m.group("src"), m.group("tag")
        split = splits.get(src_stem)
        if split != "train":
            bad.append((base, split))
            continue
        sources.add(src_stem)
        stem = f"{src_stem}__{tag}"
        dst = os.path.join(args.dataset, "images", f"{stem}.jpg")
        if not os.path.exists(dst):
            shutil.copy2(p, dst)
        with Image.open(p) as im:
            W, H = im.size
        img_id += 1
        lbl = os.path.join(args.src, "labels", base + ".txt")
        rows = [r.split() for r in open(lbl, encoding="utf-8") if r.strip()] if os.path.exists(lbl) else []
        for r in rows:
            if len(r) != 9:
                continue
            cls = int(r[0])
            norm = [float(x) for x in r[1:]]
            poly = [norm[i] * (W if i % 2 == 0 else H) for i in range(8)]
            area = poly_area(poly)
            xs, ys = poly[0::2], poly[1::2]
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            if area < args.min_area or bw <= 1 or bh <= 1:
                continue
            ann_id += 1
            train["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": cls + 1,
                                         "segmentation": [[round(v, 2) for v in poly]],
                                         "bbox": [round(min(xs), 2), round(min(ys), 2), round(bw, 2), round(bh, 2)],
                                         "area": round(area, 2), "iscrowd": 0, "ignore": 0})
            n_boxes += 1
        train["images"].append({"id": img_id, "file_name": f"{stem}.jpg", "width": W, "height": H,
                                "original_file_name": os.path.basename(p), "roboflow_split": "augmented",
                                "source": src_stem})
    if bad:
        sys.exit(f"LEAK: {len(bad)} copies come from images outside train: {bad[:5]}")

    out = os.path.join(ann_dir, f"{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=1)
    print(f"original train: {n_src_train} imgs / {n_src_ann} boxes")
    print(f"augmented:      {len(imgs)} imgs / {n_boxes} boxes, from {len(sources)} train images")
    print(f"total:          {len(train['images'])} imgs / {len(train['annotations'])} boxes -> {out}")


if __name__ == "__main__":
    main()
