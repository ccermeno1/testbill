#!/usr/bin/env python
"""
Build the selection manifest for the external "Euro Banknote Detection" dataset: for every
source photo it records whether it went into train (and under what name) or why it was left
out (duplicate, naming the matched photo and distance; close-up; no banknote).

  python extra_manifest.py
Output: data_manifests/eurobanknotes_extra_selection.csv / .json / .md
"""
import csv
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image

SRC = "Euro Banknote Detection.yolov8-obb"
EXTRA = "dataset/eurobanknotes_extra"
MERGED = "dataset/banknotes_obb/annotations/train_plus_extra.json"
PAIRS = "output/duplicates/pairs.json"
EXCLUDE = "output/duplicates/exclude.json"
OUT = "data_manifests"
NAMES = "100_1 100_2 10_1 10_2 200_1 200_2 20_1 20_2 50_1 50_2 5_1 5_2 hand".split()
MAX_SINGLE_AREA = 0.5


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = json.load(open(PAIRS, encoding="utf-8"))
    excluded = {os.path.normpath(os.path.abspath(p)) for p in json.load(open(EXCLUDE, encoding="utf-8"))["exclude_dup"]}
    ours_split = {}
    for s in ("train", "valid", "test"):
        for im in json.load(open(f"dataset/banknotes_obb/annotations/{s}.json", encoding="utf-8"))["images"]:
            ours_split[im["file_name"]] = s
    best = {}
    for p in pairs:
        a = os.path.normpath(os.path.abspath(p["a"]))
        b = os.path.basename(p["b"])
        where = ours_split.get(b, "billetesprueba" if "billetesprueba" in p["b"] else "new")
        if p["kind"] == "new-vs-new":
            b2 = os.path.normpath(os.path.abspath(p["b"]))
            for x, y in ((a, b2), (b2, a)):
                if x not in best or p["dist"] < best[x][0]:
                    best[x] = (p["dist"], os.path.basename(y), "new (same dataset)")
        elif a not in best or p["dist"] < best[a][0]:
            best[a] = (p["dist"], b, where)
    included = {im["original_file_name"]: im for im in json.load(open(f"{EXTRA}/annotations/all.json", encoding="utf-8"))["images"]}
    merged_names = {im["file_name"] for im in json.load(open(MERGED, encoding="utf-8"))["images"] if im.get("roboflow_split") == "extra"}

    rows = []
    for split in ("train", "valid", "test"):
        for p in sorted(glob.glob(os.path.join(SRC, split, "images", "*.jpg"))):
            base = os.path.basename(p)
            ap = os.path.normpath(os.path.abspath(p))
            with Image.open(p) as im:
                W, H = im.size
            lbl = os.path.join(SRC, split, "labels", os.path.splitext(base)[0] + ".txt")
            n_notes = n_hand = 0
            max_area = 0.0
            for l in open(lbl, encoding="utf-8"):
                t = l.split()
                if len(t) != 9:
                    continue
                if NAMES[int(t[0])] == "hand":
                    n_hand += 1
                    continue
                n_notes += 1
                v = np.ascontiguousarray((np.array(list(map(float, t[1:9])), dtype=np.float32).reshape(4, 2) * np.array([W, H], dtype=np.float32)).astype(np.float32))
                (_, _), (w, h), _ = cv2.minAreaRect(v)
                max_area = max(max_area, w * h / (W * H))
            row = {"original_file": base, "roboflow_split": split, "n_notes": n_notes, "n_hands": n_hand,
                   "largest_note_area": round(max_area, 3), "status": "", "reason": "", "duplicate_of": "",
                   "distance": "", "duplicate_dataset": "", "name_in_train": ""}
            if ap in excluded:
                d, b, where = best.get(ap, ("", "", ""))
                row.update(status="excluded", reason="duplicate", duplicate_of=b, distance=d, duplicate_dataset=where)
            elif n_notes == 0:
                row.update(status="excluded", reason="no banknotes (hand only)")
            elif n_notes == 1 and max_area > MAX_SINGLE_AREA:
                row.update(status="excluded", reason=f"close-up (single note over {int(MAX_SINGLE_AREA*100)}% of the image)")
            elif base in included:
                new_name = "ebd_" + included[base]["file_name"]
                row.update(status="included in train", name_in_train=new_name if new_name in merged_names else "(not found in train_plus_extra.json)")
            else:
                row.update(status="excluded", reason="?")
            rows.append(row)

    fields = list(rows[0].keys())
    with open(f"{OUT}/eurobanknotes_extra_selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(f"{OUT}/eurobanknotes_extra_selection.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    inc = [r for r in rows if r["status"].startswith("included")]
    exc = [r for r in rows if r["status"] == "excluded"]
    by_reason = {}
    for r in exc:
        by_reason.setdefault(r["reason"], []).append(r)
    dup_where = {}
    for r in by_reason.get("duplicate", []):
        dup_where[r["duplicate_dataset"]] = dup_where.get(r["duplicate_dataset"], 0) + 1
    md = ["# Picking images from *Euro Banknote Detection* for training", "",
          f"Source: `{SRC}` ({len(rows)} photos, 12 banknote classes plus `hand`). Criteria:", "",
          "1. **Duplicates**: perceptual hash (pHash/dHash, Hamming distance <= 8) against `banknotes_obb`",
          "   (train/valid/test) and `billetesprueba`, and among the new photos themselves, keeping one.",
          f"2. **Close-ups**: a single note covering more than {int(MAX_SINGLE_AREA*100)} % of the image. We have plenty already.",
          "3. **No banknotes**: photos holding only a `hand` box.",
          "4. The 12 banknote classes are merged into `euro_banknote`; `hand` boxes are dropped.", "",
          f"**Included in train: {len(inc)}**, as `images/ebd_<name>.jpg` in `annotations/train_plus_extra.json`.",
          f"**Excluded: {len(exc)}** - " + ", ".join(f"{k}: {len(v)}" for k, v in by_reason.items()) + ".",
          "Duplicates by reference dataset: " + ", ".join(f"{k}: {v}" for k, v in sorted(dup_where.items())) + ".", "",
          "Full per-photo detail, with distance and matched photo, in `eurobanknotes_extra_selection.csv`.", "",
          "## Included", "", "| source photo | roboflow split | notes | name in train |", "|---|---|---|---|"]
    md += [f"| {r['original_file']} | {r['roboflow_split']} | {r['n_notes']} | {r['name_in_train']} |" for r in inc]
    md += ["", "## Excluded", "", "| source photo | reason | duplicate of | dist. | dataset |", "|---|---|---|---|---|"]
    md += [f"| {r['original_file']} | {r['reason']} | {r['duplicate_of']} | {r['distance']} | {r['duplicate_dataset']} |" for r in exc]
    open(f"{OUT}/eurobanknotes_extra_selection.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"{len(rows)} photos: {len(inc)} included, {len(exc)} excluded ({', '.join(f'{k}: {len(v)}' for k, v in by_reason.items())})")
    print(f"-> {OUT}/eurobanknotes_extra_selection.{{csv,json,md}}")


if __name__ == "__main__":
    main()
