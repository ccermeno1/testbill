#!/usr/bin/env python
"""
Anade las imagenes de un dataset extra (convertido con convert_yolo_obb.py) al train del dataset
principal: copia las imagenes a dataset/banknotes_obb/images/<prefix><stem>.jpg y genera
annotations/<out_name>.json = <base>.json + extra.

  python scripts/merge_extra.py --extra dataset/eurobanknotes_extra --out_name train_plus_extra
"""
import argparse, json, os, shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/banknotes_obb")
    ap.add_argument("--base", default="train", help="annotations/<base>.json de partida")
    ap.add_argument("--extra", required=True, help="carpeta del dataset extra (con images/ y annotations/all.json)")
    ap.add_argument("--prefix", default="ebd_")
    ap.add_argument("--out_name", default="train_plus_extra")
    args = ap.parse_args()
    ann_dir = os.path.join(args.dataset, "annotations")
    base = json.load(open(os.path.join(ann_dir, f"{args.base}.json"), encoding="utf-8"))
    extra = json.load(open(os.path.join(args.extra, "annotations", "all.json"), encoding="utf-8"))
    if [c["name"] for c in extra["categories"]] != [c["name"] for c in base["categories"]]:
        raise SystemExit(f"clases distintas: {extra['categories']} vs {base['categories']}")
    img_id = max(im["id"] for im in base["images"]); ann_id = max(a["id"] for a in base["annotations"])
    id_map = {}
    n0, a0 = len(base["images"]), len(base["annotations"])
    for im in extra["images"]:
        img_id += 1; id_map[im["id"]] = img_id
        new_name = f"{args.prefix}{im['file_name']}"
        dst = os.path.join(args.dataset, "images", new_name)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(args.extra, "images", im["file_name"]), dst)
        base["images"].append({**im, "id": img_id, "file_name": new_name, "roboflow_split": "extra", "source_dataset": os.path.basename(args.extra.rstrip("/\\"))})
    for a in extra["annotations"]:
        ann_id += 1
        base["annotations"].append({**a, "id": ann_id, "image_id": id_map[a["image_id"]]})
    out = os.path.join(ann_dir, f"{args.out_name}.json")
    json.dump(base, open(out, "w", encoding="utf-8"), indent=1)
    print(f"{args.base}: {n0} imgs / {a0} cajas  +  extra: {len(extra['images'])} / {len(extra['annotations'])}  ->  {len(base['images'])} / {len(base['annotations'])}  ({out})")

if __name__ == "__main__":
    main()
