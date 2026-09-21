#!/usr/bin/env python
"""
Convierte el export YOLOv8-OBB de Roboflow al formato que consume PP-YOLOE-R
(PaddleDetection, COCO con `segmentation` = poligono de 4 puntos), usando los
splits congelados de v1/{train,valid,test}.txt.

Emparejamiento: los ids de v1 y los ficheros reales llevan un hash de Roboflow
distinto (`<stem>_jpg.rf.<HASH>`), asi que se empareja SOLO por `<stem>`
(p.ej. `005_Euro_013`). Las carpetas train/valid/test del export se ignoran.

Salida (por defecto dataset/banknotes_obb/):
  images/<stem>.jpg                     todas las imagenes con nombre limpio
  labelTxt/<stem>.txt                   formato DOTA: x1 y1 ... x4 y4 clase dificultad
  annotations/{train,valid,test}.json   COCO (bbox + segmentation poligonal)
  splits/{train,valid,test}.txt         stems resueltos por split
  classes.txt
"""
import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from glob import glob

from PIL import Image

RX_ROBOFLOW = re.compile(r"_jpg\.rf\.[A-Za-z0-9]+(\.jpg)?$")
SPLITS = ("train", "valid", "test")


def stem_of(name):
    return RX_ROBOFLOW.sub("", os.path.basename(name))


def read_split(path):
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(stem_of(line))
    return ids


def read_class_names(data_yaml):
    """Parser minimo del bloque `names:` de data.yaml (formato dict o lista)."""
    names = {}
    in_names = False
    with open(data_yaml, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if re.match(r"^names\s*:", line):
                in_names = True
                inline = line.split(":", 1)[1].strip()
                if inline.startswith("["):
                    return [s.strip().strip("'\"") for s in inline.strip("[]").split(",") if s.strip()]
                continue
            if in_names:
                m = re.match(r"^\s+(\d+)\s*:\s*(.+?)\s*$", line)
                if m:
                    names[int(m.group(1))] = m.group(2).strip("'\"")
                    continue
                m = re.match(r"^\s+-\s*(.+?)\s*$", line)
                if m:
                    names[len(names)] = m.group(1).strip("'\"")
                    continue
                if line.strip():
                    break
    return [names[i] for i in sorted(names)]


def poly_area(p):
    xs, ys = p[0::2], p[1::2]
    return 0.5 * abs(sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i] for i in range(4)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="Annotated banknotes 2.yolov8-obb", help="carpeta del export YOLOv8-OBB")
    ap.add_argument("--splits", default="v1", help="carpeta con train.txt/valid.txt/test.txt")
    ap.add_argument("--out", default="dataset/banknotes_obb", help="carpeta de salida")
    ap.add_argument("--min-area", type=float, default=4.0, help="area minima (px^2) para conservar una caja")
    ap.add_argument("--clip", action="store_true",
                    help="recortar coordenadas a [0,1] (por defecto NO: se conserva la geometria del rectangulo rotado aunque sobresalga del borde)")
    args = ap.parse_args()

    # 1) indexar imagenes reales por stem
    real = {}
    for split in SPLITS:
        for p in glob(os.path.join(args.src, split, "images", "*.jpg")):
            s = stem_of(p)
            if s in real:
                sys.exit(f"stem duplicado en los datos: {s} -> {real[s][1]} y {p}")
            real[s] = (split, p)
    if not real:
        sys.exit(f"no se han encontrado imagenes en {args.src}")

    # 2) leer splits v1
    assign = {}
    for split in SPLITS:
        for s in read_split(os.path.join(args.splits, f"{split}.txt")):
            if s in assign:
                sys.exit(f"{s} aparece en dos splits: {assign[s]} y {split}")
            assign[s] = split
    missing = sorted(set(assign) - set(real))
    extra = sorted(set(real) - set(assign))
    if missing:
        sys.exit(f"{len(missing)} ids de {args.splits} sin imagen: {missing[:10]}")
    if extra:
        print(f"AVISO: {len(extra)} imagenes no aparecen en ningun split y se ignoran: {extra[:10]}")

    class_names = read_class_names(os.path.join(args.src, "data.yaml"))
    if not class_names:
        sys.exit("no se pudieron leer los nombres de clase de data.yaml")
    print(f"clases: {class_names}")

    # 3) crear salida
    for d in ("images", "labelTxt", "annotations", "splits"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")

    categories = [{"id": i + 1, "name": n, "supercategory": "banknote"} for i, n in enumerate(class_names)]
    coco = {split: {"images": [], "annotations": [], "categories": categories} for split in SPLITS}
    stats = defaultdict(Counter)
    max_dev = 0.0
    img_id = 0
    ann_id = 0
    for s in sorted(assign):
        split = assign[s]
        rf_split, src_img = real[s]
        img_id += 1
        dst_img = os.path.join(args.out, "images", f"{s}.jpg")
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
        with Image.open(src_img) as im:
            W, H = im.size

        src_lbl = os.path.join(args.src, rf_split, "labels", os.path.splitext(os.path.basename(src_img))[0] + ".txt")
        if os.path.exists(src_lbl):
            with open(src_lbl, encoding="utf-8") as f:
                rows = [r.split() for r in f if r.strip()]
        else:
            rows = []
            stats[split]["images_sin_label"] += 1

        dota_lines = []
        for r in rows:
            if len(r) != 9:
                print(f"AVISO: linea con {len(r)} campos en {src_lbl}, se ignora")
                stats[split]["lineas_invalidas"] += 1
                continue
            cls = int(r[0])
            norm = [float(x) for x in r[1:]]
            max_dev = max(max_dev, max(0.0, *(-v for v in norm)), max(0.0, *(v - 1.0 for v in norm)))
            if args.clip:
                norm = [min(1.0, max(0.0, v)) for v in norm]
            poly = [norm[i] * (W if i % 2 == 0 else H) for i in range(8)]
            area = poly_area(poly)
            xs, ys = poly[0::2], poly[1::2]
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            if area < args.min_area or bw <= 1 or bh <= 1:
                print(f"AVISO: caja degenerada (area={area:.2f}) en {s}, se ignora")
                stats[split]["cajas_descartadas"] += 1
                continue
            ann_id += 1
            coco[split]["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls + 1,
                "segmentation": [[round(v, 2) for v in poly]],
                "bbox": [round(min(xs), 2), round(min(ys), 2), round(bw, 2), round(bh, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
                "ignore": 0,
            })
            dota_lines.append(" ".join(f"{v:.1f}" for v in poly) + f" {class_names[cls]} 0")
            stats[split]["cajas"] += 1
            stats[split][f"clase_{class_names[cls]}"] += 1

        with open(os.path.join(args.out, "labelTxt", f"{s}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(dota_lines) + ("\n" if dota_lines else ""))
        coco[split]["images"].append({
            "id": img_id,
            "file_name": f"{s}.jpg",
            "width": W,
            "height": H,
            "original_file_name": os.path.basename(src_img),
            "roboflow_split": rf_split,
        })
        stats[split]["imagenes"] += 1

    for split in SPLITS:
        with open(os.path.join(args.out, "annotations", f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(coco[split], f, indent=1)
        with open(os.path.join(args.out, "splits", f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(im["file_name"][:-4] for im in coco[split]["images"]) + "\n")

    print(f"\nmax desviacion de coordenadas fuera de [0,1]: {max_dev:.3f} "
          f"({'recortada' if args.clip else 'conservada'})")
    for split in SPLITS:
        print(f"{split:6s}: " + ", ".join(f"{k}={v}" for k, v in sorted(stats[split].items())))
    print(f"\nsalida en: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
