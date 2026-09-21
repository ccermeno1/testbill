#!/usr/bin/env python
"""
Exporta un dataset convertido (dataset/<x>/images + labelTxt + classes.txt) a formato YOLOv8-OBB
(coordenadas normalizadas), listo para subir a Roboflow o usar con Ultralytics.

  python scripts/export_yolo_obb.py --src dataset/eurobanknotes_extra --out export/eurobanknotes_extra_yolov8obb

Salida: <out>/train/images/*.jpg, <out>/train/labels/*.txt, <out>/data.yaml
"""
import argparse, glob, os, shutil
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train", help="nombre de la carpeta de split en la salida")
    ap.add_argument("--clip", action="store_true", help="recortar coordenadas a [0,1]")
    args = ap.parse_args()
    classes = [l.strip() for l in open(os.path.join(args.src, "classes.txt"), encoding="utf-8") if l.strip()]
    cid = {n: i for i, n in enumerate(classes)}
    img_dir = os.path.join(args.out, args.split, "images"); lbl_dir = os.path.join(args.out, args.split, "labels")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
    n_img = n_box = 0
    for p in sorted(glob.glob(os.path.join(args.src, "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        with Image.open(p) as im:
            W, H = im.size
        shutil.copy2(p, os.path.join(img_dir, f"{stem}.jpg"))
        lines = []
        lt = os.path.join(args.src, "labelTxt", f"{stem}.txt")
        if os.path.exists(lt):
            for l in open(lt, encoding="utf-8"):
                t = l.split()
                if len(t) < 9:
                    continue
                v = [float(x) for x in t[:8]]
                norm = [v[i] / (W if i % 2 == 0 else H) for i in range(8)]
                if args.clip:
                    norm = [min(1.0, max(0.0, x)) for x in norm]
                lines.append(f"{cid[t[8]]} " + " ".join(f"{x:.6f}" for x in norm))
                n_box += 1
        with open(os.path.join(lbl_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"{args.split}: {args.split}/images\n\nnc: {len(classes)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes)))
    print(f"{n_img} imagenes, {n_box} cajas -> {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
