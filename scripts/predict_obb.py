#!/usr/bin/env python
"""
Predice oriented bounding boxes de billetes con un modelo PP-YOLOE-R entrenado y las
exporta en formatos faciles de consumir. Lanzar desde la raiz del proyecto:

  python scripts/predict_obb.py -c configs/ppyoloe_r_crn_s_banknotes.yml \
      -w output/ppyoloe_r_crn_s_banknotes/best_model.pdparams --split test --vis

  python scripts/predict_obb.py -c ... -w ... --images ruta/a/carpeta_o_imagen.jpg

Salida (--output_dir, por defecto output/predictions):
  <stem>.txt            una linea por caja:  clase score x1 y1 x2 y2 x3 y3 x4 y4  (pixeles)
  yolo_obb/<stem>.txt   formato YOLOv8-OBB:   clase x1 y1 ... x4 y4  (normalizado a [0,1])
  predictions.json      todo junto: {imagen: [{class_id, class_name, score, poly}]}
  vis/<stem>.jpg        (con --vis) imagen con las OBB dibujadas
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "PaddleDetection"))

import paddle  # noqa: E402
from ppdet.core.workspace import create, load_config, merge_config  # noqa: E402
from ppdet.engine import Trainer  # noqa: E402
from ppdet.utils.check import check_config, check_gpu, check_version  # noqa: E402
from ppdet.utils.logger import setup_logger  # noqa: E402

logger = setup_logger("predict_obb")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def collect_images(args, cfg):
    if args.images:
        imgs = []
        for p in args.images:
            if os.path.isdir(p):
                for ext in IMG_EXT:
                    imgs += glob.glob(os.path.join(p, f"*{ext}"))
                    imgs += glob.glob(os.path.join(p, f"*{ext.upper()}"))
            else:
                imgs.append(p)
        return sorted(set(imgs))
    ds = cfg["TestDataset"]
    dataset_dir = ds.dataset_dir
    anno = os.path.join(dataset_dir, f"annotations/{args.split}.json")
    with open(anno, encoding="utf-8") as f:
        coco = json.load(f)
    return [os.path.join(dataset_dir, "images", im["file_name"]) for im in coco["images"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="configs/ppyoloe_r_crn_s_banknotes.yml")
    ap.add_argument("-w", "--weights", required=True, help="fichero .pdparams entrenado")
    ap.add_argument("--images", nargs="*", help="imagenes o carpetas; si se omite se usa --split")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--output_dir", default="output/predictions")
    ap.add_argument("--threshold", type=float, default=0.5, help="score minimo para exportar")
    ap.add_argument("--vis", action="store_true", help="guardar imagenes con las cajas dibujadas")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    merge_config({"weights": args.weights, "use_gpu": not args.cpu})
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    paddle.set_device("gpu" if cfg.use_gpu else "cpu")

    images = collect_images(args, cfg)
    if not images:
        sys.exit("no hay imagenes que predecir")
    logger.info(f"{len(images)} imagenes")

    with open(os.path.join(cfg["TestDataset"].dataset_dir, "classes.txt"), encoding="utf-8") as f:
        class_names = [l.strip() for l in f if l.strip()]

    trainer = Trainer(cfg, mode="test")
    trainer.load_weights(cfg.weights)
    trainer.dataset.set_images(images)
    loader = create("TestReader")(trainer.dataset, 0)
    imid2path = trainer.dataset.get_imid2path()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "yolo_obb"), exist_ok=True)
    if args.vis:
        os.makedirs(os.path.join(args.output_dir, "vis"), exist_ok=True)
        from PIL import Image, ImageDraw

    trainer.model.eval()
    all_res = {}
    n_boxes = 0
    with paddle.no_grad():
        for data in loader:
            outs = trainer.model(data)
            bboxes = outs["bbox"].numpy()          # [N, 10]: clase, score, x1..y4
            bbox_num = outs["bbox_num"].numpy()
            im_ids = data["im_id"].numpy()
            start = 0
            for i, im_id in enumerate(im_ids):
                end = start + int(bbox_num[i])
                dets = bboxes[start:end]
                start = end
                path = imid2path[int(im_id)]
                stem = os.path.splitext(os.path.basename(path))[0]
                h, w = [int(v) for v in data["im_shape"].numpy()[i]]
                sf_h, sf_w = data["scale_factor"].numpy()[i]  # im_shape = original * scale_factor
                W, H = round(w / sf_w), round(h / sf_h)

                rows = []
                for d in dets:
                    cls, score = int(d[0]), float(d[1])
                    if cls < 0 or score < args.threshold:
                        continue
                    poly = [float(v) for v in d[2:10]]
                    rows.append({"class_id": cls, "class_name": class_names[cls] if cls < len(class_names) else str(cls),
                                 "score": round(score, 4), "poly": [round(v, 2) for v in poly]})
                all_res[os.path.relpath(path, ROOT) if path.startswith(ROOT) else path] = rows
                n_boxes += len(rows)

                with open(os.path.join(args.output_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(f"{r['class_id']} {r['score']:.4f} " + " ".join(f"{v:.1f}" for v in r["poly"]) + "\n")
                with open(os.path.join(args.output_dir, "yolo_obb", f"{stem}.txt"), "w", encoding="utf-8") as f:
                    for r in rows:
                        norm = [r["poly"][k] / (W if k % 2 == 0 else H) for k in range(8)]
                        f.write(f"{r['class_id']} " + " ".join(f"{v:.6f}" for v in norm) + "\n")

                if args.vis:
                    im = Image.open(path).convert("RGB")
                    draw = ImageDraw.Draw(im)
                    for r in rows:
                        pts = [(r["poly"][k], r["poly"][k + 1]) for k in range(0, 8, 2)]
                        draw.polygon(pts, outline=(0, 255, 0), width=3)
                        draw.text((pts[0][0] + 2, pts[0][1] + 2), f"{r['class_name']} {r['score']:.2f}", fill=(255, 255, 0))
                    im.save(os.path.join(args.output_dir, "vis", f"{stem}.jpg"), quality=95)

    with open(os.path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=1)
    logger.info(f"{n_boxes} cajas (score >= {args.threshold}) en {len(all_res)} imagenes -> {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
