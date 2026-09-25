#!/usr/bin/env python
"""
Predice cajas orientadas sobre imagenes sueltas o una carpeta y guarda los resultados.

  python src/ppyoloer_mps/infer.py -w models/ppyoloe_r/ppyoloe_r_s_banknotes_extra.pt \
      --images fotos/ --out-dir runs/pred --vis --device auto

Salida:
  <out-dir>/<stem>.txt            clase score x1 y1 ... x4 y4   (pixeles de la imagen original)
  <out-dir>/yolo_obb/<stem>.txt   formato YOLOv8-OBB normalizado
  <out-dir>/predictions.json      todo junto
  <out-dir>/vis/<stem>.jpg        con --vis
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r, load_checkpoint
from ppyoloer_mps.ppyoloe_obb.data import IMG_EXT, load_image, preprocess_image
from ppyoloer_mps.ppyoloe_obb.engine import resolve_device
from ppyoloer_mps.ppyoloe_obb.ops import batched_postprocess


def collect_images(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                out += [os.path.join(root, f) for f in files if f.lower().endswith(IMG_EXT)]
        else:
            out.append(p)
    return sorted(out)


def draw(image, rows, path_out) -> None:
    from PIL import Image, ImageDraw

    im = Image.fromarray(image)
    d = ImageDraw.Draw(im)
    width = max(2, min(im.size) // 250)
    for r in rows:
        pts = [(r["poly"][i], r["poly"][i + 1]) for i in range(0, 8, 2)]
        d.polygon(pts, outline=(0, 255, 0), width=width)
        d.text((pts[0][0] + 2, pts[0][1] + 2), f"{r['class_name']} {r['score']:.2f}", fill=(255, 255, 0))
    im.save(path_out, quality=95)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="imagenes o carpetas")
    ap.add_argument("--out-dir", default="runs/pred")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    payload = torch.load(args.weights, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    classes = meta.get("classes") or [f"class_{i}" for i in range(meta.get("num_classes", 1))]
    model = build_ppyoloe_r(num_classes=meta.get("num_classes", len(classes)), size=meta.get("size", "s"))
    load_checkpoint(model, args.weights)
    model.to(device).eval()

    images = collect_images(args.images)
    if not images:
        raise SystemExit("no se han encontrado imagenes")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "yolo_obb"), exist_ok=True)
    if args.vis:
        os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)
    print(f"dispositivo: {device} | {len(images)} imagenes | conf {args.conf}")

    results: dict[str, list[dict]] = {}
    total = 0
    for path in images:
        image = load_image(path)
        h, w = image.shape[:2]
        x, scale_factor = preprocess_image(image, args.img_size)
        with torch.no_grad():
            scores, rboxes = model(x.to(device))
        det = batched_postprocess(
            scores.float(), rboxes.float(),
            score_threshold=args.conf, nms_threshold=args.nms_iou,
            scale_factor=torch.as_tensor(scale_factor, device=scores.device).unsqueeze(0),
        )[0]
        rows = []
        for poly, score, label in zip(det["polys"].cpu(), det["scores"].cpu(), det["labels"].cpu()):
            rows.append(
                {
                    "class_id": int(label),
                    "class_name": classes[int(label)] if int(label) < len(classes) else str(int(label)),
                    "score": round(float(score), 4),
                    "poly": [round(float(v), 2) for v in poly],
                }
            )
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(args.out_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r['class_id']} {r['score']:.4f} " + " ".join(f"{v:.1f}" for v in r["poly"]) + "\n")
        with open(os.path.join(args.out_dir, "yolo_obb", f"{stem}.txt"), "w", encoding="utf-8") as f:
            for r in rows:
                norm = [r["poly"][i] / (w if i % 2 == 0 else h) for i in range(8)]
                f.write(f"{r['class_id']} " + " ".join(f"{v:.6f}" for v in norm) + "\n")
        if args.vis:
            draw(image, rows, os.path.join(args.out_dir, "vis", f"{stem}.jpg"))
        results[path] = rows
        total += len(rows)

    with open(os.path.join(args.out_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"{total} cajas (score >= {args.conf}) en {len(images)} imagenes -> {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
