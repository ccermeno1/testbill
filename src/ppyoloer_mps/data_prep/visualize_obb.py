#!/usr/bin/env python
"""
Draw ground truth (red) and predictions (green, with score) over a converted dataset, reading
the predictions.json written by infer.py.

  python visualize_obb.py --dataset_dir data/billetesprueba \
      --predictions runs/pred/predictions.json --output_dir runs/pred/vis_gt_pred

Output: <output_dir>/<stem>.jpg (longest side --max_side px) plus contact sheets
<output_dir>/_sheet_NN.jpg holding --cols x --rows thumbnails, worst IoU first.
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

RED, GREEN, WHITE = (255, 40, 40), (40, 255, 40), (255, 255, 255)


def load_gt(dataset_dir, stem):
    p = os.path.join(dataset_dir, "labelTxt", f"{stem}.txt")
    if not os.path.exists(p):
        return []
    return [[float(v) for v in l.split()[:8]] for l in open(p, encoding="utf-8") if l.strip()]


def poly_of(v):
    return Polygon(list(zip(v[0::2], v[1::2])))


def worst_iou(gts, preds):
    """Best IoU reached per ground truth; returns the worst of them (1.0 with no GT, 0 if a GT went undetected)."""
    if not gts:
        return 1.0 if not preds else 0.0
    best = []
    for g in gts:
        gp = poly_of(g)
        m = 0.0
        for d in preds:
            pp = poly_of(d["poly"])
            inter = pp.intersection(gp).area
            u = pp.area + gp.area - inter
            m = max(m, inter / u if u > 0 else 0)
        best.append(m)
    return min(best)


def draw(im, gts, preds, font):
    d = ImageDraw.Draw(im)
    w = max(2, min(im.size) // 250)
    for g in gts:
        d.polygon(list(zip(g[0::2], g[1::2])), outline=RED, width=w)
    for p in preds:
        v = p["poly"]
        d.polygon(list(zip(v[0::2], v[1::2])), outline=GREEN, width=w)
        x, y = min(v[0::2]), min(v[1::2])
        txt = f"{p['score']:.2f}"
        tw, th = d.textbbox((0, 0), txt, font=font)[2:]
        d.rectangle([x, y, x + tw + 6, y + th + 4], fill=(0, 0, 0))
        d.text((x + 3, y + 2), txt, fill=GREEN, font=font)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--predictions", required=True, help="predictions.json written by infer.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_side", type=int, default=1200)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--thumb", type=int, default=360)
    args = ap.parse_args()

    preds = json.load(open(args.predictions, encoding="utf-8"))
    by_stem = {os.path.splitext(os.path.basename(k))[0]: v for k, v in preds.items()}
    os.makedirs(args.output_dir, exist_ok=True)

    items = []
    for stem in sorted(by_stem):
        src = os.path.join(args.dataset_dir, "images", f"{stem}.jpg")
        if not os.path.exists(src):
            continue
        gts = load_gt(args.dataset_dir, stem)
        dets = by_stem[stem]
        im = Image.open(src).convert("RGB")
        scale = min(1.0, args.max_side / max(im.size))
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
        s_gts = [[v * scale for v in g] for g in gts]
        s_dets = [{"poly": [v * scale for v in d["poly"]], "score": d["score"]} for d in dets]
        font = ImageFont.load_default(size=max(14, min(im.size) // 40)) if hasattr(ImageFont, "load_default") else None
        try:
            font = ImageFont.truetype("arial.ttf", max(14, min(im.size) // 40))
        except OSError:
            pass
        im = draw(im, s_gts, s_dets, font)
        im.save(os.path.join(args.output_dir, f"{stem}.jpg"), quality=90)
        items.append((worst_iou(gts, dets), stem, im, len(gts), len(dets)))

    # contact sheets, worst first
    items.sort(key=lambda t: t[0])
    per = args.cols * args.rows
    T = args.thumb
    try:
        sfont = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        sfont = ImageFont.load_default()
    for n in range(0, len(items), per):
        sheet = Image.new("RGB", (args.cols * T, args.rows * (T + 20)), WHITE)
        d = ImageDraw.Draw(sheet)
        for k, (iou, stem, im, ng, nd) in enumerate(items[n:n + per]):
            th = im.copy()
            th.thumbnail((T, T))
            x, y = (k % args.cols) * T, (k // args.cols) * (T + 20)
            sheet.paste(th, (x + (T - th.width) // 2, y))
            d.text((x + 4, y + T + 2), f"{stem[:22]}  GT {ng} / pred {nd}  IoU min {iou:.2f}", fill=(0, 0, 0), font=sfont)
        sheet.save(os.path.join(args.output_dir, f"_sheet_{n // per + 1:02d}.jpg"), quality=85)
    print(f"{len(items)} imagenes y {(len(items) + per - 1) // per} hojas de contacto en {os.path.abspath(args.output_dir)}")
    print("red = ground truth, green = prediction with score. Sheets ordered worst IoU first.")


if __name__ == "__main__":
    main()
