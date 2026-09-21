#!/usr/bin/env python
"""
Busca imagenes duplicadas o casi duplicadas entre carpetas de imagenes usando hashes perceptuales
(pHash + dHash, distancia de Hamming). Imprime los pares sospechosos y guarda un JSON y una hoja de
contacto para revisarlos a ojo.

  python scripts/find_duplicates.py --new "Euro Banknote Detection.yolov8-obb" \
      --ref dataset/banknotes_obb/images dataset/billetesprueba/images --max_dist 10
"""
import argparse, glob, json, os
import imagehash
from PIL import Image, ImageOps

def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                out += [os.path.join(root, f) for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        else:
            out.append(p)
    return sorted(out)

def hashes(files):
    h = {}
    for f in files:
        try:
            with Image.open(f) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                h[f] = (imagehash.phash(im, hash_size=8), imagehash.dhash(im, hash_size=8))
        except Exception as e:
            print("no se pudo leer", f, e)
    return h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--ref", nargs="+", default=[])
    ap.add_argument("--max_dist", type=int, default=10, help="distancia de Hamming maxima (de 64) en pHash o dHash")
    ap.add_argument("--out", default="output/duplicates")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    new = hashes(collect(args.new)); ref = hashes(collect(args.ref))
    print(f"{len(new)} imagenes nuevas, {len(ref)} de referencia")
    pairs = []
    # nuevas vs referencia
    for a, (pa, da) in new.items():
        for b, (pb, db) in ref.items():
            d = min(pa - pb, da - db)
            if d <= args.max_dist:
                pairs.append((d, a, b, "new-vs-ref"))
    # nuevas entre si
    items = list(new.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            d = min(items[i][1][0] - items[j][1][0], items[i][1][1] - items[j][1][1])
            if d <= args.max_dist:
                pairs.append((d, items[i][0], items[j][0], "new-vs-new"))
    pairs.sort()
    print(f"{len(pairs)} pares con distancia <= {args.max_dist}")
    for d, a, b, k in pairs[:40]:
        print(f"  {d:2d} {k:11s} {os.path.basename(a)[:45]:45s} <-> {os.path.basename(b)[:45]}")
    with open(os.path.join(args.out, "pairs.json"), "w", encoding="utf-8") as f:
        json.dump([{"dist": d, "a": a, "b": b, "kind": k} for d, a, b, k in pairs], f, indent=1)
    # hoja de contacto: pares en filas
    T = 200
    show = pairs[:60]
    if show:
        from PIL import ImageDraw
        cols = 2; rows_per_sheet = 15
        for n in range(0, len(show), rows_per_sheet):
            chunk = show[n:n + rows_per_sheet]
            sheet = Image.new("RGB", (cols * T * 2 + 20, len(chunk) * (T + 18)), "white")
            dr = ImageDraw.Draw(sheet)
            for r, (d, a, b, k) in enumerate(chunk):
                for c, p in enumerate((a, b)):
                    im = Image.open(p).convert("RGB"); im.thumbnail((T * 2, T))
                    sheet.paste(im, (c * (T * 2 + 20), r * (T + 18)))
                dr.text((4, r * (T + 18) + T + 2), f"dist {d} [{k}]  {os.path.basename(a)[:40]}  |  {os.path.basename(b)[:40]}", fill=(0, 0, 0))
            sheet.save(os.path.join(args.out, f"_pairs_{n // rows_per_sheet + 1:02d}.jpg"), quality=85)
    print(f"JSON y hojas en {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
