#!/usr/bin/env python
"""
Genera el manifiesto de seleccion del dataset externo "Euro Banknote Detection": para cada foto
original dice si entro en train (y con que nombre) o por que se excluyo (duplicado con nuestros
datasets -> de que foto y distancia; primer plano; sin billete).

  python scripts/extra_manifest.py
Salida: data_manifests/eurobanknotes_extra_selection.csv / .json / .md
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
        where = ours_split.get(b, "billetesprueba" if "billetesprueba" in p["b"] else "nuevo")
        if p["kind"] == "new-vs-new":
            b2 = os.path.normpath(os.path.abspath(p["b"]))
            for x, y in ((a, b2), (b2, a)):
                if x not in best or p["dist"] < best[x][0]:
                    best[x] = (p["dist"], os.path.basename(y), "nuevo (mismo dataset)")
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
            row = {"original_file": base, "roboflow_split": split, "n_billetes": n_notes, "n_manos": n_hand,
                   "area_max_billete": round(max_area, 3), "estado": "", "motivo": "", "duplicado_de": "", "distancia": "",
                   "dataset_del_duplicado": "", "nombre_en_train": ""}
            if ap in excluded:
                d, b, where = best.get(ap, ("", "", ""))
                row.update(estado="excluida", motivo="duplicado", duplicado_de=b, distancia=d, dataset_del_duplicado=where)
            elif n_notes == 0:
                row.update(estado="excluida", motivo="sin billetes (solo mano)")
            elif n_notes == 1 and max_area > MAX_SINGLE_AREA:
                row.update(estado="excluida", motivo=f"primer plano (1 billete, >{int(MAX_SINGLE_AREA*100)}% de la imagen)")
            elif base in included:
                new_name = "ebd_" + included[base]["file_name"]
                row.update(estado="incluida en train", nombre_en_train=new_name if new_name in merged_names else "(no encontrada en train_plus_extra.json)")
            else:
                row.update(estado="excluida", motivo="?")
            rows.append(row)

    fields = list(rows[0].keys())
    with open(f"{OUT}/eurobanknotes_extra_selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(f"{OUT}/eurobanknotes_extra_selection.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    inc = [r for r in rows if r["estado"].startswith("incluida")]
    exc = [r for r in rows if r["estado"] == "excluida"]
    by_reason = {}
    for r in exc:
        by_reason.setdefault(r["motivo"], []).append(r)
    dup_where = {}
    for r in by_reason.get("duplicado", []):
        dup_where[r["dataset_del_duplicado"]] = dup_where.get(r["dataset_del_duplicado"], 0) + 1
    md = ["# Selección del dataset externo *Euro Banknote Detection* para train", "",
          f"Origen: `{SRC}` ({len(rows)} fotos, 12 clases de billete + `hand`). Criterios:", "",
          "1. **Duplicados**: hash perceptual (pHash/dHash, distancia de Hamming ≤ 8) contra `dataset/banknotes_obb` (train/valid/test)",
          "   y `dataset/billetesprueba`, y entre las propias fotos nuevas (se conserva una).",
          f"2. **Primeros planos**: un solo billete que ocupa > {int(MAX_SINGLE_AREA*100)} % de la imagen (ya tenemos muchos).",
          "3. **Sin billetes**: fotos con solo `hand`.",
          "4. Las 12 clases de billete se fusionan en `euro_banknote`; las cajas `hand` se descartan.", "",
          f"**Incluidas en train: {len(inc)}** (como `dataset/banknotes_obb/images/ebd_<nombre>.jpg`, en `annotations/train_plus_extra.json`).",
          f"**Excluidas: {len(exc)}** — " + ", ".join(f"{k}: {len(v)}" for k, v in by_reason.items()) + ".",
          "Duplicados por dataset de referencia: " + ", ".join(f"{k}: {v}" for k, v in sorted(dup_where.items())) + ".", "",
          "Detalle completo (todas las fotos, con distancia y foto emparejada) en `eurobanknotes_extra_selection.csv`.", "",
          "## Incluidas", "", "| foto original | split Roboflow | billetes | nombre en train |", "|---|---|---|---|"]
    md += [f"| {r['original_file']} | {r['roboflow_split']} | {r['n_billetes']} | {r['nombre_en_train']} |" for r in inc]
    md += ["", "## Excluidas", "", "| foto original | motivo | duplicado de | dist. | dataset |", "|---|---|---|---|---|"]
    md += [f"| {r['original_file']} | {r['motivo']} | {r['duplicado_de']} | {r['distancia']} | {r['dataset_del_duplicado']} |" for r in exc]
    open(f"{OUT}/eurobanknotes_extra_selection.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"{len(rows)} fotos: {len(inc)} incluidas, {len(exc)} excluidas ({', '.join(f'{k}: {len(v)}' for k, v in by_reason.items())})")
    print(f"-> {OUT}/eurobanknotes_extra_selection.{{csv,json,md}}")


if __name__ == "__main__":
    main()
