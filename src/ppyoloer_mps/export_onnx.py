#!/usr/bin/env python
"""Exporta un checkpoint de PP-YOLOE-R a ONNX.

El grafo acepta imagenes RGB float32 en el rango 0..255, ya redimensionadas y rellenadas a
un multiplo de 32. La normalizacion (/255 y media/desviacion de ImageNet) va DENTRO del grafo,
para que el cliente no tenga que replicarla.

Devuelve dos tensores, ya decodificados:
  scores (B, C, L)  probabilidades por clase, con sigmoid aplicada
  boxes  (B, L, 5)  cajas rotadas (cx, cy, w, h, angulo_en_radianes) en pixeles de la ENTRADA

Es decir, el grafo hace el forward y la decodificacion; al cliente solo le quedan tres pasos:
umbral de score, NMS rotado y dividir por la escala del resize. Se exporta asi porque la
decodificacion de PP-YOLOE-R depende solo de las formas (es trazable), mientras que el filtro
por score y el bucle de NMS dependen de los valores y no se exportan limpiamente.

`onnx_example.py` implementa esos tres pasos y comprueba que coinciden con el modelo PyTorch.

Ejemplo::

    python src/ppyoloer_mps/export_onnx.py models/ppyoloe_r/ppyoloe_r_s_banknotes_torch.pt \
        --out models/ppyoloe_r/ppyoloe_r_s_banknotes_640.onnx --img-size 640
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import torch
import torch.nn as nn

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r, load_checkpoint  # noqa: E402
from ppyoloer_mps.ppyoloe_obb.data import MEAN, STD  # noqa: E402

OUTPUT_NAMES = ("scores", "boxes")
INPUT_NAME = "images_rgb_0_255"


class ExportWrapper(nn.Module):
    """Mete la normalizacion en el grafo y expone la salida ya decodificada."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).reshape(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(STD).reshape(1, 3, 1, 1) * 255.0)

    def forward(self, images: torch.Tensor):
        x = (images - self.mean) / self.std
        scores, boxes = self.model(x)
        return scores, boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--opset", type=int, default=16)
    ap.add_argument("--dynamic-batch", action="store_true", help="eje de batch dinamico")
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    num_classes = meta.get("num_classes", 1)
    size = meta.get("size", "s")
    model = build_ppyoloe_r(num_classes=num_classes, size=size)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    wrapper = ExportWrapper(model).eval()
    example = torch.zeros(1, 3, args.img_size, args.img_size, dtype=torch.float32)
    os.makedirs(osp.dirname(osp.abspath(args.out)) or ".", exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {INPUT_NAME: {0: "batch"}, **{n: {0: "batch"} for n in OUTPUT_NAMES}}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            args.out,
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

    metadata = {
        "model": f"PP-YOLOE-R-{size}",
        "checkpoint": osp.basename(args.checkpoint),
        "classes": meta.get("classes", [f"class_{i}" for i in range(num_classes)]),
        "input": {
            "name": INPUT_NAME,
            "dtype": "float32",
            "shape": [1, 3, args.img_size, args.img_size],
            "range": [0, 255],
            "layout": "NCHW",
            "color": "RGB",
            "resize": "keep_ratio_to_long_side",
            "resize_interpolation": "INTER_AREA al reducir, INTER_LINEAR al ampliar",
            "padding": {"value": 0, "anchor": "top_left", "to_multiple_of": 32},
            "normalization_inside_graph": True,
            "mean_0_255": [round(float(v) * 255, 3) for v in MEAN],
            "std_0_255": [round(float(v) * 255, 3) for v in STD],
        },
        "outputs": {
            "scores": "(B, C, L) probabilidad por clase, sigmoid ya aplicada",
            "boxes": "(B, L, 5) cajas rotadas (cx, cy, w, h, angulo_rad) en pixeles de la entrada",
        },
        "client_postprocess": [
            "umbral de score (0.05 para metricas, 0.5 para produccion)",
            "NMS rotado por clase (IoU 0.1 en la convencion de mmrotate; 0.5 va mejor si los objetos se solapan)",
            "dividir las coordenadas por la escala del resize para volver a la imagen original",
            "convertir (cx, cy, w, h, angulo) a poligono de 4 esquinas",
        ],
        "angle_convention": "radianes en [0, pi/2), sentido horario con el eje y hacia abajo",
    }
    metadata_path = osp.splitext(args.out)[0] + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"{args.out}: exportado ({size_mb:.1f} MB, opset {args.opset})")
    print(f"{metadata_path}: metadatos de pre/post-proceso")


if __name__ == "__main__":
    main()
