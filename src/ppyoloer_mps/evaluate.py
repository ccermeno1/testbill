#!/usr/bin/env python
"""
Evalua un checkpoint de este repo sobre uno o varios splits (JSON COCO con poligonos).

  python src/ppyoloer_mps/evaluate.py -w models/ppyoloe_r/ppyoloe_r_s_banknotes.pt \
      --data data/banknotes_obb --split valid test --device auto
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r, load_checkpoint
from ppyoloer_mps.ppyoloe_obb.data import ObbDataset, collate
from ppyoloer_mps.ppyoloe_obb.engine import evaluate_model, resolve_device
from ppyoloer_mps.ppyoloe_obb.evaluation import HEADER


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("--data", required=True, help="carpeta del dataset (images/ + annotations/)")
    ap.add_argument("--split", nargs="+", default=["valid", "test"])
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.5, help="umbral para P/R/F1")
    ap.add_argument("--score-threshold", type=float, default=0.01, help="minimo para el AP")
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="carpeta donde guardar los JSON de metricas")
    args = ap.parse_args()

    device = resolve_device(args.device)
    payload = torch.load(args.weights, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    model = build_ppyoloe_r(num_classes=meta.get("num_classes", 1), size=meta.get("size", "s"))
    load_checkpoint(model, args.weights)
    model.to(device).eval()
    print(f"dispositivo: {device} | pesos: {args.weights} | clases: {meta.get('classes')}")
    print(HEADER)

    for split in args.split:
        ds = ObbDataset(
            annotation_file=os.path.join(args.data, "annotations", f"{split}.json"),
            image_dir=os.path.join(args.data, "images"),
            img_size=args.img_size,
            train=False,
        )
        loader = DataLoader(
            ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate
        )
        res = evaluate_model(
            model, loader, device, name=split, conf=args.conf,
            score_threshold=args.score_threshold, nms_threshold=args.nms_iou, print_header=False,
        )
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            res_out = dict(res, weights=args.weights, split=split, device=str(device))
            with open(os.path.join(args.out, f"{split}.json"), "w", encoding="utf-8") as f:
                json.dump(res_out, f, indent=1)


if __name__ == "__main__":
    main()
