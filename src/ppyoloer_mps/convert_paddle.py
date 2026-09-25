#!/usr/bin/env python
"""
Convierte un checkpoint de PaddleDetection (.pdparams) al formato de este repo (.pt).

No necesita tener Paddle instalado si se le pasa un .npz exportado previamente; si el
fichero es .pdparams, requiere paddlepaddle.

  python src/ppyoloer_mps/convert_paddle.py --src model_final.pdparams \
      --out models/ppyoloe_r/ppyoloe_r_s_banknotes.pt --num-classes 1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r, convert_paddle_state_dict, save_checkpoint


def read_paddle(path: str) -> dict:
    if path.endswith(".npz"):
        return dict(np.load(path))
    import paddle  # import perezoso: solo si hace falta

    return paddle.load(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="fichero .pdparams o .npz de Paddle")
    ap.add_argument("--out", required=True, help="fichero .pt de salida")
    ap.add_argument("--num-classes", type=int, default=1)
    ap.add_argument("--size", default="s", choices=["s", "m", "l", "x"])
    ap.add_argument("--classes", nargs="*", default=None, help="nombres de clase, en orden")
    args = ap.parse_args()

    state = convert_paddle_state_dict(read_paddle(args.src))
    model = build_ppyoloe_r(num_classes=args.num_classes, size=args.size)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise SystemExit(f"el checkpoint no encaja con el modelo.\n  faltan: {missing[:8]}\n  sobran: {unexpected[:8]}")
    meta = {
        "source": os.path.basename(args.src),
        "size": args.size,
        "num_classes": args.num_classes,
        "classes": args.classes or [f"class_{i}" for i in range(args.num_classes)],
    }
    save_checkpoint(args.out, model, meta)
    n = sum(p.numel() for p in model.parameters())
    print(f"convertido: {args.src} -> {args.out}  ({n / 1e6:.2f} M parametros, {len(state)} tensores)")


if __name__ == "__main__":
    main()
