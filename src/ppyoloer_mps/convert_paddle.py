#!/usr/bin/env python
"""
Convert a PaddleDetection checkpoint (.pdparams) into this repo's format (.pt).

Paddle only needs to be installed when the input is a .pdparams file. A .npz exported
elsewhere works without it.

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
    import paddle  # imported lazily, only when actually needed

    return paddle.load(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Paddle .pdparams or .npz file")
    ap.add_argument("--out", required=True, help="output .pt file")
    ap.add_argument("--num-classes", type=int, default=1)
    ap.add_argument("--size", default="s", choices=["s", "m", "l", "x"])
    ap.add_argument("--classes", nargs="*", default=None, help="class names, in order")
    args = ap.parse_args()

    state = convert_paddle_state_dict(read_paddle(args.src))
    model = build_ppyoloe_r(num_classes=args.num_classes, size=args.size)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # angle_proj_conv is a fixed DFL projection the model initialises itself; some official
    # checkpoints, the DOTA one among them, do not store it.
    ignorable = ("num_batches_tracked", "angle_proj_conv.weight")
    missing = [k for k in missing if not k.endswith(ignorable)]
    if missing or unexpected:
        raise SystemExit(f"checkpoint does not match the model.\n  missing: {missing[:8]}\n  unexpected: {unexpected[:8]}")
    meta = {
        "source": os.path.basename(args.src),
        "size": args.size,
        "num_classes": args.num_classes,
        "classes": args.classes or [f"class_{i}" for i in range(args.num_classes)],
    }
    save_checkpoint(args.out, model, meta)
    n = sum(p.numel() for p in model.parameters())
    print(f"converted: {args.src} -> {args.out}  ({n / 1e6:.2f} M params, {len(state)} tensors)")


if __name__ == "__main__":
    main()
