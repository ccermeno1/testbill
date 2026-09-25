#!/usr/bin/env python
"""Export a PP-YOLOE-R checkpoint to ONNX.

The graph takes float32 RGB images in the 0..255 range, already resized and padded to a
multiple of 32. Normalisation (divide by 255, then ImageNet mean/std) happens INSIDE the
graph so the client does not have to reproduce it.

It returns two tensors, already decoded:
  scores (B, C, L)  per-class probabilities, sigmoid applied
  boxes  (B, L, 5)  rotated boxes (cx, cy, w, h, angle_in_radians) in INPUT pixels

So the graph covers the forward pass and the box decode, leaving the client three steps:
score threshold, rotated NMS, and dividing by the resize scale. It is exported this way
because PP-YOLOE-R's decode depends only on shapes and traces cleanly, while the score
filter and the NMS loop depend on values and do not.

onnx_example.py implements those three steps and checks them against the PyTorch model.

Example::

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
    """Folds normalisation into the graph and exposes the already-decoded output."""

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
    ap.add_argument("--dynamic-batch", action="store_true", help="make the batch axis dynamic")
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
            "resize_interpolation": "INTER_AREA when downscaling, INTER_LINEAR when upscaling",
            "padding": {"value": 0, "anchor": "top_left", "to_multiple_of": 32},
            "normalization_inside_graph": True,
            "mean_0_255": [round(float(v) * 255, 3) for v in MEAN],
            "std_0_255": [round(float(v) * 255, 3) for v in STD],
        },
        "outputs": {
            "scores": "(B, C, L) per-class probability, sigmoid already applied",
            "boxes": "(B, L, 5) rotated boxes (cx, cy, w, h, angle_rad) in input pixels",
        },
        "client_postprocess": [
            "score threshold (0.05 for metrics, 0.5 for production)",
            "per-class rotated NMS (IoU 0.1 is the mmrotate convention; 0.5 works better when objects overlap)",
            "divide the coordinates by the resize scale to get back to the original image",
            "turn (cx, cy, w, h, angle) into a 4-corner polygon",
        ],
        "angle_convention": "radians in [0, pi/2), clockwise with y pointing down",
    }
    metadata_path = osp.splitext(args.out)[0] + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"{args.out}: exported ({size_mb:.1f} MB, opset {args.opset})")
    print(f"{metadata_path}: pre/post-processing contract")


if __name__ == "__main__":
    main()
