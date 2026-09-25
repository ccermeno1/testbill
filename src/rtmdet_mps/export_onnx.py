"""Export RTMDet-R inference weights to ONNX.

The exported graph accepts BGR images as float32 tensors in the 0..255 range,
already resized and padded to a square. Model normalisation is inside the graph.
It returns nine raw maps, in this order: cls/p3, reg/p3, angle/p3, cls/p4,
reg/p4, angle/p4, cls/p5, reg/p5 and angle/p5. The output names below are kept
explicit so mobile clients do not depend on PyTorch tuple naming.

Example::

    python export_onnx.py ../../models/rtmdet/experiments/baseline_C_prelight/rtmdet_r_tiny_banknotes_C_ep72_prelight.pth \
        --out ../../models/rtmdet/experiments/baseline_C_prelight/rtmdet_r_tiny_C_ep72_prelight_800.onnx --img-size 800
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import torch
import torch.nn as nn

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from rtmdet_obb import RTMDetR, load_state_dict_file  # noqa: E402


OUTPUT_NAMES = (
    "cls_p3", "reg_p3", "angle_p3",
    "cls_p4", "reg_p4", "angle_p4",
    "cls_p5", "reg_p5", "angle_p5",
)


class RawOutputs(nn.Module):
    def __init__(self, model: RTMDetR):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        cls_scores, bbox_preds, angle_preds = self.model(images)
        return tuple(x for level in zip(cls_scores, bbox_preds, angle_preds) for x in level)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--img-size", type=int, default=800)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    model = RTMDetR(num_classes=1, size="tiny")
    model.load_state_dict(load_state_dict_file(args.checkpoint))
    model.eval()
    wrapper = RawOutputs(model)
    wrapper.eval()
    example = torch.zeros(1, 3, args.img_size, args.img_size, dtype=torch.float32)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)

    torch.onnx.export(
        wrapper,
        example,
        args.out,
        input_names=["images_bgr_0_255"],
        output_names=list(OUTPUT_NAMES),
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={"images_bgr_0_255": {0: "batch"}, **{
            name: {0: "batch"} for name in OUTPUT_NAMES
        }},
    )

    metadata = {
        "model": "RTMDet-R tiny",
        "checkpoint": osp.basename(args.checkpoint),
        "input": {
            "name": "images_bgr_0_255",
            "dtype": "float32",
            "shape": [1, 3, args.img_size, args.img_size],
            "range": [0, 255],
            "layout": "NCHW",
            "color": "BGR",
            "resize": "keep_ratio_to_long_side",
            "padding": {"square_value": 114, "anchor": "top_left"},
            "mean": [103.53, 116.28, 123.675],
            "std": [57.375, 57.12, 58.395],
        },
        "outputs": list(OUTPUT_NAMES),
        "output_semantics": {
            "cls": "raw logits, sigmoid in client",
            "reg": "left/top/right/bottom distances in feature pixels",
            "angle": "raw angle prediction",
            "strides": [8, 16, 32],
        },
        "client_postprocess": [
            "decode distance2obb with le90 convention",
            "score threshold",
            "rotated NMS (IoU)",
            "divide box coordinates by resize scale",
            "convert OBB to four-corner polygon",
        ],
    }
    metadata_path = osp.splitext(args.out)[0] + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"{args.out}: exported")
    print(f"{metadata_path}: contract written")


if __name__ == "__main__":
    main()
