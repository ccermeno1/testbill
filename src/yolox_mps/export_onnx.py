"""Export a YOLOX-OBB checkpoint to ONNX, with the box decoding inside the graph.

Input ``images``: float32 ``(1, 3, S, S)``, BGR, 0..255, no normalisation. The photo is
resized keeping its aspect ratio so the long side is ``S`` and pasted at the top-left of
an ``S x S`` canvas filled with 114 (``letterbox`` in ``onnx_example.py``).

Outputs, one row per prior (all levels, strides 8 / 16 / 32):

* ``boxes``  ``(1, N, 5)``: cx, cy, w, h in input pixels, angle in radians clockwise
* ``scores`` ``(1, N, C)``: sigmoid(obj) * sigmoid(cls), in 0..1

The client keeps the best class per row, applies the score threshold, runs rotated NMS
and divides cx, cy, w, h by the letterbox scale. ``onnx_example.py`` is the reference
implementation; the ``.metadata.json`` written next to the model describes the contract.

Example::

    python src/yolox_mps/export_onnx.py v1_nano100/best_epoch_100.pth --out model_640.onnx --img-size 640
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from yolox_obb import MODELS, OBBDetector, build_model, checkpoint_arch, load_state_dict_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "models" / "yolox_obb" / "experiments"

INPUT_NAME = "images"
OUTPUT_NAMES = ("boxes", "scores")


class DeployModel(nn.Module):
    """Network + decoding: images -> (boxes (B, N, 5), scores (B, N, C))."""

    def __init__(self, model: OBBDetector):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        cls_scores, reg_preds, obj_preds = self.model(images)
        flat_cls, flat_reg, flat_obj, priors = self.model._flatten(cls_scores, reg_preds, obj_preds)
        boxes = self.model.decode(flat_reg, priors)
        scores = flat_cls.sigmoid() * flat_obj.sigmoid()[..., None]
        return boxes, scores


def export(model: OBBDetector, path: str, img_size: int, opset: int = 18):
    torch.onnx.export(
        DeployModel(model).eval(),
        torch.zeros(1, 3, img_size, img_size, dtype=torch.float32),
        path,
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        opset_version=opset,
        do_constant_folding=True,
        external_data=False,  # one self-contained .onnx file instead of .onnx + .onnx.data
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True, help="relative paths go next to the checkpoint")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--class-names", default="euro_banknote", help="comma separated")
    parser.add_argument("--model", choices=list(MODELS), help="default: the architecture stored in the checkpoint")
    parser.add_argument("--score-thr", type=float, default=0.5, help="recommended threshold written to the metadata")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="recommended NMS IoU written to the metadata")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    # the torch exporter prints emoji; a Windows cp1252 console would raise UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        checkpoint = EXPERIMENTS / args.checkpoint
    out = Path(args.out)
    if not out.is_absolute():
        out = checkpoint.parent / out

    class_names = args.class_names.split(",")
    model = build_model(args.model or checkpoint_arch(str(checkpoint)), len(class_names))
    model.load_state_dict(load_state_dict_file(str(checkpoint)))
    model.eval()
    os.makedirs(out.parent, exist_ok=True)
    export(model, str(out), args.img_size, args.opset)

    s = args.img_size
    num_priors = sum((s // st) * (s // st) for st in model.strides)
    metadata = {
        "model": model.arch,
        "checkpoint": checkpoint.name,
        "class_names": class_names,
        "input": {
            "name": INPUT_NAME,
            "shape": [1, 3, s, s],
            "dtype": "float32",
            "layout": "NCHW",
            "color": "BGR",
            "range": [0, 255],
            "normalisation": "none",
        },
        "preprocess": [
            "decode the photo applying its EXIF orientation",
            f"scale = {s} / max(height, width)",
            "resize to (round(width * scale), round(height * scale)), INTER_AREA when shrinking, "
            "INTER_LINEAR when enlarging",
            f"paste at the top-left of a {s}x{s} canvas filled with 114",
            "HWC uint8 BGR -> NCHW float32, values unchanged (0..255)",
        ],
        "outputs": {
            "boxes": {"shape": [1, num_priors, 5],
                      "meaning": "cx, cy, w, h in input pixels; angle in radians, clockwise in image coordinates"},
            "scores": {"shape": [1, num_priors, len(class_names)],
                       "meaning": "sigmoid(objectness) * sigmoid(class), 0..1"},
        },
        "postprocess": [
            "per row: label = argmax(scores), score = max(scores)",
            "keep rows with score > score_thr",
            "rotated NMS per class with IoU > nms_iou (sort by score, greedy)",
            "cx, cy, w, h /= scale (angle unchanged)",
            "corners: cx, cy +- (w/2)(cos a, sin a) +- (h/2)(-sin a, cos a)",
        ],
        "recommended": {"score_thr": args.score_thr, "nms_iou": args.nms_iou, "max_detections": 100},
    }
    metadata_path = out.with_suffix(".metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"{out}: exported")
    print(f"{metadata_path}: contract written")


if __name__ == "__main__":
    main()
