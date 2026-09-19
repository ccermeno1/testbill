#!/usr/bin/env python3
"""Run the exported FCOS ONNX on the bundled plane image and assert NMS.

Works from the repo (``python -m export demo``) and from a copied
``onnx_export/`` bundle (``python demo.py``). No oriented-det.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BUNDLE = (_HERE / "runtime.py").is_file() and (_HERE / "nms.py").is_file()
if _BUNDLE:
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
else:
    _REPO_ROOT = _HERE.parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image  # noqa: E402

if _BUNDLE:
    from postprocess import (  # noqa: E402
        any_pair_iou_at_least,
        finalize_detections_numpy,
        max_pairwise_rbox_iou,
        meta_to_finalize_kwargs,
        ort_run_pre_nms,
        score_filter_numpy,
    )
    from preprocess import canvas_size_from_meta, preprocess_path, scale_obb_to_original  # noqa: E402
    from runtime import detections_to_records, draw_detections, load_export_meta  # noqa: E402

    _DEMO_IMAGE = _HERE / "demo" / "planes_pleiades_neo.jpg"
    _DEFAULT_ONNX = _HERE / "model.onnx"
    _DEFAULT_OUT = _HERE / "demo"
else:
    from export.postprocess import (  # noqa: E402
        any_pair_iou_at_least,
        finalize_detections_numpy,
        max_pairwise_rbox_iou,
        meta_to_finalize_kwargs,
        ort_run_pre_nms,
        score_filter_numpy,
    )
    from export.preprocess import canvas_size_from_meta, preprocess_path, scale_obb_to_original  # noqa: E402
    from export.runtime import detections_to_records, draw_detections, load_export_meta  # noqa: E402

    _DEMO_IMAGE = _HERE.parent / "demo" / "planes_pleiades_neo.jpg"
    _DEFAULT_ONNX = _HERE.parents[1] / "onnx_export" / "model.onnx"
    _DEFAULT_OUT = _HERE.parents[1] / "onnx_export" / "demo"


def assert_final_nms(
    pre_boxes: object,
    post_boxes: object,
    *,
    iou_threshold: float,
    n_pre: int,
    n_post: int,
    nms_backend: str = "python",
) -> None:
    """Fail if NMS did not suppress overlaps or left high-IoU pairs."""
    if n_pre < 2:
        raise SystemExit(f"NMS assert failed: only {n_pre} score-filtered pre-NMS box(es).")
    if not any_pair_iou_at_least(pre_boxes, iou_threshold, backend=nms_backend):
        raise SystemExit(
            f"NMS assert failed: no pre-NMS pair with IoU >= {iou_threshold:g} "
            f"(nothing for NMS to suppress)."
        )
    if n_post >= n_pre:
        raise SystemExit(
            f"NMS assert failed: post-NMS count {n_post} is not < pre-NMS {n_pre}."
        )
    max_iou = max_pairwise_rbox_iou(post_boxes, backend=nms_backend)
    # Small tolerance for polygon IoU vs the NMS implementation.
    if max_iou >= float(iou_threshold) + 1e-3:
        raise SystemExit(
            f"NMS assert failed: remaining pair IoU {max_iou:.4f} >= threshold {iou_threshold:g}."
        )
    print(
        f"NMS OK: pre={n_pre} → post={n_post}  "
        f"max_remaining_iou={max_iou:.4f} < {iou_threshold:g}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="ONNX demo on the bundled plane image; asserts NMS.")
    p.add_argument("--onnx", type=Path, default=_DEFAULT_ONNX)
    p.add_argument("--image", type=Path, default=_DEMO_IMAGE)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--score-threshold", type=float, default=None)
    p.add_argument(
        "--nms-backend",
        choices=("python", "shapely", "auto"),
        default=None,
        help="CPU NMS IoU backend (default: meta / python). shapely needs: pip install shapely",
    )
    args = p.parse_args()

    if not args.onnx.is_file():
        raise SystemExit(f"Missing ONNX: {args.onnx} (run: make export-onnx)")
    if not args.image.is_file():
        raise SystemExit(f"Missing demo image: {args.image}")

    meta = load_export_meta(args.onnx)
    fk = meta_to_finalize_kwargs(meta)
    if args.score_threshold is not None:
        fk["score_threshold"] = float(args.score_threshold)
    if args.nms_backend is not None:
        fk["nms_backend"] = args.nms_backend
    iou_thr = float(fk["final_nms_iou_threshold"])
    score_thr = float(fk["score_threshold"])
    nms_backend = str(fk.get("nms_backend") or "python")

    blob, orig_h, orig_w = preprocess_path(args.image, meta)
    names = list(meta.get("output_names") or [])
    raw_boxes, raw_scores, raw_labels, raw_count = ort_run_pre_nms(blob, str(args.onnx), names)
    pre_boxes, _, _ = score_filter_numpy(
        raw_boxes, raw_scores, raw_labels, raw_count, score_thr
    )
    padded, n_post = finalize_detections_numpy(
        raw_boxes, raw_scores, raw_labels, raw_count, **fk
    )
    post_boxes = padded[:n_post, :5]

    print(f"image  {args.image}  ({orig_w}x{orig_h})")
    print(f"onnx   {args.onnx}")
    print(
        f"thr    score={score_thr:g}  nms_iou={iou_thr:g}  "
        f"class_agnostic={fk['nms_class_agnostic']}  nms_backend={nms_backend}"
    )
    print(f"pre-NMS live={raw_count}  score-filtered={len(pre_boxes)}  post-NMS={n_post}")

    assert_final_nms(
        pre_boxes,
        post_boxes,
        iou_threshold=iou_thr,
        n_pre=int(pre_boxes.shape[0]),
        n_post=int(n_post),
        nms_backend=nms_backend,
    )

    canvas_h, canvas_w = canvas_size_from_meta(meta)
    scaled = padded.copy()
    if n_post:
        scaled[:n_post, :5] = scale_obb_to_original(
            padded[:n_post, :5], orig_w, orig_h, canvas_w, canvas_h
        )
    dets = detections_to_records(scaled, n_post, list(meta.get("class_names") or []))
    for d in dets:
        print(
            f"  {d['class_name']:28s}  score={d['score']:.3f}  "
            f"cx={d['cx']:.1f} cy={d['cy']:.1f}  {d['w']:.1f}x{d['h']:.1f}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    overlay = draw_detections(Image.open(args.image).convert("RGB"), dets)
    overlay_path = args.output / f"{args.image.stem}_overlay.jpg"
    overlay.save(overlay_path, quality=92)
    rec = {
        "image": args.image.name,
        "pre_nms_live": int(raw_count),
        "score_filtered": int(pre_boxes.shape[0]),
        "post_nms": int(n_post),
        "score_threshold": score_thr,
        "nms_iou_threshold": iou_thr,
        "nms_backend": nms_backend,
        "detections": dets,
        "overlay": overlay_path.name,
    }
    json_path = args.output / f"{args.image.stem}.json"
    json_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"Wrote {overlay_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
