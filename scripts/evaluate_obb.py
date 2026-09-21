#!/usr/bin/env python
"""
Evaluacion completa de OBB (una clase) sobre un split del dataset:
  mAP@0.5, mAP@0.75, mAP@[0.50:0.95] (AP COCO-style, 101 puntos de recall, IoU rotada exacta)
  precision / recall / F1 con score >= --conf a IoU 0.5 y 0.75, y el mejor F1 posible.

  python scripts/evaluate_obb.py -c configs/ppyoloe_r_crn_s_banknotes.yml -w output/model_final.pdparams --split valid test
  python scripts/evaluate_obb.py --dataset_dir dataset/billetesprueba --split all --output_dir output/metrics_billetesprueba

Guarda un JSON por split en --output_dir (por defecto output/metrics).
"""
import argparse
import json
import os
import sys

import numpy as np
from shapely.geometry import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "PaddleDetection"))

import paddle  # noqa: E402
from ppdet.core.workspace import create, load_config, merge_config  # noqa: E402
from ppdet.engine import Trainer  # noqa: E402
from ppdet.utils.check import check_config, check_gpu, check_version  # noqa: E402

IOU_THRS = np.round(np.arange(0.50, 0.96, 0.05), 2)


def load_gt(dataset_dir, split):
    with open(os.path.join(dataset_dir, "annotations", f"{split}.json"), encoding="utf-8") as f:
        coco = json.load(f)
    gt = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        p = a["segmentation"][0]
        gt[a["image_id"]].append(Polygon(list(zip(p[0::2], p[1::2]))))
    paths = {im["id"]: os.path.join(dataset_dir, "images", im["file_name"]) for im in coco["images"]}
    return gt, paths


def run_inference(cfg, weights, images, score_thr):
    merge_config({"weights": weights})
    cfg["PPYOLOERHead"]["nms"]["score_threshold"] = score_thr
    trainer = Trainer(cfg, mode="test")
    trainer.load_weights(cfg.weights)
    trainer.dataset.set_images(images)
    loader = create("TestReader")(trainer.dataset, 0)
    imid2path = trainer.dataset.get_imid2path()
    trainer.model.eval()
    preds = {}  # path -> list of (score, Polygon)
    with paddle.no_grad():
        for data in loader:
            outs = trainer.model(data)
            bboxes, bbox_num = outs["bbox"].numpy(), outs["bbox_num"].numpy()
            start = 0
            for i, im_id in enumerate(data["im_id"].numpy()):
                dets = bboxes[start:start + int(bbox_num[i])]
                start += int(bbox_num[i])
                rows = []
                for d in dets:
                    if int(d[0]) < 0:
                        continue
                    poly = Polygon(list(zip(d[2:10:2], d[3:10:2])))
                    if poly.is_valid and poly.area > 0:
                        rows.append((float(d[1]), poly))
                preds[imid2path[int(im_id)]] = rows
    return preds


def iou_matrix(pred_polys, gt_polys):
    m = np.zeros((len(pred_polys), len(gt_polys)), dtype=np.float64)
    for i, p in enumerate(pred_polys):
        for j, g in enumerate(gt_polys):
            inter = p.intersection(g).area
            u = p.area + g.area - inter
            m[i, j] = inter / u if u > 0 else 0.0
    return m


def match(scores, ious, thr):
    """Greedy COCO-style: detecciones por score desc., cada una al GT libre con mayor IoU >= thr."""
    order = np.argsort(-scores)
    matched_gt = set()
    tp = np.zeros(len(scores), dtype=bool)
    for i in order:
        if ious.shape[1] == 0:
            continue
        cand = ious[i].copy()
        cand[list(matched_gt)] = -1
        j = int(np.argmax(cand))
        if cand[j] >= thr:
            tp[i] = True
            matched_gt.add(j)
    return tp


def average_precision(tp, scores, n_gt):
    """AP COCO-style (101 puntos de recall, precision monotona)."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores)
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1)
    for k in range(len(precision) - 2, -1, -1):
        precision[k] = max(precision[k], precision[k + 1])
    rec_thrs = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rec_thrs, side="left")
    prec_at = np.array([precision[i] if i < len(precision) else 0.0 for i in idx])
    return float(prec_at.mean())


def prf(tp, fp, n_gt):
    p = tp / max(tp + fp, 1)
    r = tp / max(n_gt, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    return p, r, f


def evaluate_split(preds, gt, paths, conf):
    all_scores, all_tp = [], {thr: [] for thr in IOU_THRS}
    n_gt = sum(len(v) for v in gt.values())
    for im_id, gpolys in gt.items():
        rows = preds.get(paths[im_id], [])
        scores = np.array([s for s, _ in rows], dtype=np.float64)
        ious = iou_matrix([p for _, p in rows], gpolys)
        all_scores.append(scores)
        for thr in IOU_THRS:
            all_tp[thr].append(match(scores, ious, thr) if len(rows) else np.zeros(0, dtype=bool))
    scores = np.concatenate(all_scores)
    tps = {thr: np.concatenate(all_tp[thr]) for thr in IOU_THRS}

    aps = {f"AP{int(round(thr * 100))}": average_precision(tps[thr], scores, n_gt) for thr in IOU_THRS}
    res = {
        "n_images": len(gt), "n_gt": n_gt, "n_pred_total": int(len(scores)),
        "mAP50": aps["AP50"], "mAP75": aps["AP75"], "mAP50-95": float(np.mean(list(aps.values()))),
        "AP_per_iou": aps,
    }
    # P/R/F1 a un umbral de confianza fijo
    keep = scores >= conf
    for thr in (0.5, 0.75):
        tp = int(tps[thr][keep].sum())
        fp = int(keep.sum()) - tp
        p, r, f = prf(tp, fp, n_gt)
        res[f"conf{conf}_iou{thr}"] = {"TP": tp, "FP": fp, "FN": n_gt - tp, "precision": p, "recall": r, "f1": f}
    # mejor F1 alcanzable (IoU 0.5) barriendo el umbral de confianza
    best = (0, 0)
    for c in np.unique(scores):
        k = scores >= c
        tp = int(tps[0.5][k].sum())
        _, _, f = prf(tp, int(k.sum()) - tp, n_gt)
        if f > best[0]:
            best = (f, float(c))
    res["best_f1_iou0.5"] = {"f1": best[0], "conf": best[1]}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="configs/ppyoloe_r_crn_s_banknotes.yml")
    ap.add_argument("-w", "--weights", default="output/model_final.pdparams")
    ap.add_argument("--split", nargs="+", default=["valid", "test"], help="nombre(s) de annotations/<split>.json")
    ap.add_argument("--dataset_dir", default=None, help="carpeta del dataset (por defecto la de TestDataset en la config)")
    ap.add_argument("--conf", type=float, default=0.5, help="umbral de confianza para P/R/F1")
    ap.add_argument("--score_thr", type=float, default=0.01, help="score minimo en inferencia (para AP)")
    ap.add_argument("--output_dir", default="output/metrics")
    ap.add_argument("--img_size", type=int, default=None, help="lado del resize de inferencia (por defecto el de la config, 640)")
    ap.add_argument("--pre_size", type=int, default=None, help="reducir primero la imagen a este lado mayor (simula la resolucion de origen del train) y luego aplicar el resize normal")
    ap.add_argument("--interp", type=int, default=None, help="interpolacion del Resize de inferencia (cv2: 0 nearest, 1 linear, 2 cubic, 3 area)")
    ap.add_argument("--stretch", action="store_true", help="resize sin mantener proporcion (imita el stretch de Roboflow del train)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    merge_config({"use_gpu": not args.cpu})
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    paddle.set_device("gpu" if cfg.use_gpu else "cpu")
    dataset_dir = args.dataset_dir or cfg["TestDataset"].dataset_dir
    if args.pre_size:
        st = cfg["TestReader"]["sample_transforms"]
        st.insert(1, {"Resize": {"target_size": [args.pre_size, args.pre_size], "keep_ratio": True, "interp": 3}})
    for t in cfg["TestReader"]["sample_transforms"][(2 if args.pre_size else 0):]:
        if "Resize" in t:
            if args.img_size:
                t["Resize"]["target_size"] = [args.img_size, args.img_size]
            if args.stretch:
                t["Resize"]["keep_ratio"] = False
            if args.interp is not None:
                t["Resize"]["interp"] = args.interp
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {}
    for split in args.split:
        gt, paths = load_gt(dataset_dir, split)
        preds = run_inference(cfg, args.weights, [paths[i] for i in sorted(paths)], args.score_thr)
        res = evaluate_split(preds, gt, paths, args.conf)
        res["weights"] = args.weights
        res["dataset_dir"] = dataset_dir
        res["img_size"] = args.img_size or cfg["TestReader"]["sample_transforms"][1]["Resize"]["target_size"]
        res["nms_threshold"] = cfg["PPYOLOERHead"]["nms"]["nms_threshold"]
        summary[split] = res
        with open(os.path.join(args.output_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    print(f"\npesos: {args.weights}   NMS: {summary[args.split[0]]['nms_threshold']}   IoU rotada exacta, AP 101 puntos")
    print(f"{'split':6s} {'imgs':>5s} {'GT':>4s} | {'mAP50':>6s} {'mAP75':>6s} {'mAP50-95':>8s} | "
          f"{'P@.5':>5s} {'R@.5':>5s} {'F1@.5':>5s} | {'P@.75':>5s} {'R@.75':>5s} {'F1@.75':>6s} | best F1 (conf)")
    for split, r in summary.items():
        a, b = r[f"conf{args.conf}_iou0.5"], r[f"conf{args.conf}_iou0.75"]
        print(f"{split:6s} {r['n_images']:5d} {r['n_gt']:4d} | {100*r['mAP50']:6.2f} {100*r['mAP75']:6.2f} {100*r['mAP50-95']:8.2f} | "
              f"{a['precision']:5.3f} {a['recall']:5.3f} {a['f1']:5.3f} | {b['precision']:5.3f} {b['recall']:5.3f} {b['f1']:6.3f} | "
              f"{r['best_f1_iou0.5']['f1']:.3f} ({r['best_f1_iou0.5']['conf']:.2f})")
    print(f"(P/R/F1 con score >= {args.conf}; JSON en {os.path.abspath(args.output_dir)})")


if __name__ == "__main__":
    main()
