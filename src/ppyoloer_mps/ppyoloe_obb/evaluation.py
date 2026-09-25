"""
Oriented detection metrics: COCO-style AP (101 recall points) with exact rotated IoU,
mAP@0.5, mAP@0.75, mAP@[0.50:0.95], plus precision/recall/F1 at a confidence threshold.

Mirrors what scripts/evaluate_obb.py did on the Paddle side, so numbers stay comparable
between the two implementations.
"""
from __future__ import annotations

import numpy as np
import torch

from .boxes import rotated_iou

__all__ = ["IOU_THRESHOLDS", "evaluate_detections", "format_metrics"]

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)


def _match(scores: np.ndarray, ious: np.ndarray, thr: float) -> np.ndarray:
    """Greedy COCO-style matching: by descending score, to the free ground truth with the highest IoU."""
    tp = np.zeros(len(scores), dtype=bool)
    if ious.shape[1] == 0:
        return tp
    matched: set[int] = set()
    for i in np.argsort(-scores):
        cand = ious[i].copy()
        if matched:
            cand[list(matched)] = -1.0
        j = int(np.argmax(cand))
        if cand[j] >= thr:
            tp[i] = True
            matched.add(j)
    return tp


def _average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
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


def _prf(tp: int, fp: int, n_gt: int):
    p = tp / max(tp + fp, 1)
    r = tp / max(n_gt, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    return p, r, f


def evaluate_detections(
    predictions: list[dict],
    ground_truth: list[dict],
    conf: float = 0.5,
) -> dict:
    """predictions and ground_truth are aligned lists of dicts holding 'rboxes' (N, 5), plus
    'scores' (N,) on the predictions. Tensors or arrays; evaluated on CPU."""
    all_scores: list[np.ndarray] = []
    all_tp: dict[float, list[np.ndarray]] = {t: [] for t in IOU_THRESHOLDS}
    n_gt = 0
    for pred, gt in zip(predictions, ground_truth):
        p_boxes = torch.as_tensor(pred["rboxes"], dtype=torch.float32).reshape(-1, 5)
        g_boxes = torch.as_tensor(gt["rboxes"], dtype=torch.float32).reshape(-1, 5)
        scores = np.asarray(pred["scores"], dtype=np.float64).reshape(-1)
        n_gt += int(g_boxes.shape[0])
        ious = rotated_iou(p_boxes, g_boxes).cpu().numpy() if p_boxes.numel() else np.zeros((0, g_boxes.shape[0]))
        all_scores.append(scores)
        for thr in IOU_THRESHOLDS:
            all_tp[thr].append(_match(scores, ious, float(thr)) if len(scores) else np.zeros(0, bool))

    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    tps = {thr: (np.concatenate(v) if v else np.zeros(0, bool)) for thr, v in all_tp.items()}
    aps = {f"AP{int(round(t * 100))}": _average_precision(tps[t], scores, n_gt) for t in IOU_THRESHOLDS}

    res: dict = {
        "n_images": len(ground_truth),
        "n_gt": n_gt,
        "n_pred_total": int(scores.size),
        "mAP50": aps["AP50"],
        "mAP75": aps["AP75"],
        "mAP50-95": float(np.mean(list(aps.values()))),
        "AP_per_iou": aps,
    }
    keep = scores >= conf
    for thr in (0.5, 0.75):
        tp = int(tps[thr][keep].sum())
        fp = int(keep.sum()) - tp
        p, r, f = _prf(tp, fp, n_gt)
        res[f"conf{conf}_iou{thr}"] = {"TP": tp, "FP": fp, "FN": n_gt - tp, "precision": p, "recall": r, "f1": f}

    best = (0.0, 0.0)
    for c in np.unique(scores):
        k = scores >= c
        tp = int(tps[0.5][k].sum())
        _, _, f = _prf(tp, int(k.sum()) - tp, n_gt)
        if f > best[0]:
            best = (f, float(c))
    res["best_f1_iou0.5"] = {"f1": best[0], "conf": best[1]}
    return res


def format_metrics(name: str, res: dict, conf: float = 0.5) -> str:
    a = res[f"conf{conf}_iou0.5"]
    b = res[f"conf{conf}_iou0.75"]
    return (
        f"{name:16s} {res['n_images']:5d} {res['n_gt']:4d} | "
        f"{100 * res['mAP50']:6.2f} {100 * res['mAP75']:6.2f} {100 * res['mAP50-95']:8.2f} | "
        f"{a['precision']:5.3f} {a['recall']:5.3f} {a['f1']:5.3f} | "
        f"{b['precision']:5.3f} {b['recall']:5.3f} {b['f1']:6.3f} | "
        f"{res['best_f1_iou0.5']['f1']:.3f} ({res['best_f1_iou0.5']['conf']:.2f})"
    )


HEADER = (
    f"{'split':16s} {'imgs':>5s} {'GT':>4s} | {'mAP50':>6s} {'mAP75':>6s} {'mAP50-95':>8s} | "
    f"{'P@.5':>5s} {'R@.5':>5s} {'F1@.5':>5s} | {'P@.75':>5s} {'R@.75':>5s} {'F1@.75':>6s} | best F1 (conf)"
)
