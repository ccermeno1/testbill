"""Rotated-box mean AP (VOC style), port of mmrotate ``eval_rbbox_map`` (area / 11-point modes)."""
from typing import Dict, List, Sequence

import numpy as np
import torch

from .ops import box_iou_rotated


def _average_precision(recalls: np.ndarray, precisions: np.ndarray, mode: str = 'area') -> float:
    if mode == 'area':
        mrec = np.concatenate([[0.0], recalls, [1.0]])
        mpre = np.concatenate([[0.0], precisions, [0.0]])
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    if mode == '11points':
        ap = 0.0
        for thr in np.arange(0, 1.1, 0.1):
            p = precisions[recalls >= thr]
            ap += (p.max() if p.size else 0.0) / 11
        return float(ap)
    raise ValueError(mode)


def _tpfp(det: torch.Tensor, gt: torch.Tensor, iou_thr: float):
    """det (n, 6) sorted by score desc, gt (k, 5) -> tp, fp arrays (n,)."""
    n, k = det.shape[0], gt.shape[0]
    tp = np.zeros(n, np.float32)
    fp = np.zeros(n, np.float32)
    if k == 0:
        fp[:] = 1
        return tp, fp
    if n == 0:
        return tp, fp
    ious = box_iou_rotated(det[:, :5], gt).cpu().numpy()  # (n, k)
    covered = np.zeros(k, bool)
    for i in range(n):
        j = int(ious[i].argmax())
        if ious[i, j] >= iou_thr and not covered[j]:
            covered[j] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def eval_rbbox_map(det_results: List[List[torch.Tensor]], annotations: List[Dict[str, torch.Tensor]],
                   num_classes: int, iou_thr: float = 0.5, mode: str = 'area') -> Dict[str, float]:
    """
    Args:
        det_results: per image, per class ``(n, 6)`` tensors (cx cy w h a score).
        annotations: per image dict(boxes (k, 5), labels (k,)).
    Returns:
        dict with ``mAP`` and per-class ``AP<i>``, ``recall<i>``.
    """
    out = {}
    aps = []
    for c in range(num_classes):
        tps, fps, scores = [], [], []
        num_gt = 0
        for dets, ann in zip(det_results, annotations):
            gt = ann['boxes'][ann['labels'] == c]
            num_gt += gt.shape[0]
            d = dets[c]
            if d.shape[0]:
                order = d[:, 5].argsort(descending=True)
                d = d[order]
            tp, fp = _tpfp(d, gt, iou_thr)
            tps.append(tp)
            fps.append(fp)
            scores.append(d[:, 5].cpu().numpy() if d.shape[0] else np.zeros(0, np.float32))
        scores = np.concatenate(scores)
        order = np.argsort(-scores, kind='stable')
        tp = np.cumsum(np.concatenate(tps)[order])
        fp = np.cumsum(np.concatenate(fps)[order])
        recalls = tp / max(num_gt, 1)
        precisions = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
        ap = _average_precision(recalls, precisions, mode) if num_gt > 0 else 0.0
        out[f'AP{c}'] = ap
        out[f'precision{c}'] = float(precisions[-1]) if precisions.size else 0.0
        out[f'recall{c}'] = float(recalls[-1]) if recalls.size else 0.0
        out[f'num_gt{c}'] = num_gt
        out[f'num_det{c}'] = int(scores.size)
        aps.append(ap)
    out['mAP'] = float(np.mean(aps)) if aps else 0.0
    return out


def summarize(det_results, annotations, num_classes, thresholds: Sequence[float] = (0.5, 0.75), mode='area',
              report_det_results=None) -> Dict[str, float]:
    """mAP at each threshold plus mAP@.5:.95 (COCO-style average)."""
    res = {}
    for t in thresholds:
        res[f'mAP@{t:.2f}'] = eval_rbbox_map(det_results, annotations, num_classes, t, mode)['mAP']
    vals = [eval_rbbox_map(det_results, annotations, num_classes, t, mode)['mAP'] for t in np.arange(0.5, 0.96, 0.05)]
    res['mAP@.5:.95'] = float(np.mean(vals))
    # AP uses the untrimmed low-threshold detections. Operational precision,
    # recall and F1 may use a separate deployment threshold.
    operational = report_det_results if report_det_results is not None else det_results
    r = eval_rbbox_map(operational, annotations, num_classes, 0.5, mode)
    precision = float(np.mean([r[f'precision{c}'] for c in range(num_classes)]))
    recall = float(np.mean([r[f'recall{c}'] for c in range(num_classes)]))
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(np.float32).eps)
    res['precision@0.50'] = precision
    res['recall@0.50'] = recall
    res['f1@0.50'] = f1
    return res
