"""Losses of RTMDet-R: Quality Focal Loss and rotated IoU loss (un-reduced)."""
import torch
import torch.nn.functional as F

from .ops import diff_iou_rotated_2d


def quality_focal_loss(pred: torch.Tensor, label: torch.Tensor, score: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """QFL (mmdet ``quality_focal_loss``), per-sample (n,) losses, not reduced.

    Args:
        pred: (n, C) logits.
        label: (n,) long, C = background.
        score: (n,) soft target (IoU) for the positives.
    """
    pred_sigmoid = pred.sigmoid()
    zerolabel = torch.zeros_like(pred)
    loss = F.binary_cross_entropy_with_logits(pred, zerolabel, reduction='none') * pred_sigmoid.pow(beta)
    bg = pred.shape[1]
    pos = ((label >= 0) & (label < bg)).nonzero().squeeze(1)
    if pos.numel():
        pos_label = label[pos]
        p = pred[pos, pos_label]
        scale = (score[pos] - pred_sigmoid[pos, pos_label]).abs().pow(beta)
        loss[pos, pos_label] = F.binary_cross_entropy_with_logits(p, score[pos], reduction='none') * scale
    return loss.sum(dim=1)


def rotated_iou_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = 'linear', eps: float = 1e-6) -> torch.Tensor:
    """(n, 5) x (n, 5) -> (n,) losses; ``linear`` = 1 - IoU, ``square`` = 1 - IoU^2, ``log`` = -log IoU."""
    ious = diff_iou_rotated_2d(pred, target).clamp(min=eps)
    if mode == 'linear':
        return 1 - ious
    if mode == 'square':
        return 1 - ious ** 2
    if mode == 'log':
        return -ious.log()
    raise ValueError(mode)
