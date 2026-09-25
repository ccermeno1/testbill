"""
RotatedTaskAlignedAssigner in plain PyTorch.

Port of ppdet/modeling/assigners/rotated_task_aligned_assigner.py. For every ground truth it
takes the topk anchors with the highest alignment metric (score^alpha * IoU^beta), requires the
anchor centre to fall inside the box, and breaks ties (one anchor claimed by several boxes) by
keeping the ground truth with the highest IoU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .boxes import batch_rotated_iou, check_points_in_rotated_boxes

__all__ = ["RotatedTaskAlignedAssigner"]


def gather_topk_anchors(metrics: torch.Tensor, topk: int, topk_mask: torch.Tensor | None = None, eps: float = 1e-9):
    """metrics (B, n, L) -> mask (B, n, L) with 1 on the topk highest per ground truth."""
    num_anchors = metrics.shape[-1]
    topk = min(topk, num_anchors)
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=True)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(dim=-1, keepdim=True).values > eps).to(metrics.dtype)
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(dim=-2).to(metrics.dtype)
    return is_in_topk * topk_mask


def compute_max_iou_anchor(ious: torch.Tensor) -> torch.Tensor:
    """For each anchor, flag the ground truth with the highest IoU: (B, n, L) -> (B, n, L)."""
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(dim=-2)
    return F.one_hot(max_iou_index, num_max_boxes).permute(0, 2, 1).to(ious.dtype)


class RotatedTaskAlignedAssigner(torch.nn.Module):
    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,   # (B, L, C) after sigmoid
        pred_bboxes: torch.Tensor,   # (B, L, 5)
        anchor_points: torch.Tensor,  # (1, L, 2)
        gt_labels: torch.Tensor,     # (B, n, 1)
        gt_bboxes: torch.Tensor,     # (B, n, 5)
        pad_gt_mask: torch.Tensor,   # (B, n, 1)
        bg_index: int,
    ):
        batch_size, num_anchors, num_classes = pred_scores.shape
        num_max_boxes = gt_bboxes.shape[1]
        device = pred_scores.device

        if num_max_boxes == 0:
            return (
                torch.full((batch_size, num_anchors), bg_index, dtype=torch.long, device=device),
                torch.zeros((batch_size, num_anchors, 5), device=device),
                torch.zeros((batch_size, num_anchors, num_classes), device=device),
            )

        ious = batch_rotated_iou(gt_bboxes, pred_bboxes)  # (B, n, L)
        ious = torch.where(ious > 1 + self.eps, torch.zeros_like(ious), ious)

        # predicted score for each ground truth's class -> (B, n, L)
        scores_t = pred_scores.permute(0, 2, 1)
        idx = gt_labels.squeeze(-1).clamp(min=0).unsqueeze(-1).expand(-1, -1, num_anchors)
        bbox_cls_scores = torch.gather(scores_t, 1, idx)

        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)
        is_in_gts = check_points_in_rotated_boxes(anchor_points, gt_bboxes)

        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts.to(alignment_metrics.dtype), self.topk, topk_mask=pad_gt_mask
        )
        mask_positive = is_in_topk * is_in_gts.to(is_in_topk.dtype) * pad_gt_mask

        mask_positive_sum = mask_positive.sum(dim=-2)
        if bool((mask_positive_sum > 1).any()):
            mask_multiple = (mask_positive_sum.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)  # (B, L)

        batch_ind = torch.arange(batch_size, device=device).unsqueeze(-1)
        flat_index = assigned_gt_index + batch_ind * num_max_boxes

        assigned_labels = gt_labels.flatten()[flat_index.flatten()].reshape(batch_size, num_anchors)
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels, torch.full_like(assigned_labels, bg_index)
        )
        assigned_bboxes = gt_bboxes.reshape(-1, 5)[flat_index.flatten()].reshape(batch_size, num_anchors, 5)

        assigned_scores = F.one_hot(assigned_labels, num_classes + 1).float()
        keep = [i for i in range(num_classes + 1) if i != bg_index]
        assigned_scores = assigned_scores[..., keep]

        alignment_metrics = alignment_metrics * mask_positive
        max_metrics = alignment_metrics.max(dim=-1, keepdim=True).values
        max_ious = (ious * mask_positive).max(dim=-1, keepdim=True).values
        alignment_metrics = alignment_metrics / (max_metrics + self.eps) * max_ious
        alignment_metrics = alignment_metrics.max(dim=-2).values.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics
        return assigned_labels, assigned_bboxes, assigned_scores
