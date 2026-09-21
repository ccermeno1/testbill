"""DynamicSoftLabelAssigner (mmdet 3.x) for rotated boxes, MPS-safe (no topk)."""
import math
from typing import Dict

import torch
import torch.nn.functional as F

from .boxes import points_in_rboxes
from .ops import box_iou_rotated

INF = 100000000.0
EPS = 1.0e-7


class DynamicSoftLabelAssigner:

    def __init__(self, soft_center_radius: float = 3.0, topk: int = 13, iou_weight: float = 3.0):
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight

    @torch.no_grad()
    def assign(self, pred_scores: torch.Tensor, decoded_bboxes: torch.Tensor, priors: torch.Tensor,
               gt_bboxes: torch.Tensor, gt_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_scores: (n, C) logits.  decoded_bboxes: (n, 5).  priors: (n, 4) x, y, stride, stride.
            gt_bboxes: (k, 5).  gt_labels: (k,).
        Returns:
            assigned_gt_inds (n,) long: 0 = background, i+1 = gt i; max_overlaps (n,) float.
        """
        num_gt = gt_bboxes.shape[0]
        num_bboxes = decoded_bboxes.shape[0]
        assigned_gt_inds = decoded_bboxes.new_zeros((num_bboxes,), dtype=torch.long)
        max_overlaps = decoded_bboxes.new_full((num_bboxes,), -INF)
        if num_gt == 0 or num_bboxes == 0:
            return dict(assigned_gt_inds=assigned_gt_inds, max_overlaps=decoded_bboxes.new_zeros((num_bboxes,)))

        prior_center = priors[:, :2]
        is_in_gts = points_in_rboxes(prior_center, gt_bboxes)  # (n, k)
        valid_mask = is_in_gts.sum(dim=1) > 0
        valid_bbox = decoded_bboxes[valid_mask]
        valid_scores = pred_scores[valid_mask]
        num_valid = valid_bbox.shape[0]
        if num_valid == 0:
            return dict(assigned_gt_inds=assigned_gt_inds, max_overlaps=decoded_bboxes.new_zeros((num_bboxes,)))

        gt_center = gt_bboxes[:, :2]
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (valid_prior[:, None, :2] - gt_center[None, :, :]).pow(2).sum(-1).sqrt() / strides[:, None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        pairwise_ious = box_iou_rotated(valid_bbox, gt_bboxes)  # (v, k)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        gt_onehot = F.one_hot(gt_labels.long(), pred_scores.shape[-1]).to(pred_scores.dtype)  # (k, C)
        soft_label = gt_onehot[None] * pairwise_ious[..., None]  # (v, k, C)
        logits = valid_scores[:, None, :].expand(num_valid, num_gt, -1)
        scale_factor = soft_label - logits.sigmoid()
        soft_cls_cost = (F.binary_cross_entropy_with_logits(logits, soft_label, reduction='none') *
                         scale_factor.abs().pow(2.0)).sum(-1)

        cost = soft_cls_cost + iou_cost + soft_center_prior
        matched_ious, matched_gt_inds, fg_mask = self.dynamic_k_matching(cost, pairwise_ious, num_gt)

        valid_idx = valid_mask.nonzero().squeeze(1)
        fg_idx = valid_idx[fg_mask]
        assigned_gt_inds[fg_idx] = matched_gt_inds + 1
        max_overlaps[fg_idx] = matched_ious
        return dict(assigned_gt_inds=assigned_gt_inds, max_overlaps=max_overlaps)

    def dynamic_k_matching(self, cost: torch.Tensor, pairwise_ious: torch.Tensor, num_gt: int):
        matching = torch.zeros_like(cost, dtype=torch.uint8)
        candidate_topk = min(self.topk, pairwise_ious.shape[0])
        # topk via sort: torch.topk is limited to k <= 16 on some MPS builds
        sorted_ious = pairwise_ious.sort(dim=0, descending=True).values[:candidate_topk]
        dynamic_ks = torch.clamp(sorted_ious.sum(0).int(), min=1).tolist()
        for gt_idx in range(num_gt):
            order = cost[:, gt_idx].argsort()
            matching[order[:dynamic_ks[gt_idx]], gt_idx] = 1
        multi = matching.sum(1) > 1
        if multi.any():
            cost_argmin = cost[multi].argmin(dim=1)
            matching[multi] = 0
            matching[multi.nonzero().squeeze(1), cost_argmin] = 1
        fg_mask = matching.sum(1) > 0
        matched_gt_inds = matching[fg_mask].argmax(1)
        matched_ious = (matching * pairwise_ious).sum(1)[fg_mask]
        return matched_ious, matched_gt_inds, fg_mask
