"""SimOTA label assignment of YOLOX-OBB (``OBBDetectX.get_assignments``), MPS-safe (no large topk)."""
from typing import Dict

import torch
import torch.nn.functional as F

from .boxes import points_in_rboxes
from .ops import box_iou_rotated


class SimOTAAssigner:

    def __init__(self, center_radius: float = 2.5, candidate_topk: int = 10, iou_weight: float = 3.0):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight

    @torch.no_grad()
    def assign(self, cls_logits: torch.Tensor, obj_logits: torch.Tensor, decoded_bboxes: torch.Tensor,
               priors: torch.Tensor, gt_bboxes: torch.Tensor, gt_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            cls_logits: (n, C).  obj_logits: (n,).  decoded_bboxes: (n, 5).
            priors: (n, 3) = grid x, grid y (in cells), stride.
            gt_bboxes: (k, 5).  gt_labels: (k,).
        Returns:
            fg_mask (n,) bool, matched_gt_inds (num_fg,) long, matched_ious (num_fg,) float.
        """
        n = decoded_bboxes.shape[0]
        num_gt = gt_bboxes.shape[0]
        empty = dict(fg_mask=torch.zeros(n, dtype=torch.bool, device=decoded_bboxes.device),
                     matched_gt_inds=gt_labels.new_zeros((0,)),
                     matched_ious=decoded_bboxes.new_zeros((0,)))
        if num_gt == 0 or n == 0:
            return empty

        # candidates: anchor centre inside a gt box, or within center_radius strides of its centre
        stride = priors[:, 2]
        centers = (priors[:, :2] + 0.5) * stride[:, None]  # (n, 2)
        in_boxes = points_in_rboxes(centers, gt_bboxes).T  # (k, n)
        radius = self.center_radius * stride[None, :]
        delta = (centers[None, :, :] - gt_bboxes[:, None, :2]).abs()  # (k, n, 2)
        in_centers = (delta[..., 0] < radius) & (delta[..., 1] < radius)
        fg_candidates = in_boxes.any(0) | in_centers.any(0)
        if not fg_candidates.any():
            return empty
        in_boxes_and_center = in_boxes[:, fg_candidates] & in_centers[:, fg_candidates]  # (k, m)

        pair_wise_ious = box_iou_rotated(gt_bboxes, decoded_bboxes[fg_candidates])  # (k, m)
        iou_cost = -torch.log(pair_wise_ious + 1e-8)

        gt_onehot = F.one_hot(gt_labels.long(), cls_logits.shape[-1]).float()  # (k, C)
        scores = (cls_logits[fg_candidates].float().sigmoid() *
                  obj_logits[fg_candidates, None].float().sigmoid()).sqrt()  # (m, C)
        scores = scores[None].expand(num_gt, -1, -1)
        cls_cost = F.binary_cross_entropy(scores, gt_onehot[:, None, :].expand_as(scores),
                                          reduction='none').sum(-1)  # (k, m)

        cost = cls_cost + self.iou_weight * iou_cost + 100000.0 * (~in_boxes_and_center)
        matching = self.dynamic_k_matching(cost, pair_wise_ious)  # (k, m)

        fg_in_candidates = matching.sum(0) > 0
        fg_mask = empty['fg_mask']
        fg_mask[fg_candidates.nonzero().squeeze(1)[fg_in_candidates]] = True
        matched_gt_inds = matching[:, fg_in_candidates].argmax(0)
        matched_ious = (matching * pair_wise_ious).sum(0)[fg_in_candidates]
        return dict(fg_mask=fg_mask, matched_gt_inds=matched_gt_inds, matched_ious=matched_ious)

    def dynamic_k_matching(self, cost: torch.Tensor, pair_wise_ious: torch.Tensor) -> torch.Tensor:
        matching = torch.zeros_like(cost, dtype=torch.uint8)
        k = min(self.candidate_topk, pair_wise_ious.shape[1])
        # topk via sort: torch.topk is limited to k <= 16 on some MPS builds
        topk_ious = pair_wise_ious.sort(dim=1, descending=True).values[:, :k]
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1).tolist()
        order = cost.argsort(dim=1)
        for gt_idx, dk in enumerate(dynamic_ks):
            matching[gt_idx, order[gt_idx, :dk]] = 1
        multi = matching.sum(0) > 1
        if multi.any():
            cost_argmin = cost[:, multi].argmin(0)
            matching[:, multi] = 0
            matching[cost_argmin, multi.nonzero().squeeze(1)] = 1
        return matching
