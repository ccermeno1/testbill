"""
Perdida de PP-YOLOE-R: VariFocal (clasificacion) + ProbIoU (caja rotada) + DFL (angulo).

Port de `PPYOLOERHead.get_loss` de ppdet/modeling/heads/ppyoloe_r_head.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import RotatedTaskAlignedAssigner
from .boxes import probiou_loss

__all__ = ["PPYOLOERLoss"]


def weighted_bce(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """BCE con peso, escrita a mano para que el gradiente fluya TAMBIEN por `weight`.

    `F.binary_cross_entropy` de PyTorch prohibe un `weight` con gradiente, pero el
    `F.binary_cross_entropy` de Paddle si lo propaga, y la VariFocal de PP-YOLOE-R calcula
    el peso a partir de la propia prediccion. Verificado numericamente contra Paddle.
    Los logaritmos se acotan en -100 igual que hace PyTorch internamente.
    """
    log_p = torch.log(pred).clamp(min=-100)
    log_1_p = torch.log1p(-pred).clamp(min=-100)
    return (-weight * (target * log_p + (1 - target) * log_1_p)).sum()


def varifocal_loss(pred_score, gt_score, label, alpha: float = 0.75, gamma: float = 2.0):
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
    return weighted_bce(pred_score, gt_score, weight)


def df_loss(pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Distribution Focal Loss sobre el bin del angulo."""
    target_left = target.long()
    target_right = target_left + 1
    weight_left = target_right.float() - target
    weight_right = 1 - weight_left
    loss_left = F.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
    loss_right = F.cross_entropy(pred_dist, target_right.clamp(max=pred_dist.shape[-1] - 1), reduction="none") * weight_right
    return (loss_left + loss_right).mean(-1, keepdim=True)


class PPYOLOERLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        angle_max: int = 90,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        loss_weight: dict[str, float] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.angle_max = angle_max
        self.half_pi_bin = (math.pi / 2) / angle_max
        self.assigner = RotatedTaskAlignedAssigner(topk=topk, alpha=alpha, beta=beta)
        self.loss_weight = loss_weight or {"class": 1.0, "iou": 2.5, "dfl": 0.05}

    def forward(self, head_outs, gt_rboxes, gt_labels, pad_gt_mask, decode_fn):
        """head_outs: salida de `PPYOLOERHead.forward_train`.

        gt_rboxes (B, n, 5), gt_labels (B, n, 1), pad_gt_mask (B, n, 1).
        decode_fn: funcion del head que convierte (puntos, dist, angulo, stride) -> rboxes.
        """
        pred_scores, pred_dist, pred_angle, anchor_points, _num_anchors_list, stride_tensor = head_outs
        pred_bboxes = decode_fn(anchor_points, pred_dist, pred_angle, stride_tensor)

        assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
            pred_scores.detach(),
            pred_bboxes.detach(),
            anchor_points,
            gt_labels,
            gt_rboxes,
            pad_gt_mask,
            bg_index=self.num_classes,
        )

        one_hot_label = F.one_hot(assigned_labels, self.num_classes + 1)[..., :-1].to(pred_scores.dtype)
        loss_cls = varifocal_loss(pred_scores, assigned_scores, one_hot_label)
        assigned_scores_sum = assigned_scores.sum().clamp(min=1.0)
        loss_cls = loss_cls / assigned_scores_sum

        mask_positive = assigned_labels != self.num_classes
        num_pos = int(mask_positive.sum())
        if num_pos > 0:
            idx = mask_positive.reshape(-1).nonzero(as_tuple=True)[0]
            pred_pos = pred_bboxes.reshape(-1, 5)[idx]
            target_pos = assigned_bboxes.reshape(-1, 5)[idx]
            weight = assigned_scores.sum(-1).reshape(-1)[idx]
            loss_iou = (probiou_loss(pred_pos, target_pos) * weight).sum() / assigned_scores_sum

            angle_pos = pred_angle.reshape(-1, self.angle_max + 1)[idx]
            target_angle = (target_pos[:, 4] / self.half_pi_bin).clamp(0, self.angle_max - 0.01)
            loss_dfl = df_loss(angle_pos, target_angle).mean()
        else:
            loss_iou = pred_bboxes.sum() * 0.0
            loss_dfl = pred_angle.sum() * 0.0

        loss = (
            self.loss_weight["class"] * loss_cls
            + self.loss_weight["iou"] * loss_iou
            + self.loss_weight["dfl"] * loss_dfl
        )
        return {"loss": loss, "loss_cls": loss_cls, "loss_iou": loss_iou, "loss_dfl": loss_dfl}
