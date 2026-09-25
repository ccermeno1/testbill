"""Losses of YOLOX-OBB (``configs/losses/yolox_losses_obb.yaml``), un-reduced.

* obj / cls: ``CrossEntropyLoss(1, sum, bce_use_sigmoid)``
* reg:       ``PolyIoULoss(5, sum, linear)``: 1 - rotated IoU
* reg extra: ``L1Loss(1, sum, norm)`` on the raw regression, only in the last epochs

The caller sums the per-element losses, applies the weights and divides by the
number of foreground anchors, as ``DetectX.get_losses`` does.
"""
import torch
import torch.nn.functional as F

from .ops import diff_iou_rotated_2d


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sigmoid BCE on logits, not reduced."""
    return F.binary_cross_entropy_with_logits(pred, target, reduction='none')


def poly_iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(n, 5) x (n, 5) -> (n,) ``1 - IoU`` (PolyIoULoss, ``linear`` mode). Computed in float32."""
    ious = diff_iou_rotated_2d(pred.float(), target.float()).clamp(min=eps, max=1.0)
    return 1 - ious


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs()
