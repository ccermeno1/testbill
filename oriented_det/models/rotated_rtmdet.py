"""Rotated RTMDet: CSPNeXt + CSPNeXtPAFPN + SepBN head (MMRotate-style), pure PyTorch.

Mirrors MMRotate ``RotatedRTMDetSepBNHead`` (tiny/s/m/l): anchor-free points
(offset 0) on strides 8/16/32, ``ltrb * stride`` distances + raw le90 angle
(``PseudoAngleCoder``), ``DynamicSoftLabelAssigner`` with exact rotated IoU,
QualityFocalLoss (β=2) and linear rotated-IoU loss (weight 2). No mmcv; the
rotated IoU / NMS come from :mod:`oriented_det.ops`, so it runs on CUDA, MPS
and CPU.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from ..ops.diff_iou_rotated import diff_iou_rotated_2d, riou_loss_per_box
from .backbones.cspnext import (
    CSPNEXT_IMAGENET_URLS,
    RTMDET_COCO_URLS,
    ConvModule,
    build_cspnext_pafpn,
    load_mmdet_weights,
)
from .oriented_rpn import normalize_boxes_to_le90
from .rotated_fcos import _apply_fcos_nms
from .rotated_retinanet import _foreground_sigmoid_focal_weights
from .utils import prepare_targets, tensor_to_rboxes, SigmoidFocalClassWeightsMixin

RTMDET_DOTA_URLS: Dict[str, str] = {
    "tiny": "https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota/rotated_rtmdet_tiny-3x-dota-9d821076.pth",
}

_EPS = 1e-7
_INF = 1e8
# Max (prior, gt) pairs per exact rotated-IoU call in the assigner (memory bound).
_ASSIGNER_IOU_CHUNK = 32768


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def generate_rtmdet_priors(
    feature_map_sizes: Sequence[Tuple[int, int]],
    strides: Sequence[float],
    dtype: "torch.dtype",
    device: "torch.device",
) -> List["torch.Tensor"]:
    """MMDet ``MlvlPointGenerator(offset=0)``: one ``[H*W, 2]`` (x, y) tensor per level."""
    points: List[torch.Tensor] = []
    for (h, w), stride in zip(feature_map_sizes, strides):
        shift_x = torch.arange(w, device=device, dtype=dtype) * float(stride)
        shift_y = torch.arange(h, device=device, dtype=dtype) * float(stride)
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        points.append(torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1))
    return points


def decode_rtmdet_boxes(
    points: "torch.Tensor",
    ltrb: "torch.Tensor",
    angle: "torch.Tensor",
) -> "torch.Tensor":
    """(left, top, right, bottom) pixel distances + angle at ``points`` → (cx, cy, w, h, a).

    Same geometry as ``DistanceAnglePointCoder.decode`` but without le90
    re-normalization, and with ``|w|``, ``|h|``: the RTMDet head has no ReLU/exp on
    distances, so early predictions can have ``l + r < 0``; taking the absolute
    extent keeps the rotated-IoU loss differentiable there instead of clamping.
    """
    cos_a = torch.cos(angle[..., 0])
    sin_a = torch.sin(angle[..., 0])
    wh = ltrb[..., :2] + ltrb[..., 2:]
    lx = (ltrb[..., 2] - ltrb[..., 0]) * 0.5
    ly = (ltrb[..., 3] - ltrb[..., 1]) * 0.5
    cx = points[..., 0] + cos_a * lx - sin_a * ly
    cy = points[..., 1] + sin_a * lx + cos_a * ly
    return torch.stack([cx, cy, wh[..., 0].abs(), wh[..., 1].abs(), angle[..., 0]], dim=-1)


def points_in_rboxes(points: "torch.Tensor", boxes: "torch.Tensor") -> "torch.Tensor":
    """``[P, G]`` bool: point ``p`` strictly inside rotated box ``g``."""
    if boxes.numel() == 0:
        return points.new_zeros((points.size(0), 0), dtype=torch.bool)
    d = points[:, None, :] - boxes[None, :, :2]
    cos_a = torch.cos(boxes[:, 4])[None]
    sin_a = torch.sin(boxes[:, 4])[None]
    local_x = d[..., 0] * cos_a + d[..., 1] * sin_a
    local_y = -d[..., 0] * sin_a + d[..., 1] * cos_a
    return (local_x.abs() < boxes[None, :, 2] * 0.5) & (local_y.abs() < boxes[None, :, 3] * 0.5)


def _rbox_hbb(boxes: "torch.Tensor") -> "torch.Tensor":
    """Axis-aligned bounds ``[N, 4]`` (x1, y1, x2, y2) of rotated boxes."""
    cos_a = torch.cos(boxes[:, 4]).abs()
    sin_a = torch.sin(boxes[:, 4]).abs()
    half_w = (boxes[:, 2] * cos_a + boxes[:, 3] * sin_a) * 0.5
    half_h = (boxes[:, 2] * sin_a + boxes[:, 3] * cos_a) * 0.5
    return torch.stack(
        [boxes[:, 0] - half_w, boxes[:, 1] - half_h, boxes[:, 0] + half_w, boxes[:, 1] + half_h],
        dim=-1,
    )


@torch.no_grad()
def pairwise_rbox_iou_exact(boxes1: "torch.Tensor", boxes2: "torch.Tensor") -> "torch.Tensor":
    """Exact polygon-clipping rotated IoU ``[N, M]`` (pure torch; CUDA / MPS / CPU).

    Only pairs whose axis-aligned bounds overlap are evaluated, in chunks.
    """
    n, m = boxes1.size(0), boxes2.size(0)
    out = boxes1.new_zeros((n, m), dtype=torch.float32)
    if n == 0 or m == 0:
        return out
    b1 = boxes1.float()
    b2 = boxes2.float()
    h1 = _rbox_hbb(b1)
    h2 = _rbox_hbb(b2)
    overlap = (
        (torch.minimum(h1[:, None, 2], h2[None, :, 2]) > torch.maximum(h1[:, None, 0], h2[None, :, 0]))
        & (torch.minimum(h1[:, None, 3], h2[None, :, 3]) > torch.maximum(h1[:, None, 1], h2[None, :, 1]))
    )
    idx1, idx2 = overlap.nonzero(as_tuple=True)
    if idx1.numel() == 0:
        return out
    ious = torch.cat(
        [
            diff_iou_rotated_2d(
                b1[idx1[s : s + _ASSIGNER_IOU_CHUNK]], b2[idx2[s : s + _ASSIGNER_IOU_CHUNK]]
            )
            for s in range(0, idx1.numel(), _ASSIGNER_IOU_CHUNK)
        ]
    )
    out[idx1, idx2] = ious
    return out


# ---------------------------------------------------------------------------
# Assigner
# ---------------------------------------------------------------------------


@torch.no_grad()
def dynamic_soft_label_assign(
    priors: "torch.Tensor",
    prior_strides: "torch.Tensor",
    decoded_boxes: "torch.Tensor",
    cls_logits: "torch.Tensor",
    gt_boxes: "torch.Tensor",
    gt_labels: "torch.Tensor",
    *,
    topk: int = 13,
    iou_weight: float = 3.0,
    soft_center_radius: float = 3.0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """MMDet ``DynamicSoftLabelAssigner`` with rotated boxes (one image).

    Args:
        priors: ``[P, 2]`` prior points.
        prior_strides: ``[P]`` stride per prior.
        decoded_boxes: ``[P, 5]`` predicted rboxes (detached).
        cls_logits: ``[P, C]`` classification logits (detached).
        gt_boxes: ``[G, 5]``.
        gt_labels: ``[G]`` **0-indexed** classes.

    Returns:
        ``assigned_gt`` ``[P]`` (GT index, ``-1`` = background) and
        ``assigned_iou`` ``[P]`` (IoU with the assigned GT, 0 for background).
    """
    num_priors = priors.size(0)
    num_gt = gt_boxes.size(0)
    assigned_gt = torch.full((num_priors,), -1, dtype=torch.long, device=priors.device)
    assigned_iou = priors.new_zeros((num_priors,), dtype=torch.float32)
    if num_gt == 0 or num_priors == 0:
        return assigned_gt, assigned_iou

    gt_boxes = gt_boxes.float()
    is_in_gts = points_in_rboxes(priors.float(), gt_boxes)
    valid_mask = is_in_gts.any(dim=1)
    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    num_valid = valid_idx.numel()
    if num_valid == 0:
        return assigned_gt, assigned_iou

    valid_boxes = decoded_boxes[valid_idx].float()
    valid_logits = cls_logits[valid_idx].float()
    valid_priors = priors[valid_idx].float()
    valid_strides = prior_strides[valid_idx].float()

    distance = (valid_priors[:, None, :] - gt_boxes[None, :, :2]).pow(2).sum(-1).sqrt()
    distance = distance / valid_strides[:, None]
    soft_center_prior = torch.pow(10.0, distance - soft_center_radius)

    pairwise_ious = pairwise_rbox_iou_exact(valid_boxes, gt_boxes)
    iou_cost = -torch.log(pairwise_ious + _EPS) * iou_weight

    num_classes = valid_logits.size(-1)
    gt_onehot = F.one_hot(gt_labels.long(), num_classes).float()  # [G, C]
    soft_label = gt_onehot[None, :, :] * pairwise_ious[..., None]  # [V, G, C]
    logits_vgc = valid_logits[:, None, :].expand(num_valid, num_gt, num_classes)
    scale = (soft_label - logits_vgc.sigmoid()).abs().pow(2.0)
    soft_cls_cost = (
        F.binary_cross_entropy_with_logits(logits_vgc, soft_label, reduction="none") * scale
    ).sum(dim=-1)

    cost = soft_cls_cost + iou_cost + soft_center_prior  # [V, G]

    # Dynamic-k: each GT takes its k lowest-cost priors, k = floor(sum of top-13 IoUs) ≥ 1.
    candidate_topk = min(int(topk), num_valid)
    topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
    dynamic_ks = topk_ious.sum(0).int().clamp(min=1)  # [G]
    # rank of each prior within its GT column (0 = cheapest); vectorized, no per-GT syncs.
    ranks = cost.argsort(dim=0).argsort(dim=0)
    matching = ranks < dynamic_ks[None, :]

    # A prior matched to several GTs keeps only its cheapest one.
    multi = matching.sum(dim=1) > 1
    cheapest = cost.argmin(dim=1)
    one_hot_cheapest = F.one_hot(cheapest, num_gt).bool()
    matching = torch.where(multi[:, None], one_hot_cheapest, matching)

    fg = matching.any(dim=1)
    matched_gt = matching.float().argmax(dim=1)
    matched_iou = (matching.float() * pairwise_ious).sum(dim=1)

    fg_idx = valid_idx[fg]
    assigned_gt[fg_idx] = matched_gt[fg]
    assigned_iou[fg_idx] = matched_iou[fg]
    return assigned_gt, assigned_iou


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def quality_focal_loss(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    scores: "torch.Tensor",
    beta: float = 2.0,
) -> "torch.Tensor":
    """MMDet ``QualityFocalLoss(use_sigmoid=True)``, element-wise ``[N, C]``.

    ``labels`` ``[N]`` in ``0..C-1`` (foreground) or ``C`` (background);
    ``scores`` ``[N]`` soft IoU targets for foreground rows.
    """
    logits = logits.float()
    pred_sigmoid = logits.sigmoid()
    loss = F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction="none"
    ) * pred_sigmoid.pow(beta)
    num_classes = logits.size(1)
    pos = ((labels >= 0) & (labels < num_classes)).nonzero(as_tuple=True)[0]
    if pos.numel() > 0:
        pos_label = labels[pos].long()
        pos_score = scores[pos].float()
        pos_logit = logits[pos, pos_label]
        scale = (pos_score - pred_sigmoid[pos, pos_label]).abs().pow(beta)
        loss = loss.index_put(
            (pos, pos_label),
            F.binary_cross_entropy_with_logits(pos_logit, pos_score, reduction="none") * scale,
        )
    return loss


def compute_rotated_rtmdet_loss(
    cls_scores: List["torch.Tensor"],
    bbox_preds: List["torch.Tensor"],
    angle_preds: List["torch.Tensor"],
    points: List["torch.Tensor"],
    strides: Sequence[float],
    gt_boxes: List["torch.Tensor"],
    gt_labels: List["torch.Tensor"],
    gt_boxes_ignore: Optional[List["torch.Tensor"]],
    num_classes: int,
    *,
    qfl_beta: float = 2.0,
    cls_weight: float = 1.0,
    box_reg_weight: float = 2.0,
    assigner_topk: int = 13,
    assigner_iou_weight: float = 3.0,
    assigner_soft_center_radius: float = 3.0,
    class_weights: Optional["torch.Tensor"] = None,
) -> Dict[str, "torch.Tensor"]:
    """QFL (normalized by Σ assigned IoU) + IoU-weighted linear rotated-IoU loss."""
    num_imgs = cls_scores[0].size(0)
    flat_cls = torch.cat(
        [c.permute(0, 2, 3, 1).reshape(num_imgs, -1, num_classes) for c in cls_scores], dim=1
    )
    flat_ltrb = torch.cat([b.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for b in bbox_preds], dim=1)
    flat_angle = torch.cat([a.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1) for a in angle_preds], dim=1)
    all_points = torch.cat(points, dim=0)
    all_strides = torch.cat(
        [p.new_full((p.size(0),), float(s)) for p, s in zip(points, strides)], dim=0
    )
    decoded = decode_rtmdet_boxes(
        all_points[None].expand(num_imgs, -1, -1), flat_ltrb.float(), flat_angle.float()
    )  # [B, P, 5]

    bg = num_classes
    labels_l: List[torch.Tensor] = []
    label_w_l: List[torch.Tensor] = []
    iou_l: List[torch.Tensor] = []
    tgt_l: List[torch.Tensor] = []
    for i in range(num_imgs):
        gts = gt_boxes[i].to(decoded.dtype)
        gt_lab0 = gt_labels[i].long() - 1  # 1-indexed → 0-indexed
        assigned_gt, assigned_iou = dynamic_soft_label_assign(
            all_points,
            all_strides,
            decoded[i].detach(),
            flat_cls[i].detach(),
            gts,
            gt_lab0,
            topk=assigner_topk,
            iou_weight=assigner_iou_weight,
            soft_center_radius=assigner_soft_center_radius,
        )
        fg = assigned_gt >= 0
        labels_i = torch.full_like(assigned_gt, bg)
        labels_i[fg] = gt_lab0[assigned_gt[fg]]
        label_w = torch.ones_like(assigned_iou)
        if gt_boxes_ignore is not None and i < len(gt_boxes_ignore):
            ign = gt_boxes_ignore[i]
            if ign is not None and ign.numel() > 0:
                in_ign = points_in_rboxes(all_points, ign.to(all_points.dtype)).any(dim=1)
                label_w = torch.where(in_ign & ~fg, torch.zeros_like(label_w), label_w)
        tgt_i = decoded.new_zeros((assigned_gt.size(0), 5))
        if fg.any():
            tgt_i[fg] = gts[assigned_gt[fg]]
        labels_l.append(labels_i)
        label_w_l.append(label_w)
        iou_l.append(assigned_iou)
        tgt_l.append(tgt_i)

    labels = torch.cat(labels_l)
    label_w = torch.cat(label_w_l)
    ious = torch.cat(iou_l)
    targets = torch.cat(tgt_l)
    flat_cls_2d = flat_cls.reshape(-1, num_classes)
    decoded_2d = decoded.reshape(-1, 5)

    avg_factor = max(float(ious.sum().item()), 1.0)

    qfl = quality_focal_loss(flat_cls_2d, labels, ious, beta=qfl_beta)
    cw = _foreground_sigmoid_focal_weights(class_weights, num_classes)
    if cw is not None:
        qfl = qfl * cw.to(device=qfl.device, dtype=qfl.dtype)
    loss_cls = float(cls_weight) * (qfl.sum(dim=1) * label_w).sum() / avg_factor

    pos = (labels >= 0) & (labels < bg)
    if pos.any():
        per_box = riou_loss_per_box(decoded_2d[pos], targets[pos].detach())
        loss_box = float(box_reg_weight) * (per_box * ious[pos]).sum() / avg_factor
    else:
        loss_box = decoded_2d.sum() * 0.0

    return {"loss_classifier": loss_cls, "loss_box_reg": loss_box}


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


class RotatedRTMDetSepBNHead(nn.Module):
    """Per-level heads with shared conv weights and separate BN (MMRotate layout).

    Parameter names match MMRotate's ``bbox_head`` (``cls_convs.{lvl}.{i}``,
    ``rtm_cls`` / ``rtm_reg`` / ``rtm_ang``) so checkpoints load by prefix rename.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        stacked_convs: int = 2,
        strides: Sequence[float] = (8, 16, 32),
        share_conv: bool = True,
        pred_kernel_size: int = 1,
        exp_on_reg: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = list(strides)
        self.exp_on_reg = bool(exp_on_reg)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_ang = nn.ModuleList()
        pad = pred_kernel_size // 2
        for _ in self.strides:
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(stacked_convs):
                chn = in_channels if i == 0 else feat_channels
                cls_convs.append(ConvModule(chn, feat_channels, 3, stride=1, padding=1))
                reg_convs.append(ConvModule(chn, feat_channels, 3, stride=1, padding=1))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.rtm_cls.append(nn.Conv2d(feat_channels, self.num_classes, pred_kernel_size, padding=pad))
            self.rtm_reg.append(nn.Conv2d(feat_channels, 4, pred_kernel_size, padding=pad))
            self.rtm_ang.append(nn.Conv2d(feat_channels, 1, pred_kernel_size, padding=pad))
        if share_conv:
            for n in range(len(self.strides)):
                for i in range(stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        bias_cls = -math.log((1 - 0.01) / 0.01)
        for conv in self.rtm_cls:
            nn.init.constant_(conv.bias, bias_cls)

    def forward(
        self,
        feats: Sequence["torch.Tensor"],
        strides: Optional[Sequence[float]] = None,
    ) -> Tuple[List["torch.Tensor"], List["torch.Tensor"], List["torch.Tensor"]]:
        """Returns per level: cls logits ``[B, C, H, W]``, ltrb **pixels** ``[B, 4, H, W]``, angle ``[B, 1, H, W]``."""
        stride_list = list(strides) if strides is not None else self.strides
        cls_scores: List[torch.Tensor] = []
        bbox_preds: List[torch.Tensor] = []
        angle_preds: List[torch.Tensor] = []
        for idx, (x, stride) in enumerate(zip(feats, stride_list)):
            cls_feat = x
            for layer in self.cls_convs[idx]:
                cls_feat = layer(cls_feat)
            reg_feat = x
            for layer in self.reg_convs[idx]:
                reg_feat = layer(reg_feat)
            cls_scores.append(self.rtm_cls[idx](cls_feat))
            reg = self.rtm_reg[idx](reg_feat).float()
            if self.exp_on_reg:
                reg = reg.exp()
            bbox_preds.append(reg * float(stride))
            angle_preds.append(self.rtm_ang[idx](reg_feat).float())
        return cls_scores, bbox_preds, angle_preds


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def _resolve_pretrained_source(variant: str, pretrained_weights: Optional[str]) -> Optional[Tuple[str, Dict[str, str]]]:
    """Map ``pretrained_weights`` to ``(url_or_path, prefix_map)``."""
    if pretrained_weights is None:
        return None
    key = str(pretrained_weights).strip()
    low = key.lower()
    full_map = {"backbone.": "backbone.", "neck.": "neck.", "bbox_head.": "head."}
    if low in ("", "none", "scratch"):
        return None
    if low == "imagenet":
        return CSPNEXT_IMAGENET_URLS[variant], {"backbone.": "backbone."}
    if low == "coco":
        if variant not in RTMDET_COCO_URLS:
            raise ValueError(f"No COCO RTMDet weights registered for variant {variant!r}.")
        return RTMDET_COCO_URLS[variant], full_map
    if low == "dota":
        if variant not in RTMDET_DOTA_URLS:
            raise ValueError(f"No DOTA Rotated RTMDet weights registered for variant {variant!r}.")
        return RTMDET_DOTA_URLS[variant], full_map
    return key, full_map


class RotatedRTMDet(SigmoidFocalClassWeightsMixin, nn.Module):
    """Rotated RTMDet detector (``variant`` tiny / s / m / l)."""

    def __init__(
        self,
        num_classes: int,
        variant: str = "tiny",
        pretrained_backbone: bool = True,
        pretrained_weights: Optional[str] = "coco",
        input_bgr: Optional[bool] = None,
        frozen_stages: int = -1,
        fpn_strides: Optional[List[float]] = None,
        stacked_convs: int = 2,
        share_conv: bool = True,
        pred_kernel_size: int = 1,
        exp_on_reg: bool = False,
        assigner_topk: int = 13,
        assigner_iou_weight: float = 3.0,
        assigner_soft_center_radius: float = 3.0,
        qfl_beta: float = 2.0,
        cls_weight: float = 1.0,
        box_reg_weight: float = 2.0,
        score_threshold: float = 0.05,
        final_nms_iou_threshold: float = 0.1,
        max_detections_per_image: int = 2000,
        nms_pre: int = 2000,
        min_bbox_size: float = 0.0,
        final_nms_use_cpu: bool = False,
        nms_class_agnostic: bool = False,
        final_nms_iou_schedule_epochs: Optional[List[int]] = None,
        final_nms_iou_schedule_values: Optional[List[float]] = None,
        roi_class_weights: Optional[Union[Dict[str, float], "torch.Tensor"]] = None,
        **kwargs: Any,
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required for RotatedRTMDet.")
        super().__init__()
        self.num_classes = int(num_classes)
        self._init_sigmoid_focal_class_weights(roi_class_weights)
        self.variant = str(variant).strip().lower()
        self.fpn_strides = list(fpn_strides or [8, 16, 32])
        if len(self.fpn_strides) != 3:
            raise ValueError(f"RotatedRTMDet uses 3 levels (P3–P5); got fpn_strides={self.fpn_strides}.")
        self.assigner_topk = int(assigner_topk)
        self.assigner_iou_weight = float(assigner_iou_weight)
        self.assigner_soft_center_radius = float(assigner_soft_center_radius)
        self.qfl_beta = float(qfl_beta)
        self.cls_weight = float(cls_weight)
        self.box_reg_weight = float(box_reg_weight)
        self.score_threshold = float(score_threshold)
        self.final_nms_iou_threshold = float(final_nms_iou_threshold)
        self.max_detections_per_image = int(max_detections_per_image)
        self.nms_pre = int(nms_pre)
        self.min_bbox_size = float(min_bbox_size)
        self.final_nms_use_cpu = bool(final_nms_use_cpu)
        self.nms_class_agnostic = bool(nms_class_agnostic)
        self._final_nms_iou_schedule_epochs = final_nms_iou_schedule_epochs
        self._final_nms_iou_schedule_values = final_nms_iou_schedule_values
        pw = None if pretrained_weights is None else str(pretrained_weights).strip().lower()
        # MMDet/MMRotate RTMDet detectors were trained on BGR input (bgr_to_rgb=False);
        # the ImageNet CSPNeXt backbones on RGB. The repo feeds normalized RGB.
        self.input_bgr = bool(input_bgr) if input_bgr is not None else pw in ("coco", "dota")

        self.backbone, self.neck, neck_channels = build_cspnext_pafpn(
            self.variant, frozen_stages=int(frozen_stages)
        )
        self.head = RotatedRTMDetSepBNHead(
            num_classes=self.num_classes,
            in_channels=neck_channels,
            feat_channels=neck_channels,
            stacked_convs=stacked_convs,
            strides=self.fpn_strides,
            share_conv=share_conv,
            pred_kernel_size=pred_kernel_size,
            exp_on_reg=exp_on_reg,
        )

        if pretrained_backbone:
            source = _resolve_pretrained_source(self.variant, pretrained_weights)
            if source is not None:
                self._load_pretrained(*source)

    def _load_pretrained(self, src: str, prefix_map: Dict[str, str]) -> None:
        try:
            loaded, skipped = load_mmdet_weights(self, src, prefix_map)
        except Exception as exc:
            warnings.warn(
                f"Could not load RTMDet pretrained weights from {src!r} ({exc}). "
                "Continuing from random initialization.",
                RuntimeWarning,
            )
            return
        # Re-apply frozen stages (load_state_dict does not touch requires_grad, but be explicit).
        self.backbone._freeze_stages()
        print(
            f"RotatedRTMDet: loaded {len(loaded)} pretrained tensors from {src}"
            + (f" (skipped {len(skipped)} with different shape, e.g. {skipped[0]})" if skipped else "")
        )

    def set_final_nms_iou_for_epoch(self, epoch: int) -> None:
        if not self._final_nms_iou_schedule_epochs or not self._final_nms_iou_schedule_values:
            return
        idx = 0
        for boundary in self._final_nms_iou_schedule_epochs:
            if epoch < boundary:
                break
            idx += 1
        idx = min(idx, len(self._final_nms_iou_schedule_values) - 1)
        self.final_nms_iou_threshold = self._final_nms_iou_schedule_values[idx]

    def extract_features(self, images_tensor: "torch.Tensor") -> List["torch.Tensor"]:
        x = images_tensor.flip(1) if self.input_bgr else images_tensor
        return self.neck(self.backbone(x))

    def forward(
        self,
        images: Union["torch.Tensor", Sequence["torch.Tensor"]],
        targets: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Union[Dict[str, "torch.Tensor"], List[Dict[str, Any]]]:
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images_tensor = images
        else:
            if not isinstance(images, (list, tuple)):
                images = [images]
            images_tensor = torch.stack(list(images), dim=0)
        device = images_tensor.device

        feats = self.extract_features(images_tensor)
        cls_scores, bbox_preds, angle_preds = self.head(feats, strides=self.fpn_strides)
        feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feats]
        points = generate_rtmdet_priors(
            feature_map_sizes, self.fpn_strides, dtype=torch.float32, device=device
        )

        if self.training:
            if targets is None:
                raise ValueError("Targets required during training.")
            gt_boxes_list, gt_labels_list, gt_ignore_list, _ = prepare_targets(targets, device=device)
            return compute_rotated_rtmdet_loss(
                cls_scores,
                bbox_preds,
                angle_preds,
                points,
                self.fpn_strides,
                gt_boxes_list,
                gt_labels_list,
                gt_ignore_list,
                self.num_classes,
                qfl_beta=self.qfl_beta,
                cls_weight=self.cls_weight,
                box_reg_weight=self.box_reg_weight,
                assigner_topk=self.assigner_topk,
                assigner_iou_weight=self.assigner_iou_weight,
                assigner_soft_center_radius=self.assigner_soft_center_radius,
                class_weights=self.roi_class_weights_tensor,
            )

        return self._inference(cls_scores, bbox_preds, angle_preds, points, device)

    def _inference(
        self,
        cls_scores: List["torch.Tensor"],
        bbox_preds: List["torch.Tensor"],
        angle_preds: List["torch.Tensor"],
        points: List["torch.Tensor"],
        device: "torch.device",
    ) -> List[Dict[str, Any]]:
        num_imgs = cls_scores[0].size(0)
        outputs: List[Dict[str, Any]] = []
        for img_idx in range(num_imgs):
            boxes, scores, labels = rotated_rtmdet_decode_pre_nms(
                [c[img_idx] for c in cls_scores],
                [b[img_idx] for b in bbox_preds],
                [a[img_idx] for a in angle_preds],
                points,
                self.num_classes,
                self.score_threshold,
                self.nms_pre,
                self.min_bbox_size,
            )
            if boxes.numel() == 0:
                outputs.append(
                    {
                        "rboxes": [],
                        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                    }
                )
                continue
            boxes, scores, labels = _apply_fcos_nms(
                boxes,
                scores,
                labels,
                iou_threshold=self.final_nms_iou_threshold,
                max_detections_per_image=self.max_detections_per_image,
                final_nms_use_cpu=self.final_nms_use_cpu,
                class_agnostic=self.nms_class_agnostic,
            )
            outputs.append(
                {
                    "rboxes": tensor_to_rboxes(normalize_boxes_to_le90(boxes)),
                    "labels": labels.to(dtype=torch.int64),
                    "scores": scores,
                }
            )
        return outputs


def rotated_rtmdet_kwargs_from_config(model_cfg: Any) -> Dict[str, Any]:
    """``RotatedRTMDet`` kwargs from ``config.model`` (shared by training and checkpoint loading).

    ``frozen_stages`` follows MMDet CSPNeXt: ``k`` freezes the stem and stages ``1..k``;
    ``null`` → ``-1`` (all trainable). Callers add ``num_classes`` and
    ``roi_class_weights``; checkpoint loading also sets ``pretrained_backbone=False``.
    """
    m = model_cfg
    frozen = getattr(m, "frozen_stages", None)
    return dict(
        variant=getattr(m, "rtmdet_variant", "tiny"),
        pretrained_backbone=bool(getattr(m, "pretrained_backbone", True)),
        pretrained_weights=getattr(m, "rtmdet_pretrained_weights", "coco"),
        input_bgr=getattr(m, "rtmdet_input_bgr", None),
        frozen_stages=-1 if frozen is None else int(frozen),
        fpn_strides=getattr(m, "fpn_strides", None) or [8, 16, 32],
        stacked_convs=getattr(m, "rtmdet_stacked_convs", 2),
        share_conv=getattr(m, "rtmdet_share_conv", True),
        pred_kernel_size=getattr(m, "rtmdet_pred_kernel_size", 1),
        exp_on_reg=getattr(m, "rtmdet_exp_on_reg", False),
        assigner_topk=getattr(m, "rtmdet_assigner_topk", 13),
        assigner_iou_weight=getattr(m, "rtmdet_assigner_iou_weight", 3.0),
        assigner_soft_center_radius=getattr(m, "rtmdet_assigner_soft_center_radius", 3.0),
        qfl_beta=getattr(m, "rtmdet_qfl_beta", 2.0),
        box_reg_weight=getattr(m, "box_reg_weight", 2.0),
        score_threshold=getattr(m, "inference_pre_nms_score_threshold", 0.05),
        final_nms_iou_threshold=getattr(m, "final_nms_iou_threshold", 0.1),
        max_detections_per_image=getattr(m, "max_detections_per_image", 2000),
        nms_pre=getattr(m, "rtmdet_nms_pre", 2000),
        final_nms_iou_schedule_epochs=getattr(m, "final_nms_iou_schedule_epochs", None),
        final_nms_iou_schedule_values=getattr(m, "final_nms_iou_schedule_values", None),
        final_nms_use_cpu=getattr(m, "final_nms_use_cpu", False),
        nms_class_agnostic=getattr(m, "nms_class_agnostic", False),
    )


def rotated_rtmdet_decode_pre_nms(
    cls_per_level: Sequence["torch.Tensor"],
    bbox_per_level: Sequence["torch.Tensor"],
    angle_per_level: Sequence["torch.Tensor"],
    points: Sequence["torch.Tensor"],
    num_classes: int,
    score_threshold: float,
    nms_pre: int,
    min_bbox_size: float,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """One image: MMDet ``filter_scores_and_topk`` per level, then decode.

    Returns ``boxes [N, 5]``, ``scores [N]``, ``labels [N]`` (**1-indexed**).
    """
    boxes_l: List[torch.Tensor] = []
    scores_l: List[torch.Tensor] = []
    labels_l: List[torch.Tensor] = []
    for cls_map, bbox_map, angle_map, pts in zip(
        cls_per_level, bbox_per_level, angle_per_level, points
    ):
        scores = cls_map.permute(1, 2, 0).reshape(-1, num_classes).float().sigmoid()
        ltrb = bbox_map.permute(1, 2, 0).reshape(-1, 4).float()
        angle = angle_map.permute(1, 2, 0).reshape(-1, 1).float()
        valid = scores > float(score_threshold)
        cand_scores = scores[valid]
        cand_idx = valid.nonzero(as_tuple=False)  # [K, 2] (prior, class)
        if cand_scores.numel() == 0:
            continue
        k = min(int(nms_pre), cand_scores.numel())
        cand_scores, order = cand_scores.sort(descending=True)
        cand_scores = cand_scores[:k]
        prior_idx, cls_idx = cand_idx[order[:k]].unbind(dim=1)
        decoded = decode_rtmdet_boxes(pts[prior_idx], ltrb[prior_idx], angle[prior_idx])
        keep = (decoded[:, 2] > float(min_bbox_size)) & (decoded[:, 3] > float(min_bbox_size))
        keep = keep & torch.isfinite(decoded).all(dim=1)
        boxes_l.append(decoded[keep])
        scores_l.append(cand_scores[keep])
        labels_l.append(cls_idx[keep] + 1)

    device = cls_per_level[0].device
    if not boxes_l:
        return (
            torch.zeros((0, 5), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.int64, device=device),
        )
    return (
        torch.cat(boxes_l, dim=0),
        torch.cat(scores_l, dim=0),
        torch.cat(labels_l, dim=0).to(dtype=torch.int64),
    )


__all__ = [
    "RotatedRTMDet",
    "RotatedRTMDetSepBNHead",
    "RTMDET_DOTA_URLS",
    "compute_rotated_rtmdet_loss",
    "decode_rtmdet_boxes",
    "dynamic_soft_label_assign",
    "generate_rtmdet_priors",
    "pairwise_rbox_iou_exact",
    "points_in_rboxes",
    "quality_focal_loss",
    "rotated_rtmdet_decode_pre_nms",
    "rotated_rtmdet_kwargs_from_config",
]
