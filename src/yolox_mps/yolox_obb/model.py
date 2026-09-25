"""YOLOX-OBB detectors in plain PyTorch.

:class:`OBBDetector` holds everything that does not depend on the network: SimOTA
assignment, the YOLOX-OBB losses, decoding and rotated NMS. The networks only
implement ``forward`` (raw head maps): :class:`YOLOXOBB` here and the official
YOLOX variants in ``official.py``.

YOLOX-OBB-s (DDGRCF/YOLOX_OBB, ``configs/modules/yoloxs_obb.yaml``): the YAML network
is written out explicitly. Layer ``i`` of the YAML is ``self.model[i]`` and every
sub-module keeps its original name, so the YOLOX_OBB checkpoints (``ckpt['model']``)
load with matching keys:

* 0-9   backbone: YOLOv5-s style (Conv 6x6/2, C3, SPP), ReLU
* 10-23 PAN neck -> P3 (17), P4 (20), P5 (23)
* 24-32 decoupled head stems: 1x1 lateral + one 3x3 conv per cls / reg branch
* 33    ``OBBDetectX``: ``cls_preds``, ``reg_preds`` (x, y, w, h, angle), ``obj_preds``

Input images are **BGR** ``(B, 3, H, W)`` in 0..255 without normalisation, as in
YOLOX. Boxes everywhere are ``(cx, cy, w, h, angle_rad)`` with the angle clockwise
in image coordinates (the convention of ``boxes.py``). YOLOX_OBB / BboxToolkit
measure the angle counter-clockwise, so the angle channel is negated at decode time.
"""
import math
from typing import Dict, List

import torch
import torch.nn as nn

from .assigner import SimOTAAssigner
from .boxes import regularize_mintheta
from .losses import bce_loss, l1_loss, poly_iou_loss
from .ops import batched_nms_rotated

# ----------------------------------------------------------------------------- blocks


class Conv(nn.Module):
    """conv (no bias) -> BN -> ReLU (``Conv`` with ``act_func=nn.ReLU``)."""

    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class OBBDetectX(nn.Module):
    """Prediction convs of the decoupled head (1x1 on the cls / reg stems of each level)."""

    def __init__(self, num_classes, in_channels=128, num_levels=3, reg_dim=5):
        super().__init__()
        self.cls_preds = nn.ModuleList([nn.Conv2d(in_channels, num_classes, 1) for _ in range(num_levels)])
        self.reg_preds = nn.ModuleList([nn.Conv2d(in_channels, reg_dim, 1) for _ in range(num_levels)])
        self.obj_preds = nn.ModuleList([nn.Conv2d(in_channels, 1, 1) for _ in range(num_levels)])

    def forward(self, cls_feats, reg_feats):
        cls, reg, obj = [], [], []
        for k, (cx, rx) in enumerate(zip(cls_feats, reg_feats)):
            cls.append(self.cls_preds[k](cx))
            reg.append(self.reg_preds[k](rx))
            obj.append(self.obj_preds[k](rx))
        return cls, reg, obj




# ----------------------------------------------------------------------------- shared detector


class OBBDetector(nn.Module):
    """YOLOX-OBB training and inference on top of any network whose ``forward`` returns the raw
    head maps ``(cls, reg, obj)``: SimOTA assignment, losses, decoding and rotated NMS."""

    arch = None  # registry name, stored in the checkpoints

    # tiny-box filter of OBBTrainTransform (short_wh_thre / long_wh_thre)
    MIN_SHORT_SIDE = 3.0
    MIN_LONG_SIDE = 6.0

    def __init__(self, num_classes=1, strides=(8, 16, 32), loss_reg_weight=5.0, loss_obj_weight=1.0,
                 loss_cls_weight=1.0, loss_l1_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.strides = list(strides)
        self.assigner = SimOTAAssigner()
        self.loss_reg_weight = loss_reg_weight
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_l1_weight = loss_l1_weight
        self.use_l1 = False  # switched on for the last (no-mosaic) epochs, as YOLOX does

    def _init_weights(self, cls_preds, obj_preds, prior_prob=0.01):
        """YOLOX init: BN eps 1e-3 / momentum 0.03, prior bias on cls and obj."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        bias = -math.log((1 - prior_prob) / prior_prob)
        for conv in list(cls_preds) + list(obj_preds):
            nn.init.constant_(conv.bias, bias)

    def forward(self, images: torch.Tensor):
        """Raw head outputs: lists over levels of cls (B, C, H, W), reg (B, 5, H, W), obj (B, 1, H, W) logits."""
        raise NotImplementedError

    def grid_priors(self, featmap_sizes, device, dtype=torch.float32) -> torch.Tensor:
        """``(N, 3)`` = grid x, grid y (cell units) and stride for every location of every level."""
        priors = []
        for (h, w), s in zip(featmap_sizes, self.strides):
            ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                    torch.arange(w, device=device, dtype=dtype), indexing='ij')
            st = torch.full((h * w, 1), float(s), device=device, dtype=dtype)
            priors.append(torch.cat([xs.reshape(-1, 1), ys.reshape(-1, 1), st], 1))
        return torch.cat(priors, 0)

    def _flatten(self, cls_scores, reg_preds, obj_preds):
        """Concatenate levels: (B, N, C) cls logits, (B, N, 5) raw reg, (B, N) obj logits, (N, 3) priors."""
        b = cls_scores[0].shape[0]
        priors = self.grid_priors([c.shape[-2:] for c in cls_scores], cls_scores[0].device, cls_scores[0].dtype)
        flat = lambda maps, ch: torch.cat([x.permute(0, 2, 3, 1).reshape(b, -1, ch) for x in maps], 1)  # noqa: E731
        return (flat(cls_scores, self.num_classes), flat(reg_preds, 5), flat(obj_preds, 1)[..., 0], priors)

    @staticmethod
    def decode(raw: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """Raw regression ``(..., N, 5)`` -> boxes: ``(xy + grid) * stride``, ``exp(wh) * stride``, angle."""
        stride = priors[:, 2:3]
        xy = (raw[..., :2] + priors[:, :2]) * stride
        wh = raw[..., 2:4].exp() * stride
        return torch.cat([xy, wh, -raw[..., 4:5]], -1)

    # ---- training
    def loss(self, images: torch.Tensor, gt_boxes: List[torch.Tensor], gt_labels: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Args: images (B,3,H,W); per image gt_boxes (n,5) and gt_labels (n,). Returns the weighted losses."""
        cls_scores, reg_preds, obj_preds = self.forward(images)
        flat_cls, flat_reg, flat_obj, priors = self._flatten(cls_scores, reg_preds, obj_preds)
        flat_boxes = self.decode(flat_reg, priors)
        b, n = flat_obj.shape

        obj_targets = torch.zeros_like(flat_obj)
        fg_masks, cls_targets, box_targets = [], [], []
        for i in range(b):
            gtb = regularize_mintheta(gt_boxes[i].to(flat_boxes.dtype))
            gtl = gt_labels[i]
            keep = ((gtb[:, 2:4].min(1).values > self.MIN_SHORT_SIDE) &
                    (gtb[:, 2:4].max(1).values > self.MIN_LONG_SIDE))
            gtb, gtl = gtb[keep], gtl[keep]
            res = self.assigner.assign(flat_cls[i].detach(), flat_obj[i].detach(), flat_boxes[i].detach(),
                                       priors, gtb, gtl)
            fg = res['fg_mask']
            fg_masks.append(fg)
            obj_targets[i, fg] = 1.0
            idx = res['matched_gt_inds']
            cls_targets.append(nn.functional.one_hot(gtl[idx].long(), self.num_classes).to(flat_cls.dtype)
                               * res['matched_ious'][:, None].to(flat_cls.dtype))
            box_targets.append(gtb[idx])

        fg_mask = torch.stack(fg_masks)  # (B, N)
        cls_targets = torch.cat(cls_targets)
        box_targets = torch.cat(box_targets)
        num_fg = max(int(fg_mask.sum()), 1)

        loss_obj = self.loss_obj_weight * bce_loss(flat_obj, obj_targets).sum() / num_fg
        if fg_mask.any():
            loss_cls = self.loss_cls_weight * bce_loss(flat_cls[fg_mask], cls_targets).sum() / num_fg
            loss_reg = self.loss_reg_weight * poly_iou_loss(flat_boxes[fg_mask], box_targets).sum() / num_fg
        else:
            loss_cls = loss_reg = flat_cls.sum() * 0
        losses = dict(loss_obj=loss_obj, loss_cls=loss_cls, loss_reg=loss_reg)
        if self.use_l1:
            if fg_mask.any():
                pri = priors[None].expand(b, -1, -1)[fg_mask]
                stride = pri[:, 2:3]
                l1_target = torch.cat([box_targets[:, :2] / stride - pri[:, :2],
                                       torch.log(box_targets[:, 2:4] / stride + 1e-8),
                                       -box_targets[:, 4:5]], 1)
                losses['loss_l1'] = self.loss_l1_weight * l1_loss(flat_reg[fg_mask], l1_target).sum() / num_fg
            else:
                losses['loss_l1'] = flat_reg.sum() * 0
        return losses

    # ---- inference
    @torch.no_grad()
    def predict(self, images: torch.Tensor, score_thr=0.05, nms_pre=2000, nms_iou=0.1, max_per_img=2000,
                min_bbox_size=0) -> List[Dict[str, torch.Tensor]]:
        """Returns per image dict(boxes (k,5) in input-pixel coords, scores (k,), labels (k,)).

        As in ``obbpostprocess``: score = obj * best class probability, threshold, class-aware rotated NMS.
        """
        cls_scores, reg_preds, obj_preds = self.forward(images)
        flat_cls, flat_reg, flat_obj, priors = self._flatten(cls_scores, reg_preds, obj_preds)
        flat_boxes = self.decode(flat_reg, priors)
        results = []
        for i in range(flat_cls.shape[0]):
            cls_prob, labels = flat_cls[i].sigmoid().max(1)
            scores = cls_prob * flat_obj[i].sigmoid()
            valid = scores > score_thr
            boxes, scores, labels = flat_boxes[i][valid], scores[valid], labels[valid]
            if scores.numel() > nms_pre:
                scores, order = scores.sort(descending=True)
                scores, order = scores[:nms_pre], order[:nms_pre]
                boxes, labels = boxes[order], labels[order]
            if min_bbox_size >= 0 and boxes.numel():
                ok = (boxes[:, 2] > min_bbox_size) & (boxes[:, 3] > min_bbox_size)
                boxes, scores, labels = boxes[ok], scores[ok], labels[ok]
            if boxes.numel():
                _, keep = batched_nms_rotated(boxes, scores, labels, nms_iou)
                keep = keep[:max_per_img]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            results.append(dict(boxes=boxes, scores=scores, labels=labels))
        return results


# ----------------------------------------------------------------------------- YOLOX_OBB-s (DDGRCF)

PRESETS = {
    # YOLOX_OBB only ships the "s" OBB config: depth_multiple 0.33, width_multiple 0.5
    's': dict(width=0.5, depth=0.33),
}


def _ch(c, width):
    return math.ceil(c * width / 8) * 8  # make_divisible(c * width, 8)


def _n(n, depth):
    return max(round(n * depth), 1) if n > 1 else n


class YOLOXOBB(OBBDetector):
    """YOLOX-OBB-s of DDGRCF/YOLOX_OBB (YOLOv5-s style backbone, ReLU)."""

    arch = 'ddgrcf_s'

    def __init__(self, num_classes=1, size='s', strides=(8, 16, 32), prior_prob=0.01, **loss_kw):
        super().__init__(num_classes, strides, **loss_kw)
        p = PRESETS[size]
        w, d = p['width'], p['depth']
        c64, c128, c256, c512, c1024 = (_ch(c, w) for c in (64, 128, 256, 512, 1024))
        self.model = nn.ModuleList([
            Conv(3, c64, 6, 2, 2),                          # 0  P1/2
            Conv(c64, c128, 3, 2),                          # 1  P2/4
            C3(c128, c128, _n(3, d)),                       # 2
            Conv(c128, c256, 3, 2),                         # 3  P3/8
            C3(c256, c256, _n(9, d)),                       # 4
            Conv(c256, c512, 3, 2),                         # 5  P4/16
            C3(c512, c512, _n(9, d)),                       # 6
            Conv(c512, c1024, 3, 2),                        # 7  P5/32
            SPP(c1024, c1024, (5, 9, 13)),                  # 8
            C3(c1024, c1024, _n(3, d), shortcut=False),     # 9
            Conv(c1024, c512, 1, 1),                        # 10
            nn.Upsample(None, 2, 'nearest'),                # 11
            nn.Identity(),                                  # 12 concat(11, 6)
            C3(2 * c512, c512, _n(3, d), shortcut=False),   # 13
            Conv(c512, c256, 1, 1),                         # 14
            nn.Upsample(None, 2, 'nearest'),                # 15
            nn.Identity(),                                  # 16 concat(15, 4)
            C3(2 * c256, c256, _n(3, d), shortcut=False),   # 17 P3 out
            Conv(c256, c256, 3, 2),                         # 18
            nn.Identity(),                                  # 19 concat(18, 14)
            C3(2 * c256, c512, _n(3, d), shortcut=False),   # 20 P4 out
            Conv(c512, c512, 3, 2),                         # 21
            nn.Identity(),                                  # 22 concat(21, 10)
            C3(2 * c512, c1024, _n(3, d), shortcut=False),  # 23 P5 out
            Conv(c256, c256, 1, 1),                         # 24 lateral P3
            Conv(c512, c256, 1, 1),                         # 25 lateral P4
            Conv(c1024, c256, 1, 1),                        # 26 lateral P5
            Conv(c256, c256, 3, 1),                         # 27 cls stem P3
            Conv(c256, c256, 3, 1),                         # 28 reg stem P3
            Conv(c256, c256, 3, 1),                         # 29 cls stem P4
            Conv(c256, c256, 3, 1),                         # 30 reg stem P4
            Conv(c256, c256, 3, 1),                         # 31 cls stem P5
            Conv(c256, c256, 3, 1),                         # 32 reg stem P5
            OBBDetectX(num_classes, c256, len(self.strides)),  # 33
        ])
        head = self.model[33]
        self._init_weights(head.cls_preds, head.obj_preds, prior_prob)

    def forward(self, images: torch.Tensor):
        """Raw head outputs: lists over levels of cls (B, C, H, W), reg (B, 5, H, W), obj (B, 1, H, W) logits."""
        m = self.model
        x = images.float()
        x = m[1](m[0](x))
        x = m[2](x)
        p3 = m[4](m[3](x))
        p4 = m[6](m[5](p3))
        x = m[9](m[8](m[7](p4)))
        h10 = m[10](x)
        x = m[13](torch.cat([m[11](h10), p4], 1))
        h14 = m[14](x)
        out3 = m[17](torch.cat([m[15](h14), p3], 1))
        out4 = m[20](torch.cat([m[18](out3), h14], 1))
        out5 = m[23](torch.cat([m[21](out4), h10], 1))
        lat = [m[24](out3), m[25](out4), m[26](out5)]
        cls_feats = [m[27](lat[0]), m[29](lat[1]), m[31](lat[2])]
        reg_feats = [m[28](lat[0]), m[30](lat[1]), m[32](lat[2])]
        return m[33](cls_feats, reg_feats)
