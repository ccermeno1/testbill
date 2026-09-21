"""RTMDet-R (rotated RTMDet) in plain PyTorch.

Module names follow mmdet/mmrotate so the official checkpoints load with
``strict=True``:

* ``backbone`` - CSPNeXt (mmdet ``CSPNeXt``)
* ``neck``     - CSPNeXtPAFPN
* ``bbox_head`` - RotatedRTMDetSepBNHead (shared convs, per-level BN)

Boxes everywhere are ``(cx, cy, w, h, angle_rad)`` in the ``le90`` convention.
"""
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import DynamicSoftLabelAssigner
from .boxes import distance2obb, norm_angle, regularize_le90
from .losses import quality_focal_loss, rotated_iou_loss
from .ops import batched_nms_rotated

# ----------------------------------------------------------------------------- blocks


class ConvModule(nn.Module):
    """conv (no bias) -> BN -> SiLU, named like mmcv's ConvModule."""

    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)  # SyncBN in the config -> plain BN (eps 1e-5, momentum 0.1)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):

    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__()
        self.depthwise_conv = ConvModule(cin, cin, k, stride, padding, groups=cin)
        self.pointwise_conv = ConvModule(cin, cout, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        return x * self.act(self.fc(self.global_avgpool(x)))


class CSPNeXtBlock(nn.Module):

    def __init__(self, cin, cout, expansion=1.0, add_identity=True, kernel_size=5):
        super().__init__()
        hidden = int(cout * expansion)
        self.conv1 = ConvModule(cin, hidden, 3, 1, 1)
        self.conv2 = DepthwiseSeparableConvModule(hidden, cout, kernel_size, 1, kernel_size // 2)
        self.add_identity = add_identity and cin == cout

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):

    def __init__(self, cin, cout, expand_ratio=0.5, num_blocks=1, add_identity=True,
                 channel_attention=False):
        super().__init__()
        mid = int(cout * expand_ratio)
        self.main_conv = ConvModule(cin, mid, 1)
        self.short_conv = ConvModule(cin, mid, 1)
        self.final_conv = ConvModule(2 * mid, cout, 1)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid, mid, 1.0, add_identity) for _ in range(num_blocks)])
        self.channel_attention = channel_attention
        if channel_attention:
            self.attention = ChannelAttention(2 * mid)

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):

    def __init__(self, cin, cout, kernel_sizes=(5, 9, 13)):
        super().__init__()
        mid = cin // 2
        self.conv1 = ConvModule(cin, mid, 1)
        self.poolings = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in kernel_sizes])
        self.conv2 = ConvModule(mid * (len(kernel_sizes) + 1), cout, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [p(x) for p in self.poolings], dim=1)
        return self.conv2(x)


# ----------------------------------------------------------------------------- backbone / neck


class CSPNeXt(nn.Module):
    # (in, out, num_blocks, add_identity, use_spp) for the P5 arch
    ARCH = [[64, 128, 3, True, False], [128, 256, 6, True, False],
            [256, 512, 6, True, False], [512, 1024, 3, False, True]]

    def __init__(self, deepen_factor=1.0, widen_factor=1.0, out_indices=(2, 3, 4),
                 expand_ratio=0.5, channel_attention=True):
        super().__init__()
        self.out_indices = out_indices
        c0 = int(self.ARCH[0][0] * widen_factor)
        self.stem = nn.Sequential(
            ConvModule(3, c0 // 2, 3, 2, 1),
            ConvModule(c0 // 2, c0 // 2, 3, 1, 1),
            ConvModule(c0 // 2, c0, 3, 1, 1))
        self.layers = ['stem']
        self.out_channels = []
        for i, (cin, cout, n, add_identity, use_spp) in enumerate(self.ARCH):
            cin, cout = int(cin * widen_factor), int(cout * widen_factor)
            n = max(round(n * deepen_factor), 1)
            stage = [ConvModule(cin, cout, 3, 2, 1)]
            if use_spp:
                stage.append(SPPBottleneck(cout, cout))
            stage.append(CSPLayer(cout, cout, expand_ratio, n, add_identity, channel_attention))
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')
            if i + 1 in out_indices:
                self.out_channels.append(cout)

    def forward(self, x):
        outs = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class CSPNeXtPAFPN(nn.Module):

    def __init__(self, in_channels: Sequence[int], out_channels: int, num_csp_blocks=3, expand_ratio=0.5):
        super().__init__()
        self.in_channels = list(in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(ConvModule(in_channels[idx], in_channels[idx - 1], 1))
            self.top_down_blocks.append(CSPLayer(in_channels[idx - 1] * 2, in_channels[idx - 1], expand_ratio,
                                                 num_csp_blocks, add_identity=False))
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(ConvModule(in_channels[idx], in_channels[idx], 3, 2, 1))
            self.bottom_up_blocks.append(CSPLayer(in_channels[idx] * 2, in_channels[idx + 1], expand_ratio,
                                                  num_csp_blocks, add_identity=False))
        self.out_convs = nn.ModuleList([ConvModule(c, out_channels, 3, 1, 1) for c in in_channels])

    def forward(self, inputs):
        n = len(self.in_channels)
        inner_outs = [inputs[-1]]
        for idx in range(n - 1, 0, -1):
            feat_high = self.reduce_layers[n - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_high
            inner = self.top_down_blocks[n - 1 - idx](torch.cat([self.upsample(feat_high), inputs[idx - 1]], 1))
            inner_outs.insert(0, inner)
        outs = [inner_outs[0]]
        for idx in range(n - 1):
            down = self.downsamples[idx](outs[-1])
            outs.append(self.bottom_up_blocks[idx](torch.cat([down, inner_outs[idx + 1]], 1)))
        return tuple(conv(o) for conv, o in zip(self.out_convs, outs))


# ----------------------------------------------------------------------------- head


class RotatedRTMDetSepBNHead(nn.Module):
    """Shared 3x3 convs (weights tied across levels), separate BN per level,
    1x1 prediction convs per level for cls / (l, t, r, b) / angle."""

    def __init__(self, num_classes, in_channels, feat_channels, strides=(8, 16, 32), stacked_convs=2,
                 exp_on_reg=False):
        super().__init__()
        self.num_classes = num_classes
        self.strides = list(strides)
        self.exp_on_reg = exp_on_reg
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_ang = nn.ModuleList()
        for _ in self.strides:
            cls_convs, reg_convs = nn.ModuleList(), nn.ModuleList()
            for i in range(stacked_convs):
                chn = in_channels if i == 0 else feat_channels
                cls_convs.append(ConvModule(chn, feat_channels, 3, 1, 1))
                reg_convs.append(ConvModule(chn, feat_channels, 3, 1, 1))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.rtm_cls.append(nn.Conv2d(feat_channels, num_classes, 1))
            self.rtm_reg.append(nn.Conv2d(feat_channels, 4, 1))
            self.rtm_ang.append(nn.Conv2d(feat_channels, 1, 1))
        # share_conv=True: the conv weights are the same object on every level
        for n in range(1, len(self.strides)):
            for i in range(stacked_convs):
                self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        bias_cls = -math.log((1 - 0.01) / 0.01)
        for m in self.rtm_cls:
            nn.init.constant_(m.bias, bias_cls)

    def forward(self, feats):
        cls_scores, bbox_preds, angle_preds = [], [], []
        for idx, (x, stride) in enumerate(zip(feats, self.strides)):
            cls_feat, reg_feat = x, x
            for layer in self.cls_convs[idx]:
                cls_feat = layer(cls_feat)
            for layer in self.reg_convs[idx]:
                reg_feat = layer(reg_feat)
            reg = self.rtm_reg[idx](reg_feat)
            reg = reg.exp() * stride if self.exp_on_reg else reg * stride
            cls_scores.append(self.rtm_cls[idx](cls_feat))
            bbox_preds.append(reg)
            angle_preds.append(self.rtm_ang[idx](reg_feat))
        return cls_scores, bbox_preds, angle_preds


# ----------------------------------------------------------------------------- detector

PRESETS = {
    # deepen, widen, neck csp blocks, head channels, exp_on_reg
    'tiny': dict(deepen=0.167, widen=0.375, num_csp_blocks=1, exp_on_reg=False),
    's': dict(deepen=0.33, widen=0.5, num_csp_blocks=1, exp_on_reg=False),
    'm': dict(deepen=0.67, widen=0.75, num_csp_blocks=2, exp_on_reg=True),
    'l': dict(deepen=1.0, widen=1.0, num_csp_blocks=3, exp_on_reg=True),
}


class RTMDetR(nn.Module):
    """Rotated RTMDet detector with loss and post-processing.

    Input images are **BGR** uint8/float ``(B, 3, H, W)`` in 0..255 (as loaded
    by OpenCV); normalisation happens inside :meth:`preprocess` with the DOTA
    checkpoint statistics.
    """
    MEAN = (103.53, 116.28, 123.675)  # BGR
    STD = (57.375, 57.12, 58.395)

    def __init__(self, num_classes=1, size='tiny', strides=(8, 16, 32),
                 assigner_topk=13, loss_bbox_weight=2.0, loss_cls_weight=1.0):
        super().__init__()
        p = PRESETS[size]
        self.num_classes = num_classes
        self.strides = list(strides)
        self.backbone = CSPNeXt(p['deepen'], p['widen'])
        neck_in = self.backbone.out_channels
        neck_out = neck_in[0]
        self.neck = CSPNeXtPAFPN(neck_in, neck_out, p['num_csp_blocks'])
        self.bbox_head = RotatedRTMDetSepBNHead(num_classes, neck_out, neck_out, strides, 2, p['exp_on_reg'])
        self.assigner = DynamicSoftLabelAssigner(topk=assigner_topk)
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_cls_weight = loss_cls_weight
        self.register_buffer('pix_mean', torch.tensor(self.MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('pix_std', torch.tensor(self.STD).view(1, 3, 1, 1), persistent=False)

    # ---- forward pieces
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return (images.float() - self.pix_mean) / self.pix_std

    def extract_feat(self, images: torch.Tensor):
        return self.neck(self.backbone(self.preprocess(images)))

    def forward(self, images: torch.Tensor):
        """Raw head outputs (lists over levels of cls / reg / angle maps)."""
        return self.bbox_head(self.extract_feat(images))

    def grid_priors(self, featmap_sizes, device, dtype=torch.float32) -> List[torch.Tensor]:
        """Per level ``(H*W, 4)`` = (x, y, stride, stride) with offset 0 (MlvlPointGenerator)."""
        priors = []
        for (h, w), s in zip(featmap_sizes, self.strides):
            ys = (torch.arange(h, device=device, dtype=dtype) * s)[:, None].expand(h, w)
            xs = (torch.arange(w, device=device, dtype=dtype) * s)[None, :].expand(h, w)
            st = torch.full((h * w, 2), float(s), device=device, dtype=dtype)
            priors.append(torch.cat([xs.reshape(-1, 1), ys.reshape(-1, 1), st], 1))
        return priors

    def _flatten(self, cls_scores, bbox_preds, angle_preds):
        """Concatenate levels: (B, N, C) scores, (B, N, 5) decoded boxes, (N, 4) priors."""
        b = cls_scores[0].shape[0]
        featmap_sizes = [c.shape[-2:] for c in cls_scores]
        priors = self.grid_priors(featmap_sizes, cls_scores[0].device, cls_scores[0].dtype)
        flat_scores = torch.cat([c.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes) for c in cls_scores], 1)
        decoded = []
        for prior, reg, ang in zip(priors, bbox_preds, angle_preds):
            reg = reg.permute(0, 2, 3, 1).reshape(b, -1, 4)
            ang = ang.permute(0, 2, 3, 1).reshape(b, -1, 1)
            decoded.append(distance2obb(prior[None, :, :2], torch.cat([reg, ang], -1)))
        return flat_scores, torch.cat(decoded, 1), torch.cat(priors, 0)

    # ---- training
    def loss(self, images: torch.Tensor, gt_boxes: List[torch.Tensor], gt_labels: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Args: images (B,3,H,W); per image gt_boxes (n,5) le90 and gt_labels (n,)."""
        cls_scores, bbox_preds, angle_preds = self.forward(images)
        flat_scores, flat_boxes, priors = self._flatten(cls_scores, bbox_preds, angle_preds)
        b, n = flat_scores.shape[:2]

        labels = flat_scores.new_full((b, n), self.num_classes, dtype=torch.long)
        assign_metrics = flat_scores.new_zeros((b, n))
        bbox_targets = flat_scores.new_zeros((b, n, 5))
        pos_mask = torch.zeros((b, n), dtype=torch.bool, device=flat_scores.device)
        for i in range(b):
            gtb = regularize_le90(gt_boxes[i].to(flat_scores.dtype))
            res = self.assigner.assign(flat_scores[i].detach(), flat_boxes[i].detach(), priors, gtb, gt_labels[i])
            pos = res['assigned_gt_inds'] > 0
            if pos.any():
                gt_idx = res['assigned_gt_inds'][pos] - 1
                labels[i, pos] = gt_labels[i][gt_idx]
                bbox_targets[i, pos] = gtb[gt_idx]
                assign_metrics[i, pos] = res['max_overlaps'][pos]
                pos_mask[i, pos] = True

        # classification: QFL with the IoU as soft target, normalised by the sum of the metrics
        loss_cls = quality_focal_loss(flat_scores.reshape(-1, self.num_classes), labels.reshape(-1),
                                      assign_metrics.reshape(-1), beta=2.0).sum()
        cls_avg = assign_metrics.sum().clamp(min=1.0)
        loss_cls = self.loss_cls_weight * loss_cls / cls_avg

        # regression: rotated IoU loss on the decoded boxes, weighted by the assign metric
        if pos_mask.any():
            w = assign_metrics[pos_mask]
            loss_bbox = rotated_iou_loss(flat_boxes[pos_mask], bbox_targets[pos_mask], mode='linear')
            loss_bbox = self.loss_bbox_weight * (loss_bbox * w).sum() / w.sum().clamp(min=1.0)
        else:
            loss_bbox = flat_boxes.sum() * 0
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    # ---- inference
    @torch.no_grad()
    def predict(self, images: torch.Tensor, score_thr=0.05, nms_pre=2000, nms_iou=0.1, max_per_img=2000,
                min_bbox_size=0) -> List[Dict[str, torch.Tensor]]:
        """Returns per image dict(boxes (k,5) le90 in input-pixel coords, scores (k,), labels (k,))."""
        cls_scores, bbox_preds, angle_preds = self.forward(images)
        b = cls_scores[0].shape[0]
        featmap_sizes = [c.shape[-2:] for c in cls_scores]
        priors = self.grid_priors(featmap_sizes, cls_scores[0].device, cls_scores[0].dtype)
        results = []
        for i in range(b):
            lvl_boxes, lvl_scores, lvl_labels = [], [], []
            for prior, cls, reg, ang in zip(priors, cls_scores, bbox_preds, angle_preds):
                scores = cls[i].permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
                reg = reg[i].permute(1, 2, 0).reshape(-1, 4)
                ang = ang[i].permute(1, 2, 0).reshape(-1, 1)
                # filter_scores_and_topk: threshold, then keep the nms_pre best (prior, class) pairs
                valid = scores > score_thr
                sc = scores[valid]
                idx = valid.nonzero()
                if sc.numel() > nms_pre:
                    sc, order = sc.sort(descending=True)
                    sc, idx = sc[:nms_pre], idx[order[:nms_pre]]
                keep, lab = idx.unbind(1)
                boxes = distance2obb(prior[keep, :2], torch.cat([reg[keep], ang[keep]], -1))
                lvl_boxes.append(boxes)
                lvl_scores.append(sc)
                lvl_labels.append(lab)
            boxes = torch.cat(lvl_boxes)
            scores = torch.cat(lvl_scores)
            labels = torch.cat(lvl_labels)
            if min_bbox_size >= 0 and boxes.numel():
                ok = (boxes[:, 2] > min_bbox_size) & (boxes[:, 3] > min_bbox_size)
                boxes, scores, labels = boxes[ok], scores[ok], labels[ok]
            if boxes.numel():
                _, keep = batched_nms_rotated(boxes, scores, labels, nms_iou)
                keep = keep[:max_per_img]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            results.append(dict(boxes=boxes, scores=scores, labels=labels))
        return results
