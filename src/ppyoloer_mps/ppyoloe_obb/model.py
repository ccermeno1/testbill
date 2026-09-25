"""
PP-YOLOE-R en PyTorch puro (CPU / CUDA / MPS).

Port fiel de PaddleDetection release/2.8:
  backbone  ppdet/modeling/backbones/cspresnet.py  (CSPResNet)
  neck      ppdet/modeling/necks/custom_pan.py     (CustomCSPPAN)
  head      ppdet/modeling/heads/ppyoloe_r_head.py (PPYOLOERHead)

Los nombres de los submodulos coinciden con los de Paddle, asi que el state_dict se
convierte 1:1 (ver checkpoint.py): solo cambian `bn._mean`/`bn._variance` por
`bn.running_mean`/`bn.running_var`.

Notas de equivalencia entre frameworks:
  - `swish` de Paddle == `SiLU` de PyTorch.
  - `hardsigmoid` de Paddle (slope=1/6, offset=0.5) == `F.hardsigmoid` de PyTorch.
  - `F.interpolate(..., scale_factor=2)` usa modo 'nearest' por defecto en ambos.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boxes import rbox2poly

__all__ = ["PPYOLOER", "build_ppyoloe_r"]


def _act(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    return {"swish": nn.SiLU, "silu": nn.SiLU, "relu": nn.ReLU, "hardsigmoid": nn.Hardsigmoid}[name]()


class ConvBNLayer(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, groups=1, padding=0, act=None):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, filter_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = _act(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepVggBlock(nn.Module):
    """Dos ramas (3x3 y 1x1) sumadas con un peso `alpha` aprendido; sin fusionar."""

    def __init__(self, ch_in, ch_out, act="relu", alpha=False):
        super().__init__()
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act = _act(act)
        if alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("alpha", None)

    def forward(self, x):
        if self.alpha is not None:
            y = self.conv1(x) + self.alpha * self.conv2(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu", shortcut=True, use_alpha=False):
        super().__init__()
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act, alpha=use_alpha)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


class EffectiveSELayer(nn.Module):
    def __init__(self, channels, act="hardsigmoid"):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = _act(act)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        return x * self.act(self.fc(x_se))


class CSPResStage(nn.Module):
    def __init__(self, block_fn, ch_in, ch_out, n, stride, act="relu", attn="eca", use_alpha=False):
        super().__init__()
        ch_mid = (ch_in + ch_out) // 2
        self.conv_down = (
            ConvBNLayer(ch_in, ch_mid, 3, stride=2, padding=1, act=act) if stride == 2 else None
        )
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(
            *[
                block_fn(ch_mid // 2, ch_mid // 2, act=act, shortcut=True, use_alpha=use_alpha)
                for _ in range(n)
            ]
        )
        self.attn = EffectiveSELayer(ch_mid, act="hardsigmoid") if attn else None
        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], dim=1)
        if self.attn is not None:
            y = self.attn(y)
        return self.conv3(y)


class CSPResNet(nn.Module):
    def __init__(
        self,
        layers=(3, 6, 6, 3),
        channels=(64, 128, 256, 512, 1024),
        act="swish",
        return_idx=(1, 2, 3),
        use_large_stem=False,
        width_mult=1.0,
        depth_mult=1.0,
        use_alpha=False,
    ):
        super().__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]

        if use_large_stem:
            self.stem = nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act)),
                        ("conv2", ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, 1, padding=1, act=act)),
                        ("conv3", ConvBNLayer(channels[0] // 2, channels[0], 3, 1, padding=1, act=act)),
                    ]
                )
            )
        else:
            self.stem = nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act)),
                        ("conv2", ConvBNLayer(channels[0] // 2, channels[0], 3, 1, padding=1, act=act)),
                    ]
                )
            )

        n = len(channels) - 1
        self.stages = nn.Sequential(
            *[
                CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=act, use_alpha=use_alpha)
                for i in range(n)
            ]
        )
        self.return_idx = list(return_idx)
        self.out_channels = [channels[i + 1] for i in self.return_idx]

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class SPP(nn.Module):
    def __init__(self, ch_in, ch_out, k, pool_size, act="swish"):
        super().__init__()
        self.pool = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False) for size in pool_size]
        )
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

    def forward(self, x):
        outs = [x] + [p(x) for p in self.pool]
        return self.conv(torch.cat(outs, dim=1))


class CSPStage(nn.Module):
    def __init__(self, block_fn, ch_in, ch_out, n, act="swish", spp=False, use_alpha=False):
        super().__init__()
        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        convs = OrderedDict()
        next_ch_in = ch_mid
        for i in range(n):
            convs[str(i)] = block_fn(next_ch_in, ch_mid, act=act, shortcut=False, use_alpha=use_alpha)
            if i == (n - 1) // 2 and spp:
                convs["spp"] = SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act)
            next_ch_in = ch_mid
        self.convs = nn.Sequential(convs)
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.convs(self.conv2(x))
        return self.conv3(torch.cat([y1, y2], dim=1))


class CustomCSPPAN(nn.Module):
    def __init__(
        self,
        in_channels=(256, 512, 1024),
        out_channels=(1024, 512, 256),
        act="leaky",
        stage_num=1,
        block_num=3,
        spp=False,
        width_mult=1.0,
        depth_mult=1.0,
        use_alpha=False,
    ):
        super().__init__()
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        self.num_blocks = len(in_channels)
        self.out_channels = out_channels
        in_channels = list(in_channels)[::-1]

        fpn_stages, fpn_routes = [], []
        ch_pre = None
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2
            stage = nn.Sequential(
                *[
                    CSPStage(
                        BasicBlock,
                        ch_in if j == 0 else ch_out,
                        ch_out,
                        block_num,
                        act=act,
                        spp=(spp and i == 0),
                        use_alpha=use_alpha,
                    )
                    for j in range(stage_num)
                ]
            )
            fpn_stages.append(stage)
            if i < self.num_blocks - 1:
                fpn_routes.append(ConvBNLayer(ch_out, ch_out // 2, 1, stride=1, padding=0, act=act))
            ch_pre = ch_out
        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

        pan_stages, pan_routes = [], []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNLayer(out_channels[i + 1], out_channels[i + 1], 3, stride=2, padding=1, act=act)
            )
            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            pan_stages.append(
                nn.Sequential(
                    *[
                        CSPStage(
                            BasicBlock,
                            ch_in if j == 0 else ch_out,
                            ch_out,
                            block_num,
                            act=act,
                            spp=False,
                            use_alpha=use_alpha,
                        )
                        for j in range(stage_num)
                    ]
                )
            )
        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])

    def forward(self, blocks):
        blocks = blocks[::-1]
        fpn_feats = []
        route = None
        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], dim=1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(route, scale_factor=2.0, mode="nearest")

        pan_feats = [fpn_feats[-1]]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.cat([route, block], dim=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)
        return pan_feats[::-1]


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act="swish"):
        super().__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


class PPYOLOERHead(nn.Module):
    def __init__(
        self,
        in_channels=(1024, 512, 256),
        num_classes=15,
        act="swish",
        fpn_strides=(32, 16, 8),
        grid_cell_offset=0.5,
        angle_max=90,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.num_classes = num_classes
        self.fpn_strides = list(fpn_strides)
        self.grid_cell_offset = grid_cell_offset
        self.angle_max = angle_max
        self.half_pi = math.pi / 2
        self.half_pi_bin = self.half_pi / angle_max

        self.stem_cls = nn.ModuleList([ESEAttn(c, act=act) for c in self.in_channels])
        self.stem_reg = nn.ModuleList([ESEAttn(c, act=act) for c in self.in_channels])
        self.stem_angle = nn.ModuleList([ESEAttn(c, act=act) for c in self.in_channels])
        self.pred_cls = nn.ModuleList([nn.Conv2d(c, num_classes, 3, padding=1) for c in self.in_channels])
        self.pred_reg = nn.ModuleList([nn.Conv2d(c, 4, 3, padding=1) for c in self.in_channels])
        self.pred_angle = nn.ModuleList([nn.Conv2d(c, angle_max + 1, 3, padding=1) for c in self.in_channels])
        # proyeccion DFL del angulo: conv 1x1 fija (no entrenable), igual que en Paddle
        self.angle_proj_conv = nn.Conv2d(angle_max + 1, 1, 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        bias_cls = float(-math.log((1 - 0.01) / 0.01))
        for cls_, reg_, angle_ in zip(self.pred_cls, self.pred_reg, self.pred_angle):
            nn.init.normal_(cls_.weight, std=0.01)
            nn.init.constant_(cls_.bias, bias_cls)
            nn.init.normal_(reg_.weight, std=0.01)
            nn.init.constant_(reg_.bias, 0.0)
            nn.init.constant_(angle_.weight, 0.0)
            with torch.no_grad():
                angle_.bias.copy_(torch.tensor([10.0] + [1.0] * self.angle_max))
        angle_proj = torch.linspace(0, self.angle_max, self.angle_max + 1) * self.half_pi_bin
        with torch.no_grad():
            self.angle_proj_conv.weight.copy_(angle_proj.reshape(1, self.angle_max + 1, 1, 1))
        self.angle_proj_conv.weight.requires_grad_(False)
        self.register_buffer("angle_proj", angle_proj, persistent=False)

    def _generate_anchors(self, feats):
        anchor_points, stride_tensor, num_anchors_list = [], [], []
        for feat, stride in zip(feats, self.fpn_strides):
            _, _, h, w = feat.shape
            shift_x = (torch.arange(w, device=feat.device, dtype=torch.float32) + self.grid_cell_offset) * stride
            shift_y = (torch.arange(h, device=feat.device, dtype=torch.float32) + self.grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            anchor_points.append(torch.stack([shift_x, shift_y], dim=-1).reshape(1, -1, 2))
            stride_tensor.append(torch.full((1, h * w, 1), float(stride), device=feat.device))
            num_anchors_list.append(h * w)
        return torch.cat(anchor_points, dim=1), torch.cat(stride_tensor, dim=1), num_anchors_list

    def _head_outputs(self, feats):
        cls_logits, reg_dists, reg_angles = [], [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_angle = self.pred_angle[i](self.stem_angle[i](feat, avg_feat))
            cls_logits.append(cls_logit)
            reg_dists.append(reg_dist)
            reg_angles.append(reg_angle)
        return cls_logits, reg_dists, reg_angles

    def forward_train(self, feats):
        anchor_points, stride_tensor, num_anchors_list = self._generate_anchors(feats)
        cls_logits, reg_dists, reg_angles = self._head_outputs(feats)
        cls_score_list = torch.cat([torch.sigmoid(c).flatten(2).transpose(1, 2) for c in cls_logits], dim=1)
        reg_dist_list = torch.cat([r.flatten(2).transpose(1, 2) for r in reg_dists], dim=1)
        reg_angle_list = torch.cat([a.flatten(2).transpose(1, 2) for a in reg_angles], dim=1)
        return cls_score_list, reg_dist_list, reg_angle_list, anchor_points, num_anchors_list, stride_tensor

    def bbox_decode(self, points, pred_dist, pred_angle, stride_tensor):
        """(B, L, 4) + (B, L, angle_max+1) -> rboxes (B, L, 5) en pixeles de la entrada."""
        b, l = pred_angle.shape[:2]
        xy, wh = pred_dist.split(2, dim=-1)
        xy = xy * stride_tensor + points
        wh = (F.elu(wh) + 1.0) * stride_tensor
        angle = F.softmax(pred_angle.reshape(b, l, 1, self.angle_max + 1), dim=-1) @ self.angle_proj.to(
            pred_angle.dtype
        )
        return torch.cat([xy, wh, angle], dim=-1)

    def forward_eval(self, feats):
        anchor_points, _, _ = self._generate_anchors(feats)
        cls_score_list, reg_box_list = [], []
        for i, (feat, stride) in enumerate(zip(feats, self.fpn_strides)):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_xy, reg_wh = reg_dist.split(2, dim=1)
            reg_xy = reg_xy * stride
            reg_wh = (F.elu(reg_wh) + 1.0) * stride
            reg_angle = self.pred_angle[i](self.stem_angle[i](feat, avg_feat))
            reg_angle = self.angle_proj_conv(F.softmax(reg_angle, dim=1))
            reg_box = torch.cat([reg_xy, reg_wh, reg_angle], dim=1)
            cls_score_list.append(torch.sigmoid(cls_logit).reshape(b, self.num_classes, l))
            reg_box_list.append(reg_box.reshape(b, 5, l))
        cls_score_list = torch.cat(cls_score_list, dim=-1)  # (B, C, L)
        reg_box_list = torch.cat(reg_box_list, dim=-1).transpose(1, 2)  # (B, L, 5)
        reg_xy, reg_wha = reg_box_list.split([2, 3], dim=-1)
        reg_box_list = torch.cat([reg_xy + anchor_points, reg_wha], dim=-1)
        return cls_score_list, reg_box_list

    def forward(self, feats):
        return self.forward_train(feats) if self.training else self.forward_eval(feats)


class PPYOLOER(nn.Module):
    """backbone + neck + head. `forward` devuelve las salidas crudas de la cabeza."""

    def __init__(self, backbone: CSPResNet, neck: CustomCSPPAN, head: PPYOLOERHead):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = head

    @property
    def num_classes(self) -> int:
        return self.yolo_head.num_classes

    def forward(self, images: torch.Tensor):
        feats = self.neck(self.backbone(images))
        return self.yolo_head(feats)

    @torch.no_grad()
    def predict(self, images: torch.Tensor):
        """(B, 3, H, W) -> (scores (B, C, L), polys (B, L, 8)) en pixeles de la entrada."""
        self.eval()
        scores, rboxes = self.forward(images)
        return scores, rbox2poly(rboxes)


def build_ppyoloe_r(num_classes: int = 1, size: str = "s") -> PPYOLOER:
    """Construye PP-YOLOE-R con los multiplicadores oficiales (s/m/l/x)."""
    mults = {"s": (0.33, 0.50), "m": (0.67, 0.75), "l": (1.0, 1.0), "x": (1.33, 1.25)}
    if size not in mults:
        raise ValueError(f"tamano desconocido: {size} (usa {sorted(mults)})")
    depth_mult, width_mult = mults[size]
    backbone = CSPResNet(
        layers=(3, 6, 6, 3),
        channels=(64, 128, 256, 512, 1024),
        return_idx=(1, 2, 3),
        use_large_stem=True,
        use_alpha=True,
        act="swish",
        width_mult=width_mult,
        depth_mult=depth_mult,
    )
    neck = CustomCSPPAN(
        in_channels=tuple(backbone.out_channels),
        out_channels=(768, 384, 192),
        stage_num=1,
        block_num=3,
        act="swish",
        spp=True,
        use_alpha=True,
        width_mult=width_mult,
        depth_mult=depth_mult,
    )
    head = PPYOLOERHead(
        in_channels=tuple(neck.out_channels),
        num_classes=num_classes,
        act="swish",
        fpn_strides=(32, 16, 8),
        grid_cell_offset=0.5,
        angle_max=90,
    )
    return PPYOLOER(backbone, neck, head)
