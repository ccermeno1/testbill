"""Official YOLOX (Megvii-BaseDetection/YOLOX) with an oriented-box head, in plain PyTorch.

CSPDarknet + PAFPN + ``YOLOXHead`` as in ``yolox/models``, with the same module
names, so the COCO checkpoints of the 0.1.1rc0 release (``yolox_nano.pth``,
``yolox_tiny.pth``, ``yolox_s.pth``, input BGR 0..255 without normalisation)
load directly. The only change is ``reg_preds``: 5 channels (x, y, w, h, angle)
instead of 4. :meth:`YOLOXOfficialOBB.adapt_state_dict` keeps the pretrained
x, y, w, h filters and starts the angle channel at 0 (horizontal boxes).

Training, losses and post-processing are the YOLOX-OBB ones of :class:`OBBDetector`.
"""
import torch
import torch.nn as nn

from .model import OBBDetector

# ----------------------------------------------------------------------------- blocks


class BaseConv(nn.Module):
    """conv (no bias) -> BN -> SiLU, same padding."""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, (ksize - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise conv + pointwise conv."""

    def __init__(self, in_channels, out_channels, ksize, stride=1):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize, stride, groups=in_channels)
        self.pconv = BaseConv(in_channels, out_channels, 1, 1)

    def forward(self, x):
        return self.pconv(self.dconv(x))


def _conv(depthwise):
    return DWConv if depthwise else BaseConv


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.conv2 = _conv(depthwise)(hidden, out_channels, 3, 1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class SPPBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(ks, 1, ks // 2) for ks in kernel_sizes])
        self.conv2 = BaseConv(hidden * (len(kernel_sizes) + 1), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))


class CSPLayer(nn.Module):

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden, 1, 1)
        self.conv3 = BaseConv(2 * hidden, out_channels, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden, hidden, shortcut, 1.0, depthwise) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class Focus(nn.Module):
    """Space-to-depth (2x2 patches into channels), then a conv."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))


# ----------------------------------------------------------------------------- network


class CSPDarknet(nn.Module):

    def __init__(self, depth, width, depthwise=False):
        super().__init__()
        Conv = _conv(depthwise)
        c = int(width * 64)
        n = max(round(depth * 3), 1)
        self.stem = Focus(3, c, ksize=3)
        self.dark2 = nn.Sequential(Conv(c, c * 2, 3, 2), CSPLayer(c * 2, c * 2, n, depthwise=depthwise))
        self.dark3 = nn.Sequential(Conv(c * 2, c * 4, 3, 2), CSPLayer(c * 4, c * 4, n * 3, depthwise=depthwise))
        self.dark4 = nn.Sequential(Conv(c * 4, c * 8, 3, 2), CSPLayer(c * 8, c * 8, n * 3, depthwise=depthwise))
        self.dark5 = nn.Sequential(Conv(c * 8, c * 16, 3, 2), SPPBottleneck(c * 16, c * 16),
                                   CSPLayer(c * 16, c * 16, n, shortcut=False, depthwise=depthwise))

    def forward(self, x):
        x = self.dark2(self.stem(x))
        c3 = self.dark3(x)
        c4 = self.dark4(c3)
        return c3, c4, self.dark5(c4)


class YOLOPAFPN(nn.Module):

    def __init__(self, depth, width, in_channels=(256, 512, 1024), depthwise=False):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise)
        Conv = _conv(depthwise)
        c0, c1, c2 = (int(c * width) for c in in_channels)
        n = round(3 * depth)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv0 = BaseConv(c2, c1, 1, 1)
        self.C3_p4 = CSPLayer(2 * c1, c1, n, False, depthwise=depthwise)
        self.reduce_conv1 = BaseConv(c1, c0, 1, 1)
        self.C3_p3 = CSPLayer(2 * c0, c0, n, False, depthwise=depthwise)
        self.bu_conv2 = Conv(c0, c0, 3, 2)
        self.C3_n3 = CSPLayer(2 * c0, c1, n, False, depthwise=depthwise)
        self.bu_conv1 = Conv(c1, c1, 3, 2)
        self.C3_n4 = CSPLayer(2 * c1, c2, n, False, depthwise=depthwise)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.C3_p4(torch.cat([self.upsample(fpn_out0), x1], 1))
        fpn_out1 = self.reduce_conv1(f_out0)
        pan_out2 = self.C3_p3(torch.cat([self.upsample(fpn_out1), x2], 1))
        pan_out1 = self.C3_n3(torch.cat([self.bu_conv2(pan_out2), fpn_out1], 1))
        pan_out0 = self.C3_n4(torch.cat([self.bu_conv1(pan_out1), fpn_out0], 1))
        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):
    """Decoupled head: 1x1 stem, two 3x3 convs per branch, 1x1 predictions. ``reg_preds`` has 5 channels."""

    def __init__(self, num_classes, width, in_channels=(256, 512, 1024), depthwise=False, reg_dim=5):
        super().__init__()
        Conv = _conv(depthwise)
        c = int(256 * width)
        self.stems = nn.ModuleList([BaseConv(int(ch * width), c, 1, 1) for ch in in_channels])
        self.cls_convs = nn.ModuleList([nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 3, 1)) for _ in in_channels])
        self.reg_convs = nn.ModuleList([nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 3, 1)) for _ in in_channels])
        self.cls_preds = nn.ModuleList([nn.Conv2d(c, num_classes, 1) for _ in in_channels])
        self.reg_preds = nn.ModuleList([nn.Conv2d(c, reg_dim, 1) for _ in in_channels])
        self.obj_preds = nn.ModuleList([nn.Conv2d(c, 1, 1) for _ in in_channels])

    def forward(self, feats):
        cls, reg, obj = [], [], []
        for k, x in enumerate(feats):
            x = self.stems[k](x)
            cls.append(self.cls_preds[k](self.cls_convs[k](x)))
            reg_feat = self.reg_convs[k](x)
            reg.append(self.reg_preds[k](reg_feat))
            obj.append(self.obj_preds[k](reg_feat))
        return cls, reg, obj


# ----------------------------------------------------------------------------- detector

PRESETS = {
    # exps/default/yolox_{nano,tiny,s}.py
    'nano': dict(depth=0.33, width=0.25, depthwise=True),
    'tiny': dict(depth=0.33, width=0.375, depthwise=False),
    's': dict(depth=0.33, width=0.50, depthwise=False),
}


class YOLOXOfficialOBB(OBBDetector):
    """Official YOLOX network (``backbone`` = YOLOPAFPN, ``head`` = YOLOXHead) predicting rotated boxes."""

    def __init__(self, num_classes=1, size='nano', strides=(8, 16, 32), prior_prob=0.01, **loss_kw):
        super().__init__(num_classes, strides, **loss_kw)
        p = PRESETS[size]
        self.arch = f'yolox_{size}'
        self.backbone = YOLOPAFPN(p['depth'], p['width'], depthwise=p['depthwise'])
        self.head = YOLOXHead(num_classes, p['width'], depthwise=p['depthwise'])
        self._init_weights(self.head.cls_preds, self.head.obj_preds, prior_prob)
        for conv in self.head.reg_preds:  # start with horizontal boxes
            nn.init.zeros_(conv.weight[4])
            nn.init.zeros_(conv.bias[4])

    def forward(self, images: torch.Tensor):
        return self.head(self.backbone(images.float()))

    def adapt_state_dict(self, state: dict) -> dict:
        """COCO checkpoints have 4-channel ``reg_preds``: keep them as x, y, w, h, angle channel at 0."""
        own = self.state_dict()
        state = dict(state)
        for k, v in state.items():
            if '.reg_preds.' in k and k in own and v.shape[0] == 4 and own[k].shape[0] == 5:
                state[k] = torch.cat([v, torch.zeros_like(v[:1])], 0)
        return state
