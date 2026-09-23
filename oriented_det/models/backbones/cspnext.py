"""CSPNeXt backbone + CSPNeXtPAFPN neck (RTMDet), pure PyTorch.

Module / parameter names mirror MMDetection (``ConvModule.conv`` / ``.bn``,
``stem.*``, ``stage{i}.*``, ``reduce_layers``, ``top_down_blocks`` …) so MMDet
checkpoints load after stripping the ``backbone.`` / ``neck.`` prefix — no mmcv
needed. See :func:`load_mmdet_cspnext_weights`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from .utils import require_torch

# (in_channels, out_channels, num_blocks, add_identity, use_spp) before deepen/widen.
CSPNEXT_P5_ARCH: Tuple[Tuple[int, int, int, bool, bool], ...] = (
    (64, 128, 3, True, False),
    (128, 256, 6, True, False),
    (256, 512, 6, True, False),
    (512, 1024, 3, False, True),
)

# RTMDet variants: (deepen_factor, widen_factor, neck_out_channels, head_feat_channels).
RTMDET_VARIANTS: Dict[str, Tuple[float, float, int]] = {
    "tiny": (0.167, 0.375, 96),
    "s": (0.33, 0.5, 128),
    "m": (0.67, 0.75, 192),
    "l": (1.0, 1.0, 256),
}

# ImageNet-pretrained CSPNeXt weights published by MMDetection (RTMDet ``init_cfg``).
CSPNEXT_IMAGENET_URLS: Dict[str, str] = {
    "tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth",
    "s": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth",
    "m": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth",
    "l": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth",
}

# Full COCO-trained RTMDet (backbone + neck + head convs; natural photos).
RTMDET_COCO_URLS: Dict[str, str] = {
    "tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
}

# MMDet RTMDet: BN(momentum=0.03, eps=0.001) — same momentum convention as PyTorch.
_BN_MOMENTUM = 0.03
_BN_EPS = 0.001


class ConvModule(nn.Module):
    """Conv2d(bias=False) + BatchNorm2d + SiLU (``mmcv.cnn.ConvModule`` layout)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM, eps=_BN_EPS)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise ConvModule + pointwise 1x1 ConvModule."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    """Global-avg-pool → 1x1 conv → hard-sigmoid gate."""

    def __init__(self, channels: int):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.global_avgpool(x.float()).to(x.dtype)
        out = self.act(self.fc(out))
        return x * out


class CSPNeXtBlock(nn.Module):
    """3x3 ConvModule + 5x5 depthwise-separable ConvModule (+ identity)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        kernel_size: int = 5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    """Cross Stage Partial layer with CSPNeXt blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        channel_attention: bool = False,
    ):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(in_channels, mid_channels, 1)
        self.short_conv = ConvModule(in_channels, mid_channels, 1)
        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1)
        self.blocks = nn.Sequential(
            *[
                CSPNeXtBlock(mid_channels, mid_channels, 1.0, add_identity)
                for _ in range(num_blocks)
            ]
        )
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling (5/9/13 max-pools) between two 1x1 ConvModules."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (5, 9, 13),
    ):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, 1)
        self.poolings = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        self.conv2 = ConvModule(mid_channels * (len(kernel_sizes) + 1), out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        return self.conv2(x)


class CSPNeXt(nn.Module):
    """CSPNeXt backbone (RTMDet P5). Returns stage outputs for ``out_indices``.

    ``out_indices`` index ``[stem, stage1, stage2, stage3, stage4]``; the RTMDet
    default ``(2, 3, 4)`` yields strides 8 / 16 / 32.
    """

    def __init__(
        self,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        expand_ratio: float = 0.5,
        channel_attention: bool = True,
    ):
        require_torch()
        super().__init__()
        self.out_indices = tuple(out_indices)
        self.frozen_stages = int(frozen_stages)

        stem_planes = CSPNEXT_P5_ARCH[0][0]
        stem_half = int(stem_planes * widen_factor // 2)
        stem_full = int(stem_planes * widen_factor)
        self.stem = nn.Sequential(
            ConvModule(3, stem_half, 3, stride=2, padding=1),
            ConvModule(stem_half, stem_half, 3, stride=1, padding=1),
            ConvModule(stem_half, stem_full, 3, stride=1, padding=1),
        )
        self.layers = ["stem"]
        self.out_channels: List[int] = []
        channels_per_layer = [stem_full]
        for i, (in_c, out_c, num_blocks, add_identity, use_spp) in enumerate(CSPNEXT_P5_ARCH):
            in_c = int(in_c * widen_factor)
            out_c = int(out_c * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage: List[nn.Module] = [ConvModule(in_c, out_c, 3, stride=2, padding=1)]
            if use_spp:
                stage.append(SPPBottleneck(out_c, out_c))
            stage.append(
                CSPLayer(
                    out_c,
                    out_c,
                    expand_ratio=expand_ratio,
                    num_blocks=num_blocks,
                    add_identity=add_identity,
                    channel_attention=channel_attention,
                )
            )
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")
            channels_per_layer.append(out_c)
        self.out_channels = [channels_per_layer[i] for i in self.out_indices]
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        for i in range(self.frozen_stages + 1):
            m = getattr(self, self.layers[i])
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "CSPNeXt":
        super().train(mode)
        self._freeze_stages()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for i, layer_name in enumerate(self.layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class CSPNeXtPAFPN(nn.Module):
    """RTMDet path-aggregation FPN (top-down + bottom-up CSP blocks, 3x3 out convs)."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_csp_blocks: int = 3,
        expand_ratio: float = 0.5,
    ):
        require_torch()
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = int(out_channels)
        n = len(self.in_channels)

        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(n - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(self.in_channels[idx], self.in_channels[idx - 1], 1)
            )
            self.top_down_blocks.append(
                CSPLayer(
                    self.in_channels[idx - 1] * 2,
                    self.in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    expand_ratio=expand_ratio,
                )
            )

        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(n - 1):
            self.downsamples.append(
                ConvModule(self.in_channels[idx], self.in_channels[idx], 3, stride=2, padding=1)
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    self.in_channels[idx] * 2,
                    self.in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    expand_ratio=expand_ratio,
                )
            )

        self.out_convs = nn.ModuleList(
            [ConvModule(c, self.out_channels, 3, padding=1) for c in self.in_channels]
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        n = len(self.in_channels)
        if len(inputs) != n:
            raise ValueError(f"Expected {n} inputs, got {len(inputs)}")

        # Top-down path
        inner_outs = [inputs[-1]]
        for idx in range(n - 1, 0, -1):
            feat_high = self.reduce_layers[n - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_high
            feat_low = inputs[idx - 1]
            # Upsample to the exact lower-level size (== nearest x2 when padded to /32).
            upsample_feat = F.interpolate(feat_high, size=feat_low.shape[-2:], mode="nearest")
            inner_out = self.top_down_blocks[n - 1 - idx](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        # Bottom-up path
        outs = [inner_outs[0]]
        for idx in range(n - 1):
            downsample_feat = self.downsamples[idx](outs[-1])
            feat_high = inner_outs[idx + 1]
            if downsample_feat.shape[-2:] != feat_high.shape[-2:]:
                downsample_feat = F.interpolate(
                    downsample_feat, size=feat_high.shape[-2:], mode="nearest"
                )
            outs.append(
                self.bottom_up_blocks[idx](torch.cat([downsample_feat, feat_high], dim=1))
            )

        return [conv(out) for conv, out in zip(self.out_convs, outs)]


def build_cspnext_pafpn(
    variant: str = "tiny",
    *,
    frozen_stages: int = -1,
) -> Tuple["CSPNeXt", "CSPNeXtPAFPN", int]:
    """Build RTMDet backbone + neck for ``variant`` in {tiny, s, m, l}.

    Returns ``(backbone, neck, neck_out_channels)``.
    """
    require_torch()
    key = str(variant).strip().lower()
    if key not in RTMDET_VARIANTS:
        raise ValueError(
            f"Unknown RTMDet variant {variant!r}; expected one of {sorted(RTMDET_VARIANTS)}."
        )
    deepen, widen, neck_out = RTMDET_VARIANTS[key]
    backbone = CSPNeXt(
        deepen_factor=deepen,
        widen_factor=widen,
        out_indices=(2, 3, 4),
        frozen_stages=frozen_stages,
    )
    neck = CSPNeXtPAFPN(
        in_channels=backbone.out_channels,
        out_channels=neck_out,
        num_csp_blocks=max(round(3 * deepen), 1),
        expand_ratio=0.5,
    )
    return backbone, neck, neck_out


def _extract_state_dict(obj: Dict) -> Dict[str, "torch.Tensor"]:
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


class _PickleStub:
    """Placeholder for classes whose package is not installed (mmengine, mmdet …)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _LenientUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (_PickleStub,), {"__module__": module})


class _LenientPickleModule:
    """``pickle_module`` for ``torch.load`` that tolerates missing OpenMMLab classes."""

    Unpickler = _LenientUnpickler
    load = staticmethod(lambda f, **kw: _LenientUnpickler(f, **kw).load())


def load_openmmlab_checkpoint(source: Union[str, Path]) -> Dict[str, "torch.Tensor"]:
    """Read the ``state_dict`` of an MMDet / MMRotate / MMPretrain ``.pth`` without mmcv.

    ``source`` may be a local path or an ``http(s)://`` URL (downloaded once into the
    ``torch.hub`` cache). OpenMMLab checkpoints pickle ``mmengine`` objects in
    their ``message_hub`` / ``meta``; those are replaced by inert stubs. Only use
    with checkpoints from a trusted source (full unpickling, not ``weights_only``).
    """
    require_torch()
    src = str(source)
    if src.startswith(("http://", "https://")):
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        local = cache_dir / Path(src.split("?")[0]).name
        if not local.is_file():
            torch.hub.download_url_to_file(src, str(local), progress=True)
        src = str(local)
    ckpt = torch.load(
        src,
        map_location="cpu",
        pickle_module=_LenientPickleModule,
        weights_only=False,
    )
    return _extract_state_dict(ckpt)


def load_mmdet_weights(
    module: "nn.Module",
    source: Union[str, Path],
    prefix_map: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    """Partially load OpenMMLab weights into ``module``.

    ``prefix_map`` renames checkpoint prefixes to module prefixes, e.g.
    ``{"backbone.": "backbone.", "neck.": "neck.", "bbox_head.": "head."}``.
    Checkpoint keys outside the map, keys the module does not have, and tensors
    whose shape differs (e.g. an 80-class COCO ``rtm_cls``) are skipped.

    Returns ``(loaded_keys, skipped_shape_keys)``.
    """
    raw = load_openmmlab_checkpoint(source)
    own = module.state_dict()
    selected: Dict[str, torch.Tensor] = {}
    skipped_shape: List[str] = []
    for k, v in raw.items():
        name = k[7:] if k.startswith("module.") else k
        for src_prefix, dst_prefix in prefix_map.items():
            if name.startswith(src_prefix):
                name = dst_prefix + name[len(src_prefix):]
                break
        else:
            continue
        if name not in own or not torch.is_tensor(v):
            continue
        if own[name].shape != v.shape:
            skipped_shape.append(name)
            continue
        selected[name] = v
    if not selected:
        raise ValueError(
            f"No tensors from {str(source)!r} matched the module (prefix map {prefix_map}). "
            "Is this a checkpoint of the same RTMDet variant?"
        )
    module.load_state_dict(selected, strict=False)
    return sorted(selected), skipped_shape


def load_mmdet_cspnext_weights(
    backbone: "nn.Module",
    source: Union[str, Path],
) -> Tuple[List[str], List[str]]:
    """Load an ImageNet CSPNeXt checkpoint (``backbone.*`` keys) into a :class:`CSPNeXt`."""
    return load_mmdet_weights(backbone, source, {"backbone.": ""})


__all__ = [
    "ConvModule",
    "DepthwiseSeparableConvModule",
    "ChannelAttention",
    "CSPNeXtBlock",
    "CSPLayer",
    "SPPBottleneck",
    "CSPNeXt",
    "CSPNeXtPAFPN",
    "RTMDET_VARIANTS",
    "CSPNEXT_IMAGENET_URLS",
    "RTMDET_COCO_URLS",
    "build_cspnext_pafpn",
    "load_openmmlab_checkpoint",
    "load_mmdet_weights",
    "load_mmdet_cspnext_weights",
]
