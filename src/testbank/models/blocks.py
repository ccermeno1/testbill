"""YOLOX building blocks. Apache-2.0, pure PyTorch, no compiled extensions.

Implemented here instead of vendoring Megvii's repository for three reasons:

1. **No usable package.** YOLOX is not published on PyPI as a library; the
   `yolox` name on PyPI is something else. The alternative would be cloning a
   whole repository to use 15% of its code.
2. **Nothing compiled.** It is the reason this candidate exists: everything
   here is `torch.nn`. See the MMCV problem in the README.
3. **Control of the schema.** It is the only candidate whose architecture we
   control entirely, so the OBB head attaches without guessing foreign
   conventions.

Nano uses depthwise separable convolutions everywhere except the stem. That is
where the difference between 0.91M and tiny's 5.06M comes from: a 3x3
convolution from C channels to C goes from 9C^2 to 9C + C^2 parameters.

The architectures are identical to Megvii's: their COCO checkpoints
(`yolox_s.pth`, `yolox_tiny.pth`, `yolox_nano.pth`) load into backbone and
neck of every variant, only renaming modules (`models/pretrained.py`).
"""

from __future__ import annotations

import torch
from torch import nn


def _round_channels(channels: int, width: float) -> int:
    """Scale the channels and keep them a multiple of 8.

    Multiples of 8 are not cosmetics: CPU vectorized routines and the matrix
    units of phones work in blocks, and a stray channel forces padding. It is
    the same rounding YOLOX uses.
    """
    scaled = max(1, int(channels * width))
    return max(8, (scaled + 4) // 8 * 8)


class BaseConv(nn.Module):
    """Conv + BatchNorm + SiLU, the brick of the whole network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # `kernel // 2` keeps the spatial size with stride 1, which is what
        # allows adding lateral branches without cropping.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            padding=kernel // 2,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Separable convolution: one spatial per channel and one 1x1 that mixes them.

    It is the change that makes nano truly small. A 3x3 from C to C costs 9C^2;
    separated it costs 9C + C^2. With C=64 that is 36864 against 4672.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel: int, stride: int
    ) -> None:
        super().__init__()
        self.dconv = BaseConv(
            in_channels, in_channels, kernel, stride, groups=in_channels
        )
        self.pconv = BaseConv(in_channels, out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pconv(self.dconv(x))


def conv_factory(depthwise: bool):
    return DWConv if depthwise else BaseConv


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        Conv = conv_factory(depthwise)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.conv2 = Conv(hidden, out_channels, 3, 1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.use_add else y


class CSPLayer(nn.Module):
    """Cross-Stage Partial: split the channels, process one half, concatenate.

    The point is that only half goes through the bottlenecks, so the cost drops
    without losing the full gradient path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden, 1, 1)
        self.conv3 = BaseConv(2 * hidden, out_channels, 1, 1)
        self.m = nn.Sequential(
            *[
                Bottleneck(hidden, hidden, shortcut, 1.0, depthwise)
                for _ in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling: mixes context at several scales at no cost.

    Three max-pools of 5, 9 and 13 in parallel. For a large banknote in the
    foreground, the 13 gives the detector enough receptive field without
    adding parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: tuple[int, ...] = (5, 9, 13),
    ) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernels]
        )
        self.conv2 = BaseConv(hidden * (len(kernels) + 1), out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class Focus(nn.Module):
    """Downsample 2x by rearranging pixels into channels, losing nothing.

    Takes every other pixel in both directions and stacks the four mosaics as
    channels. Unlike a stride-2 conv, nothing is discarded here: the
    information changes place, it does not disappear.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel: int = 3
    ) -> None:
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, kernel, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(
            torch.cat(
                (
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ),
                dim=1,
            )
        )


__all__ = [
    "BaseConv",
    "Bottleneck",
    "CSPLayer",
    "DWConv",
    "Focus",
    "SPPBottleneck",
    "_round_channels",
    "conv_factory",
]
