"""Pure-torch port of the `DDGRCF/YOLOX_OBB` network (`yoloxs_obb.yaml`).

Why a port and not the clone
----------------------------
The clone requires compiled C++/CUDA operators and a GPU, and neither is
available here. But its network -- YOLOv5 backbone with `C3` blocks and ReLU,
PAFPN neck, YOLOX decoupled heads -- is pure torch. The only compiled parts
were the rotated IoU of the assigner and of the loss, and the NMS. So the
network is ported and those three pieces are replaced by torch versions
(`overlap.py`).

License: Apache-2.0, with attribution. The blocks below are theirs
(`yolox/models/modules/common.py` and `block.py`), reduced to what the yaml uses.

The only thing that really matters: the NAMES
---------------------------------------------
The reason for porting instead of rewriting is to load **their DOTA weights**.
Their model is built from the yaml into an `nn.Sequential` called `model`, with
layers indexed 0..33 in file order; the parameters come out as
`model.0.conv.weight`, `model.2.m.0.cv1.bn.bias`, `model.33.cls_preds.0.bias`...
Here that tree is replicated by hand, layer by layer, with the same attributes
inside each block, so that `load_state_dict(strict=True)` accepts their
checkpoint without any mapping. `tests/test_ddgrcf_port.py` compares keys and
shapes against their actually built model.

What changes with respect to the own head
-----------------------------------------
- YOLOX-style regression: `(dx, dy, log w, log h)` relative to the cell, not
  l/t/r/b distances. Decoded as `cx = (dx + i) * stride`, `w = e^{log w} *
  stride`.
- Angle: a RAW scalar in radians, no sigmoid and no folding. Their convention
  (`mintheta_obb`): the long side is `w` and the angle falls in `(-pi/4, pi/4]`,
  swapping `w`/`h` if needed.
- With objectness, like YOLOX.

All of that is described by a `HeadSpec(regression="yolox", angle="radians")`
and consumed by the same decoder as the rest of the project.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from testbank.models.yolox_obb import STRIDES, HeadOutput, HeadSpec

DEPTH_MULTIPLE = 0.33
WIDTH_MULTIPLE = 0.50


def make_divisible(x: float, divisor: int = 8) -> int:
    return int(math.ceil(x / divisor) * divisor)


def _width(channels: int) -> int:
    return make_divisible(channels * WIDTH_MULTIPLE)


def _depth(n: int) -> int:
    return max(round(n * DEPTH_MULTIPLE), 1) if n > 1 else n


# --- blocks, with THEIR attribute names (Apache-2.0, DDGRCF/YOLOX_OBB) -------


class Conv(nn.Module):
    """`conv` + `bn` + `act`. ReLU throughout this model's yaml."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden, hidden, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)) -> None:
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Concat(nn.Module):
    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


class OBBDetectX(nn.Module):
    """The final 1x1 convs of their `OBBDetectX`: `cls_preds`, `reg_preds`, `obj_preds`.

    Only the layers. Their assigner, loss and post-processing -- what carried
    the compiled operators -- live in `losses.py` and `overlap.py`.
    Returns `HeadOutput` so the common decoder does not know where it comes from.
    """

    REG_DIM = 5  # dx, dy, log w, log h, theta

    def __init__(self, in_channels: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        for cls_in, reg_in in zip(in_channels[0::2], in_channels[1::2]):
            self.cls_preds.append(nn.Conv2d(cls_in, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(reg_in, self.REG_DIM, 1))
            self.obj_preds.append(nn.Conv2d(reg_in, 1, 1))

    def initialize_biases(self, prior: float = 0.01) -> None:
        bias = -math.log((1 - prior) / prior)
        for layer in list(self.cls_preds) + list(self.obj_preds):
            nn.init.constant_(layer.bias, bias)

    def forward(self, xin: list[torch.Tensor]) -> list[HeadOutput]:
        outputs = []
        for k, (cls_x, reg_x) in enumerate(zip(xin[0::2], xin[1::2])):
            reg = self.reg_preds[k](reg_x)
            outputs.append(
                HeadOutput(
                    distances=reg[:, :4],
                    angle=reg[:, 4:5],
                    objectness=self.obj_preds[k](reg_x),
                    classes=self.cls_preds[k](cls_x),
                    stride=STRIDES[k],
                    regression="yolox",
                    angle_mode="radians",
                )
            )
        return outputs


# --- the whole network, layer by layer as in the yaml ----------------------


class DdgrcfYoloxObb(nn.Module):
    """DDGRCF's `yoloxs_obb.yaml`, built by hand with the same indices.

    `self.model[i]` is layer `i` of the yaml, and `.f` says where it takes its
    input from, exactly like their `parse_model`. The `forward` is their
    `forward_once`. Every line below carries the yaml layer number.
    """

    HEAD_SPEC = HeadSpec(regression="yolox", angle="radians", objectness=True)

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.variant = "ddgrcf-s"
        c64, c128, c256, c512, c1024 = (_width(c) for c in (64, 128, 256, 512, 1024))

        def seq(module_factory, n):
            # Like their `parse_model`: `n` goes through the depth multiplier
            # and, if it ends up as 1, the layer goes WITHOUT a Sequential
            # wrapper. The yaml puts `n=2` in the head stems and
            # `round(2 * 0.33) = 1`: it is a single conv, and its keys are
            # `model.27.conv.*`, not `model.27.0.*`.
            n = _depth(n)
            return module_factory() if n == 1 else nn.Sequential(*(module_factory() for _ in range(n)))

        layers: list[tuple[nn.Module, int | list[int]]] = [
            # ---- backbone ----
            (Conv(3, c64, 6, 2, 2), -1),                          # 0  P1/2
            (Conv(c64, c128, 3, 2), -1),                          # 1  P2/4
            (C3(c128, c128, _depth(3)), -1),                      # 2
            (Conv(c128, c256, 3, 2), -1),                         # 3  P3/8
            (C3(c256, c256, _depth(9)), -1),                      # 4
            (Conv(c256, c512, 3, 2), -1),                         # 5  P4/16
            (C3(c512, c512, _depth(9)), -1),                      # 6
            (Conv(c512, c1024, 3, 2), -1),                        # 7  P5/32
            (SPP(c1024, c1024, (5, 9, 13)), -1),                  # 8
            (C3(c1024, c1024, _depth(3), shortcut=False), -1),    # 9
            # ---- neck ----
            (Conv(c1024, c512, 1, 1), -1),                        # 10
            (nn.Upsample(None, 2, "nearest"), -1),                # 11
            (Concat(1), [-1, 6]),                                 # 12
            (C3(c512 * 2, c512, _depth(3), shortcut=False), -1),  # 13
            (Conv(c512, c256, 1, 1), -1),                         # 14
            (nn.Upsample(None, 2, "nearest"), -1),                # 15
            (Concat(1), [-1, 4]),                                 # 16
            (C3(c256 * 2, c256, _depth(3), shortcut=False), -1),  # 17 P3
            (Conv(c256, c256, 3, 2), -1),                         # 18
            (Concat(1), [-1, 14]),                                # 19
            (C3(c256 * 2, c512, _depth(3), shortcut=False), -1),  # 20 P4
            (Conv(c512, c512, 3, 2), -1),                         # 21
            (Concat(1), [-1, 10]),                                # 22
            (C3(c512 * 2, c1024, _depth(3), shortcut=False), -1), # 23 P5
            # ---- laterals and head stems ----
            (Conv(c256, c256, 1, 1), 17),                         # 24 lateral0
            (Conv(c512, c256, 1, 1), 20),                         # 25 lateral1
            (Conv(c1024, c256, 1, 1), 23),                        # 26 lateral2
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 24),         # 27 cls0
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 24),         # 28 reg0
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 25),         # 29 cls1
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 25),         # 30 reg1
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 26),         # 31 cls2
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 26),         # 32 reg2
            # ---- detection ----
            (OBBDetectX((c256,) * 6, num_classes), [27, 28, 29, 30, 31, 32]),  # 33
        ]
        modules = []
        self.save: set[int] = set()
        for index, (module, source) in enumerate(layers):
            module.i = index
            module.f = source
            modules.append(module)
            for s in [source] if isinstance(source, int) else source:
                if s != -1:
                    self.save.add(s)
        self.model = nn.Sequential(*modules)
        self.model[-1].initialize_biases()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03

    @property
    def head_spec(self) -> HeadSpec:
        return self.HEAD_SPEC

    def forward(self, x: torch.Tensor) -> list[HeadOutput]:
        saved: list[torch.Tensor | None] = []
        for m in self.model:
            if m.f != -1:
                x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x)
            saved.append(x if m.i in self.save else None)
        return x

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_pretrained(model: DdgrcfYoloxObb, path) -> dict:
    """Load THEIR DOTA checkpoint. Returns what did not fit, which must be only
    the class layer: their weights have 15 outputs and here there is one.

    Strict in everything else on purpose: if a name does not match, the port
    has drifted from the yaml, and that has to be known, not hidden.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    own = model.state_dict()
    kept, skipped = {}, {}
    for key, value in state.items():
        if key not in own:
            skipped[key] = "does not exist in the port"
        elif own[key].shape != value.shape:
            skipped[key] = f"shape {tuple(value.shape)} != {tuple(own[key].shape)}"
        else:
            kept[key] = value
    missing = sorted(set(own) - set(kept))
    unexpected_missing = [k for k in missing if "cls_preds" not in k]
    if unexpected_missing:
        raise RuntimeError(
            "the checkpoint does not cover the port: "
            f"{len(unexpected_missing)} tensors outside the class layer are missing, "
            f"the first one {unexpected_missing[0]!r}"
        )
    model.load_state_dict(kept, strict=False)
    return skipped


__all__ = ["DdgrcfYoloxObb", "OBBDetectX", "load_pretrained", "make_divisible"]
