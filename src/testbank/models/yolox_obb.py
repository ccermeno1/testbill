"""YOLOX-Nano with an ORIENTED box head. Apache-2.0, pure PyTorch.

The head is the only thing that is not standard YOLOX. YOLOX predicts
axis-aligned `x, y, w, h`; here the angle is needed as well.

How the angle is represented
----------------------------
NOT as a scalar in radians. A rectangle rotated by theta and another rotated
by theta+180 are the SAME rectangle, so a direct regression on theta punishes
the model for being right: predicting 179 when the truth is 1 would give a
huge error for what is a 2-degree error. It is the same problem the angle
metric solves with `min(|d|, 180-|d|)`.

The head predicts `(sin 2t, cos 2t)`, and the angle is recovered with
`atan2(s, c) / 2`. The factor 2 makes theta and theta+180 fall on the SAME
point of the circle, so the ambiguity disappears by construction instead of
having to be corrected afterwards. It is continuous over the whole range,
including the crossing at 0 and at 180, where a direct regression jumps.

Decoupled head
--------------
Separate branches for classification and for geometry, like YOLOX. Sharing the
trunk between "what it is" and "where it is" worsens both: they are tasks
whose gradients pull in different directions.

Anchor-free. Each cell predicts one box, with `l, t, r, b` as distances to the
cell center. With a single class and banknotes of very variable aspect due to
the 416x416 resizing, anchors would be one more hyperparameter to tune with no
clear gain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from testbank.models.blocks import (
    BaseConv,
    CSPLayer,
    Focus,
    SPPBottleneck,
    _round_channels,
    conv_factory,
)

#: Variants. `depth` scales how many bottlenecks each CSPLayer carries and
#: `width` how many channels. They are YOLOX's; nano also uses separable
#: convolution.
VARIANTS: dict[str, tuple[float, float, bool]] = {
    "nano": (0.33, 0.25, True),
    "tiny": (0.33, 0.375, False),
    "small": (0.33, 0.50, False),
}

#: Spatial reduction of each pyramid level. A P3 cell covers 8x8 pixels; a P5
#: one, 32x32.
STRIDES = (8, 16, 32)


class CSPDarknet(nn.Module):
    """Backbone. Returns the three levels the neck consumes."""

    def __init__(
        self, depth: float, width: float, depthwise: bool = False
    ) -> None:
        super().__init__()
        Conv = conv_factory(depthwise)
        base = _round_channels(64, width)
        n = max(round(3 * depth), 1)

        self.stem = Focus(3, base, kernel=3)
        self.dark2 = nn.Sequential(
            Conv(base, base * 2, 3, 2),
            CSPLayer(base * 2, base * 2, n=n, depthwise=depthwise),
        )
        self.dark3 = nn.Sequential(
            Conv(base * 2, base * 4, 3, 2),
            CSPLayer(base * 4, base * 4, n=n * 3, depthwise=depthwise),
        )
        self.dark4 = nn.Sequential(
            Conv(base * 4, base * 8, 3, 2),
            CSPLayer(base * 8, base * 8, n=n * 3, depthwise=depthwise),
        )
        self.dark5 = nn.Sequential(
            Conv(base * 8, base * 16, 3, 2),
            SPPBottleneck(base * 16, base * 16),
            CSPLayer(
                base * 16, base * 16, n=n, shortcut=False, depthwise=depthwise
            ),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        x = self.dark2(x)
        c3 = self.dark3(x)
        c4 = self.dark4(c3)
        c5 = self.dark5(c4)
        return c3, c4, c5


class PAFPN(nn.Module):
    """Neck: brings semantics down from the top and detail up from the bottom.

    Both directions are needed. Top-down only, the fine level receives context
    but loses localization precision; bottom-up only, the coarse level does
    not know what it is looking at.
    """

    def __init__(
        self, depth: float, width: float, depthwise: bool = False
    ) -> None:
        super().__init__()
        Conv = conv_factory(depthwise)
        base = _round_channels(64, width)
        c3, c4, c5 = base * 4, base * 8, base * 16
        n = max(round(3 * depth), 1)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_c5 = BaseConv(c5, c4, 1, 1)
        self.p4 = CSPLayer(2 * c4, c4, n=n, shortcut=False, depthwise=depthwise)
        self.lateral_c4 = BaseConv(c4, c3, 1, 1)
        self.p3 = CSPLayer(2 * c3, c3, n=n, shortcut=False, depthwise=depthwise)

        self.down_p3 = Conv(c3, c3, 3, 2)
        self.n4 = CSPLayer(2 * c3, c4, n=n, shortcut=False, depthwise=depthwise)
        self.down_n4 = Conv(c4, c4, 3, 2)
        self.n5 = CSPLayer(2 * c4, c5, n=n, shortcut=False, depthwise=depthwise)

    def forward(self, features) -> tuple[torch.Tensor, ...]:
        c3, c4, c5 = features
        top = self.lateral_c5(c5)
        p4 = self.p4(torch.cat([self.upsample(top), c4], dim=1))
        lateral = self.lateral_c4(p4)
        p3 = self.p3(torch.cat([self.upsample(lateral), c3], dim=1))

        n4 = self.n4(torch.cat([self.down_p3(p3), lateral], dim=1))
        n5 = self.n5(torch.cat([self.down_n4(n4), top], dim=1))
        return p3, n4, n5


@dataclass(frozen=True, slots=True)
class HeadSpec:
    """What shape the head has. The loss recipe fixes it, not the user.

    The recipes this project compares do not share a head, and pretending they
    do -- training the Ultralytics recipe on our direct regression -- would be
    comparing something else under its name. So the recipe chooses:

        own / fork       direct regression, angle (sin 2t, cos 2t), objectness
        ultralytics      DFL regression, scalar angle, NO objectness
        ddgrcf           YOLOX regression, raw angle in radians, objectness

    It is stored with the weights: loading a checkpoint rebuilds the head that
    produced it, not the one of the config at the time.
    """

    #: `direct`: four l/t/r/b distances per cell. `dfl`: a DISTRIBUTION of
    #: `reg_max` bins per distance, and the distance is its expectation (Li et
    #: al., "Generalized Focal Loss", NeurIPS 2020). `yolox`: the YOLOX
    #: original, `(dx, dy, log w, log h)` relative to the cell; used by the
    #: DDGRCF port.
    regression: str = "direct"
    reg_max: int = 16
    #: `sincos`: (sin 2t, cos 2t), no discontinuity. `scalar`: one channel,
    #: `theta = (sigmoid(x) - 1/4) * pi`, which is how Ultralytics documents
    #: it. `radians`: one raw channel in radians, untransformed (DDGRCF).
    angle: str = "sincos"
    #: "There is an object" branch. v8-style heads do not carry it: the class
    #: absorbs presence, with soft targets from the assigner.
    objectness: bool = True

    def __post_init__(self) -> None:
        if self.regression not in ("direct", "dfl", "yolox"):
            raise ValueError(f"regression: {self.regression!r}")
        if self.angle not in ("sincos", "scalar", "radians"):
            raise ValueError(f"angle: {self.angle!r}")
        if self.reg_max < 2:
            raise ValueError("reg_max must be >= 2")

    def to_dict(self) -> dict:
        return {
            "regression": self.regression,
            "reg_max": self.reg_max,
            "angle": self.angle,
            "objectness": self.objectness,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> HeadSpec:
        return cls(**data) if data else cls()


@dataclass(frozen=True, slots=True)
class HeadOutput:
    """Raw output of one level, before decoding."""

    #: (B, 4, H, W) -- l, t, r, b distances to the cell center, in stride
    #: units. With DFL it is the EXPECTATION of the distribution; the decoder
    #: does not tell one head from another.
    distances: torch.Tensor
    #: (B, 2, H, W) with `sincos`, (B, 1, H, W) with `scalar`. `decode_angle`
    #: distinguishes by the number of channels.
    angle: torch.Tensor
    #: (B, 1, H, W) -- there is an object here. None if the head has no branch.
    objectness: torch.Tensor | None
    #: (B, C, H, W) -- which class it is.
    classes: torch.Tensor
    stride: int
    #: (B, 4 * reg_max, H, W) -- raw logits of the distribution. Only with
    #: DFL, and only the loss uses them: decoding already goes via `distances`.
    distribution: torch.Tensor | None = None
    #: How to read `distances` and `angle`. With a single angle channel one
    #: cannot tell "sigmoid" from "radians" by looking at the tensor: the head
    #: says so.
    regression: str = "direct"
    angle_mode: str = "sincos"

    def to_cpu(self) -> HeadOutput:
        """For decoding: the rotated NMS goes through shapely, which lives on CPU."""
        move = lambda t: None if t is None else t.detach().cpu()
        return HeadOutput(
            distances=move(self.distances),
            angle=move(self.angle),
            objectness=move(self.objectness),
            classes=move(self.classes),
            stride=self.stride,
            distribution=move(self.distribution),
            regression=self.regression,
            angle_mode=self.angle_mode,
        )


class ObbHead(nn.Module):
    """Decoupled head with an angle branch. Shares weights across levels.

    Sharing them across the three pyramid levels is deliberate: with ~450
    images, three independent heads triple the parameters of the part that
    overfits most easily. The `stride` tells the scale apart when decoding,
    not different weights.
    """

    def __init__(
        self,
        num_classes: int = 1,
        width: float = 0.25,
        depthwise: bool = False,
        in_channels: tuple[int, ...] = (256, 512, 1024),
        spec: HeadSpec | None = None,
    ) -> None:
        super().__init__()
        Conv = conv_factory(depthwise)
        hidden = _round_channels(256, width)
        self.num_classes = num_classes
        self.spec = spec or HeadSpec()

        self.stems = nn.ModuleList(
            [
                BaseConv(_round_channels(c, width), hidden, 1, 1)
                for c in in_channels
            ]
        )
        self.cls_branch = nn.Sequential(
            Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1)
        )
        self.reg_branch = nn.Sequential(
            Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1)
        )
        self.cls_pred = nn.Conv2d(hidden, num_classes, 1)
        reg_channels = 4 * self.spec.reg_max if self.spec.regression == "dfl" else 4
        self.reg_pred = nn.Conv2d(hidden, reg_channels, 1)
        self.angle_pred = nn.Conv2d(hidden, 2 if self.spec.angle == "sincos" else 1, 1)
        self.obj_pred = nn.Conv2d(hidden, 1, 1) if self.spec.objectness else None
        if self.spec.regression == "dfl":
            # The bins 0..reg_max-1, as a buffer so they travel with the module.
            self.register_buffer(
                "bins", torch.arange(self.spec.reg_max, dtype=torch.float32)
            )
        self._init_biases()

    def _init_biases(self, prior: float = 0.01) -> None:
        """Initial bias so the network starts by predicting "almost nothing".

        Without this, at start it predicts object in every cell: thousands of
        false positives whose gradient dominates the first iterations and
        destabilizes training. The value leaves p(object) = 0.01.
        """
        bias = -math.log((1 - prior) / prior)
        for layer in (self.cls_pred, self.obj_pred):
            if layer is not None:
                nn.init.constant_(layer.bias, bias)

    def forward(self, features) -> list[HeadOutput]:
        outputs = []
        for level, (feature, stride) in enumerate(zip(features, STRIDES)):
            x = self.stems[level](feature)
            reg = self.reg_branch(x)
            raw = self.reg_pred(reg)
            if self.spec.regression == "dfl":
                distribution = raw
                distances = self.expected_distances(raw)
            else:
                distribution = None
                # softplus, not exp: distances are positive and exp blows up
                # at the start, when the weights are still noise.
                distances = nn.functional.softplus(raw)
            outputs.append(
                HeadOutput(
                    distances=distances,
                    angle=self.angle_pred(reg),
                    objectness=self.obj_pred(reg) if self.obj_pred is not None else None,
                    classes=self.cls_pred(self.cls_branch(x)),
                    stride=stride,
                    distribution=distribution,
                    regression=self.spec.regression,
                    angle_mode=self.spec.angle,
                )
            )
        return outputs

    def expected_distances(self, logits: torch.Tensor) -> torch.Tensor:
        """`(B, 4 * reg_max, H, W)` -> `(B, 4, H, W)`: the expectation over bins.

        It is the "integral" of DFL: softmax over the bins and weighted sum by
        their index. The distance comes out in stride units, in
        `[0, reg_max - 1]`, which with reg_max=16 and stride 32 is up to 480
        px: more than the image.
        """
        batch, _, height, width = logits.shape
        probabilities = logits.view(batch, 4, self.spec.reg_max, height, width).softmax(dim=2)
        return (probabilities * self.bins.view(1, 1, -1, 1, 1)).sum(dim=2)


class YoloxObb(nn.Module):
    """Backbone + neck + OBB head. The complete candidate."""

    def __init__(
        self,
        variant: str = "nano",
        num_classes: int = 1,
        head: HeadSpec | None = None,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r}; available: {sorted(VARIANTS)}"
            )
        depth, width, depthwise = VARIANTS[variant]
        self.variant = variant
        self.num_classes = num_classes
        self.backbone = CSPDarknet(depth, width, depthwise)
        self.neck = PAFPN(depth, width, depthwise)
        self.head = ObbHead(num_classes, width, depthwise, spec=head)

    @property
    def head_spec(self) -> HeadSpec:
        return self.head.spec

    def forward(self, x: torch.Tensor) -> list[HeadOutput]:
        return self.head(self.neck(self.backbone(x)))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def decode_angle(angle: torch.Tensor, mode: str = "sincos") -> torch.Tensor:
    """Raw output of the angle branch -> theta in radians, in `[0, pi)`.

    - `sincos`, 2 channels `(sin 2t, cos 2t)`: the factor 1/2 undoes the
      doubling. The result always falls within half a turn, which is the whole
      range that distinguishes rectangles: beyond it repeats.
    - `scalar`, 1 channel: `theta = (sigmoid(x) - 1/4) * pi`, in
      `[-pi/4, 3pi/4)`, which is the parametrization Ultralytics documents.
    - `radians`, 1 channel: the value as is, in radians (DDGRCF).

    Everything is taken to `[0, pi)` with the modulo, because that is what
    the rest expects.
    """
    if mode == "sincos":
        if angle.shape[1] != 2:
            raise ValueError(f"sincos needs 2 channels, got {angle.shape[1]}")
        sin2, cos2 = angle[:, 0], angle[:, 1]
        return 0.5 * torch.atan2(sin2, cos2) % math.pi
    if angle.shape[1] != 1:
        raise ValueError(f"{mode} needs 1 channel, got {angle.shape[1]}")
    if mode == "scalar":
        return ((torch.sigmoid(angle[:, 0]) - 0.25) * math.pi) % math.pi
    if mode == "radians":
        return angle[:, 0] % math.pi
    raise ValueError(f"unknown angle mode: {mode!r}")


def encode_angle(theta: torch.Tensor) -> torch.Tensor:
    """Inverse of `decode_angle`, to build the training target."""
    return torch.stack((torch.sin(2 * theta), torch.cos(2 * theta)), dim=-1)


__all__ = [
    "PAFPN",
    "STRIDES",
    "VARIANTS",
    "CSPDarknet",
    "HeadOutput",
    "HeadSpec",
    "ObbHead",
    "YoloxObb",
    "decode_angle",
    "encode_angle",
]
