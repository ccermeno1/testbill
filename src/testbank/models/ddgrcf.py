"""Port en torch puro de la red de `DDGRCF/YOLOX_OBB` (`yoloxs_obb.yaml`).

Por que un port y no el clon
----------------------------
El clon exige operadores C++/CUDA compilados y una GPU, y ninguna de las dos
cosas hay aqui. Pero su red -- backbone YOLOv5 con bloques `C3` y ReLU, cuello
PAFPN, cabezas desacopladas de YOLOX -- es torch puro. Lo unico compilado eran
el IoU rotado del asignador y de la perdida, y el NMS. Asi que la red se porta
y esas tres piezas se sustituyen por versiones en torch (`polygon.py`).

Licencia: Apache-2.0, con atribucion. Los bloques de abajo son los suyos
(`yolox/models/modules/common.py` y `block.py`), reducidos a lo que el yaml usa.

Lo unico que importa de verdad: los NOMBRES
-------------------------------------------
El motivo de portar en vez de reescribir es cargar **sus pesos de DOTA**. Su
modelo se construye desde el yaml en un `nn.Sequential` llamado `model`, con las
capas indexadas 0..33 en el orden del fichero; los parametros salen como
`model.0.conv.weight`, `model.2.m.0.cv1.bn.bias`, `model.33.cls_preds.0.bias`...
Aqui se replica ese arbol a mano, capa a capa, con los mismos atributos dentro
de cada bloque, para que `load_state_dict(strict=True)` acepte su checkpoint
sin ningun mapeo. `tests/test_ddgrcf_port.py` compara las claves y las formas
contra su modelo construido de verdad.

Que cambia respecto a la cabeza propia
--------------------------------------
- Regresion al estilo YOLOX: `(dx, dy, log w, log h)` respecto a la celda, no
  distancias l/t/r/b. Se decodifica `cx = (dx + i) * stride`, `w = e^{log w} *
  stride`.
- Angulo: un escalar CRUDO en radianes, sin sigmoide ni doblado. Su convencion
  (`mintheta_obb`): el lado largo es `w` y el angulo cae en `(-pi/4, pi/4]`,
  intercambiando `w`/`h` si hace falta.
- Con objectness, como YOLOX.

Todo eso lo describe un `HeadSpec(regression="yolox", angle="radians")` y lo
consume el mismo decodificador que el resto del proyecto.
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


# --- bloques, con SUS nombres de atributo (Apache-2.0, DDGRCF/YOLOX_OBB) -----


class Conv(nn.Module):
    """`conv` + `bn` + `act`. ReLU en todo el yaml de este modelo."""

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
    """Las 1x1 finales de su `OBBDetectX`: `cls_preds`, `reg_preds`, `obj_preds`.

    Solo las capas. Su asignador, su perdida y su postproceso -- lo que llevaba
    los operadores compilados -- viven en `recipes.py` y `polygon.py`.
    Devuelve `HeadOutput` para que el decodificador comun no sepa de donde viene.
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


# --- la red entera, capa a capa como en el yaml -----------------------------


class DdgrcfYoloxObb(nn.Module):
    """`yoloxs_obb.yaml` de DDGRCF, construido a mano con los mismos indices.

    `self.model[i]` es la capa `i` del yaml, y `.f` dice de donde toma su
    entrada, exactamente como su `parse_model`. El `forward` es su
    `forward_once`. Cada linea de abajo lleva el numero de capa del yaml.
    """

    HEAD_SPEC = HeadSpec(regression="yolox", angle="radians", objectness=True)

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.variant = "ddgrcf-s"
        c64, c128, c256, c512, c1024 = (_width(c) for c in (64, 128, 256, 512, 1024))

        def seq(module_factory, n):
            # Como su `parse_model`: `n` pasa por el multiplicador de profundidad
            # y, si queda en 1, la capa va SIN envoltorio Sequential. El yaml pone
            # `n=2` en los tallos de las cabezas y `round(2 * 0.33) = 1`: es una
            # sola conv, y sus claves son `model.27.conv.*`, no `model.27.0.*`.
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
            # ---- cuello ----
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
            # ---- laterales y tallos de las cabezas ----
            (Conv(c256, c256, 1, 1), 17),                         # 24 lateral0
            (Conv(c512, c256, 1, 1), 20),                         # 25 lateral1
            (Conv(c1024, c256, 1, 1), 23),                        # 26 lateral2
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 24),         # 27 cls0
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 24),         # 28 reg0
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 25),         # 29 cls1
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 25),         # 30 reg1
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 26),         # 31 cls2
            (seq(lambda: Conv(c256, c256, 3, 1), 2), 26),         # 32 reg2
            # ---- deteccion ----
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
    """Carga SU checkpoint de DOTA. Devuelve lo que no encajo, que debe ser solo
    la capa de clase: sus pesos tienen 15 salidas y aqui hay una.

    Estricto en todo lo demas a proposito: si un nombre no coincide es que el
    port se ha desviado del yaml, y eso hay que saberlo, no taparlo.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    own = model.state_dict()
    kept, skipped = {}, {}
    for key, value in state.items():
        if key not in own:
            skipped[key] = "no existe en el port"
        elif own[key].shape != value.shape:
            skipped[key] = f"forma {tuple(value.shape)} != {tuple(own[key].shape)}"
        else:
            kept[key] = value
    missing = sorted(set(own) - set(kept))
    unexpected_missing = [k for k in missing if "cls_preds" not in k]
    if unexpected_missing:
        raise RuntimeError(
            "el checkpoint no cubre el port: faltan "
            f"{len(unexpected_missing)} tensores fuera de la capa de clase, "
            f"el primero {unexpected_missing[0]!r}"
        )
    model.load_state_dict(kept, strict=False)
    return skipped


__all__ = ["DdgrcfYoloxObb", "OBBDetectX", "load_pretrained", "make_divisible"]
