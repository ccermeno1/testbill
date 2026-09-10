"""Bloques de YOLOX. Apache-2.0, PyTorch puro, sin extensiones compiladas.

Se implementan aqui en vez de vendorizar el repositorio de Megvii por tres
motivos:

1. **No hay paquete usable.** YOLOX no se publica en PyPI como libreria; el
   nombre `yolox` en PyPI es otra cosa. La alternativa seria clonar un
   repositorio entero para usar el 15% de su codigo.
2. **Sin compilar nada.** Es el motivo por el que este candidato existe: todo lo
   que hay aqui es `torch.nn`. Ver el problema de MMCV en el README.
3. **Control del esquema.** Es el unico candidato del que controlamos la
   arquitectura entera, asi que la cabeza OBB se acopla sin adivinar convenios
   ajenos.

Nano usa convolucion separable en profundidad (`depthwise`) en todo salvo el
tronco. Es de donde sale la diferencia de 0.91M frente a los 5.06M de tiny: una
convolucion 3x3 de C canales a C pasa de 9C^2 a 9C + C^2 parametros.
"""

from __future__ import annotations

import torch
from torch import nn


def _round_channels(channels: int, width: float) -> int:
    """Escala los canales y los deja en multiplo de 8.

    Los multiplos de 8 no son estetica: las rutinas vectorizadas de CPU y las
    unidades de matriz de los moviles trabajan por bloques, y un canal suelto
    obliga a rellenar. Es el mismo redondeo que usa YOLOX.
    """
    scaled = max(1, int(channels * width))
    return max(8, (scaled + 4) // 8 * 8)


class BaseConv(nn.Module):
    """Conv + BatchNorm + SiLU, el ladrillo de toda la red."""

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
        # `kernel // 2` mantiene el tamano espacial con stride 1, que es lo que
        # permite sumar ramas laterales sin recortar.
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
    """Convolucion separable: una espacial por canal y una 1x1 que los mezcla.

    Es el cambio que hace nano pequeno de verdad. Una 3x3 de C a C cuesta 9C^2;
    separada cuesta 9C + C^2. Con C=64 son 36864 frente a 4672.
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
    """Cross-Stage Partial: parte los canales, procesa una mitad y reconcatena.

    La gracia es que solo la mitad pasa por los bottlenecks, asi que el coste
    baja sin perder el camino de gradiente completo.
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
    """Spatial Pyramid Pooling: mezcla contexto a varias escalas sin coste.

    Tres max-pool de 5, 9 y 13 en paralelo. Para un billete grande en primer
    plano, el de 13 le da al detector campo receptivo suficiente sin anadir
    parametros.
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
    """Submuestrea 2x reordenando pixeles en canales, sin perder informacion.

    Coge uno de cada dos pixeles en las dos direcciones y apila los cuatro
    mosaicos como canales. A diferencia de una conv con stride 2, aqui no se
    descarta nada: la informacion cambia de sitio, no desaparece.
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
