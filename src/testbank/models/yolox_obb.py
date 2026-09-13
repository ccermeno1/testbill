"""YOLOX-Nano con cabeza de caja ORIENTADA. Apache-2.0, PyTorch puro.

La cabeza es lo unico que no es YOLOX estandar. YOLOX predice `x, y, w, h`
alineado al eje; aqui hace falta ademas el angulo.

Como se representa el angulo
----------------------------
NO como un escalar en radianes. Un rectangulo girado theta y otro girado
theta+180 son el MISMO rectangulo, asi que una regresion directa sobre theta
castiga al modelo por acertar: predecir 179 cuando la verdad es 1 daria un error
enorme siendo un error de 2 grados. Es el mismo problema que la metrica de
angulo resuelve con `min(|d|, 180-|d|)`.

La cabeza predice `(sin 2t, cos 2t)`, y el angulo se recupera con
`atan2(s, c) / 2`. El factor 2 hace que theta y theta+180 caigan en el MISMO
punto del circulo, asi que la ambiguedad desaparece por construccion en vez de
tener que corregirse despues. Es continuo en todo el rango, incluido el paso por
0 y por 180, donde una regresion directa da un salto.

Cabeza desacoplada
------------------
Ramas separadas para clasificacion y para geometria, como YOLOX. Compartir el
tronco entre "que es" y "donde esta" empeora las dos: son tareas con gradientes
que tiran en direcciones distintas.

Sin anclas. Cada celda predice una caja, con `l, t, r, b` como distancias al
centro de la celda. Con una sola clase y billetes de aspecto muy variable por el
redimensionado a 416x416, las anclas serian un hiperparametro mas que ajustar sin
ganancia clara.
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

#: Variantes. `depth` escala cuantos bottlenecks lleva cada CSPLayer y `width`
#: cuantos canales. Son las de YOLOX; nano ademas usa convolucion separable.
VARIANTS: dict[str, tuple[float, float, bool]] = {
    "nano": (0.33, 0.25, True),
    "tiny": (0.33, 0.375, False),
    "small": (0.33, 0.50, False),
}

#: Reduccion espacial de cada nivel de la piramide. Una celda de P3 cubre 8x8
#: pixeles; una de P5, 32x32.
STRIDES = (8, 16, 32)


class CSPDarknet(nn.Module):
    """Backbone. Devuelve los tres niveles que consume el cuello."""

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
    """Cuello: baja semantica desde arriba y sube detalle desde abajo.

    Los dos sentidos hacen falta. Solo de arriba abajo, el nivel fino recibe
    contexto pero pierde precision de localizacion; solo de abajo arriba, el
    nivel grueso no sabe que esta mirando.
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
    """Que forma tiene la cabeza. Lo fija la receta de perdida, no el usuario.

    Las tres recetas que este proyecto compara no comparten cabeza, y fingir que
    si -- entrenar la receta de Ultralytics sobre nuestra regresion directa --
    seria comparar otra cosa con su nombre puesto. Asi que la receta elige:

        propia / fork    regresion directa, angulo (sin 2t, cos 2t), objectness
        ultralytics      regresion DFL, angulo escalar, SIN objectness

    Va guardada con los pesos: cargar un checkpoint reconstruye la cabeza que
    lo produjo, no la de la config del momento.
    """

    #: `direct`: cuatro distancias l/t/r/b por celda. `dfl`: una DISTRIBUCION
    #: de `reg_max` bins por distancia, y la distancia es su esperanza (Li et
    #: al., "Generalized Focal Loss", NeurIPS 2020). `yolox`: el original de
    #: YOLOX, `(dx, dy, log w, log h)` respecto a la celda; lo usa el port de
    #: DDGRCF.
    regression: str = "direct"
    reg_max: int = 16
    #: `sincos`: (sin 2t, cos 2t), sin discontinuidad. `scalar`: un canal,
    #: `theta = (sigmoid(x) - 1/4) * pi`, que es como lo documenta Ultralytics.
    #: `radians`: un canal crudo en radianes, sin transformar (DDGRCF).
    angle: str = "sincos"
    #: Rama de "hay objeto". Las cabezas al estilo v8 no la llevan: la clase
    #: absorbe la presencia, con objetivos suaves del asignador.
    objectness: bool = True

    def __post_init__(self) -> None:
        if self.regression not in ("direct", "dfl", "yolox"):
            raise ValueError(f"regression: {self.regression!r}")
        if self.angle not in ("sincos", "scalar", "radians"):
            raise ValueError(f"angle: {self.angle!r}")
        if self.reg_max < 2:
            raise ValueError("reg_max tiene que ser >= 2")

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
    """Salida cruda de un nivel, antes de decodificar."""

    #: (B, 4, H, W) -- distancias l, t, r, b al centro de la celda, en unidades
    #: de stride. Con DFL es la ESPERANZA de la distribucion; el decodificador
    #: no distingue una cabeza de otra.
    distances: torch.Tensor
    #: (B, 2, H, W) con `sincos`, (B, 1, H, W) con `scalar`. `decode_angle`
    #: distingue por el numero de canales.
    angle: torch.Tensor
    #: (B, 1, H, W) -- hay objeto aqui. None si la cabeza no lleva la rama.
    objectness: torch.Tensor | None
    #: (B, C, H, W) -- de que clase es.
    classes: torch.Tensor
    stride: int
    #: (B, 4 * reg_max, H, W) -- logits crudos de la distribucion. Solo con
    #: DFL, y solo los usa la perdida: la decodificacion ya va en `distances`.
    distribution: torch.Tensor | None = None
    #: Como leer `distances` y `angle`. Con un solo canal de angulo no se puede
    #: distinguir "sigmoide" de "radianes" mirando el tensor: lo dice la cabeza.
    regression: str = "direct"
    angle_mode: str = "sincos"

    def to_cpu(self) -> HeadOutput:
        """Para decodificar: el NMS rotado va por shapely, que vive en CPU."""
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
    """Cabeza desacoplada con rama de angulo. Comparte pesos entre niveles.

    Compartirlos entre los tres niveles de la piramide es deliberado: con ~450
    imagenes, tres cabezas independientes triplican los parametros de la parte
    que mas facilmente se sobreajusta. El `stride` distingue la escala al
    decodificar, no pesos distintos.
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
            # Los bins 0..reg_max-1, como buffer para que viajen con el modulo.
            self.register_buffer(
                "bins", torch.arange(self.spec.reg_max, dtype=torch.float32)
            )
        self._init_biases()

    def _init_biases(self, prior: float = 0.01) -> None:
        """Sesgo inicial para que la red empiece prediciendo "casi nada".

        Sin esto, al arrancar predice objeto en todas las celdas: miles de
        falsos positivos cuyo gradiente domina las primeras iteraciones y
        desestabiliza el entrenamiento. El valor deja p(objeto) = 0.01.
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
                # softplus, no exp: las distancias son positivas y exp se
                # dispara al principio, cuando los pesos aun son ruido.
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
        """`(B, 4 * reg_max, H, W)` -> `(B, 4, H, W)`: la esperanza de cada bin.

        Es la "integral" de DFL: softmax sobre los bins y suma ponderada por su
        indice. La distancia sale en unidades de stride, en `[0, reg_max - 1]`,
        que con reg_max=16 y stride 32 son hasta 480 px: mas que la imagen.
        """
        batch, _, height, width = logits.shape
        probabilities = logits.view(batch, 4, self.spec.reg_max, height, width).softmax(dim=2)
        return (probabilities * self.bins.view(1, 1, -1, 1, 1)).sum(dim=2)


class YoloxObb(nn.Module):
    """Backbone + cuello + cabeza OBB. El candidato completo."""

    def __init__(
        self,
        variant: str = "nano",
        num_classes: int = 1,
        head: HeadSpec | None = None,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(
                f"variante desconocida {variant!r}; hay {sorted(VARIANTS)}"
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
    """Salida cruda de la rama de angulo -> theta en radianes, en `[0, pi)`.

    - `sincos`, 2 canales `(sin 2t, cos 2t)`: el factor 1/2 deshace el doblado.
      El resultado cae siempre en medio giro, que es todo el rango que
      distingue rectangulos: mas alla se repite.
    - `scalar`, 1 canal: `theta = (sigmoid(x) - 1/4) * pi`, en `[-pi/4, 3pi/4)`,
      que es la parametrizacion que documenta Ultralytics.
    - `radians`, 1 canal: el valor tal cual, en radianes (DDGRCF).

    Todo se lleva a `[0, pi)` con el modulo, porque es lo que espera lo demas.
    """
    if mode == "sincos":
        if angle.shape[1] != 2:
            raise ValueError(f"sincos necesita 2 canales, hay {angle.shape[1]}")
        sin2, cos2 = angle[:, 0], angle[:, 1]
        return 0.5 * torch.atan2(sin2, cos2) % math.pi
    if angle.shape[1] != 1:
        raise ValueError(f"{mode} necesita 1 canal, hay {angle.shape[1]}")
    if mode == "scalar":
        return ((torch.sigmoid(angle[:, 0]) - 0.25) * math.pi) % math.pi
    if mode == "radians":
        return angle[:, 0] % math.pi
    raise ValueError(f"modo de angulo desconocido: {mode!r}")


def encode_angle(theta: torch.Tensor) -> torch.Tensor:
    """Inversa de `decode_angle`, para construir el objetivo de entrenamiento."""
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
