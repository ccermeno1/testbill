"""Peso de la perdida de angulo segun lo cuadrada que sea la caja.

Por que hace falta
------------------
La representacion `(cx, cy, w, h, theta)` es ambigua cuando `w ~ h`: la caja
`(w, h, t)` y la caja `(h, w, t+90)` son el MISMO rectangulo. La codificacion
`(sin 2t, cos 2t)` resuelve que `t` y `t+180` sean el mismo, pero NO esto: manda
las dos versiones a puntos opuestos del circulo, asi que el modelo recibiria dos
objetivos contradictorios para la misma caja.

Medido sobre el export: 145 de 762 anotaciones (19%) tienen ratio < 1.1.

Por que atenuar y no arreglarlo
-------------------------------
Porque en el caso ambiguo el angulo DA IGUAL para lo que nos importa. Un
rectangulo casi cuadrado recortado con 5 grados de error tapa practicamente lo
mismo, y la politica de anotacion ya dice que un rectangulo aproximado basta.
Castigar al modelo por no acertar algo que ni esta bien definido ni cambia el
resultado es gastar capacidad en ruido.

La alternativa seria la representacion gaussiana, que absorbe la ambiguedad de
forma natural. Descartada a proposito: se aleja del IoU rotado por shapely con
el que medimos, y esa trazabilidad pesa mas que la elegancia de la formulacion.

Todo parametrizable
-------------------
Umbral, forma y suelo van en `detector.loss.angle_weight`, con `enabled` para
apagarlo. La pregunta "cuanto aporta esto" se responde entrenando con y sin, no
razonando. Ver `AngleWeightConfig`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

#: Formas de subida entre el cuadrado perfecto y el umbral. La clave es lo que
#: acepta `decay` en la config; anadir una es anadir una entrada aqui.
DECAYS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # Sube recto. La mas simple de interpretar: el peso es la fraccion del
    # camino recorrido hacia el umbral.
    "linear": lambda t: t,
    # Arranca despacio y frena al final. Deja casi sin peso la franja mas
    # ambigua en vez de subir desde el primer momento.
    "smoothstep": lambda t: t * t * (3.0 - 2.0 * t),
    # Intermedia: arranca despacio pero no frena.
    "quadratic": lambda t: t * t,
    # Escalon. Sirve de referencia para medir si la transicion suave aporta
    # algo frente a cortar por lo sano.
    "step": lambda t: (t >= 1.0).to(t.dtype),
}


def angle_weight(
    ratio: torch.Tensor,
    *,
    enabled: bool = True,
    ratio_threshold: float = 1.1,
    min_weight: float = 0.0,
    decay: str = "smoothstep",
) -> torch.Tensor:
    """Peso en `[min_weight, 1]` para la perdida de angulo de cada caja.

    `ratio` es lado largo / lado corto de la caja VERDADERA, en pixeles. En
    pixeles y no normalizado: normalizar escala x e y por factores distintos, y
    el ratio dejaria de ser el geometrico -- el mismo error que ya mordio dos
    veces en este proyecto.

    Con `enabled=False` devuelve unos: es la rama de control del experimento, y
    tiene que costar exactamente lo mismo escribirla que la otra.
    """
    if not enabled:
        return torch.ones_like(ratio)
    if ratio_threshold <= 1.0:
        raise ValueError(
            f"ratio_threshold tiene que ser > 1, se recibio {ratio_threshold}"
        )
    try:
        shape = DECAYS[decay]
    except KeyError:
        raise ValueError(
            f"forma de decaimiento desconocida {decay!r}; hay {sorted(DECAYS)}"
        ) from None

    # Un ratio por debajo de 1 no existe: es el lado largo entre el corto. Si
    # llega, es que alguien los ha intercambiado, y truncar en 1 evita pesos
    # negativos sin ocultar el problema (el peso saldria minimo, no absurdo).
    progress = ((ratio.clamp(min=1.0) - 1.0) / (ratio_threshold - 1.0)).clamp(0.0, 1.0)
    return min_weight + (1.0 - min_weight) * shape(progress)


def side_ratio_px(width: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """Lado largo / lado corto, sin asumir cual de los dos es cual.

    El orden de `w` y `h` es justo lo que la ambiguedad vuelve arbitrario, asi
    que el ratio no puede depender de el.
    """
    long_side = torch.maximum(width, height)
    short_side = torch.minimum(width, height)
    return long_side / short_side.clamp(min=1e-6)


__all__ = ["DECAYS", "angle_weight", "side_ratio_px"]
