"""Intervalos por bootstrap. Toda fila de la tabla comparativa lleva uno.

Con ~500 imagenes, validacion queda en ~100 y las diferencias entre candidatos
caen dentro del ruido. Un mAP de 0.81 frente a otro de 0.78 no dice nada si los
intervalos se solapan de par en par, y una tabla de medias pelada invita a leer
como mejora lo que es dispersion.

La unidad de remuestreo es la IMAGEN, nunca la deteccion. Varios billetes de una
misma imagen comparten fondo, iluminacion, camara y anotador: estan
correlacionados. Remuestrear detecciones los trata como independientes y
estrecha los intervalos artificialmente, que es la unica cosa peor que no
tenerlos, porque da confianza falsa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SAMPLES = 2000
DEFAULT_SEED = 20260910


@dataclass(frozen=True, slots=True)
class Interval:
    value: float
    ci_low: float
    ci_high: float
    n: int
    samples: int

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "bootstrap_samples": self.samples,
        }

    def describe(self) -> str:
        return f"{self.value:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}] (n={self.n})"


def bootstrap_images(
    units,
    statistic,
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> Interval:
    """Percentil bootstrap sobre `units`, que son IMAGENES.

    `statistic` recibe una lista de imagenes remuestreadas y devuelve un numero.
    Recibe la lista entera y no un agregado precalculado a proposito: metricas
    como el mAP tienen un ranking global y no se pueden promediar por imagen.

    Determinista: mismo `seed`, mismo intervalo.
    """
    units = list(units)
    n = len(units)
    point = float(statistic(units))
    if n < 2:
        # Con una imagen no hay dispersion que estimar. Devolver un intervalo de
        # anchura cero seria mentir; se devuelve el punto y n para que
        # `compare` lo muestre como lo que es.
        return Interval(point, float("nan"), float("nan"), n, 0)

    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        picked = rng.integers(0, n, size=n)
        values[i] = statistic([units[j] for j in picked])

    values = values[~np.isnan(values)]
    if values.size == 0:
        return Interval(point, float("nan"), float("nan"), n, samples)

    alpha = (1.0 - confidence) / 2.0
    return Interval(
        value=point,
        ci_low=float(np.percentile(values, 100 * alpha)),
        ci_high=float(np.percentile(values, 100 * (1.0 - alpha))),
        n=n,
        samples=samples,
    )
