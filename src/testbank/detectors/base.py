"""Protocolo `Detector` y registro por decorador.

Anadir un candidato es escribir `train` y `predict` en una clase y decorarla con
`@register`. Nada de `if candidato ==`.

Por que `evaluate` NO lo implementa cada adaptador
--------------------------------------------------
La especificacion pide `train`, `predict` y `evaluate` en el protocolo. Los dos
primeros son necesariamente propios de cada candidato. El tercero NO puede serlo.

Si cada adaptador trajera su `evaluate`, cada uno usaria el de su libreria:
Ultralytics calcula el mAP a su manera, MMDetection a la suya, y las cifras de la
tabla comparativa dejarian de ser comparables aunque compartieran nombre. Seria
justo el error que la tabla existe para evitar.

Asi que `evaluate` es concreto, vive aqui, y esta escrito sobre `predict`: todos
los candidatos se puntuan con NUESTRAS metricas, el mismo IoU rotado y el mismo
emparejamiento. Un adaptador puede sobrescribirlo, pero entonces sus numeros no
son comparables y mas vale que quede dicho en el sitio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from testbank.config import Config
from testbank.dataio.formats import ImageSize
from testbank.dataio.image_sizes import SizeIndex
from testbank.dataio.prepare import load_samples
from testbank.experiment.run import ComponentInfo
from testbank.geometry.quad import Quad
from testbank.metrics.core import ImageEval, Prediction
from testbank.metrics.evaluate import evaluate as evaluate_metrics


class DetectorError(RuntimeError):
    """El candidato no se puede usar en este entorno."""


@dataclass(frozen=True, slots=True)
class TrainResult:
    weights: Path
    epochs: int
    notes: tuple[str, ...] = ()


@runtime_checkable
class Detector(Protocol):
    name: ClassVar[str]
    license: ClassVar[str]
    production_ready: ClassVar[bool]

    def train(self, samples, config: Config, *, output_dir: Path) -> TrainResult: ...

    def predict(self, samples, *, weights: Path, config: Config) -> dict: ...

    def evaluate(self, samples, config: Config, *, weights: Path) -> dict: ...


class BaseDetector:
    """Implementa `evaluate` sobre `predict`. Los adaptadores heredan de aqui."""

    name: ClassVar[str] = ""
    license: ClassVar[str] = ""
    production_ready: ClassVar[bool] = False
    #: Que se dice de este candidato en el registro de la ejecucion.
    notes: ClassVar[tuple[str, ...]] = ()

    def component(self) -> ComponentInfo:
        return ComponentInfo(
            name=self.name,
            license=self.license,
            production_ready=self.production_ready,
        )

    def train(self, samples, config: Config, *, output_dir: Path) -> TrainResult:
        raise NotImplementedError

    def predict(self, samples, *, weights: Path, config: Config) -> dict:
        """`sample_id -> lista de Prediction`, en coordenadas normalizadas."""
        raise NotImplementedError

    def evaluate(self, samples, config: Config, *, weights: Path) -> dict:
        """Puntua con NUESTRAS metricas. Igual para todos los candidatos.

        La verdad se lee por la puerta unica (`load_samples`), asi que el filtro
        de area relativa se aplica aqui igual que en todas partes, y lo que el
        filtro descarta entra como `ignored`: detectarlo no penaliza.
        """
        samples = list(samples)
        predictions = self.predict(samples, weights=weights, config=config)
        return evaluate_metrics(
            build_image_evals(samples, predictions, config=config), config
        )


def build_image_evals(samples, predictions: dict, *, config: Config) -> list[ImageEval]:
    """Une verdad filtrada, descartes y predicciones en lo que comen las metricas."""
    sizes = SizeIndex.for_samples(
        samples, cache_path=config.data.derived_dir / "image_sizes.json"
    )
    loaded, _ = load_samples(
        samples,
        sizes=sizes,
        min_relative_area=config.annotation_policy.min_relative_area,
    )
    items: list[ImageEval] = []
    for item in loaded:
        width, height = sizes.size(item.sample_id)
        items.append(
            ImageEval(
                sample_id=item.sample_id,
                size=ImageSize(int(width), int(height)),
                truths=item.quads,
                predictions=tuple(predictions.get(item.sample_id, ())),
                ignored=tuple(d.quad for d in item.dropped),
            )
        )
    return items


REGISTRY: dict[str, BaseDetector] = {}


def register(cls: type) -> type:
    if not cls.name:
        raise DetectorError(f"{cls.__name__} no declara `name`")
    if cls.name in REGISTRY:
        raise DetectorError(f"detector duplicado en el registro: {cls.name!r}")
    REGISTRY[cls.name] = cls()
    return cls


def get(name: str) -> BaseDetector:
    try:
        return REGISTRY[name]
    except KeyError:
        raise DetectorError(
            f"detector desconocido: {name!r}; registrados: {sorted(REGISTRY)}"
        ) from None


def detectors() -> list[str]:
    return sorted(REGISTRY)


def production_candidates() -> list[str]:
    """Los que pueden ir a produccion. Los demas son referencia de rendimiento."""
    return sorted(n for n, d in REGISTRY.items() if d.production_ready)


__all__ = [
    "REGISTRY",
    "BaseDetector",
    "Detector",
    "DetectorError",
    "Prediction",
    "Quad",
    "TrainResult",
    "build_image_evals",
    "detectors",
    "get",
    "production_candidates",
    "register",
]
