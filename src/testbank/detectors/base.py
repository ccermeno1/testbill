"""`Detector` protocol and registration by decorator.

Adding a candidate is writing `train` and `predict` in a class and decorating
it with `@register`. No `if candidate ==`.

Why `evaluate` is NOT implemented by each adapter
-------------------------------------------------
The specification asks for `train`, `predict` and `evaluate` in the protocol.
The first two are necessarily specific to each candidate. The third CANNOT be.

If each adapter brought its own `evaluate`, each would use its library's:
Ultralytics computes mAP its way, MMDetection its own, and the figures of the
comparison table would stop being comparable even though they share a name.
It would be exactly the mistake the table exists to avoid.

So `evaluate` is concrete, lives here, and is written on top of `predict`: all
candidates are scored with OUR metrics, the same rotated IoU and the same
matching. An adapter can override it, but then its numbers are not comparable
and it had better be said in place.
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
    """The candidate cannot be used in this environment."""


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
    """Implements `evaluate` on top of `predict`. Adapters inherit from here."""

    name: ClassVar[str] = ""
    license: ClassVar[str] = ""
    production_ready: ClassVar[bool] = False
    #: What is said about this candidate in the run record.
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
        """`sample_id -> list of Prediction`, in normalized coordinates."""
        raise NotImplementedError

    def evaluate(self, samples, config: Config, *, weights: Path) -> dict:
        """Score with OUR metrics. The same for all candidates.

        The truth is read through the single door (`load_samples`), so the
        relative area filter is applied here just like everywhere else, and
        what the filter discards enters as `ignored`: detecting it does not
        penalize.
        """
        samples = list(samples)
        predictions = self.predict(samples, weights=weights, config=config)
        return evaluate_metrics(
            build_image_evals(samples, predictions, config=config), config
        )


def build_image_evals(samples, predictions: dict, *, config: Config) -> list[ImageEval]:
    """Joins filtered truth, drops and predictions into what the metrics consume."""
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
        raise DetectorError(f"{cls.__name__} does not declare `name`")
    if cls.name in REGISTRY:
        raise DetectorError(f"duplicate detector in the registry: {cls.name!r}")
    REGISTRY[cls.name] = cls()
    return cls


def get(name: str) -> BaseDetector:
    try:
        return REGISTRY[name]
    except KeyError:
        raise DetectorError(
            f"unknown detector: {name!r}; registered: {sorted(REGISTRY)}"
        ) from None


def detectors() -> list[str]:
    return sorted(REGISTRY)


def production_candidates() -> list[str]:
    """The ones that can go to production. The rest are performance references."""
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
