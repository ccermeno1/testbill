"""`Detector` protocol and registration by decorator, inference side only.

This branch serves models trained on `main`: an adapter here knows how to
`load` a run's weights and `predict` on images, nothing else. Adding a
candidate is writing those two in a class and decorating it with
`@register`; the run's `run.json` names the adapter, so the registered name
must match what `main` registered when it trained.

Every adapter imports its framework LAZILY, inside its methods: `import
testbank.detectors` works with the bare base set, and a run whose framework
is not installed lists in the app and fails to predict with a
`DetectorError` that says what is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from testbank.config import Config


class DetectorError(RuntimeError):
    """The candidate cannot be used in this environment."""


@runtime_checkable
class Detector(Protocol):
    name: ClassVar[str]
    license: ClassVar[str]

    def load(self, weights: Path, config: Config): ...

    def predict(self, samples, *, weights: Path, config: Config, model=None) -> dict: ...

    def explain(
        self, image, prediction, *, weights: Path, config: Config, model=None, method: str = "gradcam"
    ): ...


class BaseDetector:
    """Adapters inherit from here."""

    name: ClassVar[str] = ""
    license: ClassVar[str] = ""
    #: What is said about this candidate.
    notes: ClassVar[tuple[str, ...]] = ()

    def load(self, weights: Path, config: Config):
        """The model behind `weights`, ready for repeated `predict` calls
        (`predict(..., model=loaded)`), for a caller that serves many images
        one at a time and cannot pay the load each time (the app).

        None means the adapter does not support it and `predict` loads on
        every call. Thresholds that a framework bakes into the loaded object
        are those of `config` at load time: reload when they change.
        """
        return

    def predict(self, samples, *, weights: Path, config: Config, model=None) -> dict:
        """`sample_id -> list of Prediction`, in normalized coordinates,
        each list ordered by score. `samples` carry `sample_id` and
        `image_path`; `model` is what `load` returned, or None to load here.
        """
        raise NotImplementedError

    def explain(
        self, image, prediction, *, weights: Path, config: Config, model=None, method: str = "gradcam"
    ):
        """A class-activation map for `prediction` on `image` (BGR array):
        `(H, W)` in [0, 1] at the network's input size, or None when the
        adapter cannot (no access to its feature maps). `method` is
        `gradcam` (needs a prediction) or `eigencam` (prediction ignored).
        See `models/explain.py`.
        """
        return


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


__all__ = [
    "REGISTRY",
    "BaseDetector",
    "Detector",
    "DetectorError",
    "detectors",
    "get",
    "register",
]
