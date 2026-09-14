"""What an adapter returns per image."""

from __future__ import annotations

from dataclasses import dataclass

from testbank.geometry.quad import Quad


@dataclass(frozen=True, slots=True)
class Prediction:
    """A detection, quad in normalized coordinates.

    `quad` may be None: a box the network predicted so far outside the image
    that `Quad` will not accept it (more than half a frame). It is still a
    detection the model made, so `main` counts it as a false positive; here
    it has nothing to draw or crop and is skipped.
    """

    quad: Quad | None
    score: float
    class_id: int = 0


__all__ = ["Prediction"]
