"""The unit an adapter predicts on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Sample:
    """An image on disk. `sample_id` names it in the adapter's output; the
    label file is what `main` trains on and is absent for a loose photo."""

    sample_id: str
    image_path: Path
    label_path: Path | None = None


__all__ = ["Sample"]
