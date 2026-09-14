"""Candidate registry.

Importing the adapters here is what registers them. Both import their
library LAZILY, inside their methods, so these imports do not require
anything installed: `import testbank` works with the bare base set.
"""

from testbank.detectors import (
    oriented_det,  # noqa: F401  (registers)
    yolox_obb,  # noqa: F401  (registers)
)
from testbank.detectors.base import (
    BaseDetector,
    Detector,
    DetectorError,
    detectors,
    get,
    register,
)

__all__ = [
    "BaseDetector",
    "Detector",
    "DetectorError",
    "detectors",
    "get",
    "register",
]
