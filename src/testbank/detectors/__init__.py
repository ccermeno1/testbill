"""Candidate registry.

Importing the adapters here is what registers them. All of them import their
library LAZILY, inside their methods, so these imports do not require anything
installed: `import testbank` works with the bare base set.

It is what lets `list` and `compare` show candidates this environment cannot
run -- RTMDet-R needs torch 2.0 in a separate environment, and it still has
to appear in the table.
"""

from testbank.detectors import (
    rtmdet_r,  # noqa: F401  (registers)
    ultralytics_obb,  # noqa: F401  (registers)
    yolox_obb,  # noqa: F401  (registers)
)
from testbank.detectors.base import (
    BaseDetector,
    Detector,
    DetectorError,
    TrainResult,
    detectors,
    get,
    production_candidates,
    register,
)

__all__ = [
    "BaseDetector",
    "Detector",
    "DetectorError",
    "TrainResult",
    "detectors",
    "get",
    "production_candidates",
    "register",
]
