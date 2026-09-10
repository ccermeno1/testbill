"""Registro de candidatos.

Importar los adaptadores aqui es lo que los registra. El de Ultralytics importa
`ultralytics` de forma PEREZOSA, dentro de sus metodos, asi que este import no
exige tener el paquete instalado.
"""

from testbank.detectors import ultralytics_obb  # noqa: F401  (registra)
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
