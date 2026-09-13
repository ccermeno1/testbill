"""Registro de candidatos.

Importar los adaptadores aqui es lo que los registra. Todos importan su libreria
de forma PEREZOSA, dentro de sus metodos, asi que estos imports no exigen tener
nada instalado: `import testbank` funciona con el conjunto base pelado.

Es lo que permite que `list` y `compare` enseñen candidatos que este entorno no
puede ejecutar -- RTMDet-R necesita torch 2.0 en un entorno aparte, y aun asi
tiene que aparecer en la tabla.
"""

from testbank.detectors import (
    rtmdet_r,  # noqa: F401  (registra)
    ultralytics_obb,  # noqa: F401  (registra)
    yolox_obb,  # noqa: F401  (registra)
    yolox_obb_ddgrcf,  # noqa: F401  (registra)
    yolox_obb_fork,  # noqa: F401  (registra)
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
