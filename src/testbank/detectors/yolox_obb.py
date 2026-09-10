"""Adaptador del candidato propio: YOLOX-Nano con cabeza OBB.

Apache-2.0 y APTO PARA PRODUCCION, al contrario que Ultralytics. Es el unico
candidato cuyo codigo controlamos entero: sin repositorio ajeno que vendorizar,
sin operadores compilados, sin adivinar el esquema de datos de un dataloader que
no hemos leido.

Y es con diferencia el mas pequeno -- 857k parametros frente a los 2.65M de
YOLO26n -- que es lo que pedia el despliegue movil.

`torch` se importa de forma perezosa igual que `ultralytics` en el otro
adaptador: va en un grupo opcional y `import testbank` no debe exigirlo.
"""

from __future__ import annotations

from pathlib import Path

from testbank.config import Config
from testbank.dataio.formats import ImageSize
from testbank.dataio.image_sizes import SizeIndex
from testbank.detectors.base import (
    BaseDetector,
    DetectorError,
    TrainResult,
    register,
)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise DetectorError(
            "torch no esta instalado. Va en el grupo opcional `torch`: "
            "`uv pip install -e \".[torch]\"`"
        ) from exc
    return torch


@register
class YoloxObbDetector(BaseDetector):
    name = "yolox-obb-nano"
    license = "Apache-2.0"
    production_ready = True
    _NOTE = (
        "Candidato propio: 857k parametros, sin dependencias compiladas y con "
        "la arquitectura entera bajo nuestro control."
    )
    notes = (_NOTE,)

    # --- entrenamiento ----------------------------------------------------

    def train(self, samples_by_split, config: Config, *, output_dir: Path) -> TrainResult:
        _import_torch()
        from testbank.models.data import build_datasets
        from testbank.models.train import fit

        datasets = build_datasets(samples_by_split, config)
        if "train" not in datasets:
            raise DetectorError("hace falta la particion 'train' para entrenar")

        weights, history = fit(datasets["train"], config, output_dir=output_dir)
        last = history.epochs[-1] if history.epochs else {}
        return TrainResult(
            weights=weights,
            epochs=config.detector.epochs,
            notes=(
                f"{len(datasets['train'])} imagenes de entrenamiento",
                "perdidas finales: "
                + ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in last.items()
                    if k in ("box", "angle", "objectness", "classes", "total")
                ),
            ),
        )

    # --- inferencia -------------------------------------------------------

    def predict(self, samples, *, weights: Path, config: Config) -> dict:
        """`sample_id -> [Prediction]` en coordenadas NORMALIZADAS.

        Las predicciones se decodifican en el espacio de ENTRADA de la red
        (`image_size`), y como los quads son normalizados no hace falta
        reescalar: la normalizacion ya absorbe el cambio de tamano. Se pasa el
        tamano real solo para el aspecto del orden canonico.
        """
        torch = _import_torch()
        import cv2
        import numpy as np

        from testbank.models.decode import detections
        from testbank.models.train import load_model

        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        model = load_model(weights)
        side = config.detector.image_size

        out: dict[str, list] = {}
        with torch.no_grad():
            for sample in samples:
                image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise DetectorError(f"no se pudo leer {sample.image_path}")
                resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_LINEAR)
                tensor = (
                    torch.from_numpy(
                        np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))
                    )
                    .float()
                    .div_(255.0)
                    .unsqueeze(0)
                )
                outputs = model(tensor)
                width, height = sizes.size(sample.sample_id)
                out[sample.sample_id] = detections(
                    outputs,
                    ImageSize(side, side),
                    confidence=config.detector.confidence_threshold,
                    iou_threshold=config.detector.nms_iou,
                )
                # El aspecto real solo importa para el orden canonico, y los
                # quads ya salen normalizados sobre un cuadrado. Se recanonicaliza
                # con el aspecto verdadero para que el ancla del lado mas largo
                # sea la geometrica y no la del cuadrado de entrada.
                out[sample.sample_id] = _recanonicalize(
                    out[sample.sample_id], width / height
                )
        return out


def _recanonicalize(predictions, aspect: float):
    """Reordena los vertices con el aspecto REAL de la imagen.

    La red trabaja sobre un cuadrado, asi que ahi el lado mas largo normalizado
    coincide con el geometrico. En la imagen original no tiene por que: es la
    anisotropia que ya ha mordido dos veces en este proyecto.
    """
    import warnings

    from testbank.geometry.quad import (
        CoordinateRangeWarning,
        QuadShapeWarning,
        canonicalize,
    )
    from testbank.metrics.core import Prediction

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuadShapeWarning)
        warnings.simplefilter("ignore", CoordinateRangeWarning)
        for prediction in predictions:
            out.append(
                Prediction(
                    quad=canonicalize(prediction.quad, aspect=aspect),
                    score=prediction.score,
                    class_id=prediction.class_id,
                )
            )
    return out


__all__ = ["YoloxObbDetector"]
