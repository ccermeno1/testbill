"""Adaptador de Ultralytics YOLO-OBB. Linea base de rendimiento, NO produccion.

AGPL-3.0
--------
`production_ready = False`, y no como formalidad: la AGPL contamina un producto
cerrado. Este candidato existe para tener un numero de referencia rapido -- sin
el no se sabe si los demas van bien o mal -- y `compare` marca su fila como no
apta para que nadie la confunda con un candidato.

ESTE ES EL UNICO MODULO DEL PROYECTO QUE PUEDE IMPORTAR `ultralytics`.
`tests/test_detectors.py::test_aislamiento_de_ultralytics` lo verifica sobre el
arbol de fuentes. Ademas el import es PEREZOSO, dentro de los metodos, para que
`import testbank` funcione sin tener el paquete instalado: va en el grupo
opcional `ultralytics` del pyproject, no en las dependencias base.
"""

from __future__ import annotations

from pathlib import Path

from testbank.config import Config
from testbank.dataio.image_sizes import SizeIndex
from testbank.detectors import dataset as dataset_view
from testbank.detectors.base import (
    BaseDetector,
    DetectorError,
    TrainResult,
    register,
)
from testbank.geometry.quad import Quad, canonicalize
from testbank.metrics.core import Prediction

#: Variante nano por defecto: el despliegue es movil y la especificacion pide
#: priorizar nano/small.
DEFAULT_MODEL = "yolo26n-obb.pt"


def _import_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectorError(
            "ultralytics no esta instalado. Va en un grupo opcional a proposito, "
            "porque es AGPL-3.0 y no puede entrar en el conjunto base: "
            "`uv pip install -e \".[ultralytics]\"`"
        ) from exc
    return YOLO


@register
class UltralyticsObb(BaseDetector):
    name = "ultralytics-yolo-obb"
    license = "AGPL-3.0"
    production_ready = False
    _NOTE = (
        "Referencia de rendimiento, no candidato: AGPL-3.0 contamina un "
        "producto cerrado."
    )
    notes = (_NOTE,)

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    # --- entrenamiento ----------------------------------------------------

    def train(self, samples_by_split, config: Config, *, output_dir: Path) -> TrainResult:
        """`samples_by_split` es `{"train": [...], "valid": [...]}` de SplitLoader.

        Se materializa una vista con las etiquetas YA FILTRADAS: entrenar contra
        la verdad sin filtrar y medir contra la filtrada haria que las cifras no
        significaran nada. Ver `detectors/dataset.py`.
        """
        YOLO = _import_ultralytics()
        view = dataset_view.materialize(samples_by_split, config)

        model = YOLO(self.model)
        model.train(
            data=str(view.data_yaml),
            epochs=config.detector.epochs,
            imgsz=config.detector.image_size,
            batch=config.detector.batch_size,
            seed=config.metrics.seed,
            deterministic=True,
            project=str(output_dir),
            name="train",
            exist_ok=True,
            verbose=False,
        )
        weights = output_dir / "train" / "weights" / "best.pt"
        if not weights.exists():
            raise DetectorError(f"el entrenamiento no dejo pesos en {weights}")
        return TrainResult(
            weights=weights,
            epochs=config.detector.epochs,
            notes=(view.describe(),),
        )

    # --- inferencia -------------------------------------------------------

    def predict(self, samples, *, weights: Path, config: Config) -> dict:
        """Devuelve `sample_id -> [Prediction]` en coordenadas NORMALIZADAS.

        Ultralytics da los quads en pixeles (`obb.xyxyxyxy`), asi que se
        normalizan aqui, en la frontera del adaptador. Dentro de testbank todo
        quad es normalizado y todo calculo metrico es en pixeles; mezclarlo en
        medio del pipeline es el error que ya nos ha mordido dos veces.
        """
        YOLO = _import_ultralytics()
        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        model = YOLO(str(weights))

        out: dict[str, list[Prediction]] = {}
        for sample in samples:
            width, height = sizes.size(sample.sample_id)
            result = model.predict(
                str(sample.image_path),
                conf=config.detector.confidence_threshold,
                iou=config.detector.nms_iou,
                verbose=False,
            )[0]
            out[sample.sample_id] = list(
                _to_predictions(result, width, height, aspect=width / height)
            )
        return out


def _to_predictions(result, width: int, height: int, *, aspect: float):
    """Traduce el `obb` de Ultralytics a nuestros quads canonicos."""
    obb = getattr(result, "obb", None)
    if obb is None or obb.xyxyxyxy is None:
        return
    corners = obb.xyxyxyxy.cpu().numpy()
    scores = obb.conf.cpu().numpy()
    classes = obb.cls.cpu().numpy().astype(int)
    for points, score, class_id in zip(corners, scores, classes):
        normalized = [(float(x) / width, float(y) / height) for x, y in points]
        yield Prediction(
            quad=canonicalize(Quad.from_xy(normalized), aspect=aspect),
            score=float(score),
            class_id=int(class_id),
        )


__all__ = ["DEFAULT_MODEL", "UltralyticsObb"]
