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

import tempfile
from pathlib import Path

from testbank.config import Config, OutOfBoundsPolicy
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
            data=str(view.data_yaml.resolve()),
            epochs=config.detector.epochs,
            imgsz=config.detector.image_size,
            batch=config.detector.batch_size,
            seed=config.metrics.seed,
            deterministic=True,
            # ABSOLUTO a proposito. Con una ruta relativa, Ultralytics la
            # interpreta respecto a su propio `runs_dir` de settings y acaba
            # escribiendo en `runs/obb/<lo que le pasaste>`, fuera de nuestra
            # ejecucion. Medido: dejaba los pesos en
            # `runs/obb/runs/<timestamp>_<nombre>/_train`.
            project=str(output_dir.resolve()),
            name="train",
            exist_ok=True,
            verbose=False,
        )
        weights = _locate_weights(model, output_dir)
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
        trained_padded = (
            OutOfBoundsPolicy(config.detector.out_of_bounds) is OutOfBoundsPolicy.PAD
        )
        padded = trained_padded and config.detector.pad_at_inference
        fraction = config.detector.pad_fraction

        out: dict[str, list[Prediction]] = {}
        with tempfile.TemporaryDirectory() as scratch:
            for sample in samples:
                width, height = sizes.size(sample.sample_id)
                image_path = sample.image_path

                if padded:
                    # El modelo se entreno sobre imagenes padeadas: si aqui se le
                    # da la original, ve un encuadre distinto del que aprendio.
                    # Que este `if` exista es el coste real de la politica `pad`.
                    image_path = Path(scratch) / sample.image_path.name
                    dataset_view.pad_image(sample.image_path, image_path, fraction)

                result = model.predict(
                    str(image_path),
                    conf=config.detector.confidence_threshold,
                    iou=config.detector.nms_iou,
                    verbose=False,
                )[0]
                found = list(
                    _to_predictions(
                        result,
                        *_target_size(width, height, fraction if padded else 0.0),
                        aspect=width / height,
                    )
                )
                if padded:
                    found = [_unpad(p, fraction) for p in found]
                out[sample.sample_id] = found
        return out


def _target_size(width: int, height: int, fraction: float) -> tuple[int, int]:
    """Tamano sobre el que normalizar. Con padding, el de la imagen padeada."""
    if fraction <= 0.0:
        return width, height
    return (
        width + 2 * round(width * fraction),
        height + 2 * round(height * fraction),
    )


def _unpad(prediction: Prediction, fraction: float) -> Prediction:
    """Devuelve una prediccion del espacio padeado al de la imagen original.

    Inversa exacta de `dataset.pad_quad`: si aquella hace
    `x' = x/(1+2f) + f/(1+2f)`, esta hace `x = x'*(1+2f) - f`.

    El resultado puede caer fuera de [0,1], y debe: si el modelo predice que el
    billete sigue mas alla del encuadre, esa es justo la informacion que la
    politica `pad` existe para conservar. El rango tolerante del quad lo admite.
    """
    scale = 1.0 + 2.0 * fraction
    return Prediction(
        quad=canonicalize(
            Quad.from_xy(
                [(x * scale - fraction, y * scale - fraction)
                 for x, y in prediction.quad.points]
            )
        ),
        score=prediction.score,
        class_id=prediction.class_id,
    )


def _locate_weights(model, output_dir: Path) -> Path:
    """Pregunta al entrenador donde guardo, en vez de reconstruir la ruta.

    Reconstruirla es adivinar el convenio de una libreria que no controlamos, y
    ya fallo una vez. `trainer.save_dir` es lo que Ultralytics uso de verdad;
    la ruta esperada queda solo como respaldo.
    """
    candidates: list[Path] = []
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        candidates.append(Path(save_dir) / "weights" / "best.pt")
    candidates.append(output_dir / "train" / "weights" / "best.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise DetectorError(
        "el entrenamiento no dejo pesos; se buscaron: "
        + ", ".join(str(c) for c in candidates)
    )


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
