"""Ultralytics YOLO-OBB adapter. Performance baseline, NOT production.

AGPL-3.0
--------
`production_ready = False`, and not as a formality: the AGPL contaminates a
closed product. This candidate exists to have a quick reference number --
without it there is no knowing whether the others do well or badly -- and
`compare` marks its row as unfit so nobody mistakes it for a candidate.

THIS IS THE ONLY MODULE IN THE PROJECT THAT MAY IMPORT `ultralytics`.
`tests/test_detectors.py::test_ultralytics_isolation` verifies it over the
source tree. Moreover the import is LAZY, inside the methods, so that
`import testbank` works without the package installed: it is in the optional
`ultralytics` group of the pyproject, not in the base dependencies.
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

#: Nano variant by default: the deployment is mobile and the specification
#: asks to prioritize nano/small.
DEFAULT_MODEL = "yolo26n-obb.pt"


def _import_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectorError(
            "ultralytics is not installed. It is in an optional group on purpose, "
            "because it is AGPL-3.0 and cannot enter the base set: "
            "`uv pip install -e \".[ultralytics]\"`"
        ) from exc
    return YOLO


@register
class UltralyticsObb(BaseDetector):
    name = "ultralytics-yolo-obb"
    license = "AGPL-3.0"
    production_ready = False
    _NOTE = (
        "Performance reference, not a candidate: AGPL-3.0 contaminates a "
        "closed product."
    )
    notes = (_NOTE,)

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    # --- training ---------------------------------------------------------

    def train(self, samples_by_split, config: Config, *, output_dir: Path) -> TrainResult:
        """`samples_by_split` is `{"train": [...], "valid": [...]}` from SplitLoader.

        A view is materialized with the labels ALREADY FILTERED: training
        against the unfiltered truth and measuring against the filtered one
        would make the figures mean nothing. See `detectors/dataset.py`.
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
            # ABSOLUTE on purpose. With a relative path, Ultralytics interprets
            # it relative to its own settings `runs_dir` and ends up writing to
            # `runs/obb/<what you passed>`, outside our run. Measured: it left
            # the weights in `runs/obb/runs/<timestamp>_<name>/_train`.
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

    # --- inference --------------------------------------------------------

    def predict(self, samples, *, weights: Path, config: Config) -> dict:
        """Returns `sample_id -> [Prediction]` in NORMALIZED coordinates.

        Ultralytics gives the quads in pixels (`obb.xyxyxyxy`), so they are
        normalized here, at the adapter boundary. Inside testbank every quad
        is normalized and every metric computation is in pixels; mixing that
        in the middle of the pipeline is the mistake that has already bitten
        us twice.
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
                    # The model was trained on padded images: if it is given
                    # the original here, it sees a different framing from the
                    # one it learned. That this `if` exists is the real cost
                    # of the `pad` policy.
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
    """Size to normalize over. With padding, that of the padded image."""
    if fraction <= 0.0:
        return width, height
    return (
        width + 2 * round(width * fraction),
        height + 2 * round(height * fraction),
    )


def _unpad(prediction: Prediction, fraction: float) -> Prediction:
    """Takes a prediction from the padded space back to the original image's.

    Exact inverse of `dataset.pad_quad`: if that one does
    `x' = x/(1+2f) + f/(1+2f)`, this one does `x = x'*(1+2f) - f`.

    The result can fall outside [0,1], and it should: if the model predicts
    that the banknote continues beyond the framing, that is precisely the
    information the `pad` policy exists to preserve. The quad's tolerant range
    admits it.
    """
    scale = 1.0 + 2.0 * fraction
    if prediction.quad is None:
        return prediction  # no geometry: nothing to shift
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
    """Asks the trainer where it saved, instead of rebuilding the path.

    Rebuilding it is guessing the convention of a library we do not control,
    and it already failed once. `trainer.save_dir` is what Ultralytics really
    used; the expected path remains only as a fallback.
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
        "training left no weights; looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def _to_predictions(result, width: int, height: int, *, aspect: float):
    """Translates Ultralytics' `obb` into our canonical quads."""
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
