"""Inference on loose images with the models trained in `runs/`.

What the Streamlit app (`testbank app`) is built on, kept apart from it so
it can be tested without Streamlit and reused from a script. Three things:

* `discover_models(runs_dir)`: which runs have weights and which adapter
  produced them, with the frozen config the run trained with (so inference
  uses the same `image_size`, the same NMS, the same everything).
* `predict_image(model, image)`: a BGR array in, predictions out, through
  the adapter's own `predict` -- the same door the metrics went through, so
  what the app shows is what the table measured.
* `draw` / `crops`: the annotated image, and the rectified banknote crops
  that this stage exists to deliver, expanded by `crop.margin` like the
  coverage metric expands them.

Adapters load their frameworks lazily; a run trained in another environment
(RTMDet-R, Rotated FCOS, PP-YOLOE-R) lists here but fails to predict with
its `DetectorError`, which the app shows instead of crashing.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from testbank.config import Config
from testbank.data.discover import Sample
from testbank.detectors import get as get_detector
from testbank.geometry.quad import Quad
from testbank.prediction import Prediction

RUN_RECORD = "run.json"
METRICS = "metrics.json"
CONFIG = "config.yaml"

#: Outline colour (BGR) and font of the drawn predictions.
COLOR_PREDICTION = (40, 190, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Preference among the weight files a run can hold: the selected checkpoint
#: first; then a foreign framework's last epoch; then whatever is there.
WEIGHT_PREFERENCE = ("best.pt", "last.pt")
WEIGHT_SUFFIXES = (".pt", ".pth", ".pth.tar", ".pdparams")


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """A run that can be used for inference."""

    directory: Path
    name: str
    detector: str
    weights: Path
    config: Config
    #: `metrics.json` of the run, empty if it was never evaluated.
    metrics: dict = field(default_factory=dict, compare=False)
    #: Split the run trained on, e.g. `v1 (adopt)`; empty for old runs.
    split: str = ""

    @property
    def label(self) -> str:
        """What the selector shows: name, adapter and the headline numbers."""
        head = f"{self.name}  |  {self.detector}"
        summary = self.summary()
        if summary:
            head += "  |  " + "  ".join(f"{k} {v:.3f}" for k, v in summary.items())
        return head

    def summary(self) -> dict[str, float]:
        """mAP50, mAP50-95 and coverage p5 on `valid`, when they exist."""
        out = {}
        for key, name in (("map50", "mAP50"), ("map50_95", "mAP50-95"), ("coverage_p5", "cov p5")):
            entry = self.metrics.get(key)
            if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
                out[name] = float(entry["value"])
        return out


def _weights_in(directory: Path) -> Path | None:
    weights_dir = directory / "weights"
    if not weights_dir.is_dir():
        return None
    for preferred in WEIGHT_PREFERENCE:
        if (weights_dir / preferred).is_file():
            return weights_dir / preferred
    candidates = [
        p for p in weights_dir.iterdir()
        if p.is_file() and any(p.name.endswith(s) for s in WEIGHT_SUFFIXES)
    ]
    if not candidates:
        return None

    # A foreign framework numbers its epochs (`epoch_100.pth`): the highest
    # number is the last one, and `epoch_9` must not beat it lexically.
    def epoch_number(path: Path) -> tuple[int, str]:
        digits = re.findall(r"\d+", path.name)
        return (int(digits[-1]) if digits else -1, path.name)

    return max(candidates, key=epoch_number)


def load_model(directory: str | Path) -> TrainedModel | None:
    """The run at `directory` as a `TrainedModel`, or None if it has no
    record, no frozen config or no weights (an evaluation-only run, an
    inspection folder, a run that died before saving)."""
    directory = Path(directory)
    record_path = directory / RUN_RECORD
    config_path = directory / CONFIG
    weights = _weights_in(directory)
    if not (record_path.is_file() and config_path.is_file() and weights):
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    detector = (record.get("detector") or {}).get("name")
    if not detector:
        return None
    metrics_path = directory / METRICS
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    split = record.get("split") or {}
    return TrainedModel(
        directory=directory,
        name=record.get("name") or directory.name,
        detector=detector,
        weights=weights,
        config=Config.load(config_path),
        metrics=metrics,
        split=f"{split['version']} ({split.get('mode', '?')})" if split.get("version") else "",
    )


def discover_models(runs_dir: str | Path) -> list[TrainedModel]:
    """Every usable run in `runs_dir`, most recent first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    found = []
    for child in sorted(runs_dir.iterdir(), reverse=True):
        if child.is_dir() and not child.name.startswith("_"):
            model = load_model(child)
            if model is not None:
                found.append(model)
    return found


def decode_image(payload: bytes) -> np.ndarray:
    """Bytes of a JPEG/PNG (an upload, a camera frame) to a BGR array."""
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not an image I can decode (JPEG or PNG expected)")
    return image


def inference_config(
    model: TrainedModel, *, confidence: float | None = None, nms_iou: float | None = None
) -> Config:
    """The run's frozen config with the sliders applied: `confidence` and
    `nms_iou` override the run's thresholds, None keeps them."""
    updates = {}
    if confidence is not None:
        updates["confidence_threshold"] = confidence
    if nms_iou is not None:
        updates["nms_iou"] = nms_iou
    return model.config.model_copy(
        update={"detector": model.config.detector.model_copy(update=updates)}
    )


def load_weights(
    model: TrainedModel, *, confidence: float | None = None, nms_iou: float | None = None
):
    """The adapter's loaded model for `predict_image(..., loaded=)`, so a
    page serving one photo at a time does not pay the load on each (RTMDet-R
    takes ~10 s to build and load; the forward pass under a second).

    Keyed by the thresholds too: an adapter may bake them into the loaded
    object. None when the adapter does not support it; `predict_image` then
    loads on every call.
    """
    config = inference_config(model, confidence=confidence, nms_iou=nms_iou)
    return get_detector(model.detector).load(model.weights, config)


def predict_image(
    model: TrainedModel,
    image: np.ndarray,
    *,
    confidence: float | None = None,
    nms_iou: float | None = None,
    loaded=None,
) -> list[Prediction]:
    """Predictions of `model` on one BGR image, in normalized coordinates.

    Goes through the adapter's `predict`, which reads files: the image is
    written to a temporary directory that also holds the size cache, so
    nothing of the app lands in `data/derived/`. `confidence` and `nms_iou`
    override the run's thresholds for this call only (the sliders); `loaded`
    is what `load_weights` returned with the same thresholds, or None.
    """
    config = inference_config(model, confidence=confidence, nms_iou=nms_iou)
    with tempfile.TemporaryDirectory(prefix="testbank-serve-") as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "image.jpg"
        if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 97]):
            raise OSError(f"could not write {image_path}")
        config = config.model_copy(
            update={"data": config.data.model_copy(update={"derived_dir": tmp_path / "derived"})}
        )
        sample = Sample(sample_id="image", image_path=image_path)
        detector = get_detector(model.detector)
        predictions = detector.predict(
            [sample], weights=model.weights, config=config, model=loaded
        )
    return sorted(predictions.get("image", []), key=lambda p: -p.score)


def explain_image(
    model: TrainedModel,
    image: np.ndarray,
    prediction: Prediction | None,
    *,
    method: str = "gradcam",
    confidence: float | None = None,
    nms_iou: float | None = None,
    loaded=None,
) -> np.ndarray | None:
    """The heat map of `prediction` on `image` blended over it (BGR), or
    None when the adapter has no access to its feature maps. `gradcam`
    explains one detection's score; `eigencam` needs no detection and shows
    what the network looks at. See `models/explain.py` for what the map does
    and does not say."""
    from testbank.models.explain import overlay

    config = inference_config(model, confidence=confidence, nms_iou=nms_iou)
    detector = get_detector(model.detector)
    cam = detector.explain(
        image, prediction, weights=model.weights, config=config, model=loaded, method=method
    )
    if cam is None:
        return None
    return overlay(image, cam)


def _pixels(quad: Quad, width: int, height: int) -> np.ndarray:
    return np.array([(x * width, y * height) for x, y in quad.points], dtype=np.float32)


def draw(image: np.ndarray, predictions, *, thickness: int = 2) -> np.ndarray:
    """The image with every prediction outlined and its score. Amber, like
    the inspection sheets; the first vertex (the canonical anchor) marked."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = max(0.4, min(width, height) / 900)
    for index, prediction in enumerate(predictions):
        if prediction.quad is None:
            continue
        pts = _pixels(prediction.quad, width, height).astype(np.int32)
        cv2.polylines(canvas, [pts], True, COLOR_PREDICTION, thickness, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[0]), thickness + 2, COLOR_PREDICTION, -1, cv2.LINE_AA)
        tag = f"#{index + 1} {prediction.score:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, scale, 1)
        x, y = pts[0]
        cv2.rectangle(canvas, (x - 2, y - th - 6), (x + tw + 4, y + 3), (20, 20, 20), -1)
        cv2.putText(canvas, tag, (x + 1, y - 1), FONT, scale, COLOR_PREDICTION, 1, cv2.LINE_AA)
    return canvas


def expand_quad(points: np.ndarray, margin: float) -> np.ndarray:
    """Scales the four corners about their center by `1 + 2*margin`: the
    same expansion `metrics.core.expand` applies before measuring coverage,
    so the crop shown is the crop scored."""
    if margin < 0:
        raise ValueError(f"the margin cannot be negative: {margin}")
    center = points.mean(axis=0, keepdims=True)
    return (center + (points - center) * (1.0 + 2.0 * margin)).astype(np.float32)


def crop(image: np.ndarray, quad: Quad, *, margin: float = 0.0) -> np.ndarray:
    """The banknote rectified by homography: the quad (with margin) mapped
    onto an upright rectangle whose sides are the quad's side lengths.

    The canonical order puts the longest side first, so the crop comes out
    landscape with the anchor at the top-left; parts of the quad outside
    the image come out black, they are not clipped, so the crop keeps the
    banknote's proportions.
    """
    height, width = image.shape[:2]
    src = expand_quad(_pixels(quad, width, height), margin)
    long_side = float(np.linalg.norm(src[1] - src[0]))
    short_side = float(np.linalg.norm(src[2] - src[1]))
    out_w, out_h = max(1, round(long_side)), max(1, round(short_side))
    dst = np.array([(0, 0), (out_w, 0), (out_w, out_h), (0, out_h)], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)


def crops(image: np.ndarray, predictions, *, margin: float = 0.0) -> list[np.ndarray]:
    return [crop(image, p.quad, margin=margin) for p in predictions if p.quad is not None]


__all__ = [
    "WEIGHT_PREFERENCE",
    "TrainedModel",
    "crop",
    "crops",
    "decode_image",
    "discover_models",
    "draw",
    "expand_quad",
    "explain_image",
    "inference_config",
    "load_model",
    "load_weights",
    "predict_image",
]
