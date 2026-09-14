"""Adapter of the own candidate: YOLOX with an OBB head, inference side.

Apache-2.0 and fit for production: no foreign repository, no compiled
operators. The nano variant is 857k parameters. The network is in
`models/yolox_obb.py` (and the DDGRCF port in `models/ddgrcf.py`); the
weights of a run carry which one they are, so `load` needs no config.

`torch` is imported lazily: it is in an optional group and `import testbank`
must not require it.
"""

from __future__ import annotations

from pathlib import Path

from testbank.config import Config
from testbank.dataio.image_sizes import ImageSize, SizeIndex
from testbank.detectors.base import BaseDetector, DetectorError, register


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise DetectorError(
            "torch is not installed. It is in the optional group `torch`: "
            "`uv pip install -e \".[torch]\"`"
        ) from exc
    return torch


class YoloxObbDetector(BaseDetector):
    """Base of the own candidates. One registered subclass per variant; the
    name and the variant come from the same place and cannot be separated."""

    variant: str = "nano"
    license = "Apache-2.0"
    _NOTE = (
        "Own candidate: no compiled dependencies and the whole architecture "
        "under our control."
    )
    notes = (_NOTE,)

    def load(self, weights: Path, config: Config):
        from testbank.models.load import load_model

        return load_model(weights)

    def predict(self, samples, *, weights: Path, config: Config, model=None) -> dict:
        """`sample_id -> [Prediction]` in NORMALIZED coordinates.

        Predictions are decoded in the network's INPUT space (`image_size`),
        and since quads are normalized no rescaling is needed: normalization
        already absorbs the size change. The real size is passed only for the
        aspect of the canonical order.
        """
        torch = _import_torch()
        import cv2

        from testbank.models.decode import detections
        from testbank.models.load import image_to_input

        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        if model is None:
            model = self.load(weights, config)
        device = next(model.parameters()).device
        side = config.detector.image_size

        out: dict[str, list] = {}
        with torch.no_grad():
            for sample in samples:
                image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise DetectorError(f"could not read {sample.image_path}")
                resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_LINEAR)
                # The SAME conversion as in training, or the weights are useless.
                tensor = image_to_input(resized).unsqueeze(0).to(device)
                outputs = [o.to_cpu() for o in model(tensor)]
                width, height = sizes.size(sample.sample_id)
                out[sample.sample_id] = detections(
                    outputs,
                    ImageSize(side, side),
                    confidence=config.detector.confidence_threshold,
                    iou_threshold=config.detector.nms_iou,
                )
                # The real aspect only matters for the canonical order, and the
                # quads already come out normalized over a square. They are
                # re-canonicalized with the true aspect so the longest-side
                # anchor is the geometric one and not the input square's.
                out[sample.sample_id] = _recanonicalize(
                    out[sample.sample_id], width / height
                )
        return out

    def explain(
        self, image, prediction, *, weights: Path, config: Config, model=None, method: str = "gradcam"
    ):
        _import_torch()
        if model is None:
            model = self.load(weights, config)
        return _explain_yolox(model, image, prediction, config=config, method=method)


def _head_module(model):
    """The module that reads the FPN levels: `head` on the own network,
    the last yaml layer on the DDGRCF port."""
    return model.head if hasattr(model, "head") else model.model[-1]


def _explain_yolox(model, image, prediction, *, config: Config, method: str):
    """Shared by the own variants and the port: same head output shape."""
    import cv2

    from testbank.dataio.image_sizes import ImageSize
    from testbank.models.decode import decode_outputs
    from testbank.models.explain import explain, nearest_cell_score
    from testbank.models.load import image_to_input

    device = next(model.parameters()).device
    side = config.detector.image_size
    resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_LINEAR)
    tensor = image_to_input(resized).unsqueeze(0).to(device)

    target = None
    if prediction is not None and prediction.quad is not None:
        pts = [(x * side, y * side) for x, y in prediction.quad.points]
        cx = sum(p[0] for p in pts) / 4
        cy = sum(p[1] for p in pts) / 4
        long_side = ((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2) ** 0.5

        def target(seen):
            boxes, scores = decode_outputs(seen.outputs, ImageSize(side, side))
            return nearest_cell_score(scores, boxes[:, :2], (cx, cy), 0.25 * long_side)

    return explain(model, _head_module(model), tensor, method=method, target=target)


def _recanonicalize(predictions, aspect: float):
    """Reorders the vertices with the REAL aspect of the image.

    The network works on a square, so there the normalized longest side
    matches the geometric one. In the original image it need not: it is the
    anisotropy that has already bitten twice in this project.
    """
    import warnings

    from testbank.geometry.quad import (
        CoordinateRangeWarning,
        QuadShapeWarning,
        canonicalize,
    )
    from testbank.prediction import Prediction

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuadShapeWarning)
        warnings.simplefilter("ignore", CoordinateRangeWarning)
        for prediction in predictions:
            if prediction.quad is None:
                out.append(prediction)  # no geometry: nothing to reorder
                continue
            out.append(
                Prediction(
                    quad=canonicalize(prediction.quad, aspect=aspect),
                    score=prediction.score,
                    class_id=prediction.class_id,
                )
            )
    return out


@register
class YoloxObbDdgrcfPortDetector(YoloxObbDetector):
    """The pure-torch port of DDGRCF/YOLOX_OBB (`models/ddgrcf.py`, verified
    tensor by tensor against theirs). Its weights say `arch: ddgrcf`, which
    is how `load_model` tells it apart from the own head."""

    name = "yolox-obb-ddgrcf-port"
    variant = "small"
    license = "Apache-2.0"
    notes = (
        (
            "Pure-torch port of the DDGRCF/YOLOX_OBB network (yoloxs_obb.yaml): "
            "same 426 keys and shapes, identical output with the same weights."
        ),
    )


__all__ = ["PARAMETER_COUNTS", "YoloxObbDdgrcfPortDetector", "YoloxObbDetector"]


#: Parameters of each variant with one class. A TABLE and not a call to
#: `YoloxObb(...).parameter_count()`: registration happens at import, and
#: building three networks there would make `import testbank.detectors`
#: require torch. A test pins the table against the real models.
PARAMETER_COUNTS = {"nano": 856_680, "tiny": 4_366_808, "small": 7_754_696}


def _register_variants() -> dict[str, type]:
    """One registered candidate per variant, with the name derived from it,
    so the names match what `main` trained under."""
    made = {}
    for variant, params in PARAMETER_COUNTS.items():
        cls = type(
            f"YoloxObb{variant.capitalize()}Detector",
            (YoloxObbDetector,),
            {
                "variant": variant,
                "name": f"yolox-obb-{variant}",
                "notes": (
                    YoloxObbDetector._NOTE,
                    f"variant {variant}: {params:,} parameters.",
                ),
                "__doc__": (
                    f"Own OBB head on YOLOX, variant {variant} "
                    f"({params:,} parameters)."
                ),
            },
        )
        made[variant] = register(cls)
    return made


VARIANT_DETECTORS = _register_variants()
