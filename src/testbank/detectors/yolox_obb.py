"""Adapter of the own candidate: YOLOX-Nano with an OBB head.

Apache-2.0 and FIT FOR PRODUCTION, unlike Ultralytics. It is the only
candidate whose code we control entirely: no foreign repository to vendor, no
compiled operators, no guessing the data schema of a dataloader we have not
read.

And it is by far the smallest -- 857k parameters against the 2.65M of YOLO26n
-- which is what the mobile deployment asked for.

`torch` is imported lazily just like `ultralytics` in the other adapter: it is
in an optional group and `import testbank` must not require it.
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
            "torch is not installed. It is in the optional group `torch`: "
            "`uv pip install -e \".[torch]\"`"
        ) from exc
    return torch


class YoloxObbDetector(BaseDetector):
    """Base of the own candidates. One registered subclass per variant.

    The variant is NOT read from the config: the class carries it. Before it
    came from `config.detector.variant` while the registered name said "nano"
    fixed, so training with `variant: tiny` produced a 4.37M model labelled as
    the 857k one. The comparison table would have shown a name that does not
    correspond to the model -- exactly the silent lie the rest of the project
    is devoted to avoiding.

    Now the name and the variant come from the same place and cannot be
    separated, and `train` rewrites the config with its variant so the frozen
    `config.yaml` tells the truth.
    """

    variant: str = "nano"
    license = "Apache-2.0"
    production_ready = True
    _NOTE = (
        "Own candidate: no compiled dependencies and the whole architecture "
        "under our control."
    )
    notes = (_NOTE,)

    def _with_variant(self, config: Config) -> Config:
        """The class variant overrides the config's, and it is written back.

        Without rewriting it, the run's `config.yaml` would store the default
        variant while another one was trained.
        """
        if config.detector.variant == self.variant:
            return config
        return config.model_copy(
            update={"detector": config.detector.model_copy(
                update={"variant": self.variant}
            )}
        )

    # --- training ---------------------------------------------------------

    def train(self, samples_by_split, config: Config, *, output_dir: Path) -> TrainResult:
        torch = _import_torch()
        from testbank.models.data import build_datasets
        from testbank.models.train import checkpoint_payload, fit, validation_metrics

        config = self._with_variant(config)
        datasets = build_datasets(samples_by_split, config)
        if "train" not in datasets:
            raise DetectorError("the 'train' split is required to train")

        validate = None
        valid_samples = list(samples_by_split.get("valid", ()))
        if valid_samples and config.detector.eval_every:
            # Selection metrics only: a light bootstrap, because the interval
            # is not what decides which checkpoint stays. The real `valid`
            # evaluation, with the full bootstrap, happens after training.
            light = config.model_copy(
                update={"metrics": config.metrics.model_copy(update={"bootstrap_samples": 20})}
            )
            scratch = output_dir / "_eval.pt"

            def validate(model, epoch):
                torch.save(checkpoint_payload(model, config, epoch=epoch), scratch)
                return validation_metrics(self.evaluate(valid_samples, light, weights=scratch))

        weights, history = fit(
            datasets["train"], config, output_dir=output_dir, validate=validate
        )
        (output_dir / "_eval.pt").unlink(missing_ok=True)
        last = history.epochs[-1] if history.epochs else {}
        return TrainResult(
            weights=weights,
            epochs=config.detector.epochs,
            notes=(
                f"{len(datasets['train'])} training images"
                + (f", {len(valid_samples)} validation images" if validate else ""),
                f"loss recipe: {config.detector.loss.recipe}",
                *history.notes,
                "final losses: "
                + ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in last.items()
                    if k in ("box", "angle", "objectness", "classes", "dfl", "l1", "total")
                ),
            ),
        )

    # --- inference --------------------------------------------------------

    def predict(self, samples, *, weights: Path, config: Config) -> dict:
        """`sample_id -> [Prediction]` in NORMALIZED coordinates.

        Predictions are decoded in the network's INPUT space (`image_size`),
        and since quads are normalized no rescaling is needed: normalization
        already absorbs the size change. The real size is passed only for the
        aspect of the canonical order.
        """
        torch = _import_torch()
        import cv2

        from testbank.models.data import image_to_input
        from testbank.models.decode import detections
        from testbank.models.train import load_model

        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        model = load_model(weights)
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
    from testbank.metrics.core import Prediction

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
    """The pure-torch port of DDGRCF/YOLOX_OBB, with its recipe and its weights.

    It is the version of that candidate that TRAINS on CPU and on MPS: the same
    network (`models/ddgrcf.py`, verified tensor by tensor against theirs), the
    same recipe (`losses.py: ddgrcf`, with the exact IoU in torch) and their
    DOTA weights with `--pretrained weights/yolox_s_dota1_0.pth`.

    Fit for production: it depends on nothing compiled nor on a clone. What it
    does NOT reproduce from the clone is its data loop (mosaic, mixup,
    resampling) nor its optimizer: it trains with this project's loop, like
    the others.
    """

    name = "yolox-obb-ddgrcf-port"
    variant = "small"
    license = "Apache-2.0"
    production_ready = True
    notes = (
        (
            "Pure-torch port of the DDGRCF/YOLOX_OBB network (yoloxs_obb.yaml): "
            "same 426 keys and shapes, identical output with the same weights."
        ),
        (
            "ddgrcf recipe: EXACT PolyIoU x5 + obj + cls by IoU + late L1, "
            "SimOTA with -log(IoU). The polygon IoU is in torch, not compiled."
        ),
        (
            "DOTA pretraining with --pretrained; the class layer (15 -> 1) is "
            "re-initialized and the rest is loaded strictly."
        ),
        "Trains with testbank's loop, not theirs: no mosaic or mixup.",
    )

    def _with_variant(self, config: Config) -> Config:
        loss = config.detector.loss.model_validate(
            {**config.detector.loss.model_dump(), "recipe": "ddgrcf"}
        )
        return config.model_copy(
            update={"detector": config.detector.model_copy(update={"variant": self.variant, "loss": loss})}
        )


__all__ = ["PARAMETER_COUNTS", "YoloxObbDdgrcfPortDetector", "YoloxObbDetector"]


#: Parameters of each variant with one class. A TABLE and not a call to
#: `YoloxObb(...).parameter_count()`: registration happens at import, and
#: building three networks there would make `import testbank.detectors`
#: require torch -- which the Paddle environment does not have. A test pins
#: the table against the real models, so it cannot drift.
PARAMETER_COUNTS = {"nano": 856_680, "tiny": 4_366_808, "small": 7_754_696}


def _register_variants() -> dict[str, type]:
    """One registered candidate per variant, with the name derived from it.

    Generated instead of written by hand so they cannot drift out of sync:
    adding a variant to `PARAMETER_COUNTS` (and `VARIANTS`) puts it here on
    its own, and the test that pins the table catches a variant in one table
    and not the other.
    """
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
