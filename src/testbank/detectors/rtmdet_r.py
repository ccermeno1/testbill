"""RTMDet-R (MMRotate) adapter. Apache-2.0 and fit for production.

THIS IS THE ONLY MODULE IN THE PROJECT THAT MAY IMPORT `mmrotate`, `mmdet` or
`mmcv`. `tests/test_detectors.py` verifies it over the source tree, just like
with `ultralytics`.

Does not run in the main environment
------------------------------------
It needs torch 2.0 and a very narrow chain of versions that no resolver
declares. The verified recipe is in the README; the optional `rtmdet` group of
the pyproject carries it pinned. Here the import is LAZY and the error message
points to the right place instead of leaving a bare ImportError.

Starting point
--------------
`rotated_rtmdet_tiny-3x-dota`: a rotated detector ALREADY TRAINED on DOTA, not
a classification pretraining. The specification asked for COCO pretraining,
which for the tiny variant is not published -- see the deviation documented in
the README.
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

#: MMRotate config to start from. `tiny` because of the mobile deployment.
DEFAULT_CONFIG = "rotated_rtmdet_tiny-3x-dota"

#: Published weights: rotated detector trained on DOTA, mAP 75.60.
DEFAULT_CHECKPOINT = (
    "https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/"
    "rotated_rtmdet_tiny-3x-dota/rotated_rtmdet_tiny-3x-dota-9d821076.pth"
)

_INSTALL_HINT = (
    "RTMDet-R needs mmrotate, mmdet and mmcv, which do NOT fit in the main "
    "environment: they require torch 2.0 and a very narrow chain of versions "
    "(mmrotate 1.x -> mmdet <3.2 -> mmcv <2.1 -> torch 2.0 index). "
    "The verified recipe is in the README, section 'RTMDet-R environment'."
)


def _import_mmrotate():
    """Imports and REGISTERS the MMRotate modules.

    `register_all_modules(init_default_scope=True)` is not optional, and
    neither is the flag. The model is declared as `mmdet.RTMDet`, so when
    building it mmengine jumps to the mmdet registry -- where
    `RotatedRTMDetSepBNHead` is not, because it lives in mmrotate. With
    `init_default_scope=False` it fails with `RotatedRTMDetSepBNHead is not in
    the mmdet::model registry`, which says nothing about the scope and sends
    you looking in the wrong place.

    The Runner does it on its own from the config's `default_scope`; any
    manual use of the registry does not.
    """
    try:
        from mmdet.apis import inference_detector, init_detector
        from mmrotate.utils import register_all_modules
    except ImportError as exc:
        raise DetectorError(f"{_INSTALL_HINT} ({exc})") from exc
    register_all_modules(init_default_scope=True)
    return init_detector, inference_detector


@register
class RtmdetRDetector(BaseDetector):
    name = "rtmdet-r-tiny"
    license = "Apache-2.0"
    production_ready = True
    _NOTE = (
        "Starts from rotated_rtmdet_tiny-3x-dota: rotated detector already "
        "trained on DOTA. The COCO pretraining the specification asked for is "
        "not published for tiny; see the deviation in the README."
    )
    _ENV_NOTE = (
        "Runs in a SEPARATE environment with torch 2.0. Its numbers are not "
        "strictly comparable with those of candidates on torch 2.14."
    )
    notes = (_NOTE, _ENV_NOTE)

    def __init__(
        self,
        config_name: str = DEFAULT_CONFIG,
        checkpoint: str = DEFAULT_CHECKPOINT,
    ) -> None:
        self.config_name = config_name
        self.checkpoint = checkpoint

    # --- training ---------------------------------------------------------

    def train(self, samples_by_split, config: Config, *, output_dir: Path) -> TrainResult:
        """Fine-tunes on the DOTA view we export.

        MMRotate reads an `images/` + `labelTxt/` tree, which is exactly what
        our `dota` exporter writes. So it shares the area filter and border
        policy with the other candidates.
        """
        _import_mmrotate()
        from mmengine.runner import Runner

        from testbank.dataio.export import export

        view = export("dota", samples_by_split, config, out_dir=output_dir / "dota")
        cfg = build_train_config(view.root, config, output_dir=output_dir)
        Runner.from_cfg(cfg).train()

        weights = output_dir / "work" / "last_checkpoint"
        resolved = _resolve_checkpoint(output_dir / "work")
        if resolved is None:
            raise DetectorError(
                f"training left no weights in {weights.parent}"
            )
        return TrainResult(
            weights=resolved,
            epochs=config.detector.epochs,
            notes=(view.describe(), f"starting from {self.checkpoint.rsplit('/', 1)[-1]}"),
        )

    # --- inference --------------------------------------------------------

    def predict(self, samples, *, weights: Path, config: Config) -> dict:
        """`sample_id -> [Prediction]` in NORMALIZED coordinates.

        MMRotate returns `cx, cy, w, h, theta` in image pixels, so the
        conversion goes through the same `boxes_to_quads` as the own
        candidate: a single implementation of the step to canonical quad for
        both.
        """
        init_detector, inference_detector = _import_mmrotate()
        import torch

        from testbank.metrics.core import Prediction
        from testbank.models.decode import boxes_to_quads

        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        model = init_detector(self.config_name, str(weights), device="cpu")

        out: dict[str, list] = {}
        for sample in samples:
            result = inference_detector(model, str(sample.image_path))
            instances = result.pred_instances
            keep = instances.scores >= config.detector.confidence_threshold
            boxes = instances.bboxes[keep].cpu()
            scores = instances.scores[keep].cpu()

            width, height = sizes.size(sample.sample_id)
            size = ImageSize(int(width), int(height))
            if boxes.numel() == 0:
                out[sample.sample_id] = []
                continue
            quads = boxes_to_quads(torch.as_tensor(boxes, dtype=torch.float32), size)
            out[sample.sample_id] = [
                Prediction(quad=quad, score=float(score), class_id=0)
                for quad, score in zip(quads, scores)
            ]
        return out


__all__ = ["DEFAULT_CHECKPOINT", "DEFAULT_CONFIG", "RtmdetRDetector"]


# --- MMRotate config -------------------------------------------------------


def _resolve_checkpoint(work_dir: Path) -> Path | None:
    """The last checkpoint the Runner left.

    Searched for instead of rebuilding the name: MMEngine decides it from the
    scheduler and the save interval, and guessing it already failed once with
    Ultralytics. `last_checkpoint` is a text file with the path inside.
    """
    pointer = work_dir / "last_checkpoint"
    if pointer.is_file():
        candidate = Path(pointer.read_text(encoding="utf-8").strip())
        if candidate.exists():
            return candidate
    found = sorted(work_dir.glob("*.pth"))
    return found[-1] if found else None


def build_train_config(dota_root: Path, config: Config, *, output_dir: Path):
    """MMRotate config for ONE class on our DOTA view.

    Starts from the published config and only overrides what changes. Copying
    the whole config here would leave it out of sync with the package as soon
    as it updates, and besides it is 200 lines that add nothing.

    Three things that must be touched and are not obvious:

    1. `ann_file` points to `labelTxt/`. MMRotate uses `annfiles/` by default,
       but `labelTxt/` is the name of the DOTA convention and it is what our
       exporter writes. Changing the exporter to please MMRotate would break
       any other DOTA consumer.
    2. `metainfo` with a single class. Without this, `DOTADataset` expects the
       15 DOTA classes and the `euro_banknote` annotations match none.
    3. `load_from`, not `init_cfg`. We want to start from the DETECTOR trained
       on DOTA, not from the ImageNet-pretrained backbone the config ships.
    """
    from mmengine.config import Config as MMConfig
    from mmengine.utils import get_installed_path

    package = Path(get_installed_path("mmrotate"))
    base = package / ".mim" / "configs" / "rotated_rtmdet"
    if not base.exists():  # installed from git without .mim
        base = package.parent / "configs" / "rotated_rtmdet"
    cfg = MMConfig.fromfile(str(base / f"{DEFAULT_CONFIG}.py"))

    classes = ("euro_banknote",)
    cfg.work_dir = str((output_dir / "work").resolve())
    cfg.load_from = DEFAULT_CHECKPOINT
    cfg.resume = False
    cfg.randomness = {"seed": config.metrics.seed, "deterministic": True}

    for loader, split in (("train_dataloader", "train"), ("val_dataloader", "valid")):
        section = cfg.get(loader)
        if section is None:
            continue
        section["batch_size"] = config.detector.batch_size
        # No workers: with `num_workers > 0` the batch order depends on the
        # system's scheduling and two runs with the same seed stop matching.
        # It is the same decision as in the own candidate.
        section["num_workers"] = 0
        dataset = section["dataset"]
        dataset["data_root"] = str((dota_root / split).resolve()) + "/"
        dataset["ann_file"] = "labelTxt/"
        dataset["data_prefix"] = {"img_path": "images/"}
        dataset["metainfo"] = {"classes": classes}

    if cfg.get("test_dataloader") is not None:
        # The test is SEALED. It is pointed at valid so MMRotate does not try
        # to open a directory that does not exist, but nothing here evaluates
        # it: only `evaluate-test` does, with its access log.
        cfg.test_dataloader = cfg.val_dataloader

    cfg.model["bbox_head"]["num_classes"] = len(classes)
    cfg.train_cfg = {
        "type": "EpochBasedTrainLoop",
        "max_epochs": config.detector.epochs,
        "val_interval": max(1, config.detector.epochs),
    }
    return cfg
