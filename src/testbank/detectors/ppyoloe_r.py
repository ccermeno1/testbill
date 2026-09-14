"""PP-YOLOE-R (PaddleDetection) adapter. Apache-2.0, DOTA-pretrained reference.

THIS IS THE ONLY MODULE IN THE PROJECT THAT MAY IMPORT `paddle` OR `ppdet`.
`tests/test_detectors.py` verifies it over the source tree, like the other
foreign frameworks.

Its own environment
-------------------
PaddlePaddle is a second deep-learning framework, not torch: `.venv-paddle`,
with PaddleDetection cloned anywhere and installed editable
(`pip install -e PaddleDetection`), so `ppdet` is importable and its
`configs/` sit next to it. Recipe in the README, "PP-YOLOE-R environment".
Imports are LAZY and the error message says what is missing.

What runs where
---------------
- **Inference** needs nothing compiled: the head's NMS is a core Paddle op.
  Verified on this machine without `ext_op`.
- **Training** needs `ppdet/ext_op` (`rbox_iou`): the task-aligned assigner
  computes rotated IoU with it. It is a C++ extension built per machine
  (`python setup.py install` in `ppdet/ext_op`), which needs a compiler:
  MSVC on Windows, clang on macOS. The loss (ProbIoU) and everything else are
  pure Paddle. Without `ext_op`, `train` fails at the first assignment with an
  import error naming it.

Data
----
PaddleDetection's rotated configs read COCO JSON with the oriented quad in
`segmentation` (`gt_poly`), which is exactly what our `coco` exporter writes
next to the envelope `bbox`. So this candidate shares the area filter and the
border policy with every other one.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from testbank.config import Config
from testbank.dataio.image_sizes import SizeIndex
from testbank.detectors.base import (
    BaseDetector,
    DetectorError,
    TrainResult,
    register,
)

#: Published config to start from. `s` for the mobile budget (8.09M params).
DEFAULT_CONFIG = "rotate/ppyoloe_r/ppyoloe_r_crn_s_3x_dota.yml"

#: Published weights: PP-YOLOE-R-s trained on DOTA v1.0, 73.82 mAP.
DEFAULT_CHECKPOINT_URL = "https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_s_3x_dota.pdparams"
DEFAULT_CHECKPOINT = Path("weights") / "ppyoloe_r_crn_s_3x_dota.pdparams"

CLASSES = ("euro_banknote",)

_INSTALL_HINT = (
    "PP-YOLOE-R needs paddlepaddle and PaddleDetection (`ppdet`), which live "
    "in their own environment (`.venv-paddle`): PaddlePaddle is a second "
    "framework, not torch. Recipe in the README, section 'PP-YOLOE-R environment'."
)


def _import_ppdet():
    try:
        import paddle
        import ppdet
        from ppdet.core.workspace import load_config
        from ppdet.engine import Trainer
    except ImportError as exc:
        raise DetectorError(f"{_INSTALL_HINT} ({exc})") from exc
    paddle.set_device("cpu")
    return ppdet, load_config, Trainer


def _published_config_path(ppdet) -> Path:
    """`configs/` of the editable install: PaddleDetection ships them next to
    the package, not inside it, so a plain `pip install` would not bring them."""
    path = Path(ppdet.__file__).resolve().parents[1] / "configs" / DEFAULT_CONFIG
    if not path.is_file():
        raise DetectorError(
            f"PaddleDetection configs not found at {path}. Install the clone "
            "editable (`pip install -e PaddleDetection`) so `configs/` sits next to `ppdet`"
        )
    return path


def _set_resolution(node, side: int) -> int:
    """Every `target_size` of the readers to `[side, side]`.

    The published readers work at DOTA's 1024 (`RResize`/`Resize`, with
    `keep_ratio`). The network is fully convolutional and does not care; the
    same knob as every other candidate. Returns how many were rewritten.
    """
    count = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict) and key.endswith("Resize") and "target_size" in value:
                value["target_size"] = [side, side]
                count += 1
            else:
                count += _set_resolution(value, side)
    elif isinstance(node, (list, tuple)):
        for value in node:
            count += _set_resolution(value, side)
    return count


def build_config(config: Config, *, dataset_root: Path | None = None, output_dir: Path | None = None):
    """The published config with our head, resolution, schedule and data.

    Training and inference build the model from this same object: a
    one-class checkpoint does not load into the published 15-class head, and
    inferring at another resolution would change every number silently.

    What is touched, and why:

    1. CPU, one class, no VisualDL.
    2. `target_size` of every reader at `detector.image_size`.
    3. `worker_num = 0` and no shared memory, for determinism, as everywhere.
    4. `batch_size` and `epoch` from our config; the schedule rescaled: the
       published one is pinned to 36 epochs (cosine over 44, 1000 warmup
       iterations). Same shape: warmup capped at 10% of the iterations,
       cosine over `epochs * 44 / 36` so it does not reach zero, as theirs.
    5. NMS thresholds from `detector.confidence_threshold` and `nms_iou`,
       like the other candidates at inference.
    6. Datasets pointed at our COCO export (train/valid) with `gt_poly`.
    """
    ppdet, load_config, _ = _import_ppdet()
    cfg = load_config(str(_published_config_path(ppdet)))
    cfg.use_gpu = False
    cfg.use_xpu = False
    cfg.use_npu = False
    cfg.use_mlu = False
    cfg.use_vdl = False
    cfg.num_classes = len(CLASSES)
    cfg.worker_num = 0
    if _set_resolution(cfg, config.detector.image_size) == 0:
        raise DetectorError("the published config has no reader `target_size` to override")
    for reader in ("TrainReader", "EvalReader", "TestReader"):
        cfg[reader]["batch_size"] = config.detector.batch_size
        cfg[reader]["use_shared_memory"] = False
    nms = cfg["PPYOLOERHead"]["nms"]
    nms["score_threshold"] = config.detector.confidence_threshold
    nms["nms_threshold"] = config.detector.nms_iou

    epochs = config.detector.epochs
    cfg.epoch = epochs
    cfg.snapshot_epoch = max(1, epochs)
    cfg.LearningRate["schedulers"][0].max_epochs = max(1, round(epochs * 44 / 36))
    if dataset_root is not None:
        n_train = len(list((dataset_root / "train" / "images").iterdir()))
        iters = max(1, -(-n_train // config.detector.batch_size)) * epochs
        cfg.LearningRate["schedulers"][1].steps = max(1, min(1000, iters // 10))

        fields = ["image", "gt_bbox", "gt_class", "is_crowd", "gt_poly"]
        cfg["TrainDataset"].dataset_dir = str(dataset_root.resolve())
        cfg["TrainDataset"].image_dir = "train/images"
        cfg["TrainDataset"].anno_path = "train/annotations.json"
        cfg["TrainDataset"].data_fields = fields
        cfg["EvalDataset"].dataset_dir = str(dataset_root.resolve())
        cfg["EvalDataset"].image_dir = "valid/images"
        cfg["EvalDataset"].anno_path = "valid/annotations.json"
        cfg["EvalDataset"].data_fields = fields
    # The test dataset is an image folder; its annotation file only provides
    # category names, and we have one.
    cfg["TestDataset"].anno_path = None
    if output_dir is not None:
        cfg.save_dir = str((output_dir / "work").resolve())
        cfg.pretrain_weights = str(Path(config.detector.pretrained or DEFAULT_CHECKPOINT).resolve())
    return cfg


@register
class PpyoloeRDetector(BaseDetector):
    name = "ppyoloe-r-s"
    license = "Apache-2.0"
    production_ready = True
    _NOTE = (
        "Starts from ppyoloe_r_crn_s_3x_dota: PP-YOLOE-R-s already trained on "
        "DOTA (73.82 mAP). PaddlePaddle, not torch: its own environment."
    )
    _ENV_NOTE = (
        "Runs in a SEPARATE environment (PaddlePaddle). Training needs the "
        "compiled `ppdet/ext_op`; inference does not."
    )
    notes = (_NOTE, _ENV_NOTE)

    # --- training ---------------------------------------------------------

    def train(self, samples_by_split, config: Config, *, output_dir: Path) -> TrainResult:
        """Fine-tunes on our COCO export (with `segmentation` = oriented quad)."""
        _, _, Trainer = _import_ppdet()
        from testbank.dataio.export import export

        weights_in = Path(config.detector.pretrained or DEFAULT_CHECKPOINT)
        if not weights_in.is_file():
            raise DetectorError(
                f"DOTA weights not found at {weights_in}; download them from "
                f"{DEFAULT_CHECKPOINT_URL} into weights/ (or pass --pretrained)"
            )
        view = export("coco", samples_by_split, config, out_dir=output_dir / "coco")
        cfg = build_config(config, dataset_root=view.root, output_dir=output_dir)
        trainer = Trainer(cfg, mode="train")
        trainer.load_weights(str(weights_in))
        trainer.train(validate=False)

        weights = _resolve_checkpoint(Path(cfg.save_dir), cfg.filename)
        if weights is None:
            raise DetectorError(f"training left no weights under {cfg.save_dir}")
        return TrainResult(
            weights=weights,
            epochs=config.detector.epochs,
            notes=(view.describe(), f"starting from {weights_in.name}"),
        )

    # --- inference --------------------------------------------------------

    def predict(self, samples, *, weights: Path, config: Config, model=None) -> dict:
        # `model` is ignored: PaddleDetection's Trainer is built per call
        # (`load` stays None), so the app reloads this one every photo.
        """`sample_id -> [Prediction]` in NORMALIZED coordinates.

        PaddleDetection returns `[class, score, x1, y1, ..., x4, y4]` per box,
        already in pixels of the ORIGINAL image (it undoes its own resize).
        """
        _, _, Trainer = _import_ppdet()
        from testbank.geometry.quad import (
            COORD_MAX,
            COORD_MIN,
            CoordinateRangeWarning,
            Quad,
            QuadShapeWarning,
            canonicalize,
        )
        from testbank.metrics.core import Prediction

        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        cfg = build_config(config)
        trainer = Trainer(cfg, mode="test")
        trainer.load_weights(str(weights))
        scratch = config.data.derived_dir / "_ppdet_predict"
        scratch.mkdir(parents=True, exist_ok=True)
        paths = [str(s.image_path) for s in samples]
        results = trainer.predict(paths, visualize=False, output_dir=str(scratch))

        rows = []
        for outs in results:
            start = 0
            for count in outs["bbox_num"]:
                rows.append(outs["bbox"][start : start + int(count)])
                start += int(count)
        if len(rows) != len(samples):
            raise DetectorError(
                f"PaddleDetection returned {len(rows)} result groups for {len(samples)} images"
            )

        out: dict[str, list] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QuadShapeWarning)
            warnings.simplefilter("ignore", CoordinateRangeWarning)
            for sample, boxes in zip(samples, rows):
                width, height = sizes.size(sample.sample_id)
                found = []
                for row in boxes:
                    score = float(row[1])
                    if score < config.detector.confidence_threshold:
                        continue
                    points = [
                        (float(row[2 + 2 * k]) / width, float(row[3 + 2 * k]) / height)
                        for k in range(4)
                    ]
                    inside = all(COORD_MIN <= v <= COORD_MAX for xy in points for v in xy)
                    quad = (
                        canonicalize(Quad.from_xy(points), aspect=width / height)
                        if inside
                        else None  # false positive without geometry, as in decode.py
                    )
                    found.append(Prediction(quad=quad, score=score, class_id=0))
                found.sort(key=lambda p: p.score, reverse=True)
                out[sample.sample_id] = found
        return out


def _resolve_checkpoint(save_dir: Path, config_filename: str) -> Path | None:
    """`save_dir/<config name>/model_final.pdparams`, or the newest `.pdparams`
    under `save_dir` if the naming ever changes."""
    stem = Path(config_filename).stem
    final = save_dir / stem / "model_final.pdparams"
    if final.is_file():
        return final
    found = sorted(save_dir.rglob("*.pdparams"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


__all__ = [
    "CLASSES",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_CHECKPOINT_URL",
    "DEFAULT_CONFIG",
    "PpyoloeRDetector",
    "build_config",
]
