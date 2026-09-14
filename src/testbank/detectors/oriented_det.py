"""Rotated FCOS from `oriented-det` (DL4EO). Apache-2.0, pure PyTorch.

THIS IS THE ONLY MODULE IN THE PROJECT THAT MAY IMPORT `oriented_det`.
`tests/test_detectors.py` verifies it over the source tree.

Why this one, and what it is
----------------------------
`oriented-det` is a small torch-only framework for rotated detection
(Oriented R-CNN, Rotated Faster R-CNN, Rotated RetinaNet, Rotated FCOS on
ResNet + FPN). Nothing compiled: "rotated IoU and oriented NMS (CPU with
optional GPU kernels when available)". It is the only foreign reference that
meets this project's constraint as is, so it trains on CPU here and on MPS
on a Mac without any surgery. It publishes DOTA checkpoints on Hugging Face;
the one used here is Rotated FCOS R50, `rotated_fcos_dota_le90_3x_riou`,
81.58 mAP50 on DOTA.

The catch is size: Rotated FCOS R50 is 36.2M parameters, R18 (the smallest
backbone its constructor accepts) 19.7M, of which 4.7M are the FCOS head
alone. Against the 857k of the own nano these are references for what a
large model can get out of these data, not deployment candidates.

Their model, our loop
---------------------
The model is theirs, untouched: `model(images, targets)` returns its loss
terms in training and `{rboxes, scores, labels}` in inference. Everything
around it is testbank's: the dataset (area filter, border policy), the
deterministic loop, validation every `eval_every` epochs with our metrics,
`best.pt`/`last.pt`, `training.json` and `plot-training`. What is NOT
reproduced from their recipe: their augmentation (horizontal/vertical flips)
and their trainer. Their optimizer and schedule are kept in shape: SGD with
momentum 0.9, weight decay 1e-4, warmup, step decay at 2/3 and 11/12 of the
epochs (their 24 and 33 of 36), gradient clipping at 35.

Environment: `.venv-orienteddet`. `oriented-det` pins `numpy<2`, which the
main environment does not satisfy, and its `albumentations` chain needs
`--only-binary=:all:` on Windows (a transitive `stringzilla` has no wheel
for every build and wants MSVC otherwise). Recipe in the README.
"""

from __future__ import annotations

import math
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

#: Hugging Face asset (dl4eo/oriented-det-pretrained) and where we keep it,
#: with the optimizer state stripped: 145 MB instead of 289.
DEFAULT_ASSET = "rotated_fcos_dota_le90_3x_riou"
DEFAULT_CHECKPOINT = Path("weights") / "rotated_fcos_r50_dota_le90_3x_riou.pth"
#: The recipe that produced those weights, inside the package.
RECIPE = "configs/rotated_fcos/dota_le90_3x_riou.json"

#: ImageNet statistics in RGB, in [0, 1]: what their models expect.
_MEAN = (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0)
_STD = (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)

_INSTALL_HINT = (
    "oriented-det is not installed. It pins numpy<2, so it lives in its own "
    "environment (`.venv-orienteddet`); recipe in the README, section "
    "'Rotated FCOS (oriented-det) environment'."
)


def _import_oriented_det():
    try:
        import oriented_det
        import torch
        from oriented_det.models import RotatedFCOS
    except ImportError as exc:
        raise DetectorError(f"{_INSTALL_HINT} ({exc})") from exc
    return oriented_det, torch, RotatedFCOS


def _recipe():
    """Their recipe, with `_base_` resolved, as their own dataclass."""
    oriented_det, _, _ = _import_oriented_det()
    from oriented_det.train.config import TrainingExperimentConfig

    path = Path(oriented_det.__file__).resolve().parent / RECIPE
    if not path.is_file():
        raise DetectorError(f"recipe {RECIPE} not found in the installed oriented-det ({path})")
    return TrainingExperimentConfig.load(path)


def build_model(config: Config, *, backbone: str, imagenet_backbone: bool):
    """`RotatedFCOS` with one class, their recipe's head, our thresholds.

    `imagenet_backbone=True` asks torchvision for ImageNet weights (a download
    on first use); False leaves the backbone random, for when DOTA weights
    are loaded afterwards and would overwrite them anyway.
    """
    _, _, RotatedFCOS = _import_oriented_det()
    m = _recipe().model
    return RotatedFCOS(
        num_classes=1,
        backbone_name=backbone,
        pretrained_backbone=imagenet_backbone,
        trainable_layers=m.trainable_layers,
        returned_layers=list(m.fpn_returned_layers),
        fpn_strides=list(m.fpn_strides),
        fpn_extra_level=m.fpn_extra_level,
        stacked_convs=m.fcos_stacked_convs,
        center_sampling=m.fcos_center_sampling,
        center_sample_radius=m.fcos_center_sample_radius,
        norm_on_bbox=m.fcos_norm_on_bbox,
        centerness_on_reg=m.fcos_centerness_on_reg,
        scale_angle=m.fcos_scale_angle,
        focal_alpha=m.roi_focal_alpha,
        focal_gamma=m.roi_focal_gamma,
        box_reg_weight=m.box_reg_weight,
        box_reg_loss_type=m.box_reg_loss_type,
        aux_loss_type=m.aux_loss_type,
        aux_loss_weight=m.aux_loss_weight,
        angle_weight=m.fcos_angle_weight,
        score_threshold=config.detector.confidence_threshold,
        final_nms_iou_threshold=config.detector.nms_iou,
        max_detections_per_image=m.max_detections_per_image,
        nms_pre=m.fcos_nms_pre,
        final_nms_use_cpu=True,
    )


def load_dota_weights(model, path: Path) -> list[str]:
    """Their checkpoint into our one-class model. Strict except the class
    layer (`head.conv_cls`, 15 outputs there, 1 here). Returns what was
    skipped; anything else missing is an error, not a warning."""
    _, torch, _ = _import_oriented_det()
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    own = model.state_dict()
    kept, skipped = {}, []
    for key, value in state.items():
        if key in own and own[key].shape == value.shape:
            kept[key] = value
        else:
            skipped.append(key)
    missing = [k for k in own if k not in kept and "conv_cls" not in k]
    if missing:
        raise DetectorError(
            f"{path.name} does not cover the model: {len(missing)} tensors missing "
            f"outside the class layer, first {missing[0]!r}. Backbone mismatch?"
        )
    model.load_state_dict(kept, strict=False)
    return skipped


def _to_model_input(image_bgr_0_255):
    """Our dataset tensor (BGR, 0-255) -> theirs (RGB, ImageNet-normalized)."""
    _, torch, _ = _import_oriented_det()
    rgb = image_bgr_0_255[[2, 1, 0]] / 255.0
    mean = torch.tensor(_MEAN, dtype=rgb.dtype, device=rgb.device)[:, None, None]
    std = torch.tensor(_STD, dtype=rgb.dtype, device=rgb.device)[:, None, None]
    return (rgb - mean) / std


def _to_le90(boxes):
    """Our `(cx, cy, w, h, theta)` with `w` the long side and theta in [0, pi)
    -> their le90: same box, angle in [-pi/2, pi/2)."""
    _, torch, _ = _import_oriented_det()
    theta = boxes[:, 4]
    theta = torch.where(theta >= math.pi / 2, theta - math.pi, theta)
    return torch.cat((boxes[:, :4], theta[:, None]), dim=1)


def _rbox_corners(rbox) -> list[tuple[float, float]]:
    cos, sin = math.cos(rbox.angle), math.sin(rbox.angle)
    w2, h2 = rbox.width / 2, rbox.height / 2
    return [
        (rbox.cx + x * cos - y * sin, rbox.cy + x * sin + y * cos)
        for x, y in ((-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2))
    ]


def _checkpoint_payload(model, config: Config, *, backbone: str, epoch: int | None) -> dict:
    return {
        "model": model.state_dict(),
        "arch": "rotated_fcos",
        "backbone": backbone,
        "num_classes": 1,
        "image_size": config.detector.image_size,
        "epoch": epoch,
    }


class RotatedFcosDetector(BaseDetector):
    """Base of the two variants: same recipe, different backbone."""

    backbone: str = "resnet50"
    #: Whether DOTA weights are expected by default (only published for R50).
    dota_default: bool = True
    license = "Apache-2.0"
    production_ready = True

    def _weights_in(self, config: Config) -> Path | None:
        if config.detector.pretrained is not None:
            return Path(config.detector.pretrained)
        return DEFAULT_CHECKPOINT if self.dota_default else None

    # --- training ---------------------------------------------------------

    def train(self, samples_by_split, config: Config, *, output_dir: Path) -> TrainResult:
        _, torch, _ = _import_oriented_det()
        from torch.utils.data import DataLoader

        from testbank.experiment.provenance import seed_everything
        from testbank.models.data import build_datasets, collate
        from testbank.models.train import HISTORY_FILE, TrainingHistory, pick_device

        datasets = build_datasets(samples_by_split, config)
        if "train" not in datasets:
            raise DetectorError("the 'train' split is required to train")
        dataset = datasets["train"]
        output_dir.mkdir(parents=True, exist_ok=True)
        device = pick_device()
        history = TrainingHistory()
        history.notes.append(f"device: {device}")

        # EVERYTHING is seeded before building the model, as in `fit`.
        seed_everything(config.metrics.seed)
        weights_in = self._weights_in(config)
        model = build_model(config, backbone=self.backbone, imagenet_backbone=weights_in is None)
        if weights_in is not None:
            if not weights_in.is_file():
                raise DetectorError(
                    f"weights not found at {weights_in}; the DOTA checkpoint is "
                    f"asset {DEFAULT_ASSET!r} on dl4eo/oriented-det-pretrained (Git LFS here)"
                )
            skipped = load_dota_weights(model, weights_in)
            history.notes.append(
                f"DOTA pretraining loaded from {weights_in.name}; {len(skipped)} tensors "
                f"skipped (class layer): {skipped[:2]}"
            )
        else:
            history.notes.append(
                f"no DOTA weights: {self.backbone} backbone from ImageNet (torchvision), "
                "FPN and head from scratch"
            )
        model.to(device)

        generator = torch.Generator().manual_seed(config.metrics.seed)
        loader = DataLoader(
            dataset, batch_size=config.detector.batch_size, shuffle=True,
            collate_fn=collate, num_workers=0, generator=generator, drop_last=False,
        )
        recipe = _recipe().training
        base_lr = float(recipe.learning_rate)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=base_lr, momentum=float(recipe.momentum),
            weight_decay=float(recipe.weight_decay),
        )
        epochs = config.detector.epochs
        total_steps = max(1, len(loader) * epochs)
        # Their schedule, rescaled: warmup capped at 10% of the iterations,
        # x0.1 at 2/3 and at 11/12 of the epochs (24 and 33 of their 36).
        warmup = max(1, min(int(recipe.lr_warmup_steps), total_steps // 10))
        milestones = (max(1, round(epochs * 2 / 3)), max(1, round(epochs * 11 / 12)))
        gamma = float(recipe.lr_scheduler_gamma)

        def lr_at(step: int, epoch: int) -> float:
            factor = (step + 1) / warmup if step < warmup else 1.0
            return base_lr * factor * gamma ** sum(epoch >= m for m in milestones)

        validate = self._validator(samples_by_split, config, output_dir)
        eval_every = config.detector.eval_every if validate is not None else 0
        metric_name = config.detector.selection_metric
        best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
        step = 0
        for epoch in range(epochs):
            model.train()
            totals: dict[str, float] = {}
            batches = 0
            for batch in loader:
                lr = lr_at(step, epoch)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                images = [_to_model_input(img.to(device)) for img in batch.images]
                targets = [
                    {
                        "rboxes": _to_le90(boxes.to(device)),
                        "labels": torch.ones(len(boxes), dtype=torch.long, device=device),
                    }
                    for boxes in batch.boxes
                ]
                losses = model(images, targets)
                total = sum(losses.values())
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe.max_grad_norm))
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach())
                totals["total"] = totals.get("total", 0.0) + float(total.detach())
                batches += 1
                step += 1
            terms = {k: v / max(1, batches) for k, v in totals.items()}
            history.record(epoch, terms, lr_at(step, epoch))
            print(
                f"epoch {epoch + 1}/{epochs}  lr {lr_at(step, epoch):.2e}  "
                + "  ".join(f"{k} {v:.4f}" for k, v in terms.items()),
                flush=True,
            )
            is_last = epoch == epochs - 1
            if eval_every and (is_last or (epoch + 1) % eval_every == 0):
                model.eval()
                torch.save(_checkpoint_payload(model, config, backbone=self.backbone, epoch=epoch),
                           output_dir / "_eval.pt")
                metrics = validate(output_dir / "_eval.pt")
                history.validation.append({"epoch": epoch, **metrics})
                value = metrics[metric_name]
                improved = history.best is None or value > history.best[metric_name]
                print(
                    f"  valid @ {epoch + 1}: "
                    + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items())
                    + ("  <- best" if improved else ""),
                    flush=True,
                )
                if improved:
                    history.best = {"epoch": epoch, **metrics}
                    (output_dir / "_eval.pt").replace(best_path)
            history.write(output_dir / HISTORY_FILE)

        (output_dir / "_eval.pt").unlink(missing_ok=True)
        torch.save(_checkpoint_payload(model, config, backbone=self.backbone, epoch=epochs - 1), last_path)
        if history.best is None:
            torch.save(_checkpoint_payload(model, config, backbone=self.backbone, epoch=epochs - 1), best_path)
            history.notes.append("best.pt = last epoch: no validation during training")
        else:
            history.notes.append(
                f"best.pt = epoch {history.best['epoch'] + 1} of {epochs} by "
                f"{metric_name} {history.best[metric_name]:.4f}; last.pt = epoch {epochs}"
            )
        history.write(output_dir / HISTORY_FILE)
        last = history.epochs[-1] if history.epochs else {}
        return TrainResult(
            weights=best_path,
            epochs=epochs,
            notes=(
                f"{len(dataset)} training images",
                (f"recipe: oriented-det {RECIPE} (their model and losses; testbank's loop, "
                "no augmentation)"),
                *history.notes,
                "final losses: " + ", ".join(f"{k}={v:.4f}" for k, v in last.items() if k != "epoch"),
            ),
        )

    def _validator(self, samples_by_split, config: Config, output_dir: Path):
        valid = list(samples_by_split.get("valid", ()))
        if not valid or not config.detector.eval_every:
            return None
        light = config.model_copy(
            update={"metrics": config.metrics.model_copy(update={"bootstrap_samples": 20})}
        )

        def validate(weights: Path) -> dict:
            report = self.evaluate(valid, light, weights=weights)
            return {"map50": report["map50"]["value"], "coverage_p5": report["coverage_p5"]["value"]}

        return validate

    # --- inference --------------------------------------------------------

    def predict(self, samples, *, weights: Path, config: Config) -> dict:
        """`sample_id -> [Prediction]` in NORMALIZED coordinates.

        Their boxes come out in the INPUT space (`image_size` square); the
        quads are normalized by that side and re-canonicalized with the real
        aspect, exactly like the own candidate.
        """
        _, torch, _ = _import_oriented_det()
        import cv2

        from testbank.geometry.quad import (
            COORD_MAX,
            COORD_MIN,
            CoordinateRangeWarning,
            Quad,
            QuadShapeWarning,
            canonicalize,
        )
        from testbank.metrics.core import Prediction
        from testbank.models.data import image_to_input
        from testbank.models.train import pick_device

        payload = torch.load(str(weights), map_location="cpu", weights_only=False)
        if payload.get("arch") != "rotated_fcos":
            raise DetectorError(f"{weights} is not a rotated_fcos checkpoint of testbank")
        model = build_model(config, backbone=payload.get("backbone", self.backbone), imagenet_backbone=False)
        model.load_state_dict(payload["model"])
        device = pick_device()
        model.to(device).eval()
        # The thresholds are attributes of their model, set at build; they
        # follow our config, like every other candidate at inference.
        model.score_threshold = config.detector.confidence_threshold
        model.final_nms_iou_threshold = config.detector.nms_iou

        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        side = config.detector.image_size
        out: dict[str, list] = {}
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", QuadShapeWarning)
            warnings.simplefilter("ignore", CoordinateRangeWarning)
            for sample in samples:
                image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise DetectorError(f"could not read {sample.image_path}")
                resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_LINEAR)
                tensor = _to_model_input(image_to_input(resized).to(device))
                output = model([tensor])[0]
                width, height = sizes.size(sample.sample_id)
                found = []
                for rbox, score in zip(output["rboxes"], output["scores"].tolist()):
                    points = [(x / side, y / side) for x, y in _rbox_corners(rbox)]
                    inside = all(COORD_MIN <= v <= COORD_MAX for xy in points for v in xy)
                    quad = canonicalize(Quad.from_xy(points), aspect=width / height) if inside else None
                    found.append(Prediction(quad=quad, score=float(score), class_id=0))
                found.sort(key=lambda p: p.score, reverse=True)
                out[sample.sample_id] = found
        return out


@register
class RotatedFcosR50Detector(RotatedFcosDetector):
    name = "rotated-fcos-r50"
    backbone = "resnet50"
    dota_default = True
    notes = (
        ("Rotated FCOS R50-FPN from oriented-det (Apache-2.0, pure torch), starting "
        "from its DOTA checkpoint rotated_fcos_dota_le90_3x_riou (81.58 mAP50)."),
        ("36.2M parameters: a reference for what a large model gets out of these "
        "data, not a deployment candidate."),
        "Their model and losses; testbank's loop (no augmentation).",
    )


@register
class RotatedFcosR18Detector(RotatedFcosDetector):
    name = "rotated-fcos-r18"
    backbone = "resnet18"
    dota_default = False
    notes = (
        ("Rotated FCOS R18-FPN from oriented-det: the smallest backbone its "
        "constructor accepts. ImageNet backbone (torchvision), no DOTA weights "
        "published for it: FPN and head train from scratch."),
        "19.7M parameters (4.7M in the FCOS head alone): still 23x the own nano.",
        "Their model and losses; testbank's loop (no augmentation).",
    )


__all__ = [
    "DEFAULT_ASSET",
    "DEFAULT_CHECKPOINT",
    "RECIPE",
    "RotatedFcosDetector",
    "RotatedFcosR18Detector",
    "RotatedFcosR50Detector",
    "build_model",
    "load_dota_weights",
]
