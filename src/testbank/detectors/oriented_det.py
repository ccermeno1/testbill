"""Rotated FCOS from `oriented-det` (DL4EO), inference side. Apache-2.0,
pure PyTorch.

THIS IS THE ONLY MODULE IN THE PROJECT THAT MAY IMPORT `oriented_det`.

`oriented-det` is a small torch-only framework for rotated detection;
nothing compiled, so it runs on CPU and on MPS (Apple's GPU) as is. The
model is theirs, untouched: `model([image])` returns `{rboxes, scores,
labels}`. The runs served here were trained on `main` with testbank's loop;
their checkpoint says `arch: rotated_fcos` and which backbone (R50 36.2M
parameters, R18 19.7M), and `load` rebuilds the network from that.

Environment: `oriented-det` pins `numpy<2`, so it gets its own venv; recipe
in the README.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

from testbank.config import Config
from testbank.dataio.image_sizes import SizeIndex
from testbank.detectors.base import BaseDetector, DetectorError, register

#: The recipe the published DOTA weights were trained with, inside the
#: package: the head's hyper-parameters come from it.
RECIPE = "configs/rotated_fcos/dota_le90_3x_riou.json"

#: ImageNet statistics in RGB, in [0, 1]: what their models expect.
_MEAN = (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0)
_STD = (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)

_INSTALL_HINT = (
    "oriented-det is not installed. It pins numpy<2, so it lives in its own "
    "environment; recipe in the README, section 'Rotated FCOS runs'."
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


def build_model(config: Config, *, backbone: str):
    """`RotatedFCOS` with one class, their recipe's head, our thresholds.
    The backbone is left random: the run's weights overwrite it."""
    _, _, RotatedFCOS = _import_oriented_det()
    m = _recipe().model
    return RotatedFCOS(
        num_classes=1,
        backbone_name=backbone,
        pretrained_backbone=False,
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


def _to_model_input(image_bgr_0_255):
    """Our tensor (BGR, 0-255) -> theirs (RGB, ImageNet-normalized)."""
    _, torch, _ = _import_oriented_det()
    rgb = image_bgr_0_255[[2, 1, 0]] / 255.0
    mean = torch.tensor(_MEAN, dtype=rgb.dtype, device=rgb.device)[:, None, None]
    std = torch.tensor(_STD, dtype=rgb.dtype, device=rgb.device)[:, None, None]
    return (rgb - mean) / std


def _rbox_corners(rbox) -> list[tuple[float, float]]:
    cos, sin = math.cos(rbox.angle), math.sin(rbox.angle)
    w2, h2 = rbox.width / 2, rbox.height / 2
    return [
        (rbox.cx + x * cos - y * sin, rbox.cy + x * sin + y * cos)
        for x, y in ((-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2))
    ]


class RotatedFcosDetector(BaseDetector):
    """Base of the two variants: same recipe, different backbone."""

    backbone: str = "resnet50"
    license = "Apache-2.0"

    def load(self, weights: Path, config: Config):
        _, torch, _ = _import_oriented_det()

        from testbank.models.load import pick_device

        payload = torch.load(str(weights), map_location="cpu", weights_only=False)
        if payload.get("arch") != "rotated_fcos":
            raise DetectorError(f"{weights} is not a rotated_fcos checkpoint of testbank")
        model = build_model(config, backbone=payload.get("backbone", self.backbone))
        model.load_state_dict(payload["model"])
        model.to(pick_device()).eval()
        return model

    def predict(self, samples, *, weights: Path, config: Config, model=None) -> dict:
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
        from testbank.models.load import image_to_input
        from testbank.prediction import Prediction

        if model is None:
            model = self.load(weights, config)
        device = next(model.parameters()).device
        # The thresholds are attributes of their model, set at build; they
        # follow our config at every call, so a loaded model serves any
        # operating point.
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


    def explain(
        self, image, prediction, *, weights: Path, config: Config, model=None, method: str = "gradcam"
    ):
        """Grad-CAM on the FPN levels their head reads. The target is
        `sigmoid(cls) * sigmoid(centerness)` at the cell nearest the
        detection, which is their score before NMS."""
        _, torch, _ = _import_oriented_det()
        import cv2

        from testbank.models.explain import explain, nearest_cell_score
        from testbank.models.load import image_to_input

        if model is None:
            model = self.load(weights, config)
        device = next(model.parameters()).device
        side = config.detector.image_size
        resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_LINEAR)
        tensor = _to_model_input(image_to_input(resized).to(device))

        target = None
        if prediction is not None and prediction.quad is not None:
            pts = [(x * side, y * side) for x, y in prediction.quad.points]
            cx = sum(p[0] for p in pts) / 4
            cy = sum(p[1] for p in pts) / 4
            long_side = ((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2) ** 0.5

            def target(seen):
                cls_scores, _, _, centernesses = seen.outputs
                scores, centres = [], []
                for cls, cent in zip(cls_scores, centernesses):
                    _, _, h, w = cls.shape
                    stride = side / h
                    score = (torch.sigmoid(cls[0]).max(dim=0).values * torch.sigmoid(cent[0, 0]))
                    ys, xs = torch.meshgrid(
                        torch.arange(h, device=device, dtype=torch.float32),
                        torch.arange(w, device=device, dtype=torch.float32),
                        indexing="ij",
                    )
                    scores.append(score.reshape(-1))
                    centres.append(torch.stack(((xs + 0.5) * stride, (ys + 0.5) * stride), dim=-1).reshape(-1, 2))
                return nearest_cell_score(
                    torch.cat(scores), torch.cat(centres), (cx, cy), 0.25 * long_side
                )

        class _Batch(torch.nn.Module):
            """Their forward takes a list of images; the CAM runner passes a batch tensor."""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner([x[0]])

        return explain(_Batch(model).eval(), model.head, tensor.unsqueeze(0), method=method, target=target)


@register
class RotatedFcosR50Detector(RotatedFcosDetector):
    name = "rotated-fcos-r50"
    backbone = "resnet50"
    notes = (
        "Rotated FCOS R50-FPN from oriented-det (Apache-2.0, pure torch).",
        "36.2M parameters: a reference, not a deployment candidate.",
    )


@register
class RotatedFcosR18Detector(RotatedFcosDetector):
    name = "rotated-fcos-r18"
    backbone = "resnet18"
    notes = (
        "Rotated FCOS R18-FPN from oriented-det: the smallest backbone it accepts.",
        "19.7M parameters (4.7M in the FCOS head alone): still 23x the own nano.",
    )


__all__ = [
    "RECIPE",
    "RotatedFcosDetector",
    "RotatedFcosR18Detector",
    "RotatedFcosR50Detector",
    "build_model",
]
