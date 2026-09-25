"""Device selection, weight EMA, and the training and evaluation loops."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .boxes import poly2rbox
from .evaluation import HEADER, evaluate_detections, format_metrics
from .losses import PPYOLOERLoss
from .ops import batched_postprocess

__all__ = ["resolve_device", "ModelEMA", "train_one_epoch", "run_inference", "evaluate_model"]


def resolve_device(name: str = "auto") -> torch.device:
    """'auto' picks CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelEMA:
    """Exponential moving average of the weights. Port of ppdet/optimizer/ema.py.

    Follows PaddleDetection's default (ema_decay_type='threshold'): the average starts at
    ZERO, the decay grows as min(decay, (1+step)/(10+step)), and reading it applies the bias
    correction /(1 - decay**step).

    That detail matters. With the 'exponential' ramp, decay*(1-exp(-step/2000)), the decay at
    step 800 is 0.34 instead of 0.99, so the EMA tracks the model rather than averaging it and
    you lose nearly all the localisation gain averaging brings.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9998, decay_type: str = "threshold"):
        self.decay = decay
        self.decay_type = decay_type
        self.step = 0
        self._decay = 0.0
        self.shadow = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in model.state_dict().items()}
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def _next_decay(self) -> float:
        if self.decay_type == "threshold":
            return min(self.decay, (1 + self.step) / (10 + self.step))
        if self.decay_type == "exponential":
            return self.decay * (1 - math.exp(-(self.step + 1) / 2000))
        return self.decay

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self._next_decay()
        self._decay = d
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(d).add_(msd[k].detach().float(), alpha=1 - d)
        self.step += 1

    @torch.no_grad()
    def apply(self) -> torch.nn.Module:
        """The model holding the bias-corrected EMA weights, ready to evaluate or save."""
        if self.step == 0:
            return self.module
        target = self.module.state_dict()
        correction = 1.0 - self._decay ** self.step if self.decay_type != "exponential" else 1.0
        for k, v in self.shadow.items():
            value = v / correction if correction != 1.0 else v
            target[k].copy_(value.to(target[k].dtype))
        return self.module

    @property
    def ema(self) -> torch.nn.Module:
        return self.apply()


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -1.0


def _targets_from_batch(batch, device):
    polys = batch["gt_poly"].to(device)
    rboxes = poly2rbox(polys)
    return rboxes, batch["gt_class"].to(device), batch["pad_gt_mask"].to(device)


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: PPYOLOERLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler=None,
    ema: ModelEMA | None = None,
    log_interval: int = 10,
    max_grad_norm: float = 35.0,
    logger=print,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "loss_cls": 0.0, "loss_iou": 0.0, "loss_dfl": 0.0}
    n = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        gt_rboxes, gt_labels, pad_mask = _targets_from_batch(batch, device)

        head_outs = model(images)
        losses = loss_fn(head_outs, gt_rboxes, gt_labels, pad_mask, model.yolo_head.bbox_decode)

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        for k in totals:
            totals[k] += float(losses[k].detach())
        n += 1
        if log_interval and step % log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger(
                f"epoch {epoch} [{step:4d}/{len(loader)}] lr {lr:.6f} "
                + " ".join(f"{k} {float(losses[k].detach()):.4f}" for k in totals)
                + f" | {(time.time() - t0) / max(n, 1):.3f} s/iter"
            )
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.1,
    to_original: bool = True,
):
    """Returns (predictions, ground truth) in original-image coordinates."""
    model.eval()
    preds, gts = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        scores, rboxes = model(images)
        scale = batch["scale_factor"].to(device) if to_original else None
        dets = batched_postprocess(
            scores.float(),
            rboxes.float(),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            scale_factor=scale,
        )
        polys_gt = batch["gt_poly"]
        mask = batch["pad_gt_mask"][..., 0] > 0
        sf = batch["scale_factor"]
        for i, det in enumerate(dets):
            preds.append(
                {
                    "rboxes": det["rboxes"].cpu(),
                    "scores": det["scores"].cpu(),
                    "labels": det["labels"].cpu(),
                    "polys": det["polys"].cpu(),
                    "image_id": batch["image_id"][i],
                    "image_path": batch["image_path"][i],
                }
            )
            g = polys_gt[i][mask[i]].clone()
            if to_original and g.numel():
                g[:, 0::2] /= sf[i][1]
                g[:, 1::2] /= sf[i][0]
            gts.append({"rboxes": poly2rbox(g) if g.numel() else torch.zeros(0, 5), "polys": g})
    return preds, gts


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    name: str = "eval",
    conf: float = 0.5,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.1,
    logger=print,
    print_header: bool = True,
) -> dict:
    preds, gts = run_inference(
        model, loader, device, score_threshold=score_threshold, nms_threshold=nms_threshold
    )
    res = evaluate_detections(preds, gts, conf=conf)
    if print_header:
        logger(HEADER)
    logger(format_metrics(name, res, conf))
    return res
