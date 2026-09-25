"""Seleccion de dispositivo, EMA, bucle de entrenamiento y bucle de evaluacion."""
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
    """'auto' elige CUDA, luego MPS (Apple), luego CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelEMA:
    """Media movil exponencial de los pesos (PP-YOLOE-R usa decay 0.9998)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9998):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        # calentamiento: al principio sigue de cerca al modelo
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.device), alpha=1 - d)
            else:
                v.copy_(msd[k])


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
    score_threshold: float = 0.01,
    nms_threshold: float = 0.5,
    to_original: bool = True,
):
    """Devuelve (predicciones, ground truth) en coordenadas de la imagen original."""
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
    score_threshold: float = 0.01,
    nms_threshold: float = 0.5,
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
