"""Training / evaluation utilities: device selection, optimizer, LR schedule, EMA, loops.

The loop, logging and checkpoint layout follow the RTMDet-R runtime; the optimisation
recipe is the one of YOLOX (SGD nesterov, ``yoloxwarmcos``, ``ModelEMA`` 0.9998, extra
L1 loss during the last no-mosaic epochs).
"""
import copy
import json
import math
import os
import os.path as osp
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluation import summarize


def pick_device(name: str = 'auto') -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 5e-4,
                    momentum: float = 0.9) -> torch.optim.Optimizer:
    """YOLOX ``get_optimizer``: SGD nesterov; weight decay only on conv weights (not BN, not biases)."""
    bn_weights, weights, biases = [], [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if name == 'bias':
                biases.append(p)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                bn_weights.append(p)
            else:
                weights.append(p)
    return torch.optim.SGD([dict(params=bn_weights, weight_decay=0.0),
                            dict(params=weights, weight_decay=weight_decay),
                            dict(params=biases, weight_decay=0.0)],
                           lr=lr, momentum=momentum, nesterov=True)


class LRSchedule:
    """YOLOX ``yoloxwarmcos``: quadratic warmup from 0 over ``warmup_iters``, cosine down to
    ``min_lr_ratio * lr``, then constant at that minimum for the last ``no_aug_iters``."""

    def __init__(self, base_lr: float, iters_per_epoch: int, max_epochs: int, warmup_epochs: float = 1,
                 no_aug_epochs: int = 0, min_lr_ratio: float = 0.05):
        self.base_lr = base_lr
        self.total = max_epochs * iters_per_epoch
        self.warmup_iters = int(warmup_epochs * iters_per_epoch)
        self.no_aug_iters = no_aug_epochs * iters_per_epoch
        self.min_lr = base_lr * min_lr_ratio

    def __call__(self, it: int) -> float:
        if it <= self.warmup_iters:
            return self.base_lr * (it / max(self.warmup_iters, 1)) ** 2
        if it >= self.total - self.no_aug_iters:
            return self.min_lr
        t = (it - self.warmup_iters) / max(1, self.total - self.warmup_iters - self.no_aug_iters)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))


class ModelEMA:
    """YOLOX ``ModelEMA``: decay ``0.9998 * (1 - exp(-updates / 2000))``, buffers included."""

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.steps += 1
        d = self.decay * (1 - math.exp(-self.steps / 2000))
        for ema_v, v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                ema_v.copy_(v)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int,
             score_thr: float = 0.05, nms_iou: float = 0.1, max_per_img: int = 2000, mode: str = 'area',
             report_score_thr: float = 0.5) -> Dict[str, float]:
    model.eval()
    det_results, report_results, annotations = [], [], []
    for batch in loader:
        images = batch['images'].to(device)
        preds = model.predict(images, score_thr=score_thr, nms_iou=nms_iou, max_per_img=max_per_img)
        report_preds = (preds if report_score_thr == score_thr else
                        model.predict(images, score_thr=report_score_thr, nms_iou=nms_iou,
                                      max_per_img=max_per_img))
        for p, report_p, boxes, labels in zip(preds, report_preds, batch['boxes'], batch['labels']):
            pb, ps, pl = p['boxes'].cpu(), p['scores'].cpu(), p['labels'].cpu()
            det_results.append([torch.cat([pb[pl == c], ps[pl == c, None]], 1) for c in range(num_classes)])
            rpb, rps, rpl = report_p['boxes'].cpu(), report_p['scores'].cpu(), report_p['labels'].cpu()
            report_results.append([torch.cat([rpb[rpl == c], rps[rpl == c, None]], 1) for c in range(num_classes)])
            annotations.append(dict(boxes=boxes, labels=labels))
    return summarize(det_results, annotations, num_classes, mode=mode, report_det_results=report_results)


def save_checkpoint(path: str, model: nn.Module, ema: Optional[ModelEMA], optimizer, epoch: int, it: int, meta: dict):
    ckpt = dict(
        state_dict=(ema.module if ema is not None else model).state_dict(),  # weights used for inference
        model_state=model.state_dict(),
        optimizer=optimizer.state_dict(),
        epoch=epoch, iter=it, ema_steps=ema.steps if ema is not None else 0, meta=meta)
    torch.save(ckpt, path)


def train(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], device: torch.device,
          work_dir: str, epochs: int, lr: float, weight_decay: float = 5e-4, momentum: float = 0.9,
          warmup_epochs: float = 1, min_lr_ratio: float = 0.05, val_interval: int = 1, log_interval: int = 10,
          use_ema: bool = True, grad_clip: Optional[float] = None, resume: Optional[str] = None,
          val_kwargs: Optional[dict] = None, save_interval: int = 1, max_keep: int = 2,
          stage2_loader: Optional[DataLoader] = None, stage2_epoch: Optional[int] = None, accumulate: int = 1,
          tensorboard: bool = False):
    """From epoch ``stage2_epoch`` (0-based) on, the YOLOX "no aug" phase starts: the extra L1
    loss is switched on, the LR stays at its minimum and ``stage2_loader`` (if given) replaces
    ``train_loader`` (light augmentation). ``accumulate`` > 1 sums gradients over that many
    batches before each optimizer step (effective batch = batch * accumulate)."""
    os.makedirs(work_dir, exist_ok=True)
    model.to(device)
    optimizer = build_optimizer(model, lr, weight_decay, momentum)
    iters_per_epoch = len(train_loader) // accumulate
    if stage2_epoch is None:
        stage2_epoch = epochs
    schedule = LRSchedule(lr, iters_per_epoch, epochs, warmup_epochs, epochs - stage2_epoch, min_lr_ratio)
    ema = ModelEMA(model) if use_ema else None
    start_epoch, it = 0, 0
    if resume:
        ckpt = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ema is not None:
            ema.module.load_state_dict(ckpt['state_dict'])
            ema.steps = ckpt.get('ema_steps', 0)
        start_epoch, it = ckpt['epoch'], ckpt['iter']
        print(f'resumed from {resume} at epoch {start_epoch}, iter {it}')

    log_path = osp.join(work_dir, 'log.jsonl')
    writer = None
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError('TensorBoard is not installed. Run: uv sync --extra monitoring') from exc
        writer = SummaryWriter(log_dir=osp.join(work_dir, 'tensorboard'))
    mps_cache_interval = int(os.environ.get('YOLOX_MPS_CACHE_INTERVAL', '0'))

    best = -1.0
    saved = []
    for epoch in range(start_epoch, epochs):
        if epoch >= stage2_epoch and not model.use_l1:
            print(f'epoch {epoch + 1}: no-aug phase, extra L1 loss on', flush=True)
            model.use_l1 = True
            if stage2_loader is not None and train_loader is not stage2_loader:
                print(f'epoch {epoch + 1}: switching to the stage-2 (light) augmentation pipeline', flush=True)
                # shut down the persistent workers of the old loader first: on Windows every
                # worker commits several GB of virtual memory for the CUDA DLLs
                it_old = getattr(train_loader, '_iterator', None)
                if it_old is not None:
                    it_old._shutdown_workers()
                    train_loader._iterator = None
                train_loader = stage2_loader
        model.train()
        t0 = time.time()
        run, n_run = {}, 0
        optimizer.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            cur_lr = schedule(it)
            for g in optimizer.param_groups:
                g['lr'] = cur_lr
            images = batch['images'].to(device, non_blocking=True)
            boxes = [b.to(device) for b in batch['boxes']]
            labels = [l.to(device) for l in batch['labels']]
            losses = model.loss(images, boxes, labels)
            loss = sum(losses.values())
            (loss / accumulate).backward()
            if (bi + 1) % accumulate != 0:
                continue
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if (device.type == 'mps' and mps_cache_interval > 0 and
                    (it + 1) % mps_cache_interval == 0):
                torch.mps.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            it += 1
            run['loss'] = run.get('loss', 0.0) + loss.item()
            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + v.item()
            n_run += 1
            step = (bi + 1) // accumulate
            if step % log_interval == 0 or step == iters_per_epoch:
                avg = {k: v / n_run for k, v in run.items()}
                el = time.time() - t0
                parts = ' '.join(f'{k.removeprefix("loss_")} {v:.4f}' for k, v in avg.items() if k != 'loss')
                print(f'epoch {epoch + 1}/{epochs} [{step}/{iters_per_epoch}] lr {cur_lr:.2e} '
                      f'loss {avg["loss"]:.4f} {parts} {el / step:.2f}s/step', flush=True)
                with open(log_path, 'a') as f:
                    f.write(json.dumps(dict(mode='train', epoch=epoch + 1, iter=it, lr=cur_lr, **avg)) + '\n')
                if writer is not None:
                    for k, v in avg.items():
                        writer.add_scalar(f'train/{k}', v, it)
                    writer.add_scalar('train/learning_rate', cur_lr, it)
                run, n_run = {}, 0
        print(f'epoch {epoch + 1} done in {time.time() - t0:.0f}s', flush=True)

        meta = dict(epoch=epoch + 1, num_classes=model.num_classes, arch=model.arch)
        if (epoch + 1) % save_interval == 0 or epoch + 1 == epochs:
            path = osp.join(work_dir, f'epoch_{epoch + 1}.pth')
            save_checkpoint(path, model, ema, optimizer, epoch + 1, it, meta)
            saved.append(path)
            while len(saved) > max_keep:
                old = saved.pop(0)
                if osp.exists(old):
                    os.remove(old)
            save_checkpoint(osp.join(work_dir, 'latest.pth'), model, ema, optimizer, epoch + 1, it, meta)

        if val_loader is not None and ((epoch + 1) % val_interval == 0 or epoch + 1 == epochs):
            eval_model = ema.module if ema is not None else model
            t1 = time.time()
            metrics = evaluate(eval_model, val_loader, device, model.num_classes, **(val_kwargs or {}))
            metrics_str = ' '.join(f'{k} {v:.4f}' for k, v in metrics.items())
            print(f'val epoch {epoch + 1}: {metrics_str} ({time.time() - t1:.0f}s)', flush=True)
            fitness = 0.9 * metrics['mAP@.5:.95'] + 0.1 * metrics['mAP@0.50']
            metrics['fitness'] = fitness
            with open(log_path, 'a') as f:
                f.write(json.dumps(dict(mode='val', epoch=epoch + 1, iter=it, **metrics)) + '\n')
            if writer is not None:
                for name, value in metrics.items():
                    writer.add_scalar(f'valid/{name}', value, it)
                writer.flush()
            if fitness > best:
                best = fitness
                for old in [p for p in os.listdir(work_dir) if p.startswith('best_')]:
                    os.remove(osp.join(work_dir, old))
                save_checkpoint(osp.join(work_dir, f'best_epoch_{epoch + 1}.pth'), model, ema, optimizer, epoch + 1, it,
                                dict(meta, **metrics))
    if writer is not None:
        writer.close()
    return model
