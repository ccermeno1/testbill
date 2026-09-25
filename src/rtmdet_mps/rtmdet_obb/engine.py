"""Training / evaluation utilities: device selection, optimizer, LR schedule, EMA, loops."""
import copy
import json
import math
import os
import os.path as osp
import time
from typing import Dict, List, Optional

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


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.05) -> torch.optim.Optimizer:
    """AdamW with ``norm_decay_mult=0, bias_decay_mult=0`` (mmengine paramwise_cfg)."""
    decay, no_decay = [], []
    seen = set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue  # bypass_duplicate: shared head convs appear on every level
            seen.add(id(p))
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or name.endswith('bias'):
                no_decay.append(p)
            else:
                decay.append(p)
    return torch.optim.AdamW([dict(params=decay, weight_decay=weight_decay),
                              dict(params=no_decay, weight_decay=0.0)], lr=lr)


class LRSchedule:
    """Linear warmup (start_factor 1e-5 over ``warmup_iters``), constant until
    ``max_epochs // 2``, then cosine to ``eta_min`` (mmrotate ``schedule_3x``)."""

    def __init__(self, base_lr: float, iters_per_epoch: int, max_epochs: int, warmup_iters: int = 1000,
                 eta_min_ratio: float = 0.05):
        self.base_lr = base_lr
        self.warmup_iters = warmup_iters
        self.cos_begin = (max_epochs // 2) * iters_per_epoch
        self.cos_len = max(1, max_epochs * iters_per_epoch - self.cos_begin)
        self.eta_min = base_lr * eta_min_ratio

    def __call__(self, it: int) -> float:
        lr = self.base_lr
        if it >= self.cos_begin:
            t = min(it - self.cos_begin, self.cos_len) / self.cos_len
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * t))
        if it < self.warmup_iters:
            f = 1e-5 + (1 - 1e-5) * it / self.warmup_iters
            lr = lr * f
        return lr


class ExpMomentumEMA:
    """mmdet ``ExpMomentumEMA`` (momentum 2e-4, gamma 2000), buffers included."""

    def __init__(self, model: nn.Module, momentum: float = 0.0002, gamma: int = 2000):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.momentum = momentum
        self.gamma = gamma
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        m = (1 - self.momentum) * math.exp(-float(1 + self.steps) / self.gamma) + self.momentum
        for ema_v, v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(1 - m).add_(v.detach(), alpha=m)
            else:
                ema_v.copy_(v)
        self.steps += 1


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


def save_checkpoint(path: str, model: nn.Module, ema: Optional[ExpMomentumEMA], optimizer, epoch: int, it: int, meta: dict):
    ckpt = dict(
        state_dict=(ema.module if ema is not None else model).state_dict(),  # weights used for inference
        model_state=model.state_dict(),
        optimizer=optimizer.state_dict(),
        epoch=epoch, iter=it, ema_steps=ema.steps if ema is not None else 0, meta=meta)
    torch.save(ckpt, path)


def train(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], device: torch.device,
          work_dir: str, epochs: int, lr: float, weight_decay: float = 0.05, warmup_iters: int = 1000,
          val_interval: int = 1, log_interval: int = 10, use_ema: bool = True, grad_clip: Optional[float] = None,
          resume: Optional[str] = None, val_kwargs: Optional[dict] = None, save_best: str = 'mAP@.5:.95',
          save_interval: int = 1, max_keep: int = 2, stage2_loader: Optional[DataLoader] = None,
          stage2_epoch: Optional[int] = None, accumulate: int = 1, tensorboard: bool = False):
    """``stage2_loader`` replaces ``train_loader`` from epoch ``stage2_epoch`` (0-based) on
    (mmdet PipelineSwitchHook: light augmentation for the last epochs). ``accumulate`` > 1 sums
    gradients over that many batches before each optimizer step (effective batch = batch * accumulate)."""
    os.makedirs(work_dir, exist_ok=True)
    model.to(device)
    optimizer = build_optimizer(model, lr, weight_decay)
    iters_per_epoch = len(train_loader) // accumulate
    schedule = LRSchedule(lr, iters_per_epoch, epochs, warmup_iters)
    ema = ExpMomentumEMA(model) if use_ema else None
    start_epoch, it = 0, 0
    if resume:
        ckpt = torch.load(resume, map_location='cpu')
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
    debug_mps = os.environ.get('RTMDET_DEBUG_MPS') == '1' and device.type == 'mps'
    mps_cache_interval = int(os.environ.get('RTMDET_MPS_CACHE_INTERVAL', '0'))

    def debug_sync(stage: str, epoch: int, batch_index: int, batch: dict):
        if not debug_mps:
            return
        torch.mps.synchronize()
        meta = batch.get('meta', [{}])[0]
        boxes = batch.get('boxes', [])
        shapes = [tuple(b.shape) for b in boxes]
        finite = all(bool(torch.isfinite(b).all()) for b in boxes)
        print(f'mps-debug epoch={epoch + 1} batch={batch_index + 1} stage={stage} '
              f'id={meta.get("id", "?")} box_shapes={shapes} boxes_finite={finite}', flush=True)

    best = -1.0
    saved = []
    for epoch in range(start_epoch, epochs):
        if stage2_loader is not None and stage2_epoch is not None and epoch >= stage2_epoch and train_loader is not stage2_loader:
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
        run = dict(loss=0.0, loss_cls=0.0, loss_bbox=0.0, n=0)
        optimizer.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            if debug_mps:
                debug_sync('batch_loaded', epoch, bi, batch)
            cur_lr = schedule(it)
            for g in optimizer.param_groups:
                g['lr'] = cur_lr
            images = batch['images'].to(device, non_blocking=True)
            boxes = [b.to(device) for b in batch['boxes']]
            labels = [l.to(device) for l in batch['labels']]
            if debug_mps:
                debug_sync('tensors_on_device', epoch, bi, batch)
            losses = model.loss(images, boxes, labels)
            if debug_mps:
                debug_sync('loss', epoch, bi, batch)
            loss = losses['loss_cls'] + losses['loss_bbox']
            (loss / accumulate).backward()
            if debug_mps:
                debug_sync('backward', epoch, bi, batch)
            if (bi + 1) % accumulate != 0:
                continue
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if (device.type == 'mps' and mps_cache_interval > 0 and
                    (it + 1) % mps_cache_interval == 0):
                torch.mps.empty_cache()
            if debug_mps:
                debug_sync('optimizer_step', epoch, bi, batch)
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            it += 1
            run['loss'] += loss.item()
            run['loss_cls'] += losses['loss_cls'].item()
            run['loss_bbox'] += losses['loss_bbox'].item()
            run['n'] += 1
            step = (bi + 1) // accumulate
            if step % log_interval == 0 or step == iters_per_epoch:
                n = run['n']
                el = time.time() - t0
                msg = (f'epoch {epoch + 1}/{epochs} [{step}/{iters_per_epoch}] lr {cur_lr:.2e} '
                       f'loss {run["loss"] / n:.4f} cls {run["loss_cls"] / n:.4f} bbox {run["loss_bbox"] / n:.4f} '
                       f'{el / step:.2f}s/step')
                print(msg, flush=True)
                with open(log_path, 'a') as f:
                    f.write(json.dumps(dict(mode='train', epoch=epoch + 1, iter=it, lr=cur_lr, loss=run['loss'] / n,
                                             loss_cls=run['loss_cls'] / n, loss_bbox=run['loss_bbox'] / n)) + '\n')
                if writer is not None:
                    writer.add_scalar('train/loss', run['loss'] / n, it)
                    writer.add_scalar('train/loss_cls', run['loss_cls'] / n, it)
                    writer.add_scalar('train/loss_bbox', run['loss_bbox'] / n, it)
                    writer.add_scalar('train/learning_rate', cur_lr, it)
                run = dict(loss=0.0, loss_cls=0.0, loss_bbox=0.0, n=0)
        print(f'epoch {epoch + 1} done in {time.time() - t0:.0f}s', flush=True)

        meta = dict(epoch=epoch + 1, num_classes=model.num_classes)
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
