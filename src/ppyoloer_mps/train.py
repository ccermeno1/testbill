#!/usr/bin/env python
"""
Train PP-YOLOE-R in plain PyTorch. Runs on CPU, CUDA and Apple MPS.

Follows the recipe the Paddle model was trained with: SGD with momentum, linear warm-up into
a cosine schedule, weight EMA, mosaic for the first epochs and scale/translate on single
images after that, with HSV jitter, flips and rotations throughout.

  python src/ppyoloer_mps/train.py \
      --data data/banknotes_obb --train-split train_plus_extra --val-split valid \
      --init models/ppyoloe_r/ppyoloe_r_s_dota.pt \
      --work-dir runs/extra --epochs 60 --mosaic-epochs 50 --batch 4 --device auto
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r, load_checkpoint, save_checkpoint
from ppyoloer_mps.ppyoloe_obb.data import AugmentConfig, ObbDataset, collate
from ppyoloer_mps.ppyoloe_obb.engine import (
    ModelEMA,
    evaluate_model,
    resolve_device,
    train_one_epoch,
)
from ppyoloer_mps.ppyoloe_obb.losses import PPYOLOERLoss


def build_scheduler(optimizer, base_lr: float, warmup_steps: int, total_steps: int, cosine_steps: int):
    """Linear warm-up from 0, then cosine down to cosine_steps (which may exceed the total)."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(cosine_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder holding images/ and annotations/")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="valid")
    ap.add_argument("--init", default=None, help="starting weights (.pt), e.g. the converted DOTA checkpoint")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--work-dir", default="runs/train")
    ap.add_argument("--size", default="s", choices=["s", "m", "l", "x"])
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--mosaic-epochs", type=int, default=50, help="epochs using mosaic (0 disables it)")
    ap.add_argument("--cosine-epochs", type=int, default=None, help="cosine horizon (defaults to epochs*1.2)")
    ap.add_argument("--lr", type=float, default=0.004, help="base lr for a total batch of 4")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--ema-decay", type=float, default=0.9998)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="use 0 on macOS")
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--save-best", default="fitness", choices=["fitness", "mAP50", "mAP75", "mAP50-95"],
                    help="validation metric used to pick best_model.pt; "
                         "'fitness' is 0.9*mAP50-95 + 0.1*mAP50, the Ultralytics convention")
    ap.add_argument("--snapshot-interval", type=int, default=5)
    ap.add_argument("--nms-iou", type=float, default=0.1,
                    help="rotated NMS IoU used at validation (0.1 is the mmrotate default)")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-interval", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.work_dir, exist_ok=True)
    log_path = os.path.join(args.work_dir, "train.log")
    log_file = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} | device {device} | {vars(args)}")

    aug = AugmentConfig(img_size=args.img_size)
    train_ds = ObbDataset(
        annotation_file=os.path.join(args.data, "annotations", f"{args.train_split}.json"),
        image_dir=os.path.join(args.data, "images"),
        img_size=args.img_size,
        train=True,
        augment=aug,
        mosaic_epochs=args.mosaic_epochs,
        seed=args.seed,
    )
    val_ds = ObbDataset(
        annotation_file=os.path.join(args.data, "annotations", f"{args.val_split}.json"),
        image_dir=os.path.join(args.data, "images"),
        img_size=args.img_size,
        train=False,
    )
    log(f"train: {len(train_ds)} images | valid: {len(val_ds)} | classes: {train_ds.classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate
    )

    model = build_ppyoloe_r(num_classes=train_ds.num_classes, size=args.size)
    if args.init:
        # the DOTA weights carry 15 classes, so the classification layers are dropped
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        own = model.state_dict()
        skipped = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
        state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(state, strict=False)
        log(f"init weights: {args.init} ({len(state)} tensors loaded, {len(skipped)} dropped on shape)")
        for k in skipped:
            log(f"  dropped (shape mismatch): {k}")
    model.to(device)

    loss_fn = PPYOLOERLoss(num_classes=train_ds.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False
    )
    steps_per_epoch = max(1, len(train_loader))
    cosine_epochs = args.cosine_epochs or int(args.epochs * 1.2)
    scheduler = build_scheduler(
        optimizer, args.lr, args.warmup_steps, args.epochs * steps_per_epoch, cosine_epochs * steps_per_epoch
    )
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)

    start_epoch = 0
    best = -1.0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        if ema is not None and payload.get("ema"):
            ema.ema.load_state_dict(payload["ema"])
        start_epoch = int(payload.get("meta", {}).get("epoch", 0)) + 1
        best = float(payload.get("meta", {}).get("best_metric", -1.0))
        log(f"resumed from {args.resume} at epoch {start_epoch}")

    meta_base = {"size": args.size, "num_classes": train_ds.num_classes, "classes": train_ds.classes}
    history: list[dict] = []
    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        mosaic = "mosaic on" if epoch < args.mosaic_epochs else "mosaic off"
        log(f"--- epoch {epoch}/{args.epochs - 1} ({mosaic}) ---")
        stats = train_one_epoch(
            model, loss_fn, train_loader, optimizer, device, epoch,
            scheduler=scheduler, ema=ema, log_interval=args.log_interval, logger=log,
        )
        log(f"epoch {epoch} mean: " + " ".join(f"{k} {v:.4f}" for k, v in stats.items()))

        eval_model = ema.apply() if ema is not None else model
        is_last = epoch == args.epochs - 1
        if args.val_interval and ((epoch + 1) % args.val_interval == 0 or is_last):
            eval_model.to(device)
            res = evaluate_model(
                eval_model, val_loader, device, name=f"valid(ep{epoch})", conf=args.conf,
                nms_threshold=args.nms_iou, logger=log,
            )
            # Ultralytics fitness, the same criterion the RTMDet repo uses
            res["fitness"] = 0.9 * res["mAP50-95"] + 0.1 * res["mAP50"]
            history.append({"epoch": epoch, **{k: res[k] for k in ("mAP50", "mAP75", "mAP50-95", "fitness")}})
            fitness = res[args.save_best]
            if fitness > best:
                best = fitness
                save_checkpoint(
                    os.path.join(args.work_dir, "best_model.pt"), eval_model,
                    {**meta_base, "epoch": epoch, "save_best": args.save_best,
                     "metrics": {k: res[k] for k in ("mAP50", "mAP75", "mAP50-95", "fitness")}},
                )
                log(f"new best {args.save_best} = {100 * best:.2f} % -> best_model.pt")

        if args.snapshot_interval and ((epoch + 1) % args.snapshot_interval == 0 or is_last):
            payload = {
                "model": model.state_dict(),
                "ema": ema.apply().state_dict() if ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "meta": {**meta_base, "epoch": epoch, "best_metric": best},
            }
            torch.save(payload, os.path.join(args.work_dir, "last.pt"))

    final = ema.apply() if ema is not None else model
    save_checkpoint(os.path.join(args.work_dir, "model_final.pt"), final, {**meta_base, "epoch": args.epochs - 1})
    with open(os.path.join(args.work_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=1)
    log(f"training done. best {args.save_best} = {100 * best:.2f} % | weights in {args.work_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
