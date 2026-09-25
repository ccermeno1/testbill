"""Train a YOLOX-OBB detector on a YOLOv8-OBB export (pure PyTorch: CUDA, Apple MPS or CPU).

``--model``: ``ddgrcf_s`` (YOLOX_OBB-s, DOTA weights) or the official YOLOX ``yolox_nano`` /
``yolox_tiny`` / ``yolox_s`` with an oriented head (COCO weights). Example::

    python src/yolox_mps/train.py --model ddgrcf_s --work-dir dota_200 \
        --extra-train augmented --img-size 640 --batch 8 --epochs 200 --stage2-epochs 15 --strong-aug
"""
import argparse
import faulthandler
import json
import os
import os.path as osp
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from yolox_obb import MODELS, build_model, load_pretrained  # noqa: E402
from yolox_obb.data import StrongAug, YoloObbDataset, collate  # noqa: E402
from yolox_obb.engine import pick_device, train  # noqa: E402


faulthandler.enable()

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / 'models' / 'yolox_obb' / 'checkpoints'
EXPERIMENTS = ROOT / 'models' / 'yolox_obb' / 'experiments'


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() or path.exists() else base / value)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='data/dataset', help='YOLOv8-OBB dataset folder')
    p.add_argument('--split-dir', help='legacy split directory; physical train/valid/test folders are preferred')
    p.add_argument('--extra-train', nargs='*', default=[], help='optional extra folders added to train')
    p.add_argument('--classes', type=int, default=1)
    p.add_argument('--model', default='ddgrcf_s', choices=list(MODELS))
    p.add_argument('--init', help='checkpoint filename or path (default: the pretrained weights of --model); '
                                  '"none" trains from scratch')
    p.add_argument('--resume', help='latest.pth of this repo to resume')
    p.add_argument('--work-dir', default='dota_200', help='experiment name or output path')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--accumulate', type=int, default=1, help='gradient accumulation steps (effective batch = batch * accumulate)')
    p.add_argument('--lr', type=float, help='default 0.01 / 64 per image of the effective batch (YOLOX basic_lr_per_img)')
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--warmup-epochs', type=float, default=1)
    p.add_argument('--min-lr-ratio', type=float, default=0.05)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--persistent-workers', dest='persistent_workers', action='store_true', default=None)
    p.add_argument('--no-persistent-workers', dest='persistent_workers', action='store_false')
    p.add_argument('--mosaic-cache', type=int, default=40)
    p.add_argument('--mixup-cache', type=int, default=20)
    p.add_argument('--device', default='auto', help='auto | cuda | mps | cpu')
    p.add_argument('--val-interval', type=int, default=1)
    p.add_argument('--no-ema', action='store_true')
    p.add_argument('--grad-clip', type=float, default=None)
    p.add_argument('--nms-iou', type=float, default=0.1, help='rotated NMS IoU used for official mAP validation')
    p.add_argument('--report-score-thr', type=float, default=0.5,
                   help='operating threshold used for validation precision/recall/F1')
    p.add_argument('--no-rotate', action='store_true', help='disable the random rotation augmentation')
    p.add_argument('--strong-aug', action='store_true',
                   help='mosaic + random resize + rotate + crop + HSV + flip + mixup, '
                        'switching to the light pipeline for the last --stage2-epochs')
    p.add_argument('--mosaic-prob', type=float, default=1.0)
    p.add_argument('--mixup-prob', type=float, default=0.5)
    p.add_argument('--resize-range', type=float, nargs=2, default=[0.5, 1.5])
    p.add_argument('--stage2-resize-range', type=float, nargs=2, default=[0.9, 1.1])
    p.add_argument('--flip-prob', type=float, default=0.5, help='(strong aug only; the basic pipeline uses 0.75)')
    p.add_argument('--stage2-epochs', type=int, default=15,
                   help='last epochs without mosaic (YOLOX no_aug_epochs): light augmentation, extra L1 loss, '
                        'LR at its minimum')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tensorboard', action='store_true', help='write TensorBoard events under --work-dir')
    args = p.parse_args()
    args.data = _resolve_path(args.data, ROOT)
    args.extra_train = [
        _resolve_path(path, ROOT / 'data') if not Path(path).exists() else path
        for path in args.extra_train
    ]
    if args.init is None:
        args.init = MODELS[args.model][1]
    if args.init.lower() != 'none':
        args.init = _resolve_path(args.init, CHECKPOINTS)
        if not args.resume and not Path(args.init).exists():
            p.error(f'--init {args.init} not found (see README for the YOLOX_OBB weights, '
                    'or pass --init none to train from scratch)')
    else:
        args.init = None
    args.work_dir = _resolve_path(args.work_dir, EXPERIMENTS)

    torch.manual_seed(args.seed)
    if sys.platform == 'darwin' and args.workers > 0:
        # fork plus Core Graphics/Metal in a worker can segfault on macOS.
        torch.multiprocessing.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
    if args.persistent_workers is None:
        args.persistent_workers = sys.platform != 'darwin'
    import numpy as np
    np.random.seed(args.seed)
    device = pick_device(args.device)
    lr = args.lr if args.lr else 0.01 / 64 * args.batch * args.accumulate
    print(f'device {device}, lr {lr:.2e}, batch {args.batch} x {args.accumulate}, img {args.img_size}')

    aug_kw = dict(mosaic_prob=args.mosaic_prob, mixup_prob=args.mixup_prob, resize_range=args.resize_range,
                  stage2_resize_range=args.stage2_resize_range, flip_prob=args.flip_prob,
                  rotate_prob=0.0 if args.no_rotate else 0.5,
                  mosaic_max_cached=args.mosaic_cache, mixup_max_cached=args.mixup_cache)
    train_ds = YoloObbDataset(args.data, 'train', args.img_size, train=True, split_dir=args.split_dir,
                              rotate_prob=0.0 if args.no_rotate else 0.5, extra_dirs=args.extra_train,
                              strong_aug=StrongAug(**aug_kw) if args.strong_aug else None)
    stage2_loader = None
    stage2_epoch = max(0, args.epochs - args.stage2_epochs)
    if args.strong_aug:
        stage2_ds = YoloObbDataset(args.data, 'train', args.img_size, train=True, split_dir=args.split_dir,
                                   extra_dirs=args.extra_train, strong_aug=StrongAug(stage2=True, **aug_kw))
    val_ds = YoloObbDataset(args.data, 'valid', args.img_size, train=False, split_dir=args.split_dir)
    print(f'train {len(train_ds)} images, val {len(val_ds)} images')
    pin = device.type == 'cuda'
    persistent = args.workers > 0 and args.persistent_workers
    train_loader = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate,
                              drop_last=True, pin_memory=pin, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, max(1, args.batch // 2), shuffle=False, num_workers=0, collate_fn=collate,
                            pin_memory=pin)
    if args.strong_aug:
        stage2_loader = DataLoader(stage2_ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate,
                                   drop_last=True, pin_memory=pin, persistent_workers=persistent)

    model = build_model(args.model, args.classes)
    if args.init and not args.resume:
        load_pretrained(model, args.init)

    os.makedirs(args.work_dir, exist_ok=True)
    run_args = vars(args).copy()
    # Keep the effective evaluation protocol in the run record. The low score
    # threshold is intentionally fixed for AP; the report threshold is for the
    # operating-point precision/recall/F1 metrics.
    run_args.update(score_thr=0.05, report_score_thr=args.report_score_thr, lr=lr)
    with open(osp.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(run_args, f, indent=2)

    train(model, train_loader, val_loader, device, args.work_dir, args.epochs, lr, args.weight_decay,
          args.momentum, args.warmup_epochs, args.min_lr_ratio, args.val_interval, use_ema=not args.no_ema,
          grad_clip=args.grad_clip, resume=args.resume,
          val_kwargs=dict(nms_iou=args.nms_iou, report_score_thr=args.report_score_thr),
          stage2_loader=stage2_loader, stage2_epoch=stage2_epoch,
          accumulate=args.accumulate, tensorboard=args.tensorboard)


if __name__ == '__main__':
    main()
